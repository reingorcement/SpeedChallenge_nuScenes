from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from purevision_v5.model import EXPERIMENTS, VelocityModel
from purevision_v5.rgb import FrozenRGB, preprocess_rgb


torch.set_num_threads(2)
CHECKPOINT = Path(__file__).resolve().parents[2] / "PureVision_V3/checkpoints/pretrained/direct_seed_42/best.pth"


def inputs(batch_size=2):
    generator = torch.Generator().manual_seed(17)
    return dict(
        descriptor=torch.randn(batch_size, 384, generator=generator),
        flows=torch.randn(batch_size, 4, 2, 72, 128, generator=generator),
        valid=torch.ones(batch_size, 4, 1, 72, 128, dtype=torch.bool),
        fb_mask=torch.ones(batch_size, 4, 1, 72, 128, dtype=torch.bool),
        pair_dt=torch.full((batch_size, 4), 1 / 12),
    )


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_all_experiments_forward_backward(experiment):
    torch.manual_seed(42)
    model = VelocityModel(experiment)
    output = model(inputs())
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    loss = (output - torch.tensor([[3.0, -.2], [4.0, .1]])).square().mean()
    loss.backward()
    assert model.head[0].weight.grad.abs().sum() > 0
    if experiment in {"R02", "R03", "R04", "R05"}:
        assert model.flow_encoder.temporal[0].weight.grad.abs().sum() > 0


def test_reliability_models_have_identical_shapes_and_all_reliable_outputs():
    models = [VelocityModel(experiment) for experiment in ("R03", "R04", "R05")]
    state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(state, strict=True)
    batch = inputs()
    outputs = [model(batch) for model in models]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)


def test_hard_filter_only_changes_flow_channels():
    hint_model, hard_model = VelocityModel("R04"), VelocityModel("R05")
    hard_model.load_state_dict(hint_model.state_dict())
    batch = inputs(1)
    batch["fb_mask"][:, :, :, 20:40, 30:60] = False
    captured = {}
    handles = [model.flow_encoder.register_forward_pre_hook(
        lambda module, args, key=key: captured.__setitem__(key, args[0].detach().clone()))
        for key, model in (("hint", hint_model), ("hard", hard_model))]
    hint_model(batch)
    hard_model(batch)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(captured["hint"][:, :, 2:], captured["hard"][:, :, 2:], rtol=0, atol=0)
    expected = captured["hint"][:, :, :2] * batch["fb_mask"]
    torch.testing.assert_close(captured["hard"][:, :, :2], expected, rtol=0, atol=0)
    assert torch.count_nonzero(captured["hint"][:, :, :2] - expected) > 0


def test_nan_inf_flow_is_zeroed_before_arithmetic_and_validity_updated():
    model = VelocityModel("R03")
    batch = inputs(1)
    batch["flows"][0, 0, 0, 3, 4] = float("nan")
    batch["flows"][0, 1, 1, 7, 8] = float("inf")
    clean = {key: value.clone() for key, value in batch.items()}
    for pair, row, col in ((0, 3, 4), (1, 7, 8)):
        clean["flows"][0, pair, :, row, col] = 0
        clean["valid"][0, pair, :, row, col] = False
    torch.testing.assert_close(model(batch), model(clean), rtol=0, atol=0)


@pytest.mark.parametrize("field", ["target", "velocity", "pose", "angular_rate_camera", "chassis", "scene_name"])
def test_inference_whitelist_rejects_nonvisual_and_metadata_fields(field):
    batch = inputs(1)
    batch[field] = torch.zeros(1)
    with pytest.raises(ValueError, match="Forbidden inference fields"):
        VelocityModel()(batch)


@pytest.mark.parametrize("field,mutation", [
    ("pair_dt", lambda value: value.zero_()),
    ("descriptor", lambda value: value.fill_(float("nan"))),
    ("flows", lambda value: value[:, :3]),
    ("valid", lambda value: value.float().fill_(.5)),
    ("fb_mask", lambda value: value.float().fill_(float("nan"))),
])
def test_bad_shapes_and_required_values_rejected(field, mutation):
    batch = inputs(1)
    batch[field] = mutation(batch[field])
    with pytest.raises(ValueError):
        VelocityModel("R04")(batch)


def test_temporal_order_and_signed_magnitude_preserved_at_flow_encoder():
    model = VelocityModel("R01")
    batch = inputs(1)
    for pair in range(4):
        batch["flows"][:, pair, 0] = pair - 2.5
        batch["flows"][:, pair, 1] = pair + .25
    captured = []
    handle = model.flow_encoder.projection[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    model(batch)
    handle.remove()
    grid = captured[0].reshape(1, 4, 3, 4, 8)
    for pair in range(4):
        assert torch.all(grid[:, pair, 0] == pair - 2.5)
        assert torch.all(grid[:, pair, 1] == pair + .25)


def test_checkpoint_roundtrip(tmp_path):
    model = VelocityModel("R04")
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    loaded = VelocityModel("R04")
    loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    batch = inputs(1)
    torch.testing.assert_close(model(batch), loaded(batch), rtol=0, atol=0)


@pytest.fixture(scope="module")
def frozen_rgb():
    return FrozenRGB(CHECKPOINT)


def test_frozen_rgb_strict_selective_load_and_eval(frozen_rgb):
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    selected = {key: value for key, value in source.items()
                if key.startswith(("backbone.", "region_projection."))}
    assert frozen_rgb.state_dict().keys() == selected.keys()
    for key, value in frozen_rgb.state_dict().items():
        torch.testing.assert_close(value, selected[key], rtol=0, atol=0)
    nn.Sequential(frozen_rgb).train()
    assert not any(module.training for module in frozen_rgb.modules())
    assert not any(parameter.requires_grad for parameter in frozen_rgb.parameters())
    assert frozen_rgb.checkpoint_sha256 == "1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82"


def test_frozen_rgb_regional_order(frozen_rgb, monkeypatch):
    maps = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3).expand(1, 256, 2, 2, 3).clone()
    maps[:, :, 1] += 10
    monkeypatch.setattr(frozen_rgb.backbone, "forward_feature_map", lambda images: maps)
    captured = []
    handle = frozen_rgb.region_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    output = frozen_rgb(torch.zeros(1, 5, 3, 72, 128))
    handle.remove()
    expected_tokens = torch.tensor([8., 9., 10., 5., 6., 7.]).reshape(1, 6, 1).expand(1, 6, 256)
    assert output.shape == (1, 384)
    torch.testing.assert_close(captured[0], expected_tokens, rtol=0, atol=0)


def test_rgb_preprocess_matches_original_torchvision_transform():
    from torchvision import transforms
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (117, 209, 3), dtype=np.uint8))
    original = transforms.Compose([transforms.Resize((72, 128)),
                                  transforms.Lambda(lambda img: img.convert("RGB")),
                                  transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    torch.testing.assert_close(preprocess_rgb(image), original(image), rtol=0, atol=0)


@pytest.mark.parametrize("images", [torch.zeros(1, 4, 3, 72, 128),
                                   torch.full((1, 5, 3, 72, 128), 255.),
                                   torch.full((1, 5, 3, 72, 128), float("nan"))])
def test_rgb_input_contract(frozen_rgb, images):
    with pytest.raises(ValueError):
        frozen_rgb(images)
