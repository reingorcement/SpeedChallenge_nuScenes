import hashlib
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from purevision_v5.fusion_abc import (
    CACHE_FEATURE_SHAPES, EXPECTED_PARAMETERS, FrozenFusionExtractor,
    adaptive_pool_bounds, build_triplet, common_initialization_evidence,
    fixed_position_encoding, spatial_tokens,
)
from purevision_v5.common import object_hash


def features(batch=2):
    generator = torch.Generator().manual_seed(203)
    output = {key: torch.randn(batch, *shape, generator=generator)
              for key, shape in CACHE_FEATURE_SHAPES.items()}
    output["dt"].fill_(1 / 12)
    return output


def test_parameter_counts_and_module_contracts():
    models = build_triplet()
    for name, model in models.items():
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS[name]
        assert tuple(model(features()).shape) == (2, 2)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is not None
            if isinstance(module, nn.LayerNorm):
                assert module.elementwise_affine and module.eps == 1e-5
            if isinstance(module, nn.Dropout):
                assert module.p == 0
    assert len(models["C"].layers) == 2
    assert all(not layer.norm_first for layer in models["C"].layers)


def test_common_initialization_exact_and_no_shared_parameters():
    models = build_triplet()
    evidence = common_initialization_evidence(models)
    assert all(evidence[key]["exact_equal"] and evidence[key]["independent_storage"]
               for key in ("input_projection", "input_norm", "head"))
    assert evidence["position_encoding_exact_equal"]
    assert evidence["transformer_layers_independent_storage"]
    assert evidence["transformer_attention_initial_weights_differ"]
    again = build_triplet()
    for name in models:
        assert all(torch.equal(value, again[name].state_dict()[key])
                   for key, value in models[name].state_dict().items())


def test_fixed_position_encoding_order_axes_and_values():
    code = fixed_position_encoding().reshape(4, 2, 3, 64)
    assert torch.equal(code[0, 0, 0, 0::2], torch.zeros(32))
    assert torch.equal(code[0, 0, 0, 1::2], torch.ones(32))
    assert torch.equal(code[0, 0, 0, 32:], code[1, 0, 0, 32:])
    assert torch.equal(code[0, 0, 0, :32], code[0, 1, 2, :32])
    assert code[1, 0, 0, 0] == pytest.approx(torch.sin(torch.tensor(1.)).item())
    assert code[0, 1, 0, 32] == pytest.approx(torch.sin(torch.tensor(1.)).item())
    assert code[0, 0, 2, 48] == pytest.approx(torch.sin(torch.tensor(2.)).item())


def test_spatial_tokens_never_mix_time_and_never_flip_rows():
    grid = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3)
    time = 100 * torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
    output = spatial_tokens(grid + time)
    assert output.shape == (1, 5, 6, 1)
    assert torch.equal(output[0, :, :, 0], torch.arange(6).float()[None] + 100 * torch.arange(5).float()[:, None])
    assert adaptive_pool_bounds(9, 2) == [[0, 5], [4, 9]]
    assert adaptive_pool_bounds(16, 3) == [[0, 6], [5, 11], [10, 16]]
    assert adaptive_pool_bounds(128, 3) == [[0, 43], [42, 86], [85, 128]]


def test_a_explicit_flatten_orders_and_seconds_normalized_once():
    model = build_triplet()["A"]
    data = features(1)
    data["F_sp"] = torch.arange(4 * 6 * 32).float().reshape(1, 4, 6, 32)
    data["F_tm"] = torch.arange(4 * 32).float().reshape(1, 4, 32)
    observed = {}
    def capture(name):
        def hook(module, inputs):
            observed[name] = inputs[0].detach().clone()
        return hook
    h1 = model.flow_projection.register_forward_pre_hook(capture("flow"))
    h2 = model.head.register_forward_pre_hook(capture("head"))
    model(data)
    h1.remove(); h2.remove()
    assert torch.equal(observed["flow"][:, :192], data["F_sp"].mean(1).transpose(1, 2).reshape(1, 192))
    assert torch.equal(observed["flow"][:, 192:], data["F_tm"].transpose(1, 2).reshape(1, 128))
    assert torch.equal(observed["head"][:, -4:], torch.ones(1, 4))


def test_b_no_token_exchange_and_c_cross_token_exchange():
    models = build_triplet()
    base = features(1)
    changed = {key: value.clone() for key, value in base.items()}
    changed["F_sp"][0, 0, 0] += torch.linspace(-5, 5, 32)
    for name in ("B", "C"):
        model = models[name].eval()
        with torch.no_grad():
            delta = (model.tokens(base) - model.tokens(changed)).abs().amax(dim=-1)[0]
        assert delta[0] > 0
        if name == "B":
            assert torch.equal(delta[1:], torch.zeros(23))
        else:
            assert bool((delta[1:] > 1e-6).all())


def test_finite_updates_and_checkpoint_roundtrip(tmp_path):
    data = features(2)
    target = torch.tensor([[2., .01], [3., -.02]])
    for name, model in build_triplet().items():
        before = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(model(data) / torch.tensor([5., 1.]), target / torch.tensor([5., 1.]))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
        path = tmp_path / f"{name}.pt"
        torch.save(model.state_dict(), path)
        other = build_triplet()[name]
        other.load_state_dict(torch.load(path, weights_only=True), strict=True)
        assert torch.equal(model(data), other(data))


class FakeStem(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (9, 16)).expand(-1, 32, -1, -1)


class FakeSFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm3d(32)
    def forward(self, x):
        self.seen = x
        return self.bn(x) + 3


class FakeTemporal(nn.Module):
    def forward(self, x):
        self.seen = x
        return x.mean(dim=(-1, -2), keepdim=True) + 7


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_alpha = nn.Parameter(torch.tensor(.123))
        self.appearance_stem = nn.Conv3d(3, 3, 1)
        self.bn = nn.BatchNorm3d(3)
    def forward_feature_map(self, images):
        return self.bn(images.transpose(1, 2)).mean(1, keepdim=True).expand(-1, 256, -1, -1, -1)


class FakeSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_encoder = nn.Module()
        self.flow_encoder.spatial = FakeStem()
        self.flow_encoder.sfe = FakeSFE()
        self.flow_encoder.temporal = FakeTemporal()
        self.rgb_encoder = nn.Module()
        self.rgb_encoder.backbone = FakeBackbone()
        self.head = nn.Linear(452, 2)


def fake_extractor(tmp_path):
    source = FakeSource()
    path = tmp_path / "source.pt"
    config = {"experiment": "R03", "encoder_variant": "flex", "rgb_policy": "finetune",
              "rgb_variant": "appearance_difference"}
    contract = {"config": config}
    torch.save({"schema": "v5_training_state_v1", "contract": contract, "contract_hash": object_hash(contract),
                "config": config, "model": source.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with patch("purevision_v5.fusion_abc.build_model", return_value=FakeSource()):
        return FrozenFusionExtractor(path, digest)


def test_extractor_immutable_buffers_parallel_branches_and_time_selection(tmp_path):
    extractor = fake_extractor(tmp_path)
    before = {key: value.clone() for key, value in extractor.state_dict().items()}
    inputs = {"images": torch.arange(5).float().reshape(1, 5, 1, 1, 1).expand(1, 5, 3, 72, 128) / 5,
              "flows": torch.randn(1, 4, 2, 72, 128), "valid": torch.ones(1, 4, 1, 72, 128),
              "pair_dt": torch.full((1, 4), 1 / 12)}
    inputs["flows"][0, 0, 0, 0, 0] = float("nan")
    outer = nn.Sequential(extractor).train()
    assert all(not module.training for module in extractor.modules())
    output = outer(inputs)
    assert all(tuple(output[key].shape) == (1, *shape) for key, shape in CACHE_FEATURE_SHAPES.items())
    assert all(not value.requires_grad for value in output.values())
    assert all(not p.requires_grad for p in extractor.parameters())
    assert all(torch.equal(value, before[key]) for key, value in extractor.state_dict().items())
    assert extractor.source.flow_encoder.sfe.seen is extractor.source.flow_encoder.temporal.seen
    assert output["R"][0, :, 0, 0].tolist() == pytest.approx([0, .2, .4, .6], abs=2e-5)
    assert torch.equal(output["dt"], inputs["pair_dt"])
    assert extractor.provenance["strict_load"] == {"missing_keys": [], "unexpected_keys": []}
    assert extractor.provenance["appearance_alpha"] == pytest.approx(.123)
    with pytest.raises(ValueError, match="accepts only"):
        extractor({**inputs, "target": torch.zeros(1, 2)})


def test_bad_hash_rejected_before_loading(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenFusionExtractor(path)
