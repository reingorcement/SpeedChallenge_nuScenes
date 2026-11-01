import io

import torch
from torch import nn
from torch.nn import functional as F

from purevision_v5.fusion_abc import fixed_position_encoding
from purevision_v5.fusion_grid import (
    EXPECTED_PARAMETERS, GRID_VARIANTS, build_six, grid_feature_shapes,
    grid_position_encoding, initialization_evidence,
)


def feature_batch(grid, count=2):
    generator = torch.Generator().manual_seed(197)
    values = {key: torch.randn(count, *shape, generator=generator)
              for key, shape in grid_feature_shapes(grid).items()}
    values['dt'].fill_(1 / 12)
    return values


def test_initialization_preserves_history_storage_and_caller_rng():
    torch.set_num_threads(2)
    before = torch.random.get_rng_state().clone()
    models = build_six()
    assert torch.equal(torch.random.get_rng_state(), before)
    evidence = initialization_evidence(models, {grid: feature_batch(grid) for grid in ((2, 3), (4, 6))})
    assert evidence['parameters'] == EXPECTED_PARAMETERS
    assert evidence['forward']['J23']['historical_C']['exact_equal']
    assert all(evidence['forward'][name]['tokens'][1] == spec['token_count']
               for name, spec in GRID_VARIANTS.items())


def test_new_projections_and_modalities_have_recorded_seed42_initialization():
    model = build_six()['S23']
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        flow, rgb = nn.Linear(64, 64), nn.Linear(256, 64)
    assert all(torch.equal(value, flow.state_dict()[key]) for key, value in model.flow_projection.state_dict().items())
    assert all(torch.equal(value, rgb.state_dict()[key]) for key, value in model.rgb_projection.state_dict().items())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        expected = torch.empty(2, 64).normal_(0, .02)
    assert torch.equal(model.flow_modality, expected[0])
    assert torch.equal(model.rgb_modality, expected[1])


def test_position_axes_extend_integer_indices_and_sep_order_is_flow_then_rgb():
    assert torch.equal(grid_position_encoding((2, 3)), fixed_position_encoding())
    small = grid_position_encoding((2, 3)).reshape(4, 2, 3, 64)
    large = grid_position_encoding((4, 6)).reshape(4, 4, 6, 64)
    assert torch.equal(small, large[:, :2, :3])
    models, values = build_six(), feature_batch((2, 3))
    tokens = models['S23'].input_tokens(values)
    assert torch.equal(tokens[:, :24], models['F23'].input_tokens(values))
    assert torch.equal(tokens[:, 24:], models['R23'].input_tokens(values))
    assert not any('rgb' in key for key, _ in models['F23'].named_parameters())
    assert not any('flow' in key for key, _ in models['R23'].named_parameters())


def test_every_group_has_finite_gradients_and_exact_save_load_forward():
    models = build_six()
    scale = torch.tensor([5., 1.])
    targets = torch.tensor([[2., .03], [4., -.04]])
    for name, model in models.items():
        values = feature_batch(model.grid)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        prediction = model(values)
        loss = F.smooth_l1_loss(prediction / scale, targets / scale, beta=1, reduction='mean')
        axes = F.smooth_l1_loss(prediction.detach() / scale, targets / scale, beta=1, reduction='none').mean(0)
        torch.testing.assert_close(axes.mean(), loss.detach(), atol=1e-7, rtol=1e-6)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_six()[name]
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        with torch.no_grad():
            assert torch.equal(model.eval()(values), restored.eval()(values))
