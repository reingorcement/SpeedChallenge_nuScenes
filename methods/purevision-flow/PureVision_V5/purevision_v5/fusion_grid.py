"""Six frozen-feature readouts with the historical C initialization preserved.

The historical A/B/C factory remains the sole source of joint projection,
Transformer and velocity-head initialization. Grid-specific position buffers
are constructed separately and are never copied across grids.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib

import torch
from torch import nn

from .fusion_abc import build_triplet, sinusoidal_axis
from .model import DT_SCALE_SECONDS


GRID_VARIANTS = {
    "J23": {"grid": (2, 3), "organization": "joint", "token_count": 24},
    "S23": {"grid": (2, 3), "organization": "separate", "token_count": 48},
    "J46": {"grid": (4, 6), "organization": "joint", "token_count": 96},
    "S46": {"grid": (4, 6), "organization": "separate", "token_count": 192},
    "F23": {"grid": (2, 3), "organization": "flow_only", "token_count": 24},
    "R23": {"grid": (2, 3), "organization": "rgb_only", "token_count": 24},
}
EXPECTED_PARAMETERS = {"J23": 101250, "J46": 101250, "S23": 101570,
                       "S46": 101570, "F23": 84930, "R23": 97218}
INITIALIZATION_POLICY = "historical_build_triplet_C_explicit_six_group_parameter_copy_v1"


def _grid(grid):
    value = tuple(grid)
    if value not in {(2, 3), (4, 6)}:
        raise ValueError("Only the fixed 2x3 and 4x6 grids are allowed")
    return value


def grid_feature_shapes(grid):
    rows, cols = _grid(grid)
    return {"F_sp": (4, rows * cols, 32), "F_tm": (4, 32),
            "R": (4, rows * cols, 256), "dt": (4,)}


def validate_grid_features(features, grid, *, check_finite=True):
    shapes = grid_feature_shapes(grid)
    if not isinstance(features, Mapping) or set(features) != set(shapes):
        raise ValueError(f"Fusion input must contain only {sorted(shapes)}")
    dt = features["dt"]
    if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or len(dt) < 1:
        raise ValueError("Cached dt must be a nonempty [B,4] tensor")
    batch = len(dt)
    for key, shape in shapes.items():
        value = features[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, *shape)
                or value.dtype != torch.float32 or value.device != dt.device):
            raise ValueError(f"Invalid {key}: expected FP32 {(batch, *shape)} on {dt.device}")
        if check_finite and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite cached {key}")
    if check_finite and not bool((dt > 0).all()):
        raise ValueError("Cached dt must remain positive raw seconds")
    return batch


def grid_position_encoding(grid):
    """Historical integer-index time32/row16/column16 sine/cosine rule."""
    rows, cols = _grid(grid)
    time = sinusoidal_axis(4, 32)[:, None, None, :].expand(4, rows, cols, 32)
    row = sinusoidal_axis(rows, 16)[None, :, None, :].expand(4, rows, cols, 16)
    col = sinusoidal_axis(cols, 16)[None, None, :, :].expand(4, rows, cols, 16)
    return torch.cat((time, row, col), dim=-1).reshape(4 * rows * cols, 64).contiguous()


class FusionGrid(nn.Module):
    def __init__(self, group):
        super().__init__()
        if group not in GRID_VARIANTS:
            raise ValueError(f"Unknown fusion group: {group}")
        self.group = group
        self.grid = GRID_VARIANTS[group]["grid"]
        self.organization = GRID_VARIANTS[group]["organization"]
        self.token_count = GRID_VARIANTS[group]["token_count"]
        if self.organization == "joint":
            self.input_projection = nn.Linear(320, 64, bias=True)
            self.input_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
        else:
            if self.organization in {"separate", "flow_only"}:
                self.flow_projection = nn.Linear(64, 64, bias=True)
                self.flow_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
                self.flow_modality = nn.Parameter(torch.zeros(64))
            if self.organization in {"separate", "rgb_only"}:
                self.rgb_projection = nn.Linear(256, 64, bias=True)
                self.rgb_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
                self.rgb_modality = nn.Parameter(torch.zeros(64))
        self.register_buffer("position_encoding", grid_position_encoding(self.grid))
        self.head = nn.Sequential(nn.Linear(68, 192, bias=True), nn.GELU(),
                                  nn.Linear(192, 2, bias=True))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                dropout=0.0, activation="gelu", layer_norm_eps=1e-5,
                batch_first=True, norm_first=False, bias=True)
            for _ in range(2)
        ])

    def input_tokens(self, features):
        # The immutable cache/preflight performs numerical validation once.
        # Shape/dtype/device validation here avoids device synchronization at
        # every training step while retaining the exact historical arithmetic.
        batch = validate_grid_features(features, self.grid, check_finite=False)
        regions = self.grid[0] * self.grid[1]
        context = features["F_tm"].unsqueeze(2).expand(-1, -1, regions, -1)
        if self.organization == "joint":
            raw = torch.cat((features["F_sp"], context, features["R"]), dim=-1)
            raw = raw.reshape(batch, 4 * regions, 320)
            return self.input_norm(self.input_projection(raw)) + self.position_encoding
        tokens = []
        if self.organization in {"separate", "flow_only"}:
            raw = torch.cat((features["F_sp"], context), dim=-1).reshape(batch, 4 * regions, 64)
            tokens.append(self.flow_norm(self.flow_projection(raw))
                          + self.position_encoding + self.flow_modality)
        if self.organization in {"separate", "rgb_only"}:
            raw = features["R"].reshape(batch, 4 * regions, 256)
            tokens.append(self.rgb_norm(self.rgb_projection(raw))
                          + self.position_encoding + self.rgb_modality)
        return torch.cat(tokens, dim=1) if len(tokens) == 2 else tokens[0]

    def tokens(self, features):
        tokens = self.input_tokens(features)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens

    def forward(self, features):
        tokens = self.tokens(features)
        return self.head(torch.cat((tokens.mean(dim=1), features["dt"] / DT_SCALE_SECONDS), dim=1))


@torch.no_grad()
def _copy_parameters(target, source, prefixes=None):
    """Copy parameter values only; never alias storage or replace position buffers."""
    destination = dict(target.named_parameters())
    selected = {name: parameter for name, parameter in source.named_parameters()
                if prefixes is None or any(name == prefix or name.startswith(prefix + ".")
                                           for prefix in prefixes)}
    for name, parameter in selected.items():
        if name not in destination or destination[name].shape != parameter.shape:
            raise RuntimeError(f"Incompatible initialization parameter: {name}")
        destination[name].copy_(parameter)


def build_six(seed=42):
    """Fresh independent models without consuming the caller's data-order RNG.

    New S23 projections use PyTorch Linear.reset_parameters defaults in an
    isolated seed context. Its two modality vectors use a separate isolated
    seed context, Normal(0, .02), flow first then RGB. All other groups receive
    explicit copies; each retained parameter belongs to its own model.
    """
    historical = build_triplet(seed)["C"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        # S23 comes first so projection initialization has a simple recorded
        # consumption order: flow Linear(64,64), then RGB Linear(256,64).
        models = {"S23": FusionGrid("S23")}
        models.update({name: FusionGrid(name) for name in GRID_VARIANTS if name != "S23"})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        nn.init.normal_(models["S23"].flow_modality, mean=0., std=.02)
        nn.init.normal_(models["S23"].rgb_modality, mean=0., std=.02)
    _copy_parameters(models["J23"], historical)
    _copy_parameters(models["J46"], models["J23"])
    _copy_parameters(models["S23"], historical, ("layers", "head"))
    _copy_parameters(models["S46"], models["S23"])
    for group, branch in (("F23", "flow"), ("R23", "rgb")):
        _copy_parameters(models[group], models["S23"],
                         ("layers", "head", f"{branch}_projection", f"{branch}_norm", f"{branch}_modality"))
    for name, model in models.items():
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if count != EXPECTED_PARAMETERS[name]:
            raise RuntimeError(f"Fusion {name} parameter count {count} != {EXPECTED_PARAMETERS[name]}")
    return {name: models[name] for name in GRID_VARIANTS}


def _parameter_comparison(left, right, prefixes=None):
    source, target = dict(left.named_parameters()), dict(right.named_parameters())
    keys = [key for key in source if prefixes is None or any(
        key == prefix or key.startswith(prefix + ".") for prefix in prefixes)]
    return {"parameter_count": len(keys),
            "exact_equal": all(key in target and torch.equal(source[key].detach().cpu(), target[key].detach().cpu())
                               for key in keys),
            "independent_storage": all(key in target and source[key].data_ptr() != target[key].data_ptr()
                                       for key in keys)}


def _parameter_sha256(model):
    digest = hashlib.sha256()
    for key, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_evidence(models, fixed_features_by_grid=None, device="cpu", seed=42):
    """Assert one compact, disposable model preflight and return JSON evidence.

    ``fixed_features_by_grid`` maps (2,3)/(4,6) to small actual cached batches.
    If a grid is not ready, callers may run its numerical smoke with synthetic
    input explicitly labelled as such; no cache equivalence is inferred here.
    """
    if set(models) != set(GRID_VARIANTS):
        raise ValueError("Initialization evidence requires all six groups")
    historical = build_triplet(seed)["C"]
    comparisons = {
        "J23_vs_historical_C": _parameter_comparison(models["J23"], historical),
        "J46_vs_J23": _parameter_comparison(models["J46"], models["J23"]),
        "S46_vs_S23": _parameter_comparison(models["S46"], models["S23"]),
        "F23_vs_S23_flow": _parameter_comparison(models["F23"], models["S23"]),
        "R23_vs_S23_rgb": _parameter_comparison(models["R23"], models["S23"]),
    }
    comparisons.update({f"{name}_common_vs_J23": _parameter_comparison(model, models["J23"], ("layers", "head"))
                        for name, model in models.items() if name != "J23"})
    if not all(item["exact_equal"] and item["independent_storage"] for item in comparisons.values()):
        raise RuntimeError(f"Six-group initialization mismatch: {comparisons}")
    stores = [parameter.data_ptr() for model in models.values() for parameter in model.parameters()]
    if len(set(stores)) != len(stores):
        raise RuntimeError("A parameter storage is shared across the six models")
    positions = {}
    for name, model in models.items():
        if not torch.equal(model.position_encoding.cpu(), grid_position_encoding(model.grid)):
            raise RuntimeError(f"Incorrect position encoding: {name}")
        if torch.equal(model.layers[0].self_attn.in_proj_weight, model.layers[1].self_attn.in_proj_weight):
            raise RuntimeError(f"Transformer layers were cloned instead of independently initialized: {name}")
        positions[name] = {"shape": list(model.position_encoding.shape), "historical_integer_axis_rule": True}
    evidence = dict(policy=INITIALIZATION_POLICY, seed=seed, comparisons=comparisons,
        independent_parameter_storage=True, position_encoding=positions,
        parameters={name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        parameter_sha256={name: _parameter_sha256(model) for name, model in models.items()},
        new_projection_initialization=f"isolated seed{seed} PyTorch Linear.reset_parameters defaults; S23 flow then RGB",
        modality_initialization=f"separate isolated seed{seed} Normal(0,0.02); flow64 then RGB64",
        transformer_layers_independently_initialized=True,
        token_order="time,row,column within a modality; separated flow tokens then RGB tokens",
        position_axes={"time": 32, "row": 16, "column": 16, "base": 10000, "integer_start": 0})
    if fixed_features_by_grid is not None:
        evidence["forward"] = {}
        historical = historical.to(device).float().eval()
        for name, original in models.items():
            if original.grid not in fixed_features_by_grid:
                continue
            values = {key: value.to(device=device, dtype=torch.float32)
                      for key, value in fixed_features_by_grid[original.grid].items()}
            validate_grid_features(values, original.grid)
            model = copy.deepcopy(original).to(device).float().eval()
            with torch.no_grad():
                tokens = model.input_tokens(values)
                prediction = model(values)
            if tokens.shape != (len(values["dt"]), model.token_count, 64) or not torch.isfinite(prediction).all():
                raise RuntimeError(f"Invalid forward output for {name}")
            record = dict(tokens=list(tokens.shape), prediction=list(prediction.shape), finite=True)
            if name == "J23":
                with torch.no_grad():
                    reference = historical(values)
                torch.testing.assert_close(prediction, reference, atol=1e-6, rtol=1e-5)
                record["historical_C"] = dict(atol=1e-6, rtol=1e-5,
                    max_absolute_difference=float((prediction-reference).abs().max()),
                    exact_equal=torch.equal(prediction, reference), same_device=str(device), dtype="float32", mode="eval")
            evidence["forward"][name] = record
    return evidence
