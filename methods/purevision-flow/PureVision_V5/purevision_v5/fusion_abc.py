"""Frozen R03 feature taps and the three explicitly specified fusion readouts.

No V3 initialization is performed here: the entire trained R03 model is loaded
strictly before its two feature extractors are tapped. Labels and identities
belong to the cache's sample table, never to these inference-only mappings.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import object_hash
from .model import DT_SCALE_SECONDS, FLOW_SHAPE, MASK_SHAPE, _binary_mask, _finite_tensor, build_model
from .rgb import RGB_PREPROCESS_VERSION, RGB_SHAPE


SOURCE_SHA = "ab25fd6c4a7cd14a99a189f56e6b47913ed0f946a60cff2d0c9102479bc30ff8"
CACHE_FEATURE_SHAPES = {"F_sp": (4, 6, 32), "F_tm": (4, 32), "R": (4, 6, 256), "dt": (4,)}
EXPECTED_PARAMETERS = {"A": 124354, "B": 100738, "C": 101250}
FEATURE_TAP_VERSION = "frozen_r03_appearance_flex_spatial2x3_rgb_t0to3_fp32_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adaptive_pool_bounds(length, bins):
    """PyTorch adaptive average pool intervals, zero-based and end-exclusive.

    Uneven input sizes cause overlap; the two modalities share a logical grid,
    not identical pixels/receptive fields.
    """
    return [[math.floor(i * length / bins), math.ceil((i + 1) * length / bins)]
            for i in range(bins)]


def spatial_tokens(feature_map):
    """B,C,T,H,W -> B,T,6,C, pooling only H,W; row-major, without flips."""
    if feature_map.ndim != 5:
        raise ValueError("Feature map must be B,C,T,H,W")
    batch, channels, steps, height, width = feature_map.shape
    slices = feature_map.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
    pooled = F.adaptive_avg_pool2d(slices, (2, 3))
    return pooled.reshape(batch, steps, channels, 6).permute(0, 1, 3, 2).contiguous()


class FrozenFusionExtractor(nn.Module):
    """Immutable trained feature extraction, including appearance alpha and BN.

    ``train()`` on this object or its enclosing module cannot enable training.
    Both paths run without autograd and return FP32 tensors. Flow spatial and
    temporal branches are parallel children of the same stem output.
    """

    def __init__(self, checkpoint, expected_sha256=SOURCE_SHA):
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        digest = file_sha256(checkpoint)
        if digest != expected_sha256:
            raise ValueError(f"Frozen source checkpoint hash mismatch: {digest}")
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (package.get("schema") != "v5_training_state_v1"
                or package.get("contract_hash") != object_hash(package.get("contract", {}))
                or package.get("config") != package.get("contract", {}).get("config")):
            raise ValueError("Frozen extraction requires an intact V5 training checkpoint contract")
        config = package["config"]
        required = {"experiment": "R03", "encoder_variant": "flex",
                    "rgb_policy": "finetune", "rgb_variant": "appearance_difference"}
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("Source must be the complete trained appearance+FLEX R03 model")
        self.source = build_model(config)
        loaded = self.source.load_state_dict(package["model"], strict=True)
        self.source.requires_grad_(False)
        self.train(False)
        from .training import model_state_sha256
        self.provenance = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "source_run_id": package.get("run_id"), "source_code": package.get("code"),
            "source_config": copy.deepcopy(config), "source_epoch": package.get("epoch"),
            "source_initialization": copy.deepcopy(package.get("initialization")),
            "source_contract_hash": package["contract_hash"],
            "source_model_state_sha256": model_state_sha256(self.source),
            "strict_load": {"missing_keys": list(loaded.missing_keys),
                            "unexpected_keys": list(loaded.unexpected_keys)},
            "feature_tap_version": FEATURE_TAP_VERSION,
            "tap_source_sha256": file_sha256(__file__),
            "preprocessing": {"rgb": RGB_PREPROCESS_VERSION,
                              "flow": "existing_signed_model_view_pixels_uv_valid_hint_valid",
                              "dt": "raw_seconds"},
            "shapes": {key: ["N", *shape] for key, shape in CACHE_FEATURE_SHAPES.items()},
            "dtype": "float32", "region_order": "row_then_col_top_to_bottom_left_to_right",
            "token_order": "t_then_row_then_col",
            "rgb_time_positions": [0, 1, 2, 3], "rgb_backbone_input_steps": 5,
            "flow_to_rgb_output": ["F01->0", "F12->1", "F23->2", "F34->3"],
            "rgb_outputs_have_temporal_context": True,
            "spatial_alignment": "coarse_logical_2x3_grid_not_pixel_alignment",
            "pooling": {
                "implementation": "adaptive_avg_pool2d_per_temporal_slice",
                "bounds": "zero_based_end_exclusive; uneven sizes can overlap",
                "flow_sfe_input_hw": [9, 16],
                "flow_rows": adaptive_pool_bounds(9, 2),
                "flow_cols": adaptive_pool_bounds(16, 3),
                "rgb_input_hw": [72, 128],
                "rgb_rows": adaptive_pool_bounds(72, 2),
                "rgb_cols": adaptive_pool_bounds(128, 3),
                "flow_temporal": "existing_MFE_spatial_mean_preserves_four_time_positions",
                "rgb_temporal": "preserve_all_five_then_take_positions_0_1_2_3_no_time_pool",
                "rgb_legacy_row_flip": False,
            },
            "taps": {"F_sp": "flow_encoder.sfe(flow_stem)",
                     "F_tm": "flow_encoder.temporal(flow_stem)",
                     "R": "rgb_encoder.backbone.forward_feature_map(images) after DIG before projection"},
            "appearance_alpha": float(self.source.rgb_encoder.backbone.appearance_alpha.detach()),
        }

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, inputs):
        self.train(False)
        required = {"images", "flows", "valid", "pair_dt"}
        if not isinstance(inputs, Mapping) or set(inputs) != required:
            raise ValueError(f"Extractor accepts only {sorted(required)}")
        dt = inputs["pair_dt"]
        if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or dt.shape[0] < 1:
            raise ValueError("pair_dt must be a nonempty [B,4] tensor")
        batch = len(dt)
        _finite_tensor(dt, (batch, 4), "pair_dt")
        if bool((dt <= 0).any()):
            raise ValueError("pair_dt must be positive seconds")
        images = _finite_tensor(inputs["images"], (batch, *RGB_SHAPE), "images")
        if bool((images.abs() > 1.00001).any()):
            raise ValueError("RGB images must retain the existing [-1,1] preprocessing")
        flow = inputs["flows"]
        if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (batch, *FLOW_SHAPE) or not flow.is_floating_point():
            raise ValueError("flows must have shape [B,4,2,72,128]")
        valid = _binary_mask(inputs["valid"], (batch, *MASK_SHAPE), "valid")
        if any(value.device != dt.device for value in (images, flow, valid)):
            raise ValueError("All inference inputs must share one device")
        if any(value.dtype != torch.float32 for value in (images, flow, dt)):
            raise ValueError("Frozen extraction requires FP32 images, flows and seconds")
        valid = valid & torch.isfinite(flow).all(dim=2, keepdim=True)
        safe = torch.where(valid.expand_as(flow), flow, torch.zeros_like(flow))
        # Exactly the existing R03 signed u/v, valid, hint=valid organization.
        encoded = torch.cat((safe, valid.to(flow.dtype), valid.to(flow.dtype)), dim=2)
        encoder = self.source.flow_encoder
        x = encoder.spatial(encoded.reshape(batch * 4, 4, 72, 128))
        x = x.reshape(batch, 4, 32, *x.shape[-2:]).transpose(1, 2)
        if tuple(x.shape[2:]) != (4, 9, 16):
            raise RuntimeError("Unexpected flow tap geometry")
        spatial = encoder.sfe(x)
        temporal = encoder.temporal(x)
        rgb = self.source.rgb_encoder.backbone.forward_feature_map(images)
        if tuple(spatial.shape[1:]) != (32, 4, 9, 16) or tuple(temporal.shape[1:]) != (32, 4, 1, 1):
            raise RuntimeError("Unexpected SFE/MFE tap geometry")
        if tuple(rgb.shape[1:]) != (256, 5, 72, 128):
            raise RuntimeError("Unexpected RGB feature map; time pooling is forbidden")
        features = {
            "F_sp": spatial_tokens(spatial),
            "F_tm": temporal.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
            "R": spatial_tokens(rgb)[:, :4].contiguous(),
            "dt": dt.clone(),
        }
        validate_features(features)
        return features


def validate_features(features):
    if not isinstance(features, Mapping) or set(features) != set(CACHE_FEATURE_SHAPES):
        raise ValueError(f"Fusion input must contain only {sorted(CACHE_FEATURE_SHAPES)}")
    dt = features["dt"]
    if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or dt.shape[0] < 1:
        raise ValueError("Cached dt must be a nonempty [B,4] tensor")
    batch = len(dt)
    for key, shape in CACHE_FEATURE_SHAPES.items():
        value = _finite_tensor(features[key], (batch, *shape), key)
        if value.dtype != torch.float32 or value.device != dt.device:
            raise ValueError("Fusion features must be FP32 on one device")
    if bool((dt <= 0).any()):
        raise ValueError("Cached dt must remain positive raw seconds")
    return batch


class FusionA(nn.Module):
    """New late-fusion readout; no old projections or velocity weights reused."""
    def __init__(self):
        super().__init__()
        self.flow_projection = nn.Sequential(nn.Linear(320, 64, bias=True), nn.ReLU())
        self.rgb_projection = nn.Sequential(nn.Linear(256, 64, bias=True), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(452, 192, bias=True), nn.GELU(), nn.Linear(192, 2, bias=True))

    def forward(self, features):
        batch = validate_features(features)
        # Explicit C,row,col and C,T orders match the original flow descriptor.
        spatial = features["F_sp"].mean(dim=1).transpose(1, 2).reshape(batch, 192)
        temporal = features["F_tm"].transpose(1, 2).reshape(batch, 128)
        flow = self.flow_projection(torch.cat((spatial, temporal), dim=1))
        rgb = self.rgb_projection(features["R"].mean(dim=1)).reshape(batch, 384)
        return self.head(torch.cat((flow, rgb, features["dt"] / DT_SCALE_SECONDS), dim=1))


def sinusoidal_axis(count, dimension):
    positions = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / dimension))
    values = torch.empty(count, dimension, dtype=torch.float32)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies)
    return values


def fixed_position_encoding():
    time = sinusoidal_axis(4, 32)[:, None, None, :].expand(4, 2, 3, 32)
    row = sinusoidal_axis(2, 16)[None, :, None, :].expand(4, 2, 3, 16)
    col = sinusoidal_axis(3, 16)[None, None, :, :].expand(4, 2, 3, 16)
    return torch.cat((time, row, col), dim=-1).reshape(24, 64).contiguous()


class TokenReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(320, 64, bias=True)
        self.input_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
        self.register_buffer("position_encoding", fixed_position_encoding())
        self.head = nn.Sequential(nn.Linear(68, 192, bias=True), nn.GELU(), nn.Linear(192, 2, bias=True))

    def input_tokens(self, features):
        batch = validate_features(features)
        shared_context = features["F_tm"].unsqueeze(2).expand(-1, -1, 6, -1)
        raw = torch.cat((features["F_sp"], shared_context, features["R"]), dim=-1)
        raw = raw.reshape(batch, 24, 320)
        return self.input_norm(self.input_projection(raw)) + self.position_encoding

    def forward(self, features):
        tokens = self.tokens(features)
        return self.head(torch.cat((tokens.mean(dim=1), features["dt"] / DT_SCALE_SECONDS), dim=1))


class IndependentTokenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64, 256, bias=True), nn.GELU(), nn.Linear(256, 64, bias=True))
        self.norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)

    def forward(self, tokens):
        return self.norm(tokens + self.mlp(tokens))


class FusionB(TokenReadout):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([IndependentTokenBlock() for _ in range(2)])

    def tokens(self, features):
        tokens = self.input_tokens(features)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class FusionC(TokenReadout):
    def __init__(self):
        super().__init__()
        # Construct each layer independently: TransformerEncoder's cloned-layer
        # default would give both layers identical initial parameter values.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                       dropout=0.0, activation="gelu", layer_norm_eps=1e-5,
                                       batch_first=True, norm_first=False, bias=True)
            for _ in range(2)
        ])

    def tokens(self, features):
        tokens = self.input_tokens(features)
        for layer in self.layers:
            tokens = layer(tokens)  # all 24 tokens are historical; no causal mask
        return tokens


def build_triplet(seed=42):
    """Fresh one-seed models; B/C common modules copied explicitly, no sharing.

    New modules use their PyTorch default reset_parameters policy. Common
    modules are copied after construction, so random-consumption order cannot
    affect their equality. All layers remain independently trainable.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        models = {"A": FusionA(), "B": FusionB(), "C": FusionC()}
    for name in ("input_projection", "input_norm", "head"):
        getattr(models["C"], name).load_state_dict(copy.deepcopy(getattr(models["B"], name).state_dict()), strict=True)
    for name, model in models.items():
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if count != EXPECTED_PARAMETERS[name]:
            raise RuntimeError(f"Fusion {name} parameter count {count} != {EXPECTED_PARAMETERS[name]}")
    return models


def common_initialization_evidence(models):
    evidence = {}
    for name in ("input_projection", "input_norm", "head"):
        left, right = getattr(models["B"], name), getattr(models["C"], name)
        evidence[name] = {
            "exact_equal": all(torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items()),
            "independent_storage": all(a.data_ptr() != b.data_ptr() for a, b in zip(left.parameters(), right.parameters())),
        }
    evidence["position_encoding_exact_equal"] = torch.equal(models["B"].position_encoding, models["C"].position_encoding)
    evidence["transformer_layers_independent_storage"] = all(
        a.data_ptr() != b.data_ptr() for a, b in zip(models["C"].layers[0].parameters(), models["C"].layers[1].parameters()))
    evidence["transformer_attention_initial_weights_differ"] = not torch.equal(
        models["C"].layers[0].self_attn.in_proj_weight, models["C"].layers[1].self_attn.in_proj_weight)
    return evidence
