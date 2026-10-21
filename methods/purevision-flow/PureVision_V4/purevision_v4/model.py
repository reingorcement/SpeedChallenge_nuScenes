"""Frozen RGB descriptors and trainable V4 fusion heads, isolated from legacy gates."""
import copy
import inspect

import numpy as np
import torch
from torch import nn

from ._legacy.model.v3 import V3VisualOnlyVelocityModel
from .io import TARGET_VERSION, sha256

FEATURE_NAMES = (
    "vx_scaled", "vy_scaled", "vz_scaled", "log_var_x", "log_var_y", "log_var_z",
    "log_condition", "residual_scaled", "support_fraction", "valid_fraction", "dispersion_scaled",
    "measurement_age_s", "dt_0", "dt_1", "dt_2", "valid_0", "valid_1", "valid_2",
    "normal_update_fraction", "normal_hold_fraction", "normal_fallback_fraction", "normal_deviation_scaled",
)


def geometry_features(geometry, pairs, normals):
    valid = geometry["pair_valid"]
    selected = [p for p in pairs if p["valid"]]
    avg = lambda key, default=0.: np.mean([p[key] for p in selected]) if selected else default
    condition = np.mean([np.log1p(min(p["condition"], 1e12)) for p in selected]) if selected else np.log1p(1e12)
    status = [sum(n["status"] == s for n in normals) / len(normals) for s in ["update", "hold", "fixed_fallback"]]
    values = [*(geometry["velocity"] / [5., 1., 1.]), *np.log1p(np.clip(np.diag(geometry["covariance"]), 0, 1e6)),
              condition, min(avg("residual"), 100.) / 10, avg("support_fraction"), valid.mean(),
              min(geometry["dispersion"], 20.) / 5, geometry["age_s"], *[p["dt"] for p in pairs],
              *valid.astype(float), *status, np.mean([n.get("nominal_deviation_deg", 0.) for n in normals]) / 10]
    result = np.asarray(values, np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Non-finite or incompatible geometry diagnostics")
    return result


class FrozenVisual(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        args = {k: v for k, v in package["config"]["model"].items() if k in inspect.signature(V3VisualOnlyVelocityModel).parameters}
        args.update(mode="fusion")
        if args.get("pair_count", 3) != 3 or args.get("pooling", "regions") != "regions":
            raise ValueError("V4 initialization requires the 3-pair regional V3 checkpoint")
        self.legacy = V3VisualOnlyVelocityModel(**args)
        self.legacy.load_state_dict(package["model"], strict=True)
        self.requires_grad_(False)
        self.eval()
        self.digest = sha256(checkpoint)

    @torch.inference_mode()
    def forward(self, images, geometry_size):
        feature_map = self.legacy.backbone.forward_feature_map(images)
        return self.legacy._descriptor(feature_map), self.legacy._reliability(feature_map, geometry_size)


class FusionHead(nn.Module):
    """Batch-capable learning on frozen RGB and compensated geometry features."""
    def __init__(self, direct_head=None):
        super().__init__()
        self.direct = copy.deepcopy(direct_head) if direct_head is not None else nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 2))
        for p in self.direct.parameters():
            p.requires_grad_(True)
        dim = 384 + len(FEATURE_NAMES)
        self.correction = nn.Sequential(nn.Linear(dim, 192), nn.ReLU(), nn.Linear(192, 2))
        self.gate = nn.Sequential(nn.Linear(dim, 192), nn.ReLU(), nn.Linear(192, 2))
        self.uncertainty = nn.Sequential(nn.Linear(dim + 2, 192), nn.ReLU(), nn.Linear(192, 2))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer("residual_limits", torch.tensor([3., 1.5]))

    def forward(self, descriptor, diagnostic, geometry_velocity, geometry_valid, mode="fusion"):
        if descriptor.ndim != 2 or descriptor.shape[1] != 384 or diagnostic.shape != (len(descriptor), len(FEATURE_NAMES)):
            raise ValueError("Invalid frozen feature batch")
        if geometry_velocity.shape != (len(descriptor), 2) or geometry_valid.shape != (len(descriptor),):
            raise ValueError("Invalid geometry batch")
        # Use a mask before arithmetic so invalid NaNs can never poison the RGB fallback.
        safe_geometry = torch.where(geometry_valid[:, None], geometry_velocity, torch.zeros_like(geometry_velocity))
        joint = torch.cat((descriptor, diagnostic), -1)
        direct = self.direct(descriptor)
        correction = self.residual_limits * torch.tanh(self.correction(joint))
        gate = torch.sigmoid(self.gate(joint)) * geometry_valid[:, None]
        prediction = gate * (safe_geometry + correction) + (1 - gate) * direct
        if mode == "direct":
            prediction = direct
        elif mode != "fusion":
            raise ValueError("Unknown learning mode")
        log_variance = self.uncertainty(torch.cat((joint, prediction), -1)).clamp(-6, 6)
        return dict(prediction=prediction, direct=direct, gate=gate, correction=correction, log_variance=log_variance)


def load_fusion(path, feature_contract, target_spec):
    package = torch.load(path, map_location="cpu", weights_only=True)
    if package.get("schema") != "purevision_v4_fusion_v1" or not package.get("trained"):
        raise ValueError("V4 fusion requires a trained V4 checkpoint; legacy weights are initialization only")
    if package.get("feature_contract") != feature_contract or package.get("target_spec") != target_spec:
        raise ValueError("V4 checkpoint preprocessing/features/target contract mismatch")
    if package.get("target_spec", {}).get("version") != TARGET_VERSION:
        raise ValueError("Unsupported V4 target convention")
    model = FusionHead()
    model.load_state_dict(package["model"], strict=True)
    return model.eval().requires_grad_(False), package
