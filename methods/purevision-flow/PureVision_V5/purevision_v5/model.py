"""Six S2 experiments with one velocity head and explicit inference inputs."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENTS = ("R00", "R01", "R02", "R03", "R04", "R05")
INFERENCE_KEYS = frozenset({"descriptor", "flows", "valid", "fb_mask", "pair_dt"})
FLOW_SHAPE = (4, 2, 72, 128)
MASK_SHAPE = (4, 1, 72, 128)
DT_SCALE_SECONDS = 1.0 / 12.0


def _finite_tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    return value


def _binary_mask(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all():
        raise ValueError(f"{name} must be a finite binary mask")
    return value.bool()


class LightFlowEncoder(nn.Module):
    """R01: fixed signed-flow grids, ordered time concatenation, one projection."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(4 * 3 * 4 * 8, 64), nn.ReLU())

    def forward(self, flow_and_valid):
        batch, steps, channels, height, width = flow_and_valid.shape
        grid = F.adaptive_avg_pool2d(flow_and_valid.reshape(batch * steps, channels, height, width), (4, 8))
        return self.projection(grid.reshape(batch, -1))


class SpatialTemporalFlowEncoder(nn.Module):
    """R02-R05: spatial flow CNN then temporal convolution before region pooling."""

    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.temporal = nn.Sequential(nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(64 * 2 * 3, 64), nn.ReLU())

    def forward(self, flow_valid_hint):
        batch, steps, channels, height, width = flow_valid_hint.shape
        spatial = self.spatial(flow_valid_hint.reshape(batch * steps, channels, height, width))
        spatial = spatial.reshape(batch, steps, 64, *spatial.shape[-2:]).transpose(1, 2)
        temporal = self.temporal(spatial)
        regions = F.adaptive_avg_pool3d(temporal, (1, 2, 3)).flatten(1)
        return self.projection(regions)


class FlexFlowEncoder(nn.Module):
    """FlexiNet SFE/MFE adapted to four signed flow fields, without CMA.

    Stem: BT,C,H,W -> B,32,T,9,16. SFE preserves T and pools to
    (1,2,3); MFE pools H,W only, then flattens C,T in fixed order.
    MFE has three temporal k=3 convolutions (RF=7), zero-padded at
    both ends of the four-step history. No samples beyond that history
    enter the model. The complete candidate, including BN, is the ablation.
    """

    def __init__(self):
        super().__init__()
        from ._rgb_reference.modules import SpatialFeatureExtraction, MotionFeatureExtraction
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.sfe = SpatialFeatureExtraction(32, 32)
        self.temporal = MotionFeatureExtraction(32, 16, 32)
        self.projection = nn.Sequential(nn.Linear(32 * 2 * 3 + 32 * 4, 64), nn.ReLU())

    def forward(self, flow_valid_hint):
        batch, steps, channels, height, width = flow_valid_hint.shape
        if steps != 4:
            raise ValueError("FLEX requires exactly four historical pairs")
        x = self.spatial(flow_valid_hint.reshape(batch * steps, channels, height, width))
        x = x.reshape(batch, steps, 32, *x.shape[-2:]).transpose(1, 2)
        spatial = F.adaptive_avg_pool3d(self.sfe(x), (1, 2, 3)).flatten(1)
        temporal = self.temporal(x).flatten(1)
        return self.projection(torch.cat((spatial, temporal), dim=1))


def build_model(config):
    """One construction path for training, resume, evaluation and startup checks."""
    return VelocityModel(config["experiment"], encoder_variant=config.get("encoder_variant", "base"),
                         head_seed=config.get("head_seed"))


class VelocityModel(nn.Module):
    """Predict [vx,vy] m/s; labels and all unlisted inputs are rejected.

    Signed flows remain displacement pixels at 72x128, without per-frame
    standardization or absolute-value temporal differencing. The same dt vector
    is explicitly supplied to every experiment, including the RGB baseline.
    """

    def __init__(self, experiment: str = "R03", *, encoder_variant="base", head_seed=None):
        super().__init__()
        if experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}")
        if encoder_variant not in {"base", "flex"} or (encoder_variant != "base" and experiment != "R02"):
            raise ValueError("FLEX is authorized only for flow-only R02")
        self.experiment = experiment
        self.uses_rgb = experiment != "R02"
        self.uses_flow = experiment != "R00"
        self.flow_encoder = (LightFlowEncoder() if experiment == "R01"
                             else FlexFlowEncoder() if encoder_variant == "flex"
                             else SpatialTemporalFlowEncoder() if self.uses_flow else None)
        input_dim = 384 * self.uses_rgb + 64 * self.uses_flow + 4
        self.head = nn.Sequential(nn.Linear(input_dim, 192), nn.ReLU(), nn.Linear(192, 2))
        if head_seed is not None:
            if not isinstance(head_seed, int) or isinstance(head_seed, bool) or head_seed < 0:
                raise ValueError("head_seed must be a nonnegative integer")
            # Explicit isolated CPU RNG: independent of encoder RNG consumption.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(head_seed)
                for module in self.head.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise ValueError("Model input must be an inference-only mapping")
        unexpected = set(batch) - INFERENCE_KEYS
        if unexpected:
            raise ValueError(f"Forbidden inference fields: {sorted(unexpected)}")
        required = {"pair_dt"}
        if self.uses_rgb:
            required.add("descriptor")
        if self.uses_flow:
            required.update({"flows", "valid"})
        if self.experiment in {"R04", "R05"}:
            required.add("fb_mask")
        missing = required - set(batch)
        if missing:
            raise ValueError(f"Missing inference fields: {sorted(missing)}")
        dt = batch["pair_dt"]
        if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or dt.shape[1] != 4 or dt.shape[0] < 1:
            raise ValueError("pair_dt must have shape [B,4] with a nonempty batch")
        size = len(dt)
        _finite_tensor(dt, (size, 4), "pair_dt")
        if bool((dt <= 0).any()):
            raise ValueError("pair_dt must be positive seconds")
        features = []
        if self.uses_rgb:
            features.append(_finite_tensor(batch["descriptor"], (size, 384), "descriptor"))
        if self.uses_flow:
            flow = batch["flows"]
            if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (size, *FLOW_SHAPE) or not flow.is_floating_point():
                raise ValueError(f"flows must be floating point with shape {(size, *FLOW_SHAPE)}")
            valid = _binary_mask(batch["valid"], (size, *MASK_SHAPE), "valid")
            # Finite checks precede arithmetic: NaN * 0 would still be NaN.
            valid = valid & torch.isfinite(flow).all(dim=2, keepdim=True)
            safe = torch.where(valid.expand_as(flow), flow, torch.zeros_like(flow))
            if self.experiment == "R01":
                encoded_input = torch.cat((safe, valid.to(flow.dtype)), dim=2)
            else:
                if self.experiment in {"R04", "R05"}:
                    hint = _binary_mask(batch["fb_mask"], (size, *MASK_SHAPE), "fb_mask") & valid
                else:
                    # No FB-derived information: constant one at legal samples.
                    hint = valid
                if self.experiment == "R05":
                    safe = torch.where(hint.expand_as(safe), safe, torch.zeros_like(safe))
                encoded_input = torch.cat((safe, valid.to(flow.dtype), hint.to(flow.dtype)), dim=2)
            features.append(self.flow_encoder(encoded_input))
        features.append(dt / DT_SCALE_SECONDS)
        if any(value.device != dt.device for value in features):
            raise ValueError("All inference tensors must share one device")
        prediction = self.head(torch.cat(features, dim=1))
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite velocity prediction")
        return prediction
