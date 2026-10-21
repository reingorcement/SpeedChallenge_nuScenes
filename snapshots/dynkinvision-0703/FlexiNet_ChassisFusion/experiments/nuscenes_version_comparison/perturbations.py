from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from dataset.nuscenes_chassis import FAIR_CHASSIS_INDEX


SHARED_VARIANTS = (
    "clean",
    "can_missing",
    "can_stale_500ms",
    "can_dropout_250ms",
    "yaw_bias_pos",
    "yaw_bias_neg",
    "wheel_scale_low",
    "wheel_scale_high",
    "camera_frozen",
    "camera_missing",
)
V2_CALIBRATION_VARIANTS = (
    "focal_scale_low",
    "focal_scale_high",
    "camera_height_low",
    "camera_height_high",
    "camera_pitch_pos",
    "camera_pitch_neg",
)


@dataclass(frozen=True)
class PerturbationContext:
    chassis_hz: float = 20.0
    yaw_bias_radps: float = 0.02
    wheel_scale_low: float = 0.9
    wheel_scale_high: float = 1.1
    stale_seconds: float = 0.5
    dropout_seconds: float = 0.25
    focal_scale_low: float = 0.95
    focal_scale_high: float = 1.05
    camera_height_low: float = 0.9
    camera_height_high: float = 1.1
    camera_pitch_degrees: float = 1.0
    kinematic_mean: torch.Tensor | None = None
    kinematic_std: torch.Tensor | None = None

    @classmethod
    def from_config(cls, protocol: dict[str, Any], stats_path: str | Path | None):
        stress = protocol["stress"]
        statistics = None
        if stats_path:
            path = Path(stats_path)
            if path.is_file():
                statistics = json.loads(path.read_text(encoding="utf-8"))
        wheel_scales = stress["wheel_scales"]
        focal_scales = stress["focal_scales"]
        height_scales = stress["camera_height_scales"]
        return cls(
            chassis_hz=float(protocol["data"]["chassis_hz"]),
            yaw_bias_radps=float(stress["yaw_bias_radps"]),
            wheel_scale_low=float(wheel_scales[0]),
            wheel_scale_high=float(wheel_scales[1]),
            stale_seconds=float(stress["stale_seconds"]),
            dropout_seconds=float(stress["dropout_seconds"]),
            focal_scale_low=float(focal_scales[0]),
            focal_scale_high=float(focal_scales[1]),
            camera_height_low=float(height_scales[0]),
            camera_height_high=float(height_scales[1]),
            camera_pitch_degrees=float(stress["camera_pitch_degrees"]),
            kinematic_mean=(
                torch.tensor(statistics["mean"], dtype=torch.float32)
                if statistics is not None
                else None
            ),
            kinematic_std=(
                torch.tensor(statistics["std"], dtype=torch.float32)
                if statistics is not None
                else None
            ),
        )


def validate_variant(variant: str, family: str) -> None:
    supported = set(SHARED_VARIANTS)
    if family == "v2":
        supported.update(V2_CALIBRATION_VARIANTS)
    if variant not in supported:
        if variant in V2_CALIBRATION_VARIANTS:
            raise ValueError(f"Calibration variant {variant!r} is a V2-only diagnostic.")
        raise ValueError(f"Unknown perturbation variant {variant!r}.")


def _copy(batch: dict[str, Any]) -> dict[str, Any]:
    return dict(batch)


def _renormalize_v2(batch: dict[str, Any], context: PerturbationContext) -> None:
    from dataset.v2_schema import normalize_kinematic_values

    if context.kinematic_mean is None or context.kinematic_std is None:
        raise ValueError("V2 value perturbations require kinematic mean/std statistics.")
    mean = context.kinematic_mean.to(batch["kinematic_values_raw"])
    std = context.kinematic_std.to(batch["kinematic_values_raw"])
    batch["kinematic_values"] = normalize_kinematic_values(
        batch["kinematic_values_raw"], batch["kinematic_valid"], mean, std
    )


def _legacy_can(batch: dict[str, Any], variant: str, context: PerturbationContext) -> None:
    chassis = batch["chassis"].clone()
    current_index = chassis.shape[1] - 1
    if "chassis_current_index" in batch:
        current_index = int(batch["chassis_current_index"].reshape(-1)[0])
    if variant == "can_missing":
        chassis.zero_()
        batch["base_velocity"] = torch.zeros_like(batch["base_velocity"])
    elif variant == "can_stale_500ms":
        offset = max(int(round(context.stale_seconds * context.chassis_hz)), 1)
        indexes = torch.arange(chassis.shape[1], device=chassis.device) - offset
        indexes.clamp_(min=0)
        chassis = chassis.index_select(1, indexes)
        batch["base_velocity"] = batch["base_velocity"].clone()
        batch["base_velocity"][:, 0] = chassis[:, current_index, FAIR_CHASSIS_INDEX["base_vx"]]
    elif variant == "can_dropout_250ms":
        count = max(int(round(context.dropout_seconds * context.chassis_hz)), 1)
        chassis[:, -count:, :].zero_()
        batch["base_velocity"] = torch.zeros_like(batch["base_velocity"])
    elif variant in {"yaw_bias_pos", "yaw_bias_neg"}:
        sign = 1.0 if variant.endswith("pos") else -1.0
        chassis[..., FAIR_CHASSIS_INDEX["yaw_rate"]] += sign * context.yaw_bias_radps
    elif variant in {"wheel_scale_low", "wheel_scale_high"}:
        scale = context.wheel_scale_low if variant.endswith("low") else context.wheel_scale_high
        wheel_indices = [
            FAIR_CHASSIS_INDEX[name]
            for name in ("fl_wheel_speed", "fr_wheel_speed", "rl_wheel_speed", "rr_wheel_speed")
        ]
        chassis[..., wheel_indices] *= scale
        chassis[..., FAIR_CHASSIS_INDEX["wheel_speed_mean"]] *= scale
        chassis[..., FAIR_CHASSIS_INDEX["wheel_speed_std"]] *= abs(scale)
    batch["chassis"] = chassis


def _v2_can(batch: dict[str, Any], variant: str, context: PerturbationContext) -> None:
    raw = batch["kinematic_values_raw"].clone()
    valid = batch["kinematic_valid"].clone()
    age = batch["kinematic_age"].clone()
    if variant == "can_missing":
        raw.zero_()
        valid.zero_()
        age.fill_(1.0)
        batch["wheel_speed"] = torch.zeros_like(batch["wheel_speed"])
        batch["angular_rate_camera"] = torch.zeros_like(batch["angular_rate_camera"])
        batch["wheel_speed_valid"] = torch.zeros_like(batch["wheel_speed_valid"])
        batch["angular_rate_valid"] = torch.zeros_like(batch["angular_rate_valid"])
        batch["wheel_speed_age_sec"] = torch.full_like(batch["wheel_speed_age_sec"], 1.0e6)
        batch["angular_rate_age_sec"] = torch.full_like(batch["angular_rate_age_sec"], 1.0e6)
    elif variant == "can_stale_500ms":
        valid.zero_()
        age.fill_(1.0)
        batch["wheel_speed_age_sec"] = batch["wheel_speed_age_sec"].clone() + context.stale_seconds
        batch["angular_rate_age_sec"] = batch["angular_rate_age_sec"].clone() + context.stale_seconds
    elif variant == "can_dropout_250ms":
        count = max(int(round(context.dropout_seconds * context.chassis_hz)), 1)
        raw[:, -count:, :].zero_()
        valid[:, -count:, :].zero_()
        age[:, -count:, :].fill_(1.0)
        batch["wheel_speed_valid"] = torch.zeros_like(batch["wheel_speed_valid"])
        batch["angular_rate_valid"] = torch.zeros_like(batch["angular_rate_valid"])
    elif variant in {"yaw_bias_pos", "yaw_bias_neg"}:
        sign = 1.0 if variant.endswith("pos") else -1.0
        bias = sign * context.yaw_bias_radps
        raw[..., 0] = torch.where(valid[..., 0], raw[..., 0] + bias, raw[..., 0])
        ego_bias = raw.new_tensor([0.0, 0.0, bias])
        rotation_camera_from_ego = batch["rotation_ego_from_camera"].transpose(-1, -2)
        camera_bias = torch.matmul(rotation_camera_from_ego, ego_bias)
        batch["angular_rate_camera"] = batch["angular_rate_camera"].clone() + camera_bias
    elif variant in {"wheel_scale_low", "wheel_scale_high"}:
        scale = context.wheel_scale_low if variant.endswith("low") else context.wheel_scale_high
        batch["wheel_speed"] = batch["wheel_speed"].clone() * scale
    batch["kinematic_values_raw"] = raw
    batch["kinematic_valid"] = valid
    batch["kinematic_age"] = age
    _renormalize_v2(batch, context)


def _camera(batch: dict[str, Any], variant: str, family: str) -> None:
    images = batch["images"].clone()
    if variant == "camera_frozen":
        images[:] = images[:, :1]
    else:
        images.zero_()
    batch["images"] = images
    if family == "v2":
        batch["flow_v2"] = torch.zeros_like(batch["flow_v2"])
        batch["flow_confidence"] = torch.zeros_like(batch["flow_confidence"])
        batch["cached_geometry"] = None


def _pitch_rotation(batch: dict[str, Any], angle_rad: float) -> None:
    rotation = batch["rotation_ego_from_camera"]
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    error = rotation.new_tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    if rotation.ndim == 3:
        error = error.expand(rotation.shape[0], -1, -1)
    perturbed = torch.matmul(rotation, error)
    upward = rotation.new_tensor([0.0, 0.0, 1.0])
    batch["rotation_ego_from_camera"] = perturbed
    batch["plane_normal_camera"] = torch.matmul(perturbed.transpose(-1, -2), upward)
    batch["cached_geometry"] = None


def _calibration(batch: dict[str, Any], variant: str, context: PerturbationContext) -> None:
    if variant.startswith("focal_scale"):
        scale = context.focal_scale_low if variant.endswith("low") else context.focal_scale_high
        intrinsics = batch["intrinsics"].clone()
        intrinsics[..., 0, 0] *= scale
        intrinsics[..., 1, 1] *= scale
        batch["intrinsics"] = intrinsics
    elif variant.startswith("camera_height"):
        scale = context.camera_height_low if variant.endswith("low") else context.camera_height_high
        batch["plane_offset"] = batch["plane_offset"].clone() * scale
    else:
        sign = 1.0 if variant.endswith("pos") else -1.0
        _pitch_rotation(batch, math.radians(sign * context.camera_pitch_degrees))
    batch["cached_geometry"] = None


def apply_perturbation(
    batch: dict[str, Any],
    variant: str,
    family: str,
    context: PerturbationContext,
) -> dict[str, Any]:
    validate_variant(variant, family)
    if variant == "clean":
        return batch
    modified = _copy(batch)
    if variant.startswith("can_") or variant.startswith("yaw_bias") or variant.startswith("wheel_scale"):
        if family == "v2":
            _v2_can(modified, variant, context)
        else:
            _legacy_can(modified, variant, context)
    elif variant.startswith("camera_") and variant not in {
        "camera_height_low",
        "camera_height_high",
        "camera_pitch_pos",
        "camera_pitch_neg",
    }:
        _camera(modified, variant, family)
    else:
        _calibration(modified, variant, context)
    return modified
