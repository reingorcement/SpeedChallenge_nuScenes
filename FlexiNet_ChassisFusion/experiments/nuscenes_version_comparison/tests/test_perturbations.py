from __future__ import annotations

import pytest
import torch

from experiments.nuscenes_version_comparison.perturbations import (
    PerturbationContext,
    apply_perturbation,
)


def _context() -> PerturbationContext:
    return PerturbationContext(
        kinematic_mean=torch.zeros(12),
        kinematic_std=torch.ones(12),
    )


def _legacy_batch():
    chassis = torch.ones(2, 21, 16)
    return {
        "images": torch.randn(2, 5, 3, 8, 8),
        "chassis": chassis,
        "base_velocity": torch.tensor([[5.0, 0.0], [6.0, 0.0]]),
        "velocity": torch.tensor([[5.1, 0.1], [6.1, -0.1]]),
        "chassis_current_index": torch.tensor([20, 20]),
    }


def _v2_batch():
    raw = torch.ones(2, 21, 12)
    return {
        "images": torch.randn(2, 5, 3, 8, 8),
        "velocity": torch.tensor([[5.1, 0.1], [6.1, -0.1]]),
        "base_velocity": torch.tensor([[5.0, 0.0], [6.0, 0.0]]),
        "kinematic_values_raw": raw,
        "kinematic_values": raw.clone(),
        "kinematic_valid": torch.ones_like(raw, dtype=torch.bool),
        "kinematic_age": torch.zeros_like(raw),
        "wheel_speed": torch.ones(2, 4),
        "wheel_speed_valid": torch.ones(2, 4, dtype=torch.bool),
        "wheel_speed_age_sec": torch.zeros(2, 4),
        "angular_rate_camera": torch.zeros(2, 3),
        "angular_rate_valid": torch.ones(2, 3, dtype=torch.bool),
        "angular_rate_age_sec": torch.zeros(2),
        "rotation_ego_from_camera": torch.eye(3).repeat(2, 1, 1),
        "plane_normal_camera": torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        "plane_offset": torch.ones(2),
        "intrinsics": torch.eye(3).repeat(2, 1, 1),
        "flow_v2": torch.ones(2, 2, 8, 8),
        "flow_confidence": torch.ones(2, 1, 8, 8),
    }


@pytest.mark.parametrize("variant", ["can_missing", "can_stale_500ms", "can_dropout_250ms", "yaw_bias_pos", "wheel_scale_low"])
def test_shared_perturbations_preserve_labels(variant: str) -> None:
    batch = _legacy_batch()
    labels = batch["velocity"].clone()
    modified = apply_perturbation(batch, variant, "legacy", _context())
    torch.testing.assert_close(modified["velocity"], labels)


def test_camera_freeze_is_deterministic_for_legacy() -> None:
    batch = _legacy_batch()
    first = apply_perturbation(batch, "camera_frozen", "legacy", _context())
    second = apply_perturbation(batch, "camera_frozen", "legacy", _context())
    torch.testing.assert_close(first["images"], second["images"])
    torch.testing.assert_close(first["images"][:, 0], first["images"][:, -1])


def test_calibration_variants_are_v2_only() -> None:
    with pytest.raises(ValueError, match="V2-only"):
        apply_perturbation(_legacy_batch(), "focal_scale_high", "legacy", _context())
