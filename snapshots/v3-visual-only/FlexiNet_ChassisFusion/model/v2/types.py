from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimedSignal:
    """A sensor value whose availability and staleness are explicit."""

    value: torch.Tensor
    valid: torch.Tensor
    age_sec: torch.Tensor

    def __post_init__(self) -> None:
        if self.valid.dtype != torch.bool:
            raise TypeError("TimedSignal.valid must be boolean.")
        if not torch.is_floating_point(self.age_sec):
            raise TypeError("TimedSignal.age_sec must be floating point.")

    def valid_fields(self, max_age_sec: float) -> torch.Tensor:
        finite_value = torch.isfinite(self.value)
        finite_age = torch.isfinite(self.age_sec)
        valid = self.valid
        while valid.ndim < self.value.ndim:
            valid = valid.unsqueeze(-1)
        while finite_age.ndim < finite_value.ndim:
            finite_age = finite_age.unsqueeze(-1)
        age = self.age_sec
        while age.ndim < finite_value.ndim:
            age = age.unsqueeze(-1)
        return valid & finite_value & finite_age & (age <= max_age_sec) & (age >= 0.0)

    def usable(self, max_age_sec: float) -> torch.Tensor:
        return self.valid_fields(max_age_sec).any(dim=-1, keepdim=True)


@dataclass(frozen=True)
class GeometryEstimate:
    camera_velocity: torch.Tensor
    ego_velocity: torch.Tensor
    covariance: torch.Tensor
    condition_number: torch.Tensor
    residual_rms: torch.Tensor
    valid: torch.Tensor
    support_count: torch.Tensor
    translation_flow: torch.Tensor


@dataclass(frozen=True)
class KinematicEstimate:
    residual: torch.Tensor
    log_variance: torch.Tensor
    gate: torch.Tensor
    coverage: torch.Tensor
    available: torch.Tensor
    latent: torch.Tensor


@dataclass(frozen=True)
class V2VelocityOutput:
    geometry_velocity: torch.Tensor
    geometry_covariance: torch.Tensor
    geometry_condition_number: torch.Tensor
    geometry_residual_rms: torch.Tensor
    geometry_valid: torch.Tensor
    geometry_support_count: torch.Tensor
    visual_velocity: torch.Tensor
    wheel_scale: torch.Tensor
    fallback_used: torch.Tensor
    residual: torch.Tensor
    residual_gate: torch.Tensor
    kinematic_residual: torch.Tensor
    kinematic_applied_correction: torch.Tensor
    kinematic_gate: torch.Tensor
    kinematic_log_variance: torch.Tensor
    kinematic_coverage: torch.Tensor
    kinematic_available: torch.Tensor
    predicted_velocity: torch.Tensor
    output_reference: str
