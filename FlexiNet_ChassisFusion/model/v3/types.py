from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairTwistEstimate:
    twist_camera: torch.Tensor
    covariance_camera: torch.Tensor
    condition_number: torch.Tensor
    residual_rms_px: torch.Tensor
    valid: torch.Tensor
    support_count: torch.Tensor
    support_fraction: torch.Tensor
    predicted_flow: torch.Tensor
    candidate_mask: torch.Tensor


@dataclass(frozen=True)
class V3VelocityOutput:
    predicted_velocity: torch.Tensor
    predicted_log_variance: torch.Tensor
    geometry_velocity: torch.Tensor
    geometry_corrected_velocity: torch.Tensor
    direct_velocity: torch.Tensor
    visual_correction: torch.Tensor
    geometry_gate: torch.Tensor
    geometry_valid: torch.Tensor
    geometry_covariance: torch.Tensor
    geometry_condition_number: torch.Tensor
    geometry_residual_rms_px: torch.Tensor
    geometry_support_fraction: torch.Tensor
    pair_twist_camera: torch.Tensor
    pair_twist_ego: torch.Tensor
    pair_covariance_camera: torch.Tensor
    pair_covariance_ego: torch.Tensor
    pair_condition_number: torch.Tensor
    pair_residual_rms_px: torch.Tensor
    pair_support_fraction: torch.Tensor
    pair_valid: torch.Tensor
    pair_predicted_flow: torch.Tensor
    pair_dispersion: torch.Tensor
    reliability: torch.Tensor
    candidate_mask: torch.Tensor
