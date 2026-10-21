from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import V2VelocityOutput


def wrapped_angle_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


@dataclass(frozen=True)
class V2LossTerms:
    total: torch.Tensor
    velocity: torch.Tensor
    beta: torch.Tensor
    geometry: torch.Tensor
    residual: torch.Tensor
    kinematic_nll: torch.Tensor
    kinematic_residual: torch.Tensor


class V2VelocityLoss(nn.Module):
    def __init__(
        self,
        sigma_vx: float = 5.0,
        sigma_vy: float = 0.5,
        beta_weight: float = 0.1,
        geometry_weight: float = 0.25,
        residual_weight: float = 0.01,
        kinematic_nll_weight: float = 0.25,
        kinematic_residual_weight: float = 0.01,
        beta_min_speed: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("sigma", torch.tensor([sigma_vx, sigma_vy], dtype=torch.float32))
        self.beta_weight = float(beta_weight)
        self.geometry_weight = float(geometry_weight)
        self.residual_weight = float(residual_weight)
        self.kinematic_nll_weight = float(kinematic_nll_weight)
        self.kinematic_residual_weight = float(kinematic_residual_weight)
        self.beta_min_speed = float(beta_min_speed)

    def forward(self, output: V2VelocityOutput, target: torch.Tensor) -> V2LossTerms:
        velocity = F.smooth_l1_loss(output.predicted_velocity / self.sigma, target / self.sigma)
        target_speed = torch.linalg.vector_norm(target, dim=-1)
        predicted_speed = torch.linalg.vector_norm(output.predicted_velocity, dim=-1)
        beta_valid = (target_speed >= self.beta_min_speed) & (
            predicted_speed >= self.beta_min_speed
        )
        if bool(beta_valid.any()):
            predicted = output.predicted_velocity[beta_valid]
            beta_target = target[beta_valid]
            predicted_beta = torch.atan2(predicted[:, 1], predicted[:, 0])
            target_beta = torch.atan2(beta_target[:, 1], beta_target[:, 0])
            beta_error = wrapped_angle_difference(predicted_beta, target_beta)
            beta = F.smooth_l1_loss(beta_error, torch.zeros_like(beta_error))
        else:
            beta = output.predicted_velocity.new_zeros(())
        geometry_error = F.smooth_l1_loss(
            output.geometry_velocity / self.sigma, target / self.sigma, reduction="none"
        ).mean(dim=-1)
        geometry = torch.where(output.geometry_valid, geometry_error, 0.0).sum()
        geometry = geometry / output.geometry_valid.sum().clamp_min(1)
        residual = output.residual.square().mean()
        kinematic_target = target - output.visual_velocity.detach()
        normalized_error = (output.kinematic_residual - kinematic_target) / self.sigma
        axis_nll = 0.5 * (
            torch.exp(-output.kinematic_log_variance) * normalized_error.square()
            + output.kinematic_log_variance
        ).mean(dim=-1)
        kinematic_nll = torch.where(output.kinematic_available, axis_nll, 0.0).sum()
        kinematic_nll = kinematic_nll / output.kinematic_available.sum().clamp_min(1)
        kinematic_residual = output.kinematic_residual.square().mean()
        total = (
            velocity
            + self.beta_weight * beta
            + self.geometry_weight * geometry
            + self.residual_weight * residual
            + self.kinematic_nll_weight * kinematic_nll
            + self.kinematic_residual_weight * kinematic_residual
        )
        return V2LossTerms(
            total=total,
            velocity=velocity,
            beta=beta,
            geometry=geometry,
            residual=residual,
            kinematic_nll=kinematic_nll,
            kinematic_residual=kinematic_residual,
        )
