from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import V3VelocityOutput


def wrapped_angle_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


@dataclass(frozen=True)
class V3LossTerms:
    total: torch.Tensor
    nll: torch.Tensor
    direct: torch.Tensor
    geometry: torch.Tensor
    speed: torch.Tensor
    beta: torch.Tensor
    rotation: torch.Tensor
    reprojection: torch.Tensor
    residual: torch.Tensor
    support: torch.Tensor
    tv: torch.Tensor


class V3VelocityLoss(nn.Module):
    def __init__(
        self,
        nll_weight: float = 1.0,
        direct_weight: float = 0.25,
        geometry_weight: float = 0.25,
        speed_weight: float = 0.1,
        beta_weight: float = 0.1,
        rotation_weight: float = 0.05,
        reprojection_weight: float = 0.05,
        residual_weight: float = 0.01,
        support_weight: float = 0.02,
        tv_weight: float = 0.005,
        sigma_vx: float = 5.0,
        sigma_vy: float = 1.0,
        beta_min_speed: float = 2.0,
        support_floor: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "sigma", torch.tensor([sigma_vx, sigma_vy], dtype=torch.float32)
        )
        self.weights = {
            "nll": float(nll_weight),
            "direct": float(direct_weight),
            "geometry": float(geometry_weight),
            "speed": float(speed_weight),
            "beta": float(beta_weight),
            "rotation": float(rotation_weight),
            "reprojection": float(reprojection_weight),
            "residual": float(residual_weight),
            "support": float(support_weight),
            "tv": float(tv_weight),
        }
        self.beta_min_speed = float(beta_min_speed)
        self.support_floor = float(support_floor)

    def forward(
        self,
        output: V3VelocityOutput,
        target: dict[str, torch.Tensor],
        visual: dict[str, torch.Tensor],
    ) -> V3LossTerms:
        velocity_target = target["velocity"]
        zero = output.predicted_velocity.new_zeros(())
        nll = zero
        if self.weights["nll"]:
            normalized_error = (
                output.predicted_velocity - velocity_target
            ) / self.sigma
            nll = (
                0.5
                * (
                    torch.exp(-output.predicted_log_variance)
                    * normalized_error.square()
                    + output.predicted_log_variance
                ).mean()
            )
        direct = zero
        if self.weights["direct"]:
            direct = F.smooth_l1_loss(
                output.direct_velocity / self.sigma, velocity_target / self.sigma
            )
        geometry = zero
        if self.weights["geometry"]:
            geometry_axis = F.smooth_l1_loss(
                output.geometry_corrected_velocity / self.sigma,
                velocity_target / self.sigma,
                reduction="none",
            ).mean(dim=-1)
            geometry = (
                geometry_axis * output.geometry_valid
            ).sum() / output.geometry_valid.sum().clamp_min(1)
        speed = zero
        if self.weights["speed"]:
            speed = F.smooth_l1_loss(
                torch.linalg.vector_norm(output.predicted_velocity, dim=-1),
                torch.linalg.vector_norm(velocity_target, dim=-1),
            )
        beta = zero
        if self.weights["beta"]:
            target_speed = torch.linalg.vector_norm(velocity_target, dim=-1)
            beta_mask = target_speed >= self.beta_min_speed
            if bool(beta_mask.any()):
                predicted_beta = torch.atan2(
                    output.predicted_velocity[beta_mask, 1],
                    output.predicted_velocity[beta_mask, 0],
                )
                target_beta = torch.atan2(
                    velocity_target[beta_mask, 1], velocity_target[beta_mask, 0]
                )
                beta = F.smooth_l1_loss(
                    wrapped_angle_difference(predicted_beta, target_beta),
                    torch.zeros_like(predicted_beta),
                )
        rotation = zero
        if self.weights["rotation"]:
            angular_target = target["angular_velocity_camera"][
                :, -output.pair_twist_camera.shape[1] :
            ]
            rotation = F.smooth_l1_loss(
                output.pair_twist_camera[..., 3:], angular_target
            )
        reprojection = zero
        if self.weights["reprojection"]:
            observed_flow = visual["flows"][
                :, -output.pair_predicted_flow.shape[1] :
            ]
            flow_confidence = visual["flow_confidence"][
                :, -output.pair_predicted_flow.shape[1] :
            ]
            reprojection_error = torch.linalg.vector_norm(
                output.pair_predicted_flow - observed_flow, dim=2
            )
            reprojection = (
                reprojection_error
                * flow_confidence[:, :, 0]
                * output.candidate_mask[:, :, 0]
            ).sum() / (
                flow_confidence[:, :, 0] * output.candidate_mask[:, :, 0]
            ).sum().clamp_min(1.0)
        residual = zero
        if self.weights["residual"]:
            residual = output.visual_correction.square().mean()
        support = zero
        if self.weights["support"]:
            support = (
                F.relu(self.support_floor - output.pair_support_fraction)
                .square()
                .mean()
            )
        tv = zero
        if self.weights["tv"]:
            tv = (
                output.reliability[..., 1:, :] - output.reliability[..., :-1, :]
            ).abs().mean() + (
                output.reliability[..., :, 1:] - output.reliability[..., :, :-1]
            ).abs().mean()
        values = {
            "nll": nll,
            "direct": direct,
            "geometry": geometry,
            "speed": speed,
            "beta": beta,
            "rotation": rotation,
            "reprojection": reprojection,
            "residual": residual,
            "support": support,
            "tv": tv,
        }
        total = sum(
            (self.weights[name] * value for name, value in values.items()), zero
        )
        return V3LossTerms(total=total, **values)
