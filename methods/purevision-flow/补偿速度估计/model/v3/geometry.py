from __future__ import annotations

import torch
import torch.nn as nn

from model.camera_geometry import road_plane_depth
from model.camera_geometry import pixel_grid

from .types import PairTwistEstimate


def motion_design_matrix(
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Return the pixel-displacement Jacobian for [v, omega]."""

    batch_size, height, width = depth.shape
    u, v = pixel_grid(height, width, intrinsics)
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    x = (u[None] - cx) / fx
    y = (v[None] - cy) / fy
    inv_depth = depth.reciprocal()
    zeros = torch.zeros_like(x)

    row_x = torch.stack(
        (
            -fx * inv_depth,
            zeros,
            fx * x * inv_depth,
            fx * x * y,
            -fx * (1.0 + x.square()),
            fx * y,
        ),
        dim=-1,
    )
    row_y = torch.stack(
        (
            zeros,
            -fy * inv_depth,
            fy * y * inv_depth,
            fy * (1.0 + y.square()),
            -fy * x * y,
            -fy * x,
        ),
        dim=-1,
    )
    design = torch.stack((row_x, row_y), dim=-2)
    return design * dt.reshape(batch_size, 1, 1, 1, 1)


def motion_flow_from_twist(
    twist_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    design = motion_design_matrix(intrinsics, depth, dt)
    flow = torch.einsum("bhwij,bj->bhwi", design, twist_camera)
    return flow.permute(0, 3, 1, 2).contiguous()


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def transform_camera_twist_to_ego(
    twist_camera: torch.Tensor,
    covariance_camera: torch.Tensor,
    rotation_ego_from_camera: torch.Tensor,
    translation_ego_from_camera: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a camera-centre twist to the nuScenes ego reference point."""

    leading_dims = twist_camera.shape[:-1]
    rotation = rotation_ego_from_camera
    translation = translation_ego_from_camera
    while rotation.ndim < twist_camera.ndim + 1:
        rotation = rotation.unsqueeze(1)
        translation = translation.unsqueeze(1)
    rotation = rotation.expand(*leading_dims, 3, 3)
    translation = translation.expand(*leading_dims, 3)

    velocity_camera = twist_camera[..., :3]
    angular_camera = twist_camera[..., 3:]
    velocity_ego_at_camera = torch.matmul(
        rotation, velocity_camera.unsqueeze(-1)
    ).squeeze(-1)
    angular_ego = torch.matmul(rotation, angular_camera.unsqueeze(-1)).squeeze(-1)
    velocity_ego = velocity_ego_at_camera - torch.linalg.cross(
        angular_ego, translation, dim=-1
    )
    twist_ego = torch.cat((velocity_ego, angular_ego), dim=-1)

    jacobian = covariance_camera.new_zeros(*leading_dims, 6, 6)
    jacobian[..., :3, :3] = rotation
    jacobian[..., :3, 3:] = skew(translation) @ rotation
    jacobian[..., 3:, 3:] = rotation
    covariance_ego = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    return twist_ego, covariance_ego


class RobustVisualTwistSolver(nn.Module):
    def __init__(
        self,
        irls_iterations: int = 4,
        huber_delta_px: float = 1.5,
        min_points: int = 48,
        max_condition_number: float = 1.0e8,
        regularization: float = 1.0e-6,
        twist_dof: int = 6,
    ) -> None:
        super().__init__()
        if twist_dof not in {3, 6}:
            raise ValueError("twist_dof must be 3 or 6.")
        self.irls_iterations = int(irls_iterations)
        self.huber_delta_px = float(huber_delta_px)
        self.min_points = int(min_points)
        self.max_condition_number = float(max_condition_number)
        self.regularization = float(regularization)
        self.twist_dof = int(twist_dof)

    def forward(
        self,
        flow: torch.Tensor,
        confidence: torch.Tensor,
        intrinsics: torch.Tensor,
        dt: torch.Tensor,
        plane_normal_camera: torch.Tensor,
        plane_offset: torch.Tensor,
        reliability: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> PairTwistEstimate:
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("flow must have shape [B,2,H,W].")
        batch_size, _, height, width = flow.shape
        if confidence.shape == (batch_size, height, width):
            confidence = confidence.unsqueeze(1)
        if confidence.shape != (batch_size, 1, height, width):
            raise ValueError("confidence must have shape [B,1,H,W].")
        if reliability is None:
            reliability = torch.ones_like(confidence)
        if reliability.shape != confidence.shape:
            raise ValueError("reliability must match confidence.")

        depth, depth_valid = road_plane_depth(
            intrinsics, plane_normal_camera, plane_offset, height, width
        )
        candidate = depth_valid & torch.isfinite(flow).all(dim=1)
        candidate &= torch.isfinite(confidence[:, 0]) & torch.isfinite(
            reliability[:, 0]
        )
        candidate &= (torch.isfinite(dt) & (dt > 1.0e-4))[:, None, None]
        if support_mask is not None:
            mask = support_mask[:, 0] if support_mask.ndim == 4 else support_mask
            if mask.shape != candidate.shape:
                raise ValueError("support_mask must have shape [B,1,H,W] or [B,H,W].")
            candidate &= mask.bool()

        point_weight = (
            torch.nan_to_num(confidence[:, 0], nan=0.0, posinf=0.0, neginf=0.0).clamp(
                0.0, 1.0
            )
            * torch.nan_to_num(
                reliability[:, 0], nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(0.0, 1.0)
            * candidate.to(flow.dtype)
        )
        safe_dt = torch.nan_to_num(
            dt, nan=1.0e-4, posinf=1.0e-4, neginf=1.0e-4
        ).clamp_min(1.0e-4)
        design = motion_design_matrix(intrinsics, depth, safe_dt)
        estimates = [
            self._solve_one(
                flow[index], design[index], point_weight[index], candidate[index]
            )
            for index in range(batch_size)
        ]
        twist = torch.stack([item[0] for item in estimates])
        covariance = torch.stack([item[1] for item in estimates])
        condition = torch.stack([item[2] for item in estimates])
        residual = torch.stack([item[3] for item in estimates])
        valid = torch.stack([item[4] for item in estimates])
        support_count = torch.stack([item[5] for item in estimates])
        depth_count = depth_valid.flatten(1).sum(dim=-1).clamp_min(1)
        support_fraction = point_weight.flatten(1).sum(dim=-1) / depth_count.to(
            flow.dtype
        )
        predicted_flow = motion_flow_from_twist(twist, intrinsics, depth, safe_dt)
        return PairTwistEstimate(
            twist_camera=twist,
            covariance_camera=covariance,
            condition_number=condition,
            residual_rms_px=residual,
            valid=valid,
            support_count=support_count,
            support_fraction=support_fraction,
            predicted_flow=predicted_flow,
            candidate_mask=candidate[:, None],
        )

    def _solve_one(
        self,
        flow: torch.Tensor,
        design: torch.Tensor,
        point_weight: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        parameter_count = self.twist_dof
        design_rows = design[..., :parameter_count].reshape(-1, parameter_count)
        observation = torch.nan_to_num(flow).permute(1, 2, 0).reshape(-1)
        base_weight = torch.where(
            candidate, point_weight, torch.zeros_like(point_weight)
        ).reshape(-1)
        weights = base_weight.repeat_interleave(2)
        support_count = (base_weight > 0.0).sum()
        eye = torch.eye(parameter_count, device=flow.device, dtype=flow.dtype)
        parameters = torch.zeros(parameter_count, device=flow.device, dtype=flow.dtype)

        for _ in range(self.irls_iterations):
            normal = (
                design_rows.T @ (weights[:, None] * design_rows)
                + self.regularization * eye
            )
            rhs = design_rows.T @ (weights * observation)
            parameters = torch.linalg.solve(normal, rhs)
            residual_vector = (design_rows @ parameters - observation).reshape(-1, 2)
            residual_norm = torch.linalg.vector_norm(residual_vector, dim=-1)
            robust = torch.where(
                residual_norm <= self.huber_delta_px,
                torch.ones_like(residual_norm),
                self.huber_delta_px / residual_norm.clamp_min(1.0e-6),
            )
            weights = (base_weight * robust).repeat_interleave(2)

        normal_unregularized = design_rows.T @ (weights[:, None] * design_rows)
        singular_values = torch.linalg.svdvals(normal_unregularized)
        condition = singular_values[0] / singular_values[-1].clamp_min(1.0e-12)
        final_residual = design_rows @ parameters - observation
        weighted_sse = (weights * final_residual.square()).sum()
        effective_rows = (weights > 0.0).sum()
        variance = weighted_sse / (effective_rows - parameter_count).clamp_min(1)
        covariance_small = variance * torch.linalg.inv(
            normal_unregularized + self.regularization * eye
        )
        residual_rms = torch.sqrt(weighted_sse / weights.sum().clamp_min(1.0))
        is_valid = (
            (support_count >= self.min_points)
            & (singular_values[-1] > self.regularization)
            & torch.isfinite(condition)
            & (condition <= self.max_condition_number)
            & torch.isfinite(parameters).all()
            & torch.isfinite(covariance_small).all()
        )

        twist = flow.new_zeros(6)
        twist[:parameter_count] = torch.where(
            is_valid, parameters, torch.zeros_like(parameters)
        )
        covariance = torch.eye(6, device=flow.device, dtype=flow.dtype) * 1.0e6
        if parameter_count == 6:
            covariance = torch.where(is_valid, covariance_small, covariance)
        else:
            covariance = covariance.clone()
            covariance[:3, :3] = torch.where(
                is_valid, covariance_small, covariance[:3, :3]
            )
        finite_condition = torch.where(
            is_valid, condition, torch.full_like(condition, float("inf"))
        )
        return (
            twist,
            covariance,
            finite_condition,
            residual_rms,
            is_valid,
            support_count,
        )


def fuse_twist_estimates(
    twist: torch.Tensor,
    covariance: torch.Tensor,
    valid: torch.Tensor,
    huber_delta_mps: float = 2.0,
    regularization: float = 1.0e-6,
    residual_rms_px: torch.Tensor | None = None,
    condition_number: torch.Tensor | None = None,
    aggregation: str = "precision",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse pair estimates and include between-pair dispersion.

    ``precision`` preserves the original covariance-only fusion.  The optional
    ``quality_weighted`` mode additionally weights each pair by its optical-flow
    residual and geometric condition number.
    """

    if aggregation not in {"precision", "quality_weighted"}:
        raise ValueError("aggregation must be precision or quality_weighted.")
    if aggregation == "quality_weighted":
        if residual_rms_px is None or condition_number is None:
            raise ValueError(
                "quality_weighted aggregation requires residual and condition inputs."
            )
        if residual_rms_px.shape != valid.shape or condition_number.shape != valid.shape:
            raise ValueError("quality inputs must have shape [B,P].")

    batch_size, _, state_dim = twist.shape
    eye = torch.eye(state_dim, device=twist.device, dtype=twist.dtype)
    fused_twists = []
    fused_covariances = []
    fused_valid = []
    dispersions = []
    for index in range(batch_size):
        selected = valid[index]
        if not bool(selected.any()):
            fused_twists.append(twist.new_zeros(state_dim))
            fused_covariances.append(eye * 1.0e6)
            fused_valid.append(torch.tensor(False, device=twist.device))
            dispersions.append(twist.new_tensor(0.0))
            continue
        values = twist[index, selected]
        covariances = covariance[index, selected]
        precision = torch.linalg.inv(covariances + regularization * eye)
        if aggregation == "quality_weighted":
            residual = torch.nan_to_num(
                residual_rms_px[index, selected], nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            ).clamp_min(0.0)
            condition = torch.nan_to_num(
                condition_number[index, selected], nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            ).clamp_min(0.0)
            quality = 1.0 / ((residual + 0.05) * torch.sqrt(condition + 1.0))
            quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
            if bool((quality > 0.0).any()):
                quality = quality / quality.mean().clamp_min(1.0e-12)
            else:
                quality = torch.ones_like(quality)
            precision = precision * quality[:, None, None]
        precision_sum = precision.sum(dim=0) + regularization * eye
        rhs = torch.einsum("nij,nj->i", precision, values)
        fused = torch.linalg.solve(precision_sum, rhs)

        translation_error = torch.linalg.vector_norm(values[:, :3] - fused[:3], dim=-1)
        robust = torch.where(
            translation_error <= huber_delta_mps,
            torch.ones_like(translation_error),
            huber_delta_mps / translation_error.clamp_min(1.0e-6),
        )
        precision = precision * robust[:, None, None]
        precision_sum = precision.sum(dim=0) + regularization * eye
        rhs = torch.einsum("nij,nj->i", precision, values)
        fused = torch.linalg.solve(precision_sum, rhs)
        delta = values - fused
        between = torch.einsum("ni,nj->ij", delta, delta) / max(values.shape[0], 1)
        fused_covariance = torch.linalg.inv(precision_sum) + between
        dispersion = torch.sqrt(delta[:, :3].square().sum(dim=-1).mean())
        fused_twists.append(fused)
        fused_covariances.append(fused_covariance)
        fused_valid.append(torch.tensor(True, device=twist.device))
        dispersions.append(dispersion)
    return (
        torch.stack(fused_twists),
        torch.stack(fused_covariances),
        torch.stack(fused_valid).bool(),
        torch.stack(dispersions),
    )
