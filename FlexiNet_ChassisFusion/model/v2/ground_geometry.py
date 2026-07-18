from __future__ import annotations

import torch
import torch.nn as nn

from .lever_arm import camera_to_reference_velocity
from .rotation_flow import pixel_grid, rotational_flow
from .types import GeometryEstimate


def road_plane_depth(
    intrinsics: torch.Tensor,
    plane_normal_camera: torch.Tensor,
    plane_offset: torch.Tensor,
    height: int,
    width: int,
    min_depth: float = 0.5,
    max_depth: float = 150.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ray depth Z for n^T X + d = 0 and its valid support mask."""

    u, v = pixel_grid(height, width, intrinsics)
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    x = (u[None] - cx) / fx
    y = (v[None] - cy) / fy
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    denominator = (rays * plane_normal_camera[:, None, None, :]).sum(dim=-1)
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-8,
        denominator,
        torch.full_like(denominator, 1.0e-8),
    )
    depth = -plane_offset[:, None, None] / safe_denominator
    valid = (
        torch.isfinite(depth)
        & (denominator.abs() > 1.0e-8)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )
    depth = torch.nan_to_num(depth, nan=min_depth, posinf=max_depth, neginf=min_depth)
    return depth.clamp(min=min_depth, max=max_depth), valid


def translation_flow_from_velocity(
    camera_velocity: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Generate translational pixel flow for synthetic tests and diagnostics."""

    height, width = depth.shape[-2:]
    u, v = pixel_grid(height, width, intrinsics)
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    x = (u[None] - cx) / fx
    y = (v[None] - cy) / fy
    tx, ty, tz = camera_velocity.unbind(dim=-1)
    inv_depth = depth.reciprocal()
    flow_x = fx * (-tx[:, None, None] + x * tz[:, None, None]) * inv_depth
    flow_y = fy * (-ty[:, None, None] + y * tz[:, None, None]) * inv_depth
    return torch.stack((flow_x * dt[:, None, None], flow_y * dt[:, None, None]), dim=1)


class RobustRoadPlaneVelocitySolver(nn.Module):
    def __init__(
        self,
        irls_iterations: int = 4,
        huber_delta_px: float = 1.5,
        min_points: int = 32,
        max_condition_number: float = 1.0e6,
        regularization: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.irls_iterations = int(irls_iterations)
        self.huber_delta_px = float(huber_delta_px)
        self.min_points = int(min_points)
        self.max_condition_number = float(max_condition_number)
        self.regularization = float(regularization)

    def forward(
        self,
        flow: torch.Tensor,
        confidence: torch.Tensor,
        intrinsics: torch.Tensor,
        dt: torch.Tensor,
        angular_rate_camera: torch.Tensor,
        angular_rate_valid: torch.Tensor,
        plane_normal_camera: torch.Tensor,
        plane_offset: torch.Tensor,
        rotation_ego_from_camera: torch.Tensor,
        road_mask: torch.Tensor | None = None,
    ) -> GeometryEstimate:
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("flow must have shape [B,2,H,W].")
        batch_size, _, height, width = flow.shape
        if confidence.shape == (batch_size, height, width):
            confidence = confidence.unsqueeze(1)
        if confidence.shape != (batch_size, 1, height, width):
            raise ValueError("confidence must have shape [B,1,H,W].")
        if angular_rate_valid.shape != angular_rate_camera.shape:
            raise ValueError("angular_rate_valid must provide a mask for each angular-rate axis.")
        gyro = torch.where(angular_rate_valid, angular_rate_camera, torch.zeros_like(angular_rate_camera))
        rotation = rotational_flow(intrinsics, gyro, height, width, dt)
        translation_flow = flow - rotation
        depth, depth_valid = road_plane_depth(
            intrinsics, plane_normal_camera, plane_offset, height, width
        )
        valid = depth_valid & torch.isfinite(translation_flow).all(dim=1)
        valid &= torch.isfinite(confidence[:, 0]) & (confidence[:, 0] > 0.0)
        valid &= (torch.isfinite(dt) & (dt > 1.0e-4))[:, None, None]
        if road_mask is not None:
            valid &= road_mask[:, 0] if road_mask.ndim == 4 else road_mask

        estimates = [
            self._solve_one(
                translation_flow[i], confidence[i, 0], intrinsics[i], depth[i], dt[i], valid[i]
            )
            for i in range(batch_size)
        ]
        camera_velocity = torch.stack([item[0] for item in estimates])
        covariance = torch.stack([item[1] for item in estimates])
        condition = torch.stack([item[2] for item in estimates])
        residual = torch.stack([item[3] for item in estimates])
        solve_valid = torch.stack([item[4] for item in estimates])
        support_count = torch.stack([item[5] for item in estimates])
        ego_3d = camera_to_reference_velocity(camera_velocity, rotation_ego_from_camera)
        ego_velocity = ego_3d[..., :2]
        return GeometryEstimate(
            camera_velocity=camera_velocity,
            ego_velocity=ego_velocity,
            covariance=covariance,
            condition_number=condition,
            residual_rms=residual,
            valid=solve_valid,
            support_count=support_count,
            translation_flow=translation_flow,
        )

    def _solve_one(
        self,
        flow: torch.Tensor,
        confidence: torch.Tensor,
        intrinsics: torch.Tensor,
        depth: torch.Tensor,
        dt: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = depth.shape
        u, v = pixel_grid(height, width, intrinsics)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        inv_depth = depth.reciprocal()
        zeros = torch.zeros_like(x)
        row_x = torch.stack((-fx * inv_depth, zeros, fx * x * inv_depth), dim=-1)
        row_y = torch.stack((zeros, -fy * inv_depth, fy * y * inv_depth), dim=-1)
        design = torch.stack((row_x, row_y), dim=-2).reshape(-1, 3)
        safe_dt = torch.nan_to_num(dt, nan=1.0e-4, posinf=1.0e-4, neginf=1.0e-4).clamp_min(1.0e-4)
        observation = torch.nan_to_num(flow / safe_dt).permute(1, 2, 0).reshape(-1)
        point_weight = torch.where(
            valid,
            torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0),
            torch.zeros_like(confidence),
        ).reshape(-1)
        weights = point_weight.repeat_interleave(2)
        support_count = (point_weight > 0.0).sum()
        eye = torch.eye(3, device=flow.device, dtype=flow.dtype)
        velocity = torch.zeros(3, device=flow.device, dtype=flow.dtype)

        for _ in range(self.irls_iterations):
            normal = design.T @ (weights[:, None] * design) + self.regularization * eye
            rhs = design.T @ (weights * observation)
            velocity = torch.linalg.solve(normal, rhs)
            residual_vector = (design @ velocity - observation).reshape(-1, 2)
            residual_norm = torch.linalg.vector_norm(residual_vector, dim=-1)
            robust = torch.where(
                residual_norm <= self.huber_delta_px,
                torch.ones_like(residual_norm),
                self.huber_delta_px / residual_norm.clamp_min(1.0e-6),
            )
            weights = (point_weight * robust).repeat_interleave(2)

        normal_unregularized = design.T @ (weights[:, None] * design)
        singular_values = torch.linalg.svdvals(normal_unregularized)
        condition = singular_values[0] / singular_values[-1].clamp_min(1.0e-12)
        final_residual = design @ velocity - observation
        weighted_sse = (weights * final_residual.square()).sum()
        effective_rows = (weights > 0.0).sum()
        variance = weighted_sse / (effective_rows - 3).clamp_min(1)
        covariance = variance * torch.linalg.inv(normal_unregularized + self.regularization * eye)
        residual_rms = torch.sqrt(weighted_sse / weights.sum().clamp_min(1.0))
        is_valid = (
            (support_count >= self.min_points)
            & (singular_values[-1] > self.regularization)
            & torch.isfinite(condition)
            & (condition <= self.max_condition_number)
            & torch.isfinite(velocity).all()
            & torch.isfinite(covariance).all()
        )
        finite_velocity = torch.where(is_valid, velocity, torch.zeros_like(velocity))
        finite_covariance = torch.where(
            is_valid,
            covariance,
            eye * torch.as_tensor(1.0e6, device=flow.device, dtype=flow.dtype),
        )
        finite_condition = torch.where(is_valid, condition, torch.full_like(condition, float("inf")))
        return finite_velocity, finite_covariance, finite_condition, residual_rms, is_valid, support_count
