from __future__ import annotations

from typing import Optional

import torch


def camera_to_reference_velocity(
    camera_velocity: torch.Tensor,
    rotation_reference_from_camera: torch.Tensor,
) -> torch.Tensor:
    if camera_velocity.shape[-1] != 3:
        raise ValueError("camera_velocity must end with three components.")
    return torch.matmul(rotation_reference_from_camera, camera_velocity.unsqueeze(-1)).squeeze(-1)


def correct_reference_to_cg(
    reference_velocity: torch.Tensor,
    angular_rate_reference: torch.Tensor,
    reference_lever_arm_from_cg: Optional[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """Convert a reference-point velocity to CG only when a lever arm is supplied."""

    if reference_lever_arm_from_cg is None:
        return reference_velocity, False
    if reference_lever_arm_from_cg.shape != reference_velocity.shape:
        raise ValueError("reference lever arm must match the velocity shape.")
    correction = torch.linalg.cross(angular_rate_reference, reference_lever_arm_from_cg, dim=-1)
    return reference_velocity - correction, True
