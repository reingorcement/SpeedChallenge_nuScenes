from __future__ import annotations

import torch


def scale_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Scale camera intrinsics from (height, width) to a resized image."""

    source_h, source_w = source_size
    target_h, target_w = target_size
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("Image dimensions must be positive.")
    scaled = intrinsics.clone()
    scaled[..., 0, :] *= float(target_w) / float(source_w)
    scaled[..., 1, :] *= float(target_h) / float(source_h)
    scaled[..., 2, :] = intrinsics[..., 2, :]
    return scaled


def pixel_grid(height: int, width: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v, u = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    return u, v


def rotational_flow(
    intrinsics: torch.Tensor,
    angular_rate_camera: torch.Tensor,
    height: int,
    width: int,
    dt: torch.Tensor,
) -> torch.Tensor:
    """First-order pixel flow caused by camera angular velocity [wx, wy, wz]."""

    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B,3,3].")
    if angular_rate_camera.shape != (intrinsics.shape[0], 3):
        raise ValueError("angular_rate_camera must have shape [B,3].")
    dt = dt.reshape(-1, 1, 1)
    u, v = pixel_grid(height, width, intrinsics)
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    x = (u[None] - cx) / fx
    y = (v[None] - cy) / fy
    wx, wy, wz = angular_rate_camera.unbind(dim=-1)
    wx = wx[:, None, None]
    wy = wy[:, None, None]
    wz = wz[:, None, None]
    dx = x * y * wx - (1.0 + x.square()) * wy + y * wz
    dy = (1.0 + y.square()) * wx - x * y * wy - x * wz
    return torch.stack((fx * dx * dt, fy * dy * dt), dim=1)
