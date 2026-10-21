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
