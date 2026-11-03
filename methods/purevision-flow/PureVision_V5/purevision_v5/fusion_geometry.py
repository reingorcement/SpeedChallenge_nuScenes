"""Training-only finite camera motion from planar endpoint velocity.

The yaw-plane velocity is at the target ego origin, in target yaw axes.
Full offline orientations and both calibrated camera lever arms are constants.
The endpoint velocity times the last actual interval approximates displacement;
no four-pair aggregation, learned rotation, or inference-time correction exists.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SCHEMA = 'v5_s23_geometry_contract_v1'
FIELDS = ('points', 'flow', 'mask', 'K', 'R21', 'base', 'velocity_map', 'dt', 'valid')


def theoretical_flow(velocity, geometry):
    """X_c2 = R21 X_c1 + base + velocity_map @ [vx,vy]."""
    if velocity.ndim != 2 or velocity.shape[1] != 2 or velocity.dtype != torch.float32:
        raise ValueError('Expected FP32 endpoint velocity [B,2]')
    g = {k: geometry[k].detach() for k in FIELDS}
    points = g['points']
    translated = torch.einsum('bij,bpj->bpi', g['R21'], points)
    translated = translated + (g['base'] + torch.einsum('bij,bj->bi', g['velocity_map'], velocity))[:, None]
    projected = torch.einsum('bij,bpj->bpi', g['K'], translated)
    # A fixed support is never changed by the prediction. The positive-depth
    # clamp is only numerical protection, and its incidence is reported.
    xy = projected[..., :2] / projected[..., 2:].clamp_min(.1)
    source = torch.einsum('bij,bpj->bpi', g['K'], points)
    source_xy = source[..., :2] / source[..., 2:].clamp_min(.1)
    return xy - source_xy, translated[..., 2]


def geometry_loss(velocity, geometry, return_details=False):
    flow, z = theoretical_flow(velocity, geometry)
    mask = geometry['mask'].detach().bool() & geometry['valid'].detach().bool()[:, None]
    dt = geometry['dt'].detach()
    focal = torch.stack((geometry['K'][:, 0, 0], geometry['K'][:, 1, 1]), -1).detach()
    residual = (flow - geometry['flow'].detach()) / (focal * dt[:, None])[:, None]
    if not torch.isfinite(residual).all():
        raise ValueError('Nonfinite geometric residual')
    robust = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.01, reduction='none')
    # Equal mean over points and axes per window, then ALL batch windows.
    counts = mask.sum(1)
    per_window = (robust * mask[..., None]).sum((1, 2)) / (2 * counts.clamp_min(1))
    value = per_window.mean()
    if not return_details:
        return value
    return value, dict(valid_windows=counts.gt(0).sum().detach(), valid_points=counts.sum().detach(),
        residual_abs_sum=(residual.abs() * mask[..., None]).sum().detach(),
        nonpositive_predicted_depth=((z <= .1) & mask).sum().detach(),
        normalization='pixel residual / focal_pixels / actual_pair_seconds = image motion rate s^-1')


def validate_packet(packet):
    required = set(FIELDS)
    if set(packet) != required:
        raise ValueError('Incomplete geometry, missing quantities cannot be zero-filled')
    n, p, d = packet['points'].shape
    shapes = dict(flow=(n,p,2),mask=(n,p),K=(n,3,3),R21=(n,3,3),base=(n,3),
                  velocity_map=(n,3,2),dt=(n,),valid=(n,))
    if d != 3 or not 1 <= p <= 256:
        raise ValueError('At most256 source road points per window')
    for key, shape in shapes.items():
        if tuple(packet[key].shape) != shape:
            raise ValueError(f'Geometry shape differs: {key}')
    for key, value in packet.items():
        if value.requires_grad or (key not in {'mask','valid'} and
                                  (value.dtype != torch.float32 or not torch.isfinite(value).all())):
            raise ValueError('Geometry must be finite, FP32 and stop-gradient')
    if packet['mask'].dtype != torch.bool or packet['valid'].dtype != torch.bool or (packet['dt'] <= 0).any():
        raise ValueError('Invalid fixed mask or actual interval')
    if (packet['valid'] & packet['mask'].sum(1).lt(32)).any():
        raise ValueError('Valid geometry requires32 fixed road points')


def to_device(packet, device):
    return {key:value.to(device,non_blocking=True) for key,value in packet.items()}


def select(packet, indices):
    return {key:value[indices] for key,value in packet.items()}
