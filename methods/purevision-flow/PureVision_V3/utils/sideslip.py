"""Sideslip at the velocity label's reference point, not a verified CoG label."""
from __future__ import annotations

import torch


def sideslip_values(
    prediction: torch.Tensor, target: torch.Tensor, min_speed: float = 2.0
) -> dict[str, torch.Tensor]:
    if min_speed <= 0:
        raise ValueError("Sideslip minimum speed must be positive.")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("Non-finite velocity in sideslip evaluation.")
    pred_defined = torch.linalg.vector_norm(prediction, dim=-1) > 1.0e-6
    target_defined = torch.linalg.vector_norm(target, dim=-1) > 1.0e-6
    eligible = torch.linalg.vector_norm(target, dim=-1) >= min_speed
    pred = torch.atan2(prediction[:, 1], prediction[:, 0])
    truth = torch.atan2(target[:, 1], target[:, 0])
    difference = pred - truth
    error = torch.atan2(torch.sin(difference), torch.cos(difference))
    # Undefined predicted direction must not silently improve the common-set MAE.
    error = torch.where(pred_defined, error, torch.full_like(error, torch.pi))
    return {
        "pred_beta_deg": torch.where(pred_defined, torch.rad2deg(pred), float("nan")),
        "target_beta_deg": torch.where(target_defined, torch.rad2deg(truth), float("nan")),
        "beta_error_deg": torch.where(eligible, torch.rad2deg(error), float("nan")),
        "beta_eval_valid": eligible,
        "pred_beta_defined": pred_defined,
    }
