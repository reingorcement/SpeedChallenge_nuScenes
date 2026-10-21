"""Finite, non-differentiable geometry with explicit per-pair inputs.

All six twist components are estimated jointly. Do not subtract rotation flow
again. Training consumes frozen geometry features, never a fictitious gradient.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ._reference.finite_geometry import camera_motion, left_jacobian, project, skew
from ._reference.finite_geometry import solve as reference_solve

SOLVER_KEYS = {"max_points", "min_points", "huber_delta_px", "max_condition", "max_nfev"}
NUMERICAL_ERRORS = (cv2.error, np.linalg.LinAlgError, FloatingPointError, ValueError)


def invalid(reason, count=0):
    return dict(valid=False, reason=reason, count=count, condition=float("inf"), residual=float("nan"),
                twist=np.zeros(6), covariance=np.eye(6) * 1e6, r21=np.eye(3), t21=np.zeros(3),
                candidate_count=0, ambiguity_gap_px=float("nan"))


def _ambiguous(p, q, k, n, height, weights, result, config):
    """Refine competitive distinct planar poses; duplicate initializers are harmless."""
    gap = config.get("min_ambiguity_gap_px", .05)
    if gap <= 0 or result.get("ambiguity_gap_px", float("inf")) > gap:
        return False
    rays = np.c_[p, np.ones(len(p))] @ np.linalg.inv(k).T
    depth = -height / (rays @ n)
    keep = np.isfinite(depth) & (depth >= .5) & (depth <= 150) & (weights > 0)
    points, obs, wt = rays[keep] * depth[keep, None], q[keep], np.sqrt(weights[keep].clip(0, 1))
    candidates = cv2.solvePnPGeneric(np.ascontiguousarray(points), np.ascontiguousarray(obs), k, None, flags=cv2.SOLVEPNP_IPPE)
    if not candidates[0]:
        return False
    def residual(a):
        return ((project(points, cv2.Rodrigues(a[:3])[0], a[3:], k) - obs) * wt[:, None]).ravel()
    for rv, tv in zip(candidates[1], candidates[2]):
        guess = np.r_[rv.ravel(), tv.ravel()]
        r = cv2.Rodrigues(guess[:3])[0]
        if np.mean((points @ r.T + guess[3:])[:, 2] > .1) < .99:
            continue
        initial_rms = np.sqrt(np.sum(residual(guess) ** 2) / max(2 * np.sum(wt ** 2), 1e-12))
        if initial_rms > result["residual"] + gap:
            continue
        fit = least_squares(residual, guess, loss="huber", f_scale=config["huber_delta_px"],
                            max_nfev=config["max_nfev"], ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if not fit.success:
            continue
        r = cv2.Rodrigues(fit.x[:3])[0]
        rms = np.sqrt(np.sum(residual(fit.x) ** 2) / max(2 * np.sum(wt ** 2), 1e-12))
        angle = np.degrees(np.linalg.norm(Rotation.from_matrix(r @ result["r21"].T).as_rotvec()))
        distance = np.linalg.norm(fit.x[3:] - result["t21"])
        distinct = angle > config.get("ambiguity_rotation_deg", 2.) or distance > config.get("ambiguity_translation_m", .1)
        if distinct and rms <= result["residual"] + gap and np.mean((points @ r.T + fit.x[3:])[:, 2] > .1) >= .99:
            return True
    return False


def solve(p, q, k, n, height, dt, weights=None, **config):
    p, q, k, n = (np.asarray(x, float) for x in (p, q, k, n))
    if p.ndim != 2 or p.shape[1:] != (2,) or q.shape != p.shape or k.shape != (3, 3) or n.shape != (3,):
        raise ValueError("Invalid point or calibration shape")
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(p),):
        raise ValueError("Weights must match points")
    if not np.isfinite(dt) or dt <= 1e-4:
        return invalid("invalid_interval")
    if not np.isfinite(k).all() or not np.isfinite(n).all() or not np.isfinite(height) or height <= 0:
        return invalid("invalid_calibration")
    keep = np.isfinite(p).all(1) & np.isfinite(q).all(1) & np.isfinite(weights) & (weights > 0)
    p, q, weights = p[keep], q[keep], weights[keep].clip(0, 1)
    max_points = config.get("max_points", 1024)
    # Match the deployed adapter's deterministic sampling; no truth-dependent selection.
    if len(p) > max_points:
        ids = np.linspace(0, len(p) - 1, max_points).astype(int)
        p, q, weights = p[ids], q[ids], weights[ids]
    try:
        result = reference_solve(p, q, k, n, height, dt, weights=weights,
                                 **{key: value for key, value in config.items() if key in SOLVER_KEYS})
        if not result["valid"]:
            return result
        reason = None
        if not np.isfinite(result["residual"]) or result["residual"] > config.get("max_residual_px", 2.):
            reason = "large_reprojection_residual"
        elif not np.isfinite(result["covariance"]).all():
            reason = "invalid_covariance"
        elif _ambiguous(p, q, k, n, height, weights, result, config):
            reason = "ambiguous_pose"
        if reason:
            result.update(valid=False, reason=reason, twist=np.zeros(6), covariance=np.eye(6) * 1e6)
        else:
            cov = (result["covariance"] + result["covariance"].T) / 2
            eig, basis = np.linalg.eigh(cov)
            result["covariance"] = (basis * np.maximum(eig, config.get("covariance_floor", 1e-8))) @ basis.T
        return result
    except NUMERICAL_ERRORS as error:
        result = invalid("numerical_failure", len(p))
        result["error_type"] = type(error).__name__
        return result


def to_body(twist, covariance, cal):
    c = np.asarray(cal["rotation_body_from_camera"], float)
    lever = np.asarray(cal["camera_lever_body_m"], float)
    j = np.zeros((6, 6))
    j[:3, :3] = c
    j[:3, 3:] = skew(lever) @ c
    j[3:, 3:] = c
    return j @ twist, j @ covariance @ j.T


def solve_pair(flow, confidence, reliability, k, normal, cal, dt, config):
    flow, confidence, reliability = (np.asarray(x, float) for x in (flow, confidence, reliability))
    if flow.ndim != 3 or flow.shape[0] != 2 or confidence.shape != flow.shape[1:] or reliability.shape != confidence.shape:
        raise ValueError("Expected flow [2,H,W] and weights [H,W]")
    h, w = confidence.shape
    y, x = np.mgrid[:h, :w]
    pixels = np.stack((x, y), -1).reshape(-1, 2)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(k).T
    denominator = rays @ normal
    depth = -cal["height_m"] / np.where(abs(denominator) > 1e-12, denominator, 1e-12)
    weight = np.nan_to_num(confidence, nan=0., posinf=0., neginf=0.).clip(0, 1) * np.nan_to_num(reliability, nan=0., posinf=0., neginf=0.).clip(0, 1)
    valid = (depth >= .5) & (depth <= 150) & np.isfinite(flow).all(0).ravel() & (weight.ravel() > 0)
    p, f = pixels[valid], flow.transpose(1, 2, 0).reshape(-1, 2)[valid]
    result = solve(p, p + f, k, normal, cal["height_m"], dt, weights=weight.ravel()[valid], **config)
    result["twist_body"], result["covariance_body"] = to_body(result["twist"], result["covariance"], cal)
    result.update(normal=np.asarray(normal).tolist(), dt=float(dt), support_fraction=float(weight.ravel()[valid].sum() / max(np.count_nonzero((depth >= .5) & (depth <= 150)), 1)))
    return result


def aggregate(pairs, source_times, target_times, output_time):
    if len(pairs) != len(source_times) or len(pairs) != len(target_times):
        raise ValueError("Each pair must have its own timestamps")
    valid = np.asarray([p["valid"] for p in pairs], bool)
    result = dict(valid=bool(valid.any()), valid_pairs=int(valid.sum()), pair_valid=valid,
                  velocity=np.zeros(3), covariance=np.eye(3) * 1e6, dispersion=0.,
                  effective_time_s=None, age_s=0., reason="all_pairs_invalid")
    if not valid.any():
        return result
    values = np.array([p["twist_body"][:3] for p in pairs])[valid]
    covariances = np.array([p["covariance_body"][:3, :3] for p in pairs])[valid]
    mean = values.mean(0)
    delta = values - mean
    # Approximate covariance; include temporal dispersion rather than claim independent pairs.
    covariance = covariances.mean(0) + delta.T @ delta / len(values)
    times = ((np.asarray(source_times) + np.asarray(target_times)) / 2)[valid]
    effective = float(times.mean())
    if effective > output_time + 1e-9:
        raise ValueError("Future pair supplied to causal aggregation")
    result.update(velocity=mean, covariance=covariance, dispersion=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
                  effective_time_s=effective, age_s=float(output_time - effective),
                  reason="all_pairs_valid" if valid.all() else "partial_valid_pairs")
    return result


def solve_batch(flows, confidences, reliabilities, intrinsics, normals, calibrations, intervals, config):
    """Batch API for frozen geometry. No mutable pair index or cross-sample state."""
    import torch
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    flows, confidences, reliabilities, intrinsics, normals, intervals = map(array, (flows, confidences, reliabilities, intrinsics, normals, intervals))
    if flows.ndim != 5 or normals.shape != (*flows.shape[:2], 3) or intervals.shape != flows.shape[:2] or len(calibrations) != len(flows):
        raise ValueError("Batch geometry requires [B,P,2,H,W], normals [B,P,3], and dt [B,P]")
    return [[solve_pair(flows[b, p], confidences[b, p].reshape(flows.shape[-2:]),
                        reliabilities[b, p].reshape(flows.shape[-2:]), intrinsics[b], normals[b, p],
                        calibrations[b], intervals[b, p], config) for p in range(flows.shape[1])] for b in range(len(flows))]
