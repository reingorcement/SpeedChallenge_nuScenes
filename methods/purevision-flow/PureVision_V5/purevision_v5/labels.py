"""Offline labels only: these functions must never be called by visual inference."""
from __future__ import annotations

import hashlib
import json

import numpy as np


LABEL_VERSION = "nuscenes_rear_axle_planar_nonuniform3_v1"
LABEL_CONTRACT = {
    "version": LABEL_VERSION,
    "mode": "PLANAR_PROXY",
    "reference_point": "nuscenes_rear_axle_midpoint",
    "reference_source": "https://www.nuscenes.org/data-format",
    "axes": "x_forward_y_left_in_current_yaw_frame",
    "units": "m/s",
    "time": "current_camera_timestamp",
    "derivative": "nonuniform_three_point_quadratic_world_xy_then_current_yaw",
    "neighbors": "previous_current_next_CAM_FRONT_sample_data_in_same_scene",
    "smoothing": "none_beyond_three_point_derivative",
    "endpoints": "excluded",
    "lever_arm": "none_ego_pose_already_at_rear_axle_midpoint",
    "future_pose": "offline_supervision_only_never_inference",
    "limitations": "planar_localization_z_zero_not_true_3D_body_velocity",
}
LABEL_SHA256 = hashlib.sha256(json.dumps(LABEL_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def yaw_from_quaternions(quaternions):
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite [N,4] wxyz quaternions")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("Invalid zero quaternion")
    w, x, y, z = (q / norm).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _validate(times, positions, yaw):
    t, p, y = (np.asarray(a, dtype=np.float64) for a in (times, positions, yaw))
    if t.ndim != 1 or len(t) < 3 or p.shape != (len(t), 2) or y.shape != t.shape:
        raise ValueError("Expected >=3 ordered times, [N,2] positions and [N] yaw")
    if not all(np.isfinite(a).all() for a in (t, p, y)) or np.any(np.diff(t) <= 0):
        raise ValueError("Times/positions/yaw must be finite and times strictly increasing")
    return t, p, y


def rotate_world_xy(velocity, yaw):
    v, y = np.asarray(velocity), np.asarray(yaw)
    c, s = np.cos(y), np.sin(y)
    return np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), axis=-1)


def planar_velocity_labels(times, positions, yaw):
    """Return [N,2], with NaN boundaries (exported as null, never zero labels)."""
    t, p, y = _validate(times, positions, yaw)
    h_minus, h_plus = np.diff(t)[:-1], np.diff(t)[1:]
    backward = (p[1:-1] - p[:-2]) / h_minus[:, None]
    forward = (p[2:] - p[1:-1]) / h_plus[:, None]
    world = (h_plus[:, None] * backward + h_minus[:, None] * forward) / (h_minus + h_plus)[:, None]
    result = np.full_like(p, np.nan)
    result[1:-1] = rotate_world_xy(world, y[1:-1])
    return result


def legacy_chord_labels(times, positions, yaw):
    t, p, y = _validate(times, positions, yaw)
    result = np.full_like(p, np.nan)
    result[1:-1] = rotate_world_xy((p[2:] - p[:-2]) / (t[2:] - t[:-2])[:, None], y[1:-1])
    return result


def five_point_quadratic_labels(times, positions, yaw):
    """Audit alternative only; not used to create targets."""
    t, p, y = _validate(times, positions, yaw)
    result = np.full_like(p, np.nan)
    for i in range(2, len(t) - 2):
        delta = t[i - 2:i + 3] - t[i]
        design = np.stack((np.ones(5), delta, delta * delta), axis=1)
        coefficients = np.linalg.lstsq(design, p[i - 2:i + 3] - p[i], rcond=None)[0]
        result[i] = rotate_world_xy(coefficients[1], y[i])
    return result


def synthetic_audit():
    t = np.array([-.05, 0., .1])
    cases = {}
    for name, velocity in (("straight", [15., 0.]), ("leftward", [0., 2.]), ("rightward", [0., -2.])):
        positions = t[:, None] * np.asarray(velocity)
        cases[name] = {"expected": velocity, "new": planar_velocity_labels(t, positions, np.zeros(3))[1].tolist()}
    speed, rate = 15., .4
    yaw = rate * t
    positions = np.stack((speed / rate * np.sin(yaw), speed / rate * (1 - np.cos(yaw))), axis=1)
    cases["no_slip_circle"] = {
        "speed_m_s": speed, "yaw_rate_rad_s": rate, "times_s": t.tolist(),
        "expected": [speed, 0.], "legacy": legacy_chord_labels(t, positions, yaw)[1].tolist(),
        "new": planar_velocity_labels(t, positions, yaw)[1].tolist(),
    }
    # A camera two metres forward has omega x lever = +0.8 m/s leftward.
    lever = np.array([2., 0.])
    camera_positions = positions + np.stack((np.cos(yaw) * lever[0], np.sin(yaw) * lever[0]), axis=1)
    cases["lever_arm"] = {"reference": "rear_axle_midpoint", "camera_lever_m": lever.tolist(),
        "expected_camera_minus_reference": [0., rate * lever[0]],
        "new_camera_minus_reference": (planar_velocity_labels(t, camera_positions, yaw)[1] - planar_velocity_labels(t, positions, yaw)[1]).tolist()}
    return cases


def sensitivity_audit(times, positions, yaw, seed=42, noise_std_m=.01):
    """Bounded train-only diagnostic; assumed noise, not a measured label error."""
    t, p, y = _validate(times, positions, yaw)
    noise = np.random.default_rng(seed).normal(0, noise_std_m, p.shape)
    three = planar_velocity_labels(t, p, y)
    five = five_point_quadratic_labels(t, p, y)
    three_noise = planar_velocity_labels(t, p + noise, y) - three
    five_noise = five_point_quadratic_labels(t, p + noise, y) - five
    valid = np.isfinite(five).all(axis=1)
    return {"frames": len(t), "seed": seed, "assumed_independent_position_noise_std_m": noise_std_m,
        "three_point_noise_rmse_vx_vy": np.sqrt(np.mean(three_noise[valid] ** 2, axis=0)).tolist(),
        "five_point_noise_rmse_vx_vy": np.sqrt(np.mean(five_noise[valid] ** 2, axis=0)).tolist(),
        "five_minus_three_rmse_vx_vy": np.sqrt(np.mean((five[valid] - three[valid]) ** 2, axis=0)).tolist(),
        "decision": "lock_three_point_v1_for_shared_first_round; diagnostic_is_not_ground_truth_uncertainty"}
