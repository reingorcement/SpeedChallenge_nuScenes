import cv2
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from purevision_v4 import geometry as g
from purevision_v4._reference.finite_geometry import solve as legacy_solve
from purevision_v4.io import intrinsics, read_json, ROOT


def observations(cal, velocity, omega, dt, roll=0., pitch=0.):
    c = np.asarray(cal["rotation_body_from_camera"])
    lever = np.asarray(cal["camera_lever_body_m"])
    r0 = Rotation.from_euler("xyz", [roll, pitch, 0], degrees=True).as_matrix()
    w = np.asarray(omega) * dt
    r1 = r0 @ Rotation.from_rotvec(w).as_matrix()
    p0 = np.array([0., 0., cal["height_m"] - (r0 @ lever)[2]])
    p1 = p0 + r0 @ g.left_jacobian(w) @ (np.asarray(velocity) * dt)
    camera0, camera1 = r0 @ c, r1 @ c
    pos0, pos1 = p0 + r0 @ lever, p1 + r1 @ lever
    n, height = camera0[2], pos0[2]
    r21, t21 = camera1.T @ camera0, camera1.T @ (pos0 - pos1)
    y, x = np.mgrid[155:280:8, 24:489:16]
    p = np.c_[x.ravel(), y.ravel()].astype(float)
    k = intrinsics(cal)
    rays = np.c_[p, np.ones(len(p))] @ np.linalg.inv(k).T
    depth = -height / (rays @ n)
    q = g.project(rays * depth[:, None], r21, t21, k)
    keep = (depth > .5) & (depth < 150) & (q[:, 0] >= 0) & (q[:, 0] < 512) & (q[:, 1] >= 0) & (q[:, 1] < 288)
    return p[keep], q[keep], k, n, height


@pytest.mark.parametrize("velocity,omega", [([0, 0, 0], [0, 0, 0]), ([0, 0, 0], [.1, .06, .2]),
                                           ([20, 0, 0], [0, 0, 0]), ([12, .6, .2], [.12, -.07, .25]),
                                           ([-3, -.4, 0], [-.06, .03, -.1])])
@pytest.mark.parametrize("dt,roll,pitch", [(.025, -3, 2), (.05, 0, 0), (.1, 4, -2)])
def test_finite_motion_body_reference(cal, cfg, velocity, omega, dt, roll, pitch):
    p, q, k, n, h = observations(cal, velocity, omega, dt, roll, pitch)
    result = g.solve(p, q, k, n, h, dt, **cfg["finite_solver"])
    assert result["valid"], result["reason"]
    twist, covariance = g.to_body(result["twist"], result["covariance"], cal)
    np.testing.assert_allclose(twist, np.r_[velocity, omega], atol=1e-4)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_reference_solver_regression(cal, cfg):
    p, q, k, n, h = observations(cal, [14, -.3, .1], [.07, .02, -.13], .05)
    q = q + np.random.default_rng(12).normal(0, .015, q.shape)
    old = legacy_solve(p, q, k, n, h, .05, **{k: v for k, v in cfg["finite_solver"].items() if k in g.SOLVER_KEYS})
    new = g.solve(p, q, k, n, h, .05, **cfg["finite_solver"])
    assert old["valid"] and new["valid"]
    np.testing.assert_allclose(new["twist"], old["twist"], atol=1e-8)


def test_intrinsics_match_pixel_center_resize(cal):
    p = np.array([[30., 80., 1], [400, 300, 1], [799, 599, 1]])
    rays = p @ np.linalg.inv(np.asarray(cal["K"])).T
    projected = rays @ intrinsics(cal).T
    np.testing.assert_allclose(projected[:, :2], (p[:, :2] + .5) * [.64, .48] - .5)


def test_legacy_calibration_equivalence(cal):
    old = read_json(ROOT / "tests/calibration_legacy.json")
    c = np.asarray(old["rotation_ego_from_camera"])
    b = np.asarray(cal["rotation_body_from_camera"]) @ c.T
    twist = np.array([1., 2., 3., .2, .3, .4])
    expected = b @ (c @ twist[:3] - np.cross(c @ twist[3:], old["translation_ego_from_camera"]))
    actual, _ = g.to_body(twist, np.eye(6), cal)
    np.testing.assert_allclose(actual[:3], expected)


def test_bad_motion_and_numerical_exceptions_are_invalid(cal, cfg, monkeypatch):
    p, q, k, n, h = observations(cal, [10, 0, 0], [0, 0, 0], .05)
    assert g.solve(p, q, k, n, h, 0, **cfg["finite_solver"])["reason"] == "invalid_interval"
    assert not g.solve(p[:10], q[:10], k, n, h, .05, **cfg["finite_solver"])["valid"]
    def failure(*args, **kwargs):
        raise cv2.error("synthetic numerical failure")
    monkeypatch.setattr(g, "reference_solve", failure)
    assert g.solve(p, q, k, n, h, .05, **cfg["finite_solver"])["reason"] == "numerical_failure"


def test_residual_and_covariance_gates(cal, cfg, monkeypatch):
    p, q, k, n, h = observations(cal, [10, 0, 0], [0, 0, 0], .05)
    base = legacy_solve(p, q, k, n, h, .05)
    monkeypatch.setattr(g, "reference_solve", lambda *a, **kw: dict(base, residual=10.))
    assert g.solve(p, q, k, n, h, .05, **cfg["finite_solver"])["reason"] == "large_reprojection_residual"
    monkeypatch.setattr(g, "reference_solve", lambda *a, **kw: dict(base, covariance=np.full((6, 6), np.nan)))
    assert g.solve(p, q, k, n, h, .05, **cfg["finite_solver"])["reason"] == "invalid_covariance"


def test_aggregation_time_and_failure_mask():
    pairs = [dict(valid=True, twist_body=np.array([10, 2, 3, 0, 0, 0]), covariance_body=np.eye(6)),
             dict(valid=False, twist_body=np.full(6, 1e6), covariance_body=np.eye(6)),
             dict(valid=True, twist_body=np.array([12, 4, 5, 0, 0, 0]), covariance_body=np.eye(6))]
    result = g.aggregate(pairs, [.05, .1, .15], [.1, .15, .2], .2)
    np.testing.assert_allclose(result["velocity"], [11, 3, 4])
    assert result["effective_time_s"] == pytest.approx(.125)
    assert result["age_s"] == pytest.approx(.075)
    for pair in pairs:
        pair["valid"] = False
    assert not g.aggregate(pairs, [.05, .1, .15], [.1, .15, .2], .2)["valid"]


def test_batch_geometry_accepts_gradient_inputs_without_state(cal, cfg):
    flow = torch.zeros(2, 3, 2, 32, 48, requires_grad=True)
    weight = torch.zeros(2, 3, 1, 32, 48, requires_grad=True)
    result = g.solve_batch(flow, weight, weight, np.stack([intrinsics(cal, (32, 48))] * 2),
                           np.tile(cal["plane_normal_camera"], (2, 3, 1)), [cal, cal], np.full((2, 3), .05), cfg["finite_solver"])
    assert len(result) == 2 and all(len(pairs) == 3 for pairs in result)
    assert all(not p["valid"] for pairs in result for p in pairs)
    assert flow.grad is None
