import itertools
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roll_velocity.finite_geometry import solve, project, left_jacobian
from roll_velocity.io import ROOT, intrinsics, read_json
from roll_velocity.normal_policy import constrained, pair_planes
from roll_velocity.reliable_normal import ReliableNormal
from roll_velocity.standalone_geometry import aggregate, body_reference


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cal = read_json(ROOT / "data/HW_DEV_SINE40_A12/calibration.json")
        self.config = read_json(ROOT / "config/default.json")
        self.k = intrinsics(self.cal)

    def test_exact_projection_roll_pitch_motion_and_lever(self):
        """24 combinations with known 6-DoF motion; tests sign, scale and lever."""
        c = np.array(self.cal["rotation_ego_from_camera"])
        lever = np.array(self.cal["translation_ego_from_camera"])
        y, x = np.mgrid[150:281:8, 24:489:16]
        pixels = np.c_[x.ravel(), y.ravel()].astype(float)
        maximum = 0.
        for roll, pitch, dt, speed in itertools.product([-4., 0., 4.], [0., 2.], [.025, .05], [25.]):
            for stationary in (False, True):
                with self.subTest(roll=roll, pitch=pitch, dt=dt, stationary=stationary):
                    rb = Rotation.from_euler("xyz", [roll, pitch, 0], degrees=True).as_matrix()
                    velocity = np.array([speed, .5 if roll >= 0 else -.5, .2]) if not stationary else np.zeros(3)
                    omega = np.radians([8, 5, 12]) if not stationary else np.zeros(3)
                    w = omega * dt
                    rb1 = rb @ Rotation.from_rotvec(w).as_matrix()
                    cg0 = np.array([0, 0, 1.4 - (rb @ lever)[2]])
                    cg1 = cg0 + rb @ left_jacobian(w) @ (velocity * dt)
                    rc0, rc1 = rb @ c, rb1 @ c
                    pc0, pc1 = cg0 + rb @ lever, cg1 + rb1 @ lever
                    normal, height = rc0[2], pc0[2]
                    r21, t21 = rc1.T @ rc0, rc1.T @ (pc0 - pc1)
                    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.k).T
                    depth = -height / (rays @ normal)
                    points = rays * depth[:, None]
                    targets = project(points, r21, t21, self.k)
                    keep = (depth > .5) & (depth < 150) & (targets[:, 0] >= 0) & (targets[:, 0] < 512)
                    keep &= (targets[:, 1] >= 0) & (targets[:, 1] < 288)
                    result = solve(pixels[keep], targets[keep], self.k, normal, height, dt, **self.config["finite_solver"])
                    self.assertTrue(result["valid"], result["reason"])
                    twist = result["twist"]
                    estimated = c @ twist[:3] - np.cross(c @ twist[3:], lever)
                    error = np.max(abs(estimated - velocity))
                    maximum = max(maximum, error)
                    self.assertLess(error, 1e-4)
                    np.testing.assert_allclose(c @ twist[3:], omega, atol=1e-5)
        print(f"24 exact motion combinations: max velocity error {maximum:.3g} m/s")

    def test_degenerate_support_is_invalid(self):
        normal = np.array(self.cal["plane_normal_camera"])
        for p in (np.zeros((0, 2)), np.full((10, 2), [200., 230.]),
                  np.c_[np.linspace(30, 480, 80), np.full(80, 220.)]):
            result = solve(p, p, self.k, normal, 1.4, .05)
            self.assertFalse(result["valid"])
        p = np.full((100, 2), [200., 230.])
        self.assertEqual(solve(p, p, self.k, normal, 1.4, 0)["reason"], "invalid_interval")

    def test_pixel_center_projection(self):
        points = np.array([[-1, .2, 3], [2, -.3, 8], [.1, .5, 2.5]])
        original = points @ np.array(self.cal["K"]).T
        original = original[:, :2] / original[:, 2:]
        expected = (original + .5) * [.64, .48] - .5
        actual = points @ self.k.T
        np.testing.assert_allclose(actual[:, :2] / actual[:, 2:], expected, atol=1e-10)

    def test_roll_constraint_and_source_frame_routing(self):
        nominal = np.array(self.cal["plane_normal_camera"])
        normals = [dict(source_id=i, normal=constrained(np.array([i * .01, -1, -.3]), nominal).tolist()) for i in range(6)]
        selected = pair_planes("F_RC", nominal, normals, 5, 1.4)
        for i, (normal, height) in zip([2, 3, 4], selected):
            np.testing.assert_allclose(normal, normals[i]["normal"])
            self.assertAlmostEqual(normal[2], nominal[2])
            self.assertAlmostEqual(height, 1.4)
        for normal, _ in pair_planes("F_R", nominal, normals, 5, 1.4):
            np.testing.assert_allclose(normal, nominal)
        with self.assertRaises(ValueError):
            pair_planes("O_R", nominal, normals, 5, 1.4)

    def test_normal_loss_holds_then_falls_back(self):
        state = ReliableNormal(self.k, np.array(self.cal["plane_normal_camera"]), self.config["normal"])
        state.ever_valid = True
        state.previous = constrained(np.array([.02, -1, -.3]), state.nominal)
        for _ in range(self.config["normal"]["max_hold_pairs"]):
            self.assertEqual(state.step(np.zeros((2, 288, 512)), np.zeros((288, 512)))["status"], "hold")
        result = state.step(np.zeros((2, 288, 512)), np.zeros((288, 512)))
        self.assertEqual(result["status"], "fixed_fallback")
        np.testing.assert_allclose(result["normal"], state.nominal)

    def test_aggregation_excludes_invalid_pairs_and_uses_full_3d(self):
        twists = np.array([[10, 2, 3, 0, 0, 0], [1000, 1000, 1000, 0, 0, 0], [12, 4, 5, 0, 0, 0]])
        result = aggregate(twists, [True, False, True], self.cal)
        expected = body_reference(self.cal) @ [11, 3, 4]
        np.testing.assert_allclose([result[x] for x in ("vx", "vy", "vz")], expected)
        result = aggregate(twists, [False, False, False], self.cal)
        self.assertFalse(result["valid"])
        self.assertTrue(np.isnan(result["vx"]))


if __name__ == "__main__":
    unittest.main()
