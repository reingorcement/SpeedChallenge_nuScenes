import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from evaluate import join_labels, metrics
from roll_velocity.io import ROOT, load_sequence, read_json, validate_calibration


class InputTests(unittest.TestCase):
    def make_input(self, directory):
        cal = read_json(ROOT / "data/HW_DEV_SINE40_A12/calibration.json")
        cal["image_size"] = [80, 60]
        cal["K"] = [[40, 0, 40], [0, 40, 30], [0, 0, 1]]
        cal["dynamic_roll_truth"] = 999
        (directory / "calibration.json").write_text(json.dumps(cal))
        frames = []
        for i in range(5):
            Image.new("RGB", (80, 60)).save(directory / f"{i}.png")
            frames.append(dict(image_id=i + 100, file=f"{i}.png", solver_time=i * .05, truth_vx=999))
        (directory / "frames.json").write_text(json.dumps(frames))
        return frames, cal

    def test_visual_input_allowlist_and_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            self.make_input(path)
            frames, cal = load_sequence(path)
            self.assertNotIn("truth_vx", frames[0])
            self.assertNotIn("dynamic_roll_truth", cal)
            old = frames[0]["sha256"]
            Image.new("RGB", (80, 60), color="red").save(path / "0.png")
            self.assertNotEqual(load_sequence(path)[0][0]["sha256"], old)

    def test_invalid_timestamp_and_escaped_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            frames, _ = self.make_input(path)
            frames[1]["solver_time"] = 0
            (path / "frames.json").write_text(json.dumps(frames))
            with self.assertRaisesRegex(ValueError, "increasing"):
                load_sequence(path)
            frames[1]["solver_time"] = .05
            frames[0]["file"] = "../outside.png"
            (path / "frames.json").write_text(json.dumps(frames))
            with self.assertRaisesRegex(ValueError, "inside"):
                load_sequence(path)

    def test_calibration_requires_metric_scale_and_mount(self):
        cal = read_json(ROOT / "data/HW_DEV_SINE40_A12/calibration.json")
        cal["height_m"] = 0
        with self.assertRaisesRegex(ValueError, "height_m"):
            validate_calibration(cal)
        cal["height_m"] = 1.4
        cal["body_mount_rotation"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "rotation"):
            validate_calibration(cal)

    def test_evaluation_preserves_failures_and_no_time_shift(self):
        rows = []
        for method in ["F_R", "F_RC"]:
            for i in range(3):
                valid = not (method == "F_RC" and i == 2)
                rows.append(dict(method=method, image_id=i, time_s=i * .05, valid=valid,
                                 reason="ok" if valid else "all_three_pairs_invalid",
                                 vx_mps=10.1 if valid else np.nan, vy_mps=.3 if valid else np.nan))
        truth = pd.DataFrame(dict(time=[0, .05, .1], vx=[10] * 3, vy=[.2] * 3, roll=[.03] * 3))
        joined = join_labels(pd.DataFrame(rows), truth)
        table = metrics(joined)
        own = table.query("method == 'F_RC' and subset == 'all' and population == 'own_valid' and axis == 'vx'").iloc[0]
        self.assertEqual(own.eligible, 3)
        self.assertEqual(own.valid, 2)
        self.assertAlmostEqual(own.mae, .1)
        common = table.query("subset == 'all' and population == 'common_valid'")
        self.assertTrue((common.valid == 2).all())
        truth.time += .01
        with self.assertRaisesRegex(ValueError, "timestamp"):
            join_labels(pd.DataFrame(rows), truth)


if __name__ == "__main__":
    unittest.main()
