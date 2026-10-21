from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from roll_velocity.flow import save_packet
from roll_velocity.io import ROOT, read_json
from roll_velocity.pipeline import prepare_region, infer


class FailurePipelineTests(unittest.TestCase):
    def test_textureless_sequence_returns_explicit_invalid(self):
        """No optical flow/support must not become a zero-speed success."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, flow, out = root / "raw", root / "flow", root / "out"
            for p in (raw, flow, out):
                p.mkdir()
            frames = []
            for i in range(5):
                Image.new("RGB", (800, 600), "black").save(raw / f"{i}.png")
                frames.append(dict(image_id=i, file=f"{i}.png", solver_time=i * .05))
                if i < 4:
                    save_packet(flow / f"{i:06d}.pt", dict(flow=torch.zeros(2, 288, 512).half(),
                                confidence=torch.zeros(1, 288, 512).half(), dt=.05, source_id=i, target_id=i + 1))
            cal = read_json(ROOT / "data/HW_DEV_SINE40_A12/calibration.json")
            config = read_json(ROOT / "config/default.json")
            normals = prepare_region(raw, frames, cal, config, flow, out / "region")
            self.assertTrue(all(n["status"] == "fixed_fallback" for n in normals))
            coverage = infer(raw, frames, cal, config, out / "region", normals,
                             ROOT / "weights/fusion_seed42_best.pth", out, ["F_RC", "F_R"], torch.device("cpu"))
            self.assertEqual(coverage["F_RC"]["coverage"], 0)
            result = pd.read_csv(out / "speeds.csv")
            self.assertFalse(result.valid.any())
            self.assertTrue(result.vx_mps.isna().all())
            self.assertTrue(result.vy_mps.isna().all())
            self.assertTrue(result[result.output_index == 4].reason.eq("all_three_pairs_invalid").all())


if __name__ == "__main__":
    unittest.main()
