"""Verification-only historical flow replay; production RAFT is not patched."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
import torch

from purevision_v4.io import ROOT, artifact_path, code_hashes, load_config, load_sequence, sha256, target_spec, write_json
from purevision_v4.pipeline import _infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--flow-cache", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    frames, cal = load_sequence(args.input, count=args.frames)
    out = artifact_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    used = {}

    class FixedFlow:
        def __init__(self, *args, **kwargs):
            pass

        def pair(self, source, target, unused_output_path):
            path = args.flow_cache / f"{source['image_id']:06d}.pt"
            packet = torch.load(path, map_location="cpu", weights_only=True)
            if packet["source_id"] != source["image_id"] or packet["target_id"] != target["image_id"] or abs(packet["dt"] - (target["time_s"] - source["time_s"])) > 1e-8:
                raise ValueError("Historical fixture pair mismatch")
            used[path.name] = sha256(path)
            return packet

    weights = dict(visual=ROOT / "weights/v3_fusion_seed42.pth", raft=ROOT / "weights/raft_large_C_T_SKHT_V2.pth")
    torch.set_num_threads(4)
    cv2.setRNGSeed(42)
    with patch("purevision_v4.pipeline.RaftFlow", FixedFlow):
        result = _infer(frames, cal, load_config(), out, args.input.name, ("F_RC", "F_R", "F_B"), weights,
                        torch.device("cpu"), None, False, {}, target_spec(cal))
    old, new = pd.read_csv(args.reference_csv), pd.read_csv(out / "speeds.csv")
    reports = {}
    for method in ["F_RC", "F_R", "F_B"]:
        a = new[new.method == method].set_index("output_index")
        b = old[old.method == method].set_index("output_index").loc[a.index]
        validity = a.valid.equals(b.valid)
        difference = (a[["vx_mps", "vy_mps"]] - b[["vx_mps", "vy_mps"]]).abs().max().to_dict()
        reports[method] = dict(valid_equal=validity, max_abs_difference_mps=difference,
                               passed=validity and max(difference.values()) < args.tolerance)
    write_json(out / "comparison.json", dict(scope="Historical fixed flow regression, not an accuracy test",
               methods=reports, flow_hashes=used, code=code_hashes(), coverage=result["coverage"], tolerance=args.tolerance))
    print(reports)
    if not all(report["passed"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
