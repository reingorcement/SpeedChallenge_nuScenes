"""Compare this release's RGB rerun to the bundled historical v2 evidence."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roll_velocity.io import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "verification/full_rgb")
    parser.add_argument("--output", type=Path, default=ROOT / "verification/full_rgb_comparison.json")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true", help="Require pointwise equality (use with identical flow inputs)")
    parser.add_argument("--input-kind", choices=["recomputed_rgb_flow", "historical_flow"], default="recomputed_rgb_flow")
    args = parser.parse_args()
    historical = pd.read_csv(ROOT / "evidence/method_evidence/per_window_body.csv")
    historical = historical.query("version == 'corrected_interface'")
    keys = ["case", "group", "output_id"]
    current = []
    for case in sorted(historical.case.unique()):
        frame = pd.read_csv(args.runs / case / "speeds.csv")
        frame = frame[frame.reason != "warmup"].rename(columns={
            "method": "group", "output_index": "output_id", "vx_mps": "vx", "vy_mps": "vy"})
        current.append(frame)
    current = pd.concat(current, ignore_index=True)
    merged = historical.merge(current, on=keys, suffixes=("_old", "_new"), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Historical and new output keys do not match")
    valid_equal = np.array_equal(merged.valid_old, merged.valid_new)
    entries = []
    for group, part in merged.groupby("group"):
        common = part.valid_old & part.valid_new
        entries.append(dict(method=group, windows=len(part), valid=int(part.valid_new.sum()),
                            max_abs_vx_difference_mps=float((part.loc[common, "vx_old"] - part.loc[common, "vx_new"]).abs().max()),
                            max_abs_vy_difference_mps=float((part.loc[common, "vy_old"] - part.loc[common, "vy_new"]).abs().max())))
    passed = bool(valid_equal and all(max(r["max_abs_vx_difference_mps"], r["max_abs_vy_difference_mps"]) <= args.tolerance for r in entries))
    result = dict(pointwise_match=passed, tolerance_mps=args.tolerance, valid_flags_equal=valid_equal,
                  input_kind=args.input_kind, strict_check=args.strict,
                  origin="Historical comparison is evaluation-only; regenerated RAFT may differ across runtimes",
                  cases=int(merged.case.nunique()), unique_windows=int(merged[["case", "output_id"]].drop_duplicates().shape[0]),
                  methods=entries)
    write_json(args.output, result)
    print(pd.DataFrame(entries).to_string(index=False))
    if not passed and args.strict:
        raise SystemExit("Regression exceeded tolerance; inspect before claiming equivalence")
    print("POINTWISE MATCH" if passed else "POINTWISE DIFFERENCE RECORDED; see matched-flow regression")


if __name__ == "__main__":
    main()
