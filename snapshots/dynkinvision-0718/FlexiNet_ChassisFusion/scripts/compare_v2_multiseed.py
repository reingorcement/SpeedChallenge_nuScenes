from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_multiseed(
    seeds: list[int],
    visual_reports: list[dict],
    joint_reports: list[dict],
) -> dict:
    if not (len(seeds) == len(visual_reports) == len(joint_reports)):
        raise ValueError("seeds, visual metrics, and joint metrics must have the same length.")
    if len(seeds) < 3:
        raise ValueError("Final V2 acceptance requires at least three paired seeds.")
    rows = []
    for seed, visual, joint in zip(seeds, visual_reports, joint_reports):
        turning_improved = float(joint.get("turning_mae_vy_mps", float("inf"))) < float(
            visual.get("turning_mae_vy_mps", float("inf"))
        )
        lateral_improved = float(
            joint.get("high_lateral_mae_vy_mps", float("inf"))
        ) < float(visual.get("high_lateral_mae_vy_mps", float("inf")))
        rows.append(
            {
                "seed": seed,
                "visual_epe_mps": float(visual["final_endpoint_error_mps"]),
                "joint_epe_mps": float(joint["final_endpoint_error_mps"]),
                "turning_vy_improved": turning_improved,
                "high_lateral_vy_improved": lateral_improved,
                "group_improved": turning_improved or lateral_improved,
            }
        )
    mean_visual = sum(row["visual_epe_mps"] for row in rows) / len(rows)
    mean_joint = sum(row["joint_epe_mps"] for row in rows) / len(rows)
    improved_seeds = sum(int(row["group_improved"]) for row in rows)
    return {
        "paired_seed_count": len(rows),
        "mean_visual_epe_mps": mean_visual,
        "mean_joint_epe_mps": mean_joint,
        "mean_epe_within_2_percent": mean_joint <= 1.02 * mean_visual,
        "group_improved_seed_count": improved_seeds,
        "kinematics_claim_allowed": mean_joint <= 1.02 * mean_visual and improved_seeds >= 2,
        "seeds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paired multi-seed V2 acceptance.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--visual", nargs="+", required=True)
    parser.add_argument("--joint", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/velocity_xy_v2_multiseed_comparison.json")
    args = parser.parse_args()
    visual_reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.visual]
    joint_reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.joint]
    report = compare_multiseed(args.seeds, visual_reports, joint_reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["kinematics_claim_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
