from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check visual+kinematics acceptance against visual-only.")
    parser.add_argument("--visual", required=True, help="visual-only metrics.json")
    parser.add_argument("--joint", required=True, help="visual+kinematics metrics.json")
    parser.add_argument("--output", default="outputs/velocity_xy_v2_stage_comparison.json")
    args = parser.parse_args()
    visual = json.loads(Path(args.visual).read_text(encoding="utf-8"))
    joint = json.loads(Path(args.joint).read_text(encoding="utf-8"))
    epe_limit = 1.02 * float(visual["final_endpoint_error_mps"])
    overall_pass = float(joint["final_endpoint_error_mps"]) <= epe_limit
    turning_improved = float(joint.get("turning_mae_vy_mps", float("inf"))) < float(
        visual.get("turning_mae_vy_mps", float("inf"))
    )
    lateral_improved = float(joint.get("high_lateral_mae_vy_mps", float("inf"))) < float(
        visual.get("high_lateral_mae_vy_mps", float("inf"))
    )
    report = {
        "overall_epe_within_2_percent": overall_pass,
        "turning_vy_improved": turning_improved,
        "high_lateral_vy_improved": lateral_improved,
        "kinematics_claim_allowed": overall_pass and (turning_improved or lateral_improved),
        "visual_final_epe_mps": visual["final_endpoint_error_mps"],
        "joint_final_epe_mps": joint["final_endpoint_error_mps"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["kinematics_claim_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
