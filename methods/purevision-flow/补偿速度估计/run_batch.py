"""Run bundled development cases sequentially on one GPU, then evaluate."""
import argparse
from pathlib import Path
import subprocess
import sys

from roll_velocity.io import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Optional evaluation-only labels root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", action="store_true", help="Also run F_B and V3_matched_R")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = sorted(p for p in args.data.resolve().iterdir() if (p / "frames.json").is_file())
    if not cases:
        parser.error("No case directories found")
    methods = ["F_RC", "F_R"] + (["F_B", "V3_matched_R"] if args.ablation else [])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in cases:
        output = args.output / case.name
        command = [sys.executable, "-B", str(ROOT / "run.py"), "--input", str(case), "--output",
                   str(output), "--device", args.device, "--methods", *methods]
        if args.resume:
            command.append("--resume")
        # Inherit stdout/stderr: supports terminal, nohup and Slurm without buffering.
        subprocess.run(command, check=True)
        if args.labels:
            from evaluate import evaluate_run
            import pandas as pd
            destination = output / "evaluation"
            if destination.exists() and args.resume:
                # Metrics can be reused only after run.py verified the run identity and outputs.
                from roll_velocity.io import read_json, sha256
                audit = read_json(destination / "provenance.json")
                truth = args.labels / case.name / "truth_20hz.csv"
                if audit != {"predictions": sha256(output / "speeds.csv"), "truth": sha256(truth)}:
                    raise ValueError(f"Evaluation inputs changed: {destination}")
                report = pd.read_csv(destination / "metrics.csv")
            else:
                from roll_velocity.io import sha256
                truth = args.labels / case.name / "truth_20hz.csv"
                report = evaluate_run(output / "speeds.csv", truth, destination)
                write_json(destination / "provenance.json",
                           {"predictions": sha256(output / "speeds.csv"), "truth": sha256(truth)})
            report["case"] = case.name
            reports.append(report)
    if reports:
        all_metrics = pd.concat(reports, ignore_index=True)
        all_metrics.to_csv(args.output / "per_case_metrics.csv", index=False)
        macro = all_metrics[all_metrics.valid > 0].groupby(["method", "subset", "population", "axis"]).agg(
            mae=("mae", "mean"), rmse=("rmse", "mean"), runs=("case", "nunique"),
            min_coverage=("coverage", "min"), worst_case_p95=("p95", "max")).reset_index()
        macro.to_csv(args.output / "macro_metrics.csv", index=False)
        print(macro.query("subset == 'active' and population == 'own_valid'").to_string(index=False))
    write_json(args.output / "batch.json", {"cases": [p.name for p in cases], "methods": methods,
                                         "evaluation": bool(args.labels), "status": "complete"})


if __name__ == "__main__":
    main()
