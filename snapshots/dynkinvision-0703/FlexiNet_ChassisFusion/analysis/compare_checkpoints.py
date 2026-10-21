"""Side-by-side comparison of two evaluation runs.

The most common use case is comparing ``best.pth`` and ``last.pth`` from the
same training run, which the post-training analysis guide explicitly calls out
as the final step. The script can also compare any two existing eval folders
(e.g. a baseline ablation vs. the residual-fusion variant) when ``--skip_inference``
is set.

Outputs (under ``--report_dir``):

- ``comparison.md`` — Markdown table of every shared metric, plus deltas.
- ``comparison.csv`` — same data in CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config_from_yaml  # noqa: E402
from utils.evaluation import write_evaluation_artifacts  # noqa: E402

from analysis.inference import run_inference_from_checkpoint  # noqa: E402
from analysis.postprocess import write_analysis_bundle  # noqa: E402


_LOWER_IS_BETTER_PREFIXES = (
    "loss",
    "mae_",
    "rmse_",
    "vector_endpoint",
    "speed_magnitude",
    "endpoint_error",
)
_HIGHER_IS_BETTER_PREFIXES = ("r2_", "gain_")


def _direction_for(metric: str) -> str:
    for prefix in _HIGHER_IS_BETTER_PREFIXES:
        if metric.startswith(prefix):
            return "higher"
    for prefix in _LOWER_IS_BETTER_PREFIXES:
        if metric.startswith(prefix) or prefix in metric:
            return "lower"
    return "?"


def _format_float(value, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two checkpoints/eval runs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_a", type=str, default="checkpoints/best.pth")
    parser.add_argument("--checkpoint_b", type=str, default="checkpoints/last.pth")
    parser.add_argument("--label_a", type=str, default="best")
    parser.add_argument("--label_b", type=str, default="last")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--output_dir_a",
        type=str,
        default="outputs/FlexiNet_ChassisFusion/eval_best",
    )
    parser.add_argument(
        "--output_dir_b",
        type=str,
        default="outputs/FlexiNet_ChassisFusion/eval_last",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="outputs/FlexiNet_ChassisFusion/eval_compare",
    )
    parser.add_argument("--prefix", type=str, default="val")
    parser.add_argument("--allow_pose_fallback", action="store_true")
    parser.add_argument("--bad_case_top_k", type=int, default=50)
    parser.add_argument("--report_top_k", type=int, default=10)
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference and just compare the metrics that already exist under --output_dir_*",
    )
    return parser.parse_args()


def _run_and_analyse(cfg: dict, checkpoint: str, output_dir: Path, prefix: str, args: argparse.Namespace) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run_inference_from_checkpoint(
        cfg,
        checkpoint_path=checkpoint,
        split=args.split,
        allow_pose_fallback=args.allow_pose_fallback,
    )
    metrics = write_evaluation_artifacts(artifacts.result, output_dir, prefix=prefix)
    write_analysis_bundle(
        output_dir,
        prefix=prefix,
        bad_case_top_k=int(args.bad_case_top_k),
        report_top_k=int(args.report_top_k),
    )
    return metrics


def _load_metrics(eval_dir: Path, prefix: str) -> dict:
    path = eval_dir / f"{prefix}_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_comparison(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    report_dir: Path,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "comparison.csv"
    md_path = report_dir / "comparison.md"

    keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", label_a, label_b, "delta_b_minus_a", "preferred_direction"])
        for key in keys:
            value_a = metrics_a.get(key)
            value_b = metrics_b.get(key)
            delta = (
                float(value_b) - float(value_a)
                if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float))
                else ""
            )
            writer.writerow(
                [key, value_a if value_a is not None else "", value_b if value_b is not None else "", delta, _direction_for(key)]
            )

    md_lines = [
        f"# Checkpoint comparison: {label_a} vs {label_b}",
        "",
        f"| Metric | {label_a} | {label_b} | Δ ({label_b} - {label_a}) | Preferred |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in keys:
        value_a = metrics_a.get(key)
        value_b = metrics_b.get(key)
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            delta = float(value_b) - float(value_a)
            md_lines.append(
                f"| `{key}` | {_format_float(value_a)} | {_format_float(value_b)} | "
                f"{_format_float(delta)} | {_direction_for(key)} |"
            )
        else:
            md_lines.append(
                f"| `{key}` | {value_a if value_a is not None else ''} | "
                f"{value_b if value_b is not None else ''} |  | {_direction_for(key)} |"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def main() -> None:
    args = parse_args()
    output_dir_a = Path(args.output_dir_a)
    output_dir_b = Path(args.output_dir_b)

    if not args.skip_inference:
        cfg = load_config_from_yaml(Path(args.config))
        print(f"[{args.label_a}] {args.checkpoint_a} -> {output_dir_a}")
        _run_and_analyse(cfg, args.checkpoint_a, output_dir_a, args.prefix, args)
        # Reload cfg for the second run because the helper may mutate it (allow_pose_fallback).
        cfg = load_config_from_yaml(Path(args.config))
        print(f"[{args.label_b}] {args.checkpoint_b} -> {output_dir_b}")
        _run_and_analyse(cfg, args.checkpoint_b, output_dir_b, args.prefix, args)

    metrics_a = _load_metrics(output_dir_a, args.prefix)
    metrics_b = _load_metrics(output_dir_b, args.prefix)
    paths = _write_comparison(metrics_a, metrics_b, args.label_a, args.label_b, Path(args.report_dir))
    print("Comparison written to:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
