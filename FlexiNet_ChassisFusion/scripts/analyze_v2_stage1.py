from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_v2_holdout import json_safe, metrics_from_rows  # noqa: E402


def load_prediction_variants(path: str | Path) -> dict[str, list[dict[str, str]]]:
    variants: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            key = (variant, row["scene_name"], row["center_token"])
            if key in seen:
                raise ValueError(f"Duplicate holdout prediction key in {path}: {key}")
            seen.add(key)
            variants.setdefault(variant, []).append(row)
    if "baseline" not in variants:
        raise ValueError(f"Holdout predictions have no baseline rows: {path}")
    return variants


def align_prediction_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def keyed(rows):
        return {(str(row["scene_name"]), str(row["center_token"])): row for row in rows}

    left_map = keyed(left)
    right_map = keyed(right)
    if set(left_map) != set(right_map):
        missing_left = sorted(set(right_map) - set(left_map))[:3]
        missing_right = sorted(set(left_map) - set(right_map))[:3]
        raise ValueError(
            f"Prediction keys do not align; missing_left={missing_left}, missing_right={missing_right}."
        )
    pairs = []
    for key in sorted(left_map):
        left_row, right_row = left_map[key], right_map[key]
        for target in ("target_vx_mps", "target_vy_mps"):
            if not math.isclose(float(left_row[target]), float(right_row[target]), abs_tol=1.0e-6):
                raise ValueError(f"Target mismatch for {key} field {target}.")
        pairs.append((left_row, right_row))
    return pairs


def _row_error(row: dict[str, Any]) -> tuple[float, float, float]:
    error_vx = float(row["pred_vx_mps"]) - float(row["target_vx_mps"])
    error_vy = float(row["pred_vy_mps"]) - float(row["target_vy_mps"])
    return error_vx, error_vy, math.hypot(error_vx, error_vy)


def _subset(name: str, row: dict[str, Any], high_lateral_threshold: float) -> bool:
    if name == "all":
        return True
    if name == "turning":
        return abs(float(row["yaw_rate_radps"])) >= 0.05
    if name == "high_lateral":
        return abs(float(row["target_vy_mps"])) >= high_lateral_threshold
    raise ValueError(f"Unknown comparison subset {name!r}.")


def scene_metric_deltas(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    subset: str,
    high_lateral_threshold: float,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for left_row, right_row in align_prediction_rows(left, right):
        if not _subset(subset, left_row, high_lateral_threshold):
            continue
        left_error = _row_error(left_row)
        right_error = _row_error(right_row)
        if metric == "endpoint_error_mps":
            delta = right_error[2] - left_error[2]
        elif metric == "mae_vy_mps":
            delta = abs(right_error[1]) - abs(left_error[1])
        elif metric == "mae_vx_mps":
            delta = abs(right_error[0]) - abs(left_error[0])
        else:
            raise ValueError(f"Unsupported paired metric {metric!r}.")
        grouped.setdefault(str(left_row["scene_name"]), []).append(delta)
    return {scene: sum(values) / len(values) for scene, values in grouped.items() if values}


def paired_scene_bootstrap(
    scene_deltas: dict[str, float],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    values = [float(scene_deltas[name]) for name in sorted(scene_deltas)]
    if not values:
        return {
            "scene_count": 0,
            "mean_delta": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        }
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    lower = estimates[max(int(0.025 * samples) - 1, 0)]
    upper = estimates[min(int(0.975 * samples), samples - 1)]
    return {
        "scene_count": len(values),
        "mean_delta": sum(values) / len(values),
        "ci95_low": lower,
        "ci95_high": upper,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _baseline_metrics(
    paths: list[str], high_lateral_threshold: float
) -> tuple[list[dict], list[list[dict[str, str]]], list[dict[str, list[dict[str, str]]]]]:
    all_variants = [load_prediction_variants(path) for path in paths]
    baseline_rows = [variants["baseline"] for variants in all_variants]
    metrics = [metrics_from_rows(rows, high_lateral_threshold) for rows in baseline_rows]
    return metrics, baseline_rows, all_variants


def _aggregate_scene_deltas(per_seed: list[dict[str, float]]) -> dict[str, float]:
    common = set(per_seed[0])
    for values in per_seed[1:]:
        common &= set(values)
    return {
        scene: statistics.fmean(values[scene] for values in per_seed)
        for scene in sorted(common)
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _per_scene_rows(
    label: str,
    seed: int,
    rows: list[dict[str, Any]],
    high_lateral_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene_name"]), []).append(row)
    output = []
    for scene in sorted(grouped):
        metrics = metrics_from_rows(grouped[scene], high_lateral_threshold)
        output.append({"model": label, "seed": seed, "scene_name": scene, **metrics})
    return output


def _markdown(report: dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    aggregate = report["aggregate_metrics"]
    lines = [
        "# nuScenes Stage-1 Independent Test Report",
        "",
        f"- Overall result: **{'PASS' if acceptance['stage1_pass'] else 'FAIL'}**",
        f"- Experimental unit: scene",
        f"- Paired seeds: {', '.join(str(seed) for seed in report['seeds'])}",
        f"- Holdout scenes: {report['stage1_test_scene_count']}",
        "- `final_system_test` remained sealed and was not evaluated.",
        "",
        "## Primary Metrics",
        "",
        "| Model | EPE mean +/- std (m/s) | vx MAE (m/s) | vy MAE (m/s) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("geometry", "visual", "joint"):
        row = aggregate[name]
        lines.append(
            f"| {name} | {row['endpoint_error_mps']['mean']:.4f} +/- {row['endpoint_error_mps']['std']:.4f} "
            f"| {row['mae_vx_mps']['mean']:.4f} | {row['mae_vy_mps']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| Check | Result |",
            "|---|---:|",
        ]
    )
    for key, value in acceptance.items():
        if key != "stage1_pass":
            lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines.extend(["", "## Paired Scene Bootstrap", ""])
    for key, value in report["paired_scene_bootstrap"].items():
        lines.append(
            f"- {key}: delta={value['mean_delta']:.4f}, 95% CI "
            f"[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}], n={value['scene_count']} scenes"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative paired deltas favor the joint model. Acceptance thresholds are engineering gates; "
            "the bootstrap intervals quantify scene-to-scene uncertainty and are not computed from adjacent frames as independent samples.",
            "",
            "Ablation results, per-scene metrics, and ranked bad cases are stored beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    seeds: list[int],
    geometry_path: str,
    visual_paths: list[str],
    joint_paths: list[str],
    high_lateral_threshold: float,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(seeds) == len(visual_paths) == len(joint_paths)) or len(seeds) < 3:
        raise ValueError("Stage-1 analysis requires at least three paired visual/joint seeds.")
    geometry_variants = load_prediction_variants(geometry_path)
    geometry_rows = geometry_variants["baseline"]
    geometry_metric = metrics_from_rows(geometry_rows, high_lateral_threshold)
    visual_metrics, visual_rows, _ = _baseline_metrics(visual_paths, high_lateral_threshold)
    joint_metrics, joint_rows, joint_variants = _baseline_metrics(joint_paths, high_lateral_threshold)
    for left, right in zip(visual_rows, joint_rows):
        align_prediction_rows(left, right)
        align_prediction_rows(geometry_rows, left)

    aggregate = {}
    metric_names = ("endpoint_error_mps", "mae_vx_mps", "mae_vy_mps")
    for label, metrics in (
        ("geometry", [geometry_metric]),
        ("visual", visual_metrics),
        ("joint", joint_metrics),
    ):
        aggregate[label] = {
            metric: _mean_std([float(row[metric]) for row in metrics]) for metric in metric_names
        }

    paired_bootstrap = {}
    for metric, subset in (
        ("endpoint_error_mps", "all"),
        ("mae_vy_mps", "turning"),
        ("mae_vy_mps", "high_lateral"),
    ):
        per_seed = [
            scene_metric_deltas(left, right, metric, subset, high_lateral_threshold)
            for left, right in zip(visual_rows, joint_rows)
        ]
        paired_bootstrap[f"joint_minus_visual_{subset}_{metric}"] = paired_scene_bootstrap(
            _aggregate_scene_deltas(per_seed), bootstrap_samples, 42
        )

    geometry_pass = {
        "geometry_valid_rate": float(geometry_metric["geometry_valid_rate"]) >= 0.60,
        "geometry_condition": float(geometry_metric["geometry_condition_median"]) < 1.0e5,
        "geometry_vx_sign": float(geometry_metric["geometry_straight_vx_sign_agreement"]) >= 0.95,
    }
    visual_epe = aggregate["visual"]["endpoint_error_mps"]["mean"]
    geometry_epe = aggregate["geometry"]["endpoint_error_mps"]["mean"]
    visual_pass = {
        "visual_epe_within_2_percent": visual_epe <= 1.02 * geometry_epe,
        "visual_axis_mae_improved": (
            aggregate["visual"]["mae_vx_mps"]["mean"] < aggregate["geometry"]["mae_vx_mps"]["mean"]
            or aggregate["visual"]["mae_vy_mps"]["mean"] < aggregate["geometry"]["mae_vy_mps"]["mean"]
        ),
    }
    improved_seed_count = sum(
        int(
            float(joint.get("turning_mae_vy_mps", math.inf))
            < float(visual.get("turning_mae_vy_mps", math.inf))
            or float(joint.get("high_lateral_mae_vy_mps", math.inf))
            < float(visual.get("high_lateral_mae_vy_mps", math.inf))
        )
        for visual, joint in zip(visual_metrics, joint_metrics)
    )
    joint_pass = {
        "joint_epe_within_2_percent": (
            aggregate["joint"]["endpoint_error_mps"]["mean"] <= 1.02 * visual_epe
        ),
        "joint_group_improved_at_least_2_seeds": improved_seed_count >= 2,
        "joint_outputs_finite": all(bool(row["all_outputs_finite"]) for row in joint_metrics),
        "joint_correction_bounds": all(
            float(row["max_abs_kinematic_correction_vx_mps"]) <= 1.000001
            and float(row["max_abs_kinematic_correction_vy_mps"]) <= 0.500001
            for row in joint_metrics
        ),
    }
    acceptance = {**geometry_pass, **visual_pass, **joint_pass}
    acceptance["stage1_pass"] = all(acceptance.values())

    per_scene = _per_scene_rows("geometry", seeds[0], geometry_rows, high_lateral_threshold)
    for seed, rows in zip(seeds, visual_rows):
        per_scene.extend(_per_scene_rows("visual", seed, rows, high_lateral_threshold))
    for seed, rows in zip(seeds, joint_rows):
        per_scene.extend(_per_scene_rows("joint", seed, rows, high_lateral_threshold))

    ablations = []
    for seed, variants in zip(seeds, joint_variants):
        baseline = metrics_from_rows(variants["baseline"], high_lateral_threshold)
        for variant, rows in sorted(variants.items()):
            metrics = metrics_from_rows(rows, high_lateral_threshold)
            ablations.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "endpoint_error_mps": metrics["endpoint_error_mps"],
                    "delta_endpoint_error_mps": float(metrics["endpoint_error_mps"])
                    - float(baseline["endpoint_error_mps"]),
                    "mae_vx_mps": metrics["mae_vx_mps"],
                    "mae_vy_mps": metrics["mae_vy_mps"],
                }
            )

    token_errors: dict[tuple[str, str], list[float]] = {}
    target_by_token = {}
    for rows in joint_rows:
        for row in rows:
            key = (str(row["scene_name"]), str(row["center_token"]))
            token_errors.setdefault(key, []).append(_row_error(row)[2])
            target_by_token[key] = row
    bad_cases = []
    for key, values in token_errors.items():
        row = target_by_token[key]
        bad_cases.append(
            {
                "scene_name": key[0],
                "center_token": key[1],
                "mean_joint_endpoint_error_mps": statistics.fmean(values),
                "target_vx_mps": row["target_vx_mps"],
                "target_vy_mps": row["target_vy_mps"],
                "geometry_valid": row["geometry_valid"],
                "fallback_used": row["fallback_used"],
            }
        )
    bad_cases.sort(key=lambda row: float(row["mean_joint_endpoint_error_mps"]), reverse=True)
    report = {
        "partition": "stage1_test",
        "sealed_partition_evaluated": False,
        "experimental_unit": "scene",
        "seeds": seeds,
        "stage1_test_scene_count": int(geometry_metric["scene_count"]),
        "high_lateral_threshold_mps": high_lateral_threshold,
        "aggregate_metrics": aggregate,
        "per_seed_metrics": {"visual": visual_metrics, "joint": joint_metrics},
        "paired_scene_bootstrap": paired_bootstrap,
        "group_improved_seed_count": improved_seed_count,
        "acceptance": acceptance,
    }
    return report, per_scene, ablations, bad_cases[:100]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired stage-1 holdout predictions by scene.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--geometry_predictions", required=True)
    parser.add_argument("--visual_predictions", nargs="+", required=True)
    parser.add_argument("--joint_predictions", nargs="+", required=True)
    parser.add_argument("--training_stats", default="logs/v2_training_stats.json")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--output_dir", default="outputs/stage1_test_report")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    statistics_report = json.loads(Path(args.training_stats).read_text(encoding="utf-8"))
    threshold = float(statistics_report["high_lateral_threshold_mps"])
    report, per_scene, ablations, bad_cases = analyze(
        args.seeds,
        args.geometry_predictions,
        args.visual_predictions,
        args.joint_predictions,
        threshold,
        args.bootstrap_samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_report = json_safe(report)
    (output_dir / "stage1_report.json").write_text(
        json.dumps(safe_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "stage1_report.md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(output_dir / "per_scene_metrics.csv", per_scene)
    _write_csv(output_dir / "ablation_metrics.csv", ablations)
    _write_csv(output_dir / "bad_cases.csv", bad_cases)
    print(json.dumps(safe_report, indent=2, allow_nan=False))
    if args.enforce and not report["acceptance"]["stage1_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
