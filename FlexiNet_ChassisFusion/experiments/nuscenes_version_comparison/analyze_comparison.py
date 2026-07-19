from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable

import numpy as np


HOST_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_PROJECT_ROOT))

from experiments.nuscenes_version_comparison.bootstrap import SOURCE_PROJECT_ROOT  # noqa: E402

PROJECT_ROOT = SOURCE_PROJECT_ROOT

from experiments.nuscenes_version_comparison.splits import load_split_manifest  # noqa: E402
from experiments.nuscenes_version_comparison.config import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    load_yaml,
)


PRIMARY_ALGORITHMS = ("legacy_kd", "legacy_vkd", "v2_visual", "v2_visual_kinematic")
COMPARISONS = (
    ("v2_visual_kinematic", "legacy_vkd"),
    ("v2_visual_kinematic", "legacy_kd"),
    ("v2_visual", "legacy_vkd"),
    ("v2_visual", "legacy_kd"),
    ("legacy_vkd", "legacy_kd"),
    ("v2_visual_kinematic", "v2_visual"),
)
ROBUSTNESS_COMPARISONS = (
    ("v2_visual_kinematic", "legacy_vkd"),
    ("v2_visual_kinematic", "legacy_kd"),
    ("v2_visual", "legacy_vkd"),
    ("v2_visual", "legacy_kd"),
)


def read_rows(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No prediction rows were loaded.")
    return rows


def _key(row: dict[str, str]) -> tuple[str, str]:
    return str(row["scene_name"]), str(row["center_token"])


def align_prediction_rows(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> list[tuple[dict[str, str], dict[str, str]]]:
    left_map = {_key(row): row for row in left}
    right_map = {_key(row): row for row in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ValueError("Duplicate scene/token keys exist within a prediction group.")
    if set(left_map) != set(right_map):
        raise ValueError(
            "Prediction keys do not align: "
            f"left_only={sorted(set(left_map) - set(right_map))[:3]}, "
            f"right_only={sorted(set(right_map) - set(left_map))[:3]}."
        )
    pairs = [(left_map[key], right_map[key]) for key in sorted(left_map)]
    for left_row, right_row in pairs:
        for field in ("target_vx_mps", "target_vy_mps"):
            if abs(float(left_row[field]) - float(right_row[field])) > 1.0e-8:
                raise ValueError(f"Target mismatch for {_key(left_row)} field {field}.")
    return pairs


def paired_scene_bootstrap(
    scene_deltas: dict[str, float], samples: int = 10_000, seed: int = 42
) -> dict[str, float | int | str]:
    values = [float(scene_deltas[name]) for name in sorted(scene_deltas)]
    if not values:
        raise ValueError("Paired bootstrap requires at least one scene.")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    rng = random.Random(seed)
    boot = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    boot.sort()
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    mean = statistics.fmean(values)
    conclusion = "clear_improvement" if high < 0.0 else "clear_regression" if low > 0.0 else "trend_only"
    return {
        "mean_delta_mps": mean,
        "ci95_low_mps": low,
        "ci95_high_mps": high,
        "scene_count": len(values),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "conclusion": conclusion,
    }


def _quantile(
    rows: list[dict[str, str]],
    field: str,
    *,
    absolute: bool = False,
    positive_only: bool = False,
    q: float = 0.9,
):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if absolute:
        values = np.abs(values)
    if positive_only:
        values = values[values > 0.0]
    if values.size == 0:
        return math.inf
    return float(np.quantile(values, q))


def fit_thresholds(rows: list[dict[str, str]], quantile: float, turning: float) -> dict[str, float]:
    clean = [row for row in rows if row.get("variant", "clean") == "clean"]
    if not clean:
        raise ValueError("Threshold fitting requires clean train_core predictions.")
    return {
        "source_partition": str(clean[0].get("partition", "unknown")),
        "sample_count": len(clean),
        "quantile": quantile,
        "turning_yaw_rate_radps": turning,
        "high_lateral_velocity_mps": _quantile(clean, "target_vy_mps", absolute=True, q=quantile),
        "wheel_inconsistent_relative_std": _quantile(clean, "wheel_relative_std", q=quantile),
        "hard_accel_mps2": _quantile(clean, "longitudinal_accel_mps2", absolute=True, q=quantile),
        "high_lateral_accel_mps2": _quantile(clean, "lateral_accel_mps2", absolute=True, q=quantile),
        "high_steering_feedback": _quantile(clean, "steering_feedback", absolute=True, q=quantile),
        "high_throttle": _quantile(clean, "throttle", positive_only=True, q=quantile),
        "braking": _quantile(clean, "brake", positive_only=True, q=quantile),
    }


def _endpoint(row: dict[str, str]) -> float:
    return float(row["endpoint_error_mps"])


def _scenario_names(
    row: dict[str, str], thresholds: dict[str, Any], metadata: dict[str, dict[str, Any]]
) -> list[str]:
    speed = math.hypot(float(row["target_vx_mps"]), float(row["target_vy_mps"]))
    names = ["all"]
    if speed < 2.0:
        names.append("low_speed")
    elif speed < 10.0:
        names.append("mid_speed")
    else:
        names.append("high_speed")
    conditions = {
        "turning": abs(float(row["yaw_rate_radps"])) >= thresholds["turning_yaw_rate_radps"],
        "high_lateral": abs(float(row["target_vy_mps"])) >= thresholds["high_lateral_velocity_mps"],
        "wheel_inconsistent": float(row["wheel_relative_std"]) >= thresholds["wheel_inconsistent_relative_std"],
        "hard_accel": abs(float(row["longitudinal_accel_mps2"])) >= thresholds["hard_accel_mps2"],
        "high_lateral_accel": abs(float(row["lateral_accel_mps2"])) >= thresholds["high_lateral_accel_mps2"],
        "high_steering": abs(float(row["steering_feedback"])) >= thresholds["high_steering_feedback"],
        "high_throttle": float(row["throttle"]) >= thresholds["high_throttle"],
        "braking": float(row["brake"]) >= thresholds["braking"],
    }
    names.extend(name for name, selected in conditions.items() if selected)
    scene = metadata.get(str(row["scene_name"]), {})
    for field in ("location", "lighting", "weather", "motion_regime"):
        value = str(scene.get(field, "unknown")).strip().lower().replace(" ", "_")
        names.append(f"{field}:{value}")
    return names


def _metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    error_x = np.asarray([float(row["error_vx_mps"]) for row in rows])
    error_y = np.asarray([float(row["error_vy_mps"]) for row in rows])
    endpoint = np.asarray([_endpoint(row) for row in rows])
    target_speed = np.hypot(
        [float(row["target_vx_mps"]) for row in rows],
        [float(row["target_vy_mps"]) for row in rows],
    )
    predicted_speed = np.hypot(
        [float(row["pred_vx_mps"]) for row in rows],
        [float(row["pred_vy_mps"]) for row in rows],
    )
    return {
        "sample_count": len(rows),
        "scene_count": len({row["scene_name"] for row in rows}),
        "mae_vx_mps": float(np.mean(np.abs(error_x))),
        "rmse_vx_mps": float(np.sqrt(np.mean(error_x**2))),
        "mae_vy_mps": float(np.mean(np.abs(error_y))),
        "rmse_vy_mps": float(np.sqrt(np.mean(error_y**2))),
        "endpoint_error_mps": float(np.mean(endpoint)),
        "endpoint_p90_mps": float(np.quantile(endpoint, 0.90)),
        "endpoint_p95_mps": float(np.quantile(endpoint, 0.95)),
        "endpoint_p99_mps": float(np.quantile(endpoint, 0.99)),
        "speed_magnitude_mae_mps": float(np.mean(np.abs(predicted_speed - target_speed))),
    }


def _group_rows(rows: list[dict[str, str]], fields: tuple[str, ...]):
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_required(rows: list[dict[str, str]], required: list[str], seeds: list[int]) -> None:
    available = {(row["algorithm"], int(row["seed"])) for row in rows if row["variant"] == "clean"}
    missing = [(algorithm, seed) for algorithm in required for seed in seeds if (algorithm, seed) not in available]
    if missing:
        raise ValueError(f"Missing required clean algorithm/seed runs: {missing}.")


def _validate_clean_alignment(rows: list[dict[str, str]]) -> None:
    groups = _group_rows(
        [row for row in rows if row["variant"] == "clean"],
        ("algorithm", "seed"),
    )
    if not groups:
        raise ValueError("No clean prediction groups were provided.")
    reference_key = sorted(groups)[0]
    reference = groups[reference_key]
    for key, group in sorted(groups.items()):
        try:
            align_prediction_rows(reference, group)
        except ValueError as error:
            raise ValueError(
                f"Clean prediction alignment failed for {reference_key} versus {key}: {error}"
            ) from error


def analyze(
    rows: list[dict[str, str]],
    thresholds: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _validate_clean_alignment(rows)
    metric_rows: list[dict[str, Any]] = []
    base_groups = _group_rows(rows, ("algorithm", "seed", "variant"))
    for (algorithm, seed, variant), group in sorted(base_groups.items()):
        scenario_groups: dict[str, list[dict[str, str]]] = {}
        for row in group:
            for scenario in _scenario_names(row, thresholds, metadata):
                scenario_groups.setdefault(scenario, []).append(row)
        for scenario, selected in sorted(scenario_groups.items()):
            metric_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "variant": variant,
                    "scenario": scenario,
                    **_metrics(selected),
                }
            )

    clean_overall = [
        row for row in metric_rows if row["variant"] == "clean" and row["scenario"] == "all"
    ]
    aggregate = []
    for (algorithm,), group in sorted(_group_rows(clean_overall, ("algorithm",)).items()):
        aggregate.append(
            {
                "algorithm": algorithm,
                "seed_count": len(group),
                "sample_count": statistics.fmean(row["sample_count"] for row in group),
                "scene_count": statistics.fmean(row["scene_count"] for row in group),
                "endpoint_error_mean_mps": statistics.fmean(row["endpoint_error_mps"] for row in group),
                "endpoint_error_std_mps": statistics.stdev(row["endpoint_error_mps"] for row in group) if len(group) > 1 else 0.0,
                "mae_vx_mean_mps": statistics.fmean(row["mae_vx_mps"] for row in group),
                "mae_vy_mean_mps": statistics.fmean(row["mae_vy_mps"] for row in group),
                "endpoint_p95_mean_mps": statistics.fmean(row["endpoint_p95_mps"] for row in group),
            }
        )

    per_scene_deltas: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    available_algorithms = {row["algorithm"] for row in rows}
    variants = sorted({row["variant"] for row in rows})
    for left_name, right_name in COMPARISONS:
        if left_name not in available_algorithms or right_name not in available_algorithms:
            continue
        for variant in variants:
            per_seed_scene: list[dict[str, float]] = []
            seeds = sorted(
                {int(row["seed"]) for row in rows if row["algorithm"] == left_name and row["variant"] == variant}
                & {int(row["seed"]) for row in rows if row["algorithm"] == right_name and row["variant"] == variant}
            )
            for seed in seeds:
                left = [row for row in rows if row["algorithm"] == left_name and row["variant"] == variant and int(row["seed"]) == seed]
                right = [row for row in rows if row["algorithm"] == right_name and row["variant"] == variant and int(row["seed"]) == seed]
                pairs = align_prediction_rows(left, right)
                grouped: dict[str, list[float]] = {}
                for left_row, right_row in pairs:
                    grouped.setdefault(left_row["scene_name"], []).append(_endpoint(left_row) - _endpoint(right_row))
                scene_values = {scene: statistics.fmean(values) for scene, values in grouped.items()}
                per_seed_scene.append(scene_values)
                per_scene_deltas.extend(
                    {
                        "comparison": f"{left_name}_minus_{right_name}",
                        "variant": variant,
                        "seed": seed,
                        "scene_name": scene,
                        "endpoint_delta_mps": value,
                    }
                    for scene, value in sorted(scene_values.items())
                )
            if per_seed_scene:
                common = set.intersection(*(set(values) for values in per_seed_scene))
                averaged = {
                    scene: statistics.fmean(values[scene] for values in per_seed_scene)
                    for scene in sorted(common)
                }
                bootstrap[f"{left_name}_minus_{right_name}:{variant}"] = paired_scene_bootstrap(
                    averaged, bootstrap_samples, bootstrap_seed
                )

    stress_rows = []
    metric_lookup = {
        (row["algorithm"], row["seed"], row["variant"]): row
        for row in metric_rows
        if row["scenario"] == "all"
    }
    for (algorithm, seed, variant), row in sorted(metric_lookup.items()):
        clean = metric_lookup.get((algorithm, seed, "clean"))
        if clean is None:
            continue
        stress_rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "variant": variant,
                "endpoint_error_mps": row["endpoint_error_mps"],
                "degradation_vs_clean_mps": row["endpoint_error_mps"] - clean["endpoint_error_mps"],
            }
        )

    stress_robustness = []
    stress_lookup = {
        (row["algorithm"], int(row["seed"]), row["variant"]): row
        for row in stress_rows
    }
    for left_name, right_name in ROBUSTNESS_COMPARISONS:
        for variant in sorted({row["variant"] for row in stress_rows if row["variant"] != "clean"}):
            seeds = sorted(
                {seed for algorithm, seed, name in stress_lookup if algorithm == left_name and name == variant}
                & {seed for algorithm, seed, name in stress_lookup if algorithm == right_name and name == variant}
            )
            if not seeds:
                continue
            absolute_delta = statistics.fmean(
                stress_lookup[(left_name, seed, variant)]["endpoint_error_mps"]
                - stress_lookup[(right_name, seed, variant)]["endpoint_error_mps"]
                for seed in seeds
            )
            degradation_delta = statistics.fmean(
                stress_lookup[(left_name, seed, variant)]["degradation_vs_clean_mps"]
                - stress_lookup[(right_name, seed, variant)]["degradation_vs_clean_mps"]
                for seed in seeds
            )
            stress_robustness.append(
                {
                    "comparison": f"{left_name}_minus_{right_name}",
                    "variant": variant,
                    "seed_count": len(seeds),
                    "absolute_error_delta_mps": absolute_delta,
                    "degradation_delta_mps": degradation_delta,
                    "robust_by_dual_criterion": absolute_delta <= 0.0 and degradation_delta <= 0.0,
                }
            )

    available_clean_algorithms = {
        row["algorithm"] for row in rows if row["variant"] == "clean"
    }
    current_name = (
        "v2_visual_kinematic"
        if "v2_visual_kinematic" in available_clean_algorithms
        else "v2_visual"
        if "v2_visual" in available_clean_algorithms
        else sorted(available_clean_algorithms)[0]
    )
    current_clean = [row for row in rows if row["algorithm"] == current_name and row["variant"] == "clean"]
    bad_group = _group_rows(current_clean, ("scene_name", "center_token"))
    bad_cases = [
        {
            "scene_name": key[0],
            "center_token": key[1],
            "mean_endpoint_error_mps": statistics.fmean(_endpoint(row) for row in group),
            "seed_count": len(group),
        }
        for key, group in bad_group.items()
    ]
    bad_cases.sort(key=lambda row: row["mean_endpoint_error_mps"], reverse=True)
    report = {
        "experimental_unit": "scene",
        "aggregate_clean": aggregate,
        "paired_scene_bootstrap": bootstrap,
        "stress_robustness": stress_robustness,
        "interpretation_boundary": {
            "verified": "Open-loop nuScenes accuracy, subgroup behavior, sensor perturbations, calibration diagnostics, and location shift.",
            "not_verified": "Physical vehicle-parameter mismatch, closed-loop estimator-controller coupling, CG sideslip truth, and cross-vehicle dynamics generalization.",
            "current_v2": "Visual geometry plus weak kinematic correction; no dynamics student branch is implemented.",
        },
    }
    return report, metric_rows, stress_rows, per_scene_deltas, bad_cases


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# nuScenes Version Comparison",
        "",
        "## Clean Accuracy",
        "",
        "| Algorithm | Seeds | Endpoint EE (mean +/- std) | MAE vx | MAE vy | P95 EE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate_clean"]:
        lines.append(
            f"| {row['algorithm']} | {row['seed_count']} | "
            f"{row['endpoint_error_mean_mps']:.5f} +/- {row['endpoint_error_std_mps']:.5f} | "
            f"{row['mae_vx_mean_mps']:.5f} | {row['mae_vy_mean_mps']:.5f} | "
            f"{row['endpoint_p95_mean_mps']:.5f} |"
        )
    lines.extend(["", "## Paired Scene Bootstrap", ""])
    for name, result in report["paired_scene_bootstrap"].items():
        if not name.endswith(":clean"):
            continue
        lines.append(
            f"- `{name}`: delta {result['mean_delta_mps']:.5f} m/s, "
            f"95% CI [{result['ci95_low_mps']:.5f}, {result['ci95_high_mps']:.5f}], "
            f"{result['conclusion']}."
        )
    boundary = report["interpretation_boundary"]
    lines.extend(
        [
            "",
            "## Conclusion Boundary",
            "",
            f"- Verified scope: {boundary['verified']}",
            f"- Trend-only results are those whose paired 95% interval crosses zero.",
            f"- Not verified: {boundary['not_verified']}",
            f"- Current V2 definition: {boundary['current_v2']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aligned comparison predictions by scene.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-thresholds")
    fit.add_argument("--predictions", nargs="+", required=True)
    fit.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    fit.add_argument("--output", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--predictions", nargs="+", required=True)
    report_parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    report_parser.add_argument("--manifest", required=True)
    report_parser.add_argument("--thresholds", required=True)
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--required-algorithms", nargs="+", default=list(PRIMARY_ALGORITHMS))
    report_parser.add_argument("--allow-incomplete", action="store_true")
    report_parser.add_argument("--bootstrap-samples", type=int)
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    rows = read_rows(args.predictions)
    if args.command == "fit-thresholds":
        analysis = protocol["analysis"]
        thresholds = fit_thresholds(
            rows,
            float(analysis["quantile"]),
            float(analysis["turning_yaw_rate_radps"]),
        )
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(json.dumps(thresholds, indent=2))
        return

    if not args.allow_incomplete:
        _validate_required(rows, args.required_algorithms, [int(seed) for seed in protocol["protocol"]["seeds"]])
    manifest = load_split_manifest(args.manifest)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    result = analyze(
        rows,
        thresholds,
        manifest.get("scene_metadata", {}),
        bootstrap_samples=int(args.bootstrap_samples or protocol["protocol"]["bootstrap_samples"]),
        bootstrap_seed=int(protocol["protocol"]["bootstrap_seed"]),
    )
    report, metric_rows, stress_rows, per_scene_deltas, bad_cases = result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "comparison_report.md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(output_dir / "metrics_by_scenario.csv", metric_rows)
    _write_csv(output_dir / "stress_metrics.csv", stress_rows)
    _write_csv(output_dir / "stress_comparisons.csv", report["stress_robustness"])
    _write_csv(output_dir / "per_scene_deltas.csv", per_scene_deltas)
    _write_csv(
        output_dir / "bad_cases.csv",
        bad_cases[: int(protocol["analysis"]["bad_case_count"])],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
