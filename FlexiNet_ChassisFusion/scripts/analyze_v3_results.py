from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_v3_paired_bootstrap import (  # noqa: E402
    bootstrap,
    load_predictions,
    paired_scene_deltas,
)


PRIMARY_METRICS = ("mae_vx_mps", "mae_vy_mps", "epe_mps")
ALL_METRICS = (
    *PRIMARY_METRICS,
    "p95_epe_mps",
    "speed_mae_mps",
    "sideslip_mae_deg",
    "nll",
    "regression_calibration_error",
    "risk_coverage_auc",
    "geometry_valid_rate",
    "visual_fallback_rate",
)


def _rows(path: str | Path) -> list[dict[str, str]]:
    return list(load_predictions(path).values())


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot analyze an empty prediction table.")
    target = np.asarray(
        [[float(row["target_vx_mps"]), float(row["target_vy_mps"])] for row in rows]
    )
    prediction = np.asarray(
        [[float(row["pred_vx_mps"]), float(row["pred_vy_mps"])] for row in rows]
    )
    log_variance = np.asarray(
        [
            [float(row["log_variance_vx"]), float(row["log_variance_vy"])]
            for row in rows
        ]
    )
    error = prediction - target
    endpoint = np.linalg.norm(error, axis=1)
    speed_error = np.abs(np.linalg.norm(prediction, axis=1) - np.linalg.norm(target, axis=1))
    beta_mask = np.linalg.norm(target, axis=1) >= 2.0
    beta_error = np.asarray([], dtype=np.float64)
    if beta_mask.any():
        difference = np.arctan2(prediction[beta_mask, 1], prediction[beta_mask, 0])
        difference -= np.arctan2(target[beta_mask, 1], target[beta_mask, 0])
        beta_error = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))

    scale = np.asarray([5.0, 1.0])
    normalized = error / scale
    nll = 0.5 * (np.exp(-log_variance) * normalized**2 + log_variance)
    uncertainty = np.linalg.norm(np.exp(0.5 * log_variance) * scale, axis=1)
    order = np.argsort(uncertainty)
    calibration_bins = []
    for indices in np.array_split(order, min(10, len(order))):
        if len(indices):
            predicted_sigma = uncertainty[indices].mean()
            empirical_rmse = np.sqrt(np.mean(endpoint[indices] ** 2))
            calibration_bins.append(abs(predicted_sigma - empirical_rmse))
    ordered_error = endpoint[order]
    risk_curve = np.cumsum(ordered_error) / np.arange(1, len(ordered_error) + 1)
    geometry_valid = np.asarray([float(row["geometry_valid"]) for row in rows])
    return {
        "mae_vx_mps": float(np.mean(np.abs(error[:, 0]))),
        "mae_vy_mps": float(np.mean(np.abs(error[:, 1]))),
        "epe_mps": float(np.mean(endpoint)),
        "p95_epe_mps": float(np.quantile(endpoint, 0.95)),
        "speed_mae_mps": float(np.mean(speed_error)),
        "sideslip_mae_deg": float(np.rad2deg(beta_error).mean())
        if beta_error.size
        else math.nan,
        "nll": float(nll.mean()),
        "regression_calibration_error": float(np.mean(calibration_bins)),
        "risk_coverage_auc": float(np.mean(risk_curve)),
        "geometry_valid_rate": float(np.mean(geometry_valid)),
        "visual_fallback_rate": float(np.mean(1.0 - geometry_valid)),
    }


def _aggregate(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in ALL_METRICS:
        values = [row[metric] for row in per_seed if math.isfinite(row[metric])]
        result[metric] = {
            "mean": statistics.fmean(values) if values else math.nan,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_count": len(values),
        }
    return result


def _effect_sizes(
    seed_deltas: list[dict[str, tuple[float, float, float]]],
) -> tuple[dict[str, float], dict[str, float]]:
    scenes = sorted(set.intersection(*(set(item) for item in seed_deltas)))
    names = PRIMARY_METRICS
    mean_deltas: dict[str, float] = {}
    effect_sizes: dict[str, float] = {}
    for axis, name in enumerate(names):
        values = [
            statistics.fmean(seed_delta[scene][axis] for seed_delta in seed_deltas)
            for scene in scenes
        ]
        mean_deltas[name] = statistics.fmean(values)
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        effect_sizes[name] = (
            mean_deltas[name] / standard_deviation
            if standard_deviation > 0.0
            else math.nan
        )
    return mean_deltas, effect_sizes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(
    output_dir: Path,
    seeds: list[int],
    direct_metrics: list[dict[str, float]],
    fusion_metrics: list[dict[str, float]],
    scene_deltas: list[dict[str, tuple[float, float, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = ("vx MAE", "vy MAE", "2D EPE")
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, metric, label in zip(axes, PRIMARY_METRICS, labels):
        for index, seed in enumerate(seeds):
            axis.plot(
                [0, 1],
                [direct_metrics[index][metric], fusion_metrics[index][metric]],
                marker="o",
                linewidth=1.5,
                label=f"seed {seed}" if metric == PRIMARY_METRICS[0] else None,
            )
        axis.set_xticks([0, 1], ["Direct", "Fusion"])
        axis.set_ylabel(f"{label} (m/s)")
        axis.grid(axis="y", alpha=0.25)
    figure.legend(loc="upper center", ncol=len(seeds), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(figure_dir / "figure-01-seed-comparison.png", dpi=180)
    plt.close(figure)

    scenes = sorted(set.intersection(*(set(item) for item in scene_deltas)))
    values = sorted(
        statistics.fmean(item[scene][2] for item in scene_deltas) for scene in scenes
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    colors = ["#2b7a4b" if value < 0.0 else "#b94a48" for value in values]
    axis.bar(np.arange(len(values)), values, color=colors, width=0.9)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Scenes sorted by fusion - direct EPE")
    axis.set_ylabel("EPE difference (m/s)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "figure-02-scene-epe-deltas.png", dpi=180)
    plt.close(figure)


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    bootstrap_metrics = report["paired_scene_bootstrap"]["metrics"]
    epe = bootstrap_metrics["epe_mps"]
    if epe["significant_improvement"]:
        decision = "Fusion improves scene-level EPE over direct visual estimation."
        allowed = "Fusion reduced EPE versus the paired direct baseline on this split."
    else:
        decision = "The scene-level EPE interval does not establish an improvement."
        allowed = "Fusion and direct performance are not distinguishable at the 95% scene-bootstrap level."
    analysis = f"""# V3 Visual-Only Analysis Report

## Analysis Question
Does pure-visual V3 fusion improve velocity estimation over its direct visual expert on the independent `stage1_test` split?

## Evidence
- Seeds: {', '.join(str(seed) for seed in report['seeds'])}
- Experimental unit: scene
- Paired scenes: {report['paired_scene_bootstrap']['scene_count']}
- Bootstrap samples: {report['paired_scene_bootstrap']['bootstrap_samples']}
- Fusion minus direct EPE: {epe['candidate_minus_baseline']:.6f} m/s
- 95% scene-bootstrap CI: [{epe['ci95'][0]:.6f}, {epe['ci95'][1]:.6f}] m/s

## Interpretation
{decision}

## Claim Candidate
- Allowed wording: {allowed}
- Forbidden stronger wording: the method is generally superior on nuScenes or other datasets.
- Uncertainty: three training seeds quantify limited seed variability; the interval resamples scenes on this fixed split.
- Decision: {'keep' if epe['significant_improvement'] else 'weaken'}

## Limitations
- This report compares fusion only with its strict direct visual baseline.
- Smoke results are excluded from the analysis.
- No CAN, wheel-speed, IMU, kinematic, or dynamic input is evaluated.
"""
    (output_dir / "analysis-report.md").write_text(analysis, encoding="utf-8")

    lines = [
        "# Statistical Appendix",
        "",
        "All differences are fusion minus direct; negative errors favor fusion.",
        "Scenes, rather than adjacent frames, are the resampling units.",
        "",
        "| Metric | Difference | 95% CI | Paired Cohen dz | Significant improvement |",
        "|---|---:|---:|---:|:---:|",
    ]
    for metric in PRIMARY_METRICS:
        item = bootstrap_metrics[metric]
        effect = report["paired_scene_effect_size_dz"][metric]
        lines.append(
            f"| {metric} | {item['candidate_minus_baseline']:.6f} | "
            f"[{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] | "
            f"{effect:.4f} | {item['significant_improvement']} |"
        )
    lines.extend(
        [
            "",
            "The paired Cohen dz is computed from per-scene differences averaged across seeds. ",
            "With only three training seeds, seed-level normality tests and seed-level significance tests are not reported.",
        ]
    )
    (output_dir / "stats-appendix.md").write_text("\n".join(lines), encoding="utf-8")

    catalog = """# Figure Catalog

## Figure 1: Seed Comparison
- File: `figures/figure-01-seed-comparison.png`
- Purpose: compare direct and fusion errors for every paired training seed.
- Error bars: none; each line is one seed.
- Interpretation: consistent downward lines indicate improvement that is stable across seeds.

## Figure 2: Scene EPE Differences
- File: `figures/figure-02-scene-epe-deltas.png`
- Purpose: show heterogeneity of fusion-minus-direct EPE across scenes.
- Unit: one bar per scene after averaging the paired difference across seeds.
- Interpretation: negative bars favor fusion; both signs indicate scene-dependent behavior.
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def analyze_results(
    seeds: list[int],
    direct_paths: list[str | Path],
    fusion_paths: list[str | Path],
    samples: int,
    bootstrap_seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("Formal V3 analysis requires at least three paired seeds.")
    if not (len(seeds) == len(direct_paths) == len(fusion_paths)):
        raise ValueError("Seeds, direct predictions, and fusion predictions must align.")

    direct_metrics = [compute_metrics(_rows(path)) for path in direct_paths]
    fusion_metrics = [compute_metrics(_rows(path)) for path in fusion_paths]
    seed_deltas = [
        paired_scene_deltas(fusion, direct)
        for fusion, direct in zip(fusion_paths, direct_paths)
    ]
    bootstrap_report = bootstrap(seed_deltas, samples, bootstrap_seed)
    mean_deltas, effect_sizes = _effect_sizes(seed_deltas)
    report = {
        "seeds": seeds,
        "experimental_unit": "scene",
        "direct_per_seed": direct_metrics,
        "fusion_per_seed": fusion_metrics,
        "direct_aggregate": _aggregate(direct_metrics),
        "fusion_aggregate": _aggregate(fusion_metrics),
        "paired_scene_mean_delta": mean_deltas,
        "paired_scene_effect_size_dz": effect_sizes,
        "paired_scene_bootstrap": bootstrap_report,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    table_rows = []
    for seed, direct, fusion in zip(seeds, direct_metrics, fusion_metrics):
        for method, metrics in (("direct", direct), ("fusion", fusion)):
            table_rows.append({"seed": seed, "method": method, **metrics})
    _write_csv(output_path / "per-seed-metrics.csv", table_rows)
    _make_figures(output_path, seeds, direct_metrics, fusion_metrics, seed_deltas)
    _write_reports(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired three-seed V3 direct and fusion holdout results."
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--direct", nargs="+", required=True)
    parser.add_argument("--fusion", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = analyze_results(
        args.seeds,
        args.direct,
        args.fusion,
        args.samples,
        args.bootstrap_seed,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
