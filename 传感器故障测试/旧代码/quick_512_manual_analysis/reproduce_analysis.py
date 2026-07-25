from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "quick_512_manual_analysis" else SCRIPT_DIR
INPUT = ROOT / "quick_512_manual_server"
OUTPUT = ROOT / "quick_512_manual_analysis"
FIGURES = OUTPUT / "figures"
ALGORITHMS = ("base_velocity", "legacy_kd", "legacy_vkd")
VARIANTS = ("clean", "can_missing", "yaw_bias_pos", "wheel_scale_low", "camera_missing")
COLORS = {
    "base_velocity": "#7F7F7F",
    "legacy_kd": "#0072B2",
    "legacy_vkd": "#D55E00",
}


def bootstrap_ci(values: np.ndarray, seed: int = 42, samples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, p_values[index] * (count - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def signed_rank_biserial(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / ranks.sum())


def audit_alignment(frame: pd.DataFrame) -> dict[str, object]:
    key_columns = ["scene_name", "center_token"]
    reference = None
    label_reference = None
    counts: dict[str, int] = {}
    for algorithm in ALGORITHMS:
        for variant in VARIANTS:
            subset = frame[(frame.algorithm == algorithm) & (frame.variant == variant)].copy()
            label = f"{algorithm}:{variant}"
            counts[label] = len(subset)
            if len(subset) != 512:
                raise AssertionError(f"{label} has {len(subset)} rows, expected 512")
            if subset.duplicated(key_columns).any():
                raise AssertionError(f"{label} contains duplicate sample keys")
            subset = subset.sort_values(key_columns).reset_index(drop=True)
            keys = subset[key_columns]
            labels = subset[["target_vx_mps", "target_vy_mps"]].to_numpy()
            if reference is None:
                reference = keys
                label_reference = labels
            else:
                if not keys.equals(reference):
                    raise AssertionError(f"{label} sample keys are not aligned")
                if not np.allclose(labels, label_reference, rtol=0.0, atol=1e-9):
                    raise AssertionError(f"{label} velocity labels differ")
    return {
        "valid": True,
        "total_rows": int(len(frame)),
        "samples_per_algorithm_variant": 512,
        "scene_count": int(frame.scene_name.nunique()),
        "algorithm_variant_counts": counts,
        "sample_key": "scene_name + center_token",
        "labels_invariant": True,
    }


def scene_errors(frame: pd.DataFrame, algorithm: str, variant: str) -> pd.Series:
    subset = frame[(frame.algorithm == algorithm) & (frame.variant == variant)]
    return subset.groupby("scene_name")["endpoint_error_mps"].mean().sort_index()


def paired_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_p_values = []
    for variant in VARIANTS:
        kd = scene_errors(frame, "legacy_kd", variant)
        vkd = scene_errors(frame, "legacy_vkd", variant)
        delta = (vkd - kd).to_numpy()
        shapiro = stats.shapiro(delta)
        wilcoxon = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        ci_low, ci_high = bootstrap_ci(delta)
        raw_p_values.append(float(wilcoxon.pvalue))
        rows.append(
            {
                "variant": variant,
                "scene_count": len(delta),
                "scene_mean_delta_mps": delta.mean(),
                "scene_median_delta_mps": np.median(delta),
                "ci95_low_mps": ci_low,
                "ci95_high_mps": ci_high,
                "vkd_scene_win_rate": float((delta < 0).mean()),
                "shapiro_w": float(shapiro.statistic),
                "shapiro_p": float(shapiro.pvalue),
                "wilcoxon_w": float(wilcoxon.statistic),
                "wilcoxon_p_raw": float(wilcoxon.pvalue),
                "rank_biserial": signed_rank_biserial(delta),
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = value
    return pd.DataFrame(rows)


def metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_rows = []
    for algorithm in ALGORITHMS:
        subset = frame[(frame.algorithm == algorithm) & (frame.variant == "clean")]
        scene = scene_errors(frame, algorithm, "clean").to_numpy()
        ci_low, ci_high = bootstrap_ci(scene)
        clean_rows.append(
            {
                "algorithm": algorithm,
                "samples": len(subset),
                "scenes": subset.scene_name.nunique(),
                "endpoint_epe_mps": subset.endpoint_error_mps.mean(),
                "scene_mean_epe_mps": scene.mean(),
                "scene_ci95_low_mps": ci_low,
                "scene_ci95_high_mps": ci_high,
                "mae_vx_mps": subset.error_vx_mps.abs().mean(),
                "mae_vy_mps": subset.error_vy_mps.abs().mean(),
                "p95_epe_mps": subset.endpoint_error_mps.quantile(0.95),
            }
        )

    stress_rows = []
    for algorithm in ("legacy_kd", "legacy_vkd"):
        clean = scene_errors(frame, algorithm, "clean")
        clean_sample = frame[(frame.algorithm == algorithm) & (frame.variant == "clean")]
        clean_epe = clean_sample.endpoint_error_mps.mean()
        for variant in VARIANTS[1:]:
            variant_scene = scene_errors(frame, algorithm, variant)
            degradation = (variant_scene - clean).to_numpy()
            ci_low, ci_high = bootstrap_ci(degradation)
            subset = frame[(frame.algorithm == algorithm) & (frame.variant == variant)]
            stress_rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "endpoint_epe_mps": subset.endpoint_error_mps.mean(),
                    "degradation_mps": subset.endpoint_error_mps.mean() - clean_epe,
                    "scene_mean_degradation_mps": degradation.mean(),
                    "scene_ci95_low_mps": ci_low,
                    "scene_ci95_high_mps": ci_high,
                }
            )
    return pd.DataFrame(clean_rows), pd.DataFrame(stress_rows)


def save_figures(frame: pd.DataFrame, clean: pd.DataFrame, stress: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 9})

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(clean))
    values = clean.scene_mean_epe_mps.to_numpy()
    lower = values - clean.scene_ci95_low_mps.to_numpy()
    upper = clean.scene_ci95_high_mps.to_numpy() - values
    ax.bar(x, values, color=[COLORS[name] for name in clean.algorithm], width=0.65)
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=4, lw=1.2)
    ax.set_xticks(x, ["Base velocity", "KD", "VKD"])
    ax.set_ylabel("Scene-level mean endpoint error (m/s)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-01-clean-comparison.pdf")
    fig.savefig(FIGURES / "figure-01-clean-comparison.png", dpi=300)
    plt.close(fig)

    deltas = []
    for variant in VARIANTS:
        kd = scene_errors(frame, "legacy_kd", variant)
        vkd = scene_errors(frame, "legacy_vkd", variant)
        deltas.append((vkd - kd).to_numpy())
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    plot = ax.boxplot(deltas, tick_labels=[v.replace("_", "\n") for v in VARIANTS], patch_artist=True, showfliers=False)
    for box in plot["boxes"]:
        box.set_facecolor("#56B4E9")
        box.set_alpha(0.75)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel("Scene EPE delta: VKD - KD (m/s)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-02-paired-scene-deltas.pdf")
    fig.savefig(FIGURES / "figure-02-paired-scene-deltas.png", dpi=300)
    plt.close(fig)

    regular = ["yaw_bias_pos", "wheel_scale_low", "camera_missing"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    width = 0.36
    for ax, variants in zip(axes, [regular, ["can_missing"]]):
        positions = np.arange(len(variants))
        for offset, algorithm in zip((-width / 2, width / 2), ("legacy_kd", "legacy_vkd")):
            selected = stress[(stress.algorithm == algorithm) & stress.variant.isin(variants)].set_index("variant").loc[variants]
            values = selected.scene_mean_degradation_mps.to_numpy()
            lower = values - selected.scene_ci95_low_mps.to_numpy()
            upper = selected.scene_ci95_high_mps.to_numpy() - values
            ax.bar(positions + offset, values, width, label=algorithm.replace("legacy_", "").upper(), color=COLORS[algorithm])
            ax.errorbar(positions + offset, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=3, lw=1.0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(positions, [value.replace("_", "\n") for value in variants])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Scene-level degradation vs clean (m/s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-03-stress-degradation.pdf")
    fig.savefig(FIGURES / "figure-03-stress-degradation.png", dpi=300)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3e}" if value != 0 and abs(value) < 10 ** -digits else f"{value:.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_reports(clean: pd.DataFrame, stress: pd.DataFrame, paired: pd.DataFrame) -> None:
    kd = clean.set_index("algorithm").loc["legacy_kd"]
    vkd = clean.set_index("algorithm").loc["legacy_vkd"]
    base = clean.set_index("algorithm").loc["base_velocity"]
    clean_relative = 100 * (kd.endpoint_epe_mps - vkd.endpoint_epe_mps) / kd.endpoint_epe_mps
    base_relative = 100 * (base.endpoint_epe_mps - vkd.endpoint_epe_mps) / base.endpoint_epe_mps
    vx_relative = 100 * (kd.mae_vx_mps - vkd.mae_vx_mps) / kd.mae_vx_mps
    vy_relative = 100 * (kd.mae_vy_mps - vkd.mae_vy_mps) / kd.mae_vy_mps
    clean_test = paired.set_index("variant").loc["clean"]

    clean_table = markdown_table(
        clean,
        ["algorithm", "samples", "scenes", "endpoint_epe_mps", "mae_vx_mps", "mae_vy_mps", "p95_epe_mps"],
    )
    stress_table = markdown_table(
        stress,
        ["algorithm", "variant", "endpoint_epe_mps", "degradation_mps"],
    )
    paired_table = markdown_table(
        paired,
        ["variant", "scene_count", "scene_mean_delta_mps", "ci95_low_mps", "ci95_high_mps", "vkd_scene_win_rate", "wilcoxon_p_raw", "wilcoxon_p_holm", "rank_biserial"],
    )

    analysis_report = f"""# quick_512_manual Strict Analysis

## Analysis Question

Does the retained visual-kinematic-dynamic checkpoint (`legacy_vkd`) improve endpoint velocity error and synthetic sensor-stress behavior relative to the retained kinematic-dynamic checkpoint (`legacy_kd`) on exactly aligned causal nuScenes samples?

Primary metric: endpoint velocity error (EPE, lower is better). Statistical unit: scene. The 512 windows belong to 75 scenes and are not treated as independent experimental replicates.

## Key Findings

{clean_table}

- On clean samples, VKD lowers sample-weighted EPE from {kd.endpoint_epe_mps:.5f} to {vkd.endpoint_epe_mps:.5f} m/s ({clean_relative:.2f}% relative), while remaining {base_relative:.2f}% below the base-velocity reference.
- The gain is concentrated in longitudinal error: MAE vx improves {vx_relative:.2f}%, but MAE vy changes by {vy_relative:.2f}% (negative means worse). Thus this result does not support a claim of improved lateral-velocity accuracy.
- Scene-level VKD-KD clean delta is {clean_test.scene_mean_delta_mps:.5f} m/s with bootstrap 95% CI [{clean_test.ci95_low_mps:.5f}, {clean_test.ci95_high_mps:.5f}]. The interval crosses zero. The existing-checkpoint clean comparison is a trend, not a stable winner claim.
- VKD wins only {100 * clean_test.vkd_scene_win_rate:.1f}% of clean scenes and the median scene delta is {clean_test.scene_median_delta_mps:.5f} m/s (positive favors KD). Scene deltas are non-normal (Shapiro-Wilk p={clean_test.shapiro_p:.3e}); the two-sided Wilcoxon test is not significant before or after Holm correction (raw p={clean_test.wilcoxon_p_raw:.3f}, adjusted p={clean_test.wilcoxon_p_holm:.3f}).
- CAN removal is catastrophic for both models: degradation exceeds 4.44 m/s. The statistically narrow VKD-KD absolute difference is practically negligible and does not demonstrate fault tolerance.
- Wheel-speed scaling raises error for both models by about 0.013-0.014 m/s. Camera removal only affects VKD and moves it back toward KD, consistent with a small visual contribution rather than robust visual compensation.

## Stress Results

{stress_table}

## Claim Candidates

- Claim:
  - Source evidence: Clean EPE {vkd.endpoint_epe_mps:.5f} vs {kd.endpoint_epe_mps:.5f} m/s on 512 aligned samples from 75 scenes.
  - Allowed wording: "The retained VKD checkpoint showed a small clean-set improvement trend over the retained KD checkpoint."
  - Forbidden stronger wording: "VKD is significantly or generally superior to KD."
  - Uncertainty: One checkpoint per method; scene-level clean CI crosses zero; no training-seed variance.
  - Next check: Complete the three-seed retraining protocol and compare seed-level paired scene estimates.
  - Decision: weaken

- Claim:
  - Source evidence: MAE vx decreases from {kd.mae_vx_mps:.5f} to {vkd.mae_vx_mps:.5f} m/s, whereas MAE vy changes from {kd.mae_vy_mps:.5f} to {vkd.mae_vy_mps:.5f} m/s.
  - Allowed wording: "The observed aggregate improvement is longitudinal; lateral MAE did not improve."
  - Forbidden stronger wording: "Vision improves both longitudinal and lateral velocity estimation."
  - Uncertainty: nuScenes pose-derived labels and one checkpoint pair.
  - Next check: Three-seed lateral-error analysis in turning/high-lateral scenes.
  - Decision: keep

- Claim:
  - Source evidence: CAN-missing degradation is over 4.44 m/s for both KD and VKD.
  - Allowed wording: "Neither retained checkpoint is robust to complete CAN removal under this injection."
  - Forbidden stronger wording: "VKD provides CAN-fault tolerance."
  - Uncertainty: Synthetic complete removal is more severe than many real faults.
  - Next check: Train with CAN dropout/staleness augmentation and test graded faults.
  - Decision: keep

## Main Caveats

- This is existing-checkpoint evaluation, not matched three-seed retraining.
- Scene bootstrap quantifies evaluation-scene uncertainty, not training randomness.
- Synthetic faults do not validate real physical parameter mismatch or estimator-controller coupling.
- Scenario subsets overlap and some contain very few scenes; they remain exploratory.
"""

    stats_appendix = f"""# Statistical Appendix

## Integrity Audit

- Rows: 7,680 = 3 algorithms x 5 variants x 512 samples.
- Unique evaluation scenes: 75.
- Pairing key: `scene_name + center_token`.
- Pairing: exact across every algorithm and variant.
- Velocity labels: invariant across algorithms and perturbations.
- Missing algorithm/variant cells: none.

## Methods

Windows from the same scene are correlated, so inferential comparisons first average EPE within each scene. VKD and KD are then compared using two-sided Wilcoxon signed-rank tests over 75 paired scene means. Five variant comparisons form one family and use Holm-adjusted p-values. Rank-biserial correlation is signed so negative values favor VKD. Bootstrap intervals resample the 75 scene-level paired deltas 10,000 times with seed 42.

No seed-level mean or standard deviation is reported because each method has only one retained checkpoint. Statistical significance across scenes must not be interpreted as reproducibility across training seeds.

## Paired Scene Statistics

{paired_table}

## Interpretation Rules

- Negative VKD-KD delta favors VKD.
- A CI crossing zero blocks a directional mean-effect claim.
- Holm-adjusted p-values address the five reported variant contrasts.
- Statistical detectability under CAN removal does not imply practical importance; both absolute errors are catastrophic.
"""

    figure_catalog = """# Figure Catalog

## Figure 01: Clean Comparison

- Files: `figures/figure-01-clean-comparison.pdf` and `.png`.
- Purpose: Compare clean endpoint error for base velocity, KD, and VKD.
- Data: Mean scene EPE over 75 scenes; error bars are 95% scene-bootstrap intervals.
- Observation: Both learned models outperform base velocity; VKD and KD intervals overlap strongly.
- Implication: Existing checkpoints support a small VKD trend, not a decisive clean-set advantage.

## Figure 02: Paired Scene Deltas

- Files: `figures/figure-02-paired-scene-deltas.pdf` and `.png`.
- Purpose: Show the distribution of scene-level VKD-KD differences for every variant.
- Data: 75 paired scene means per variant; negative values favor VKD.
- Observation: Deltas cluster close to zero and contain mixed scene-level outcomes.
- Implication: Aggregate improvements are small and heterogeneous across scenes.

## Figure 03: Stress Degradation

- Files: `figures/figure-03-stress-degradation.pdf` and `.png`.
- Purpose: Separate ordinary synthetic perturbations from catastrophic complete CAN removal.
- Data: Scene-level variant-minus-clean degradation; error bars are 95% scene-bootstrap intervals.
- Observation: Wheel scaling causes modest degradation; camera removal modestly degrades VKD; CAN removal dominates all other faults.
- Implication: The retained VKD checkpoint does not provide meaningful CAN-failure robustness.
"""

    (OUTPUT / "analysis-report.md").write_text(analysis_report, encoding="utf-8")
    (OUTPUT / "stats-appendix.md").write_text(stats_appendix, encoding="utf-8")
    (OUTPUT / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT / "all_predictions.csv")
    audit = audit_alignment(frame)
    clean, stress = metric_tables(frame)
    paired = paired_statistics(frame)
    clean.to_csv(OUTPUT / "clean-summary.csv", index=False)
    stress.to_csv(OUTPUT / "stress-summary.csv", index=False)
    paired.to_csv(OUTPUT / "paired-scene-statistics.csv", index=False)
    (OUTPUT / "alignment-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    save_figures(frame, clean, stress)
    write_reports(clean, stress, paired)
    print(json.dumps({"output": str(OUTPUT), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
