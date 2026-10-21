from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
INPUT = ROOT / "quick_512_manual_server"
OUTPUT = SCRIPT_DIR
FIGURES = OUTPUT / "figures"

ALGORITHMS = (
    "base_velocity",
    "legacy_kd",
    "legacy_vkd",
    "v2_visual",
    "v2_visual_kinematic",
)
VARIANTS = ("clean", "can_missing", "yaw_bias_pos", "wheel_scale_low", "camera_missing")
PRIMARY = "v2_visual_kinematic"
COLORS = {
    "base_velocity": "#777777",
    "legacy_kd": "#0072B2",
    "legacy_vkd": "#D55E00",
    "v2_visual": "#009E73",
    "v2_visual_kinematic": "#CC79A7",
}


def bootstrap_ci(values: np.ndarray, seed: int = 42, samples: int = 10_000) -> tuple[float, float]:
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


def rank_biserial(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / ranks.sum())


def scene_errors(frame: pd.DataFrame, algorithm: str, variant: str) -> pd.Series:
    subset = frame[(frame.algorithm == algorithm) & (frame.variant == variant)]
    return subset.groupby("scene_name")["endpoint_error_mps"].mean().sort_index()


def audit_alignment(frame: pd.DataFrame) -> dict[str, object]:
    key_columns = ["scene_name", "center_token"]
    metadata_fields = [
        "yaw_rate_radps",
        "wheel_relative_std",
        "longitudinal_accel_mps2",
        "lateral_accel_mps2",
        "steering_feedback",
        "throttle",
        "brake",
    ]
    reference_keys = None
    reference_labels = None
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
            if reference_keys is None:
                reference_keys = keys
                reference_labels = labels
            else:
                if not keys.equals(reference_keys):
                    raise AssertionError(f"{label} sample keys are not aligned")
                if not np.allclose(labels, reference_labels, rtol=0.0, atol=1e-9):
                    raise AssertionError(f"{label} velocity labels differ")

    clean = {
        algorithm: frame[(frame.algorithm == algorithm) & (frame.variant == "clean")]
        .sort_values(key_columns)
        .reset_index(drop=True)
        for algorithm in ALGORITHMS
    }
    metadata_differences: dict[str, dict[str, float | int]] = {}
    reference = clean["legacy_kd"]
    for algorithm, subset in clean.items():
        if algorithm == "legacy_kd":
            continue
        for field in metadata_fields:
            left = pd.to_numeric(reference[field], errors="coerce").fillna(0.0).to_numpy()
            right = pd.to_numeric(subset[field], errors="coerce").fillna(0.0).to_numpy()
            delta = np.abs(right - left)
            metadata_differences[f"{algorithm}:{field}"] = {
                "different_rows": int((delta > 1e-6).sum()),
                "max_abs_difference": float(delta.max()),
            }
    metadata_invariant = all(item["different_rows"] == 0 for item in metadata_differences.values())
    return {
        "valid_sample_and_label_alignment": True,
        "total_rows": int(len(frame)),
        "samples_per_algorithm_variant": 512,
        "scene_count": int(frame.scene_name.nunique()),
        "algorithm_variant_counts": counts,
        "sample_key": "scene_name + center_token",
        "labels_invariant": True,
        "scenario_metadata_invariant": metadata_invariant,
        "scenario_metadata_differences_vs_legacy_kd": metadata_differences,
        "conclusion": (
            "Sample keys and labels align, but scenario metadata does not align across adapters; "
            "cross-algorithm subgroup comparisons are invalid."
            if not metadata_invariant
            else "Sample keys, labels, and scenario metadata align."
        ),
    }


def fault_path_audit(frame: pd.DataFrame) -> dict[str, object]:
    camera = frame[(frame.algorithm == PRIMARY) & (frame.variant == "camera_missing")]
    clean = frame[(frame.algorithm == PRIMARY) & (frame.variant == "clean")]
    fallback_count = int((pd.to_numeric(camera.fallback_used) > 0.5).sum())
    invalid_count = int((pd.to_numeric(camera.geometry_valid) < 0.5).sum())
    return {
        "camera_missing_rows": int(len(camera)),
        "geometry_invalid_rows": invalid_count,
        "fallback_flag_rows": fallback_count,
        "camera_missing_mean_base_vx_mps": float(camera.base_vx_mps.mean()),
        "camera_missing_mean_pred_vx_mps": float(camera.pred_vx_mps.mean()),
        "camera_missing_epe_mps": float(camera.endpoint_error_mps.mean()),
        "clean_epe_mps": float(clean.endpoint_error_mps.mean()),
        "fallback_output_consistent_with_base_velocity": bool(
            np.allclose(camera.pred_vx_mps, camera.base_vx_mps, atol=1e-4, rtol=0.0)
        ),
        "conclusion": (
            "Camera removal marks every sample as geometry-invalid and fallback-used, but the final "
            "prediction does not use the available base velocity. The fallback flag and output path "
            "are inconsistent."
        ),
    }


def checkpoint_audit() -> dict[str, object]:
    visual = yaml.safe_load((INPUT / "v2_visual" / "resolved_config.yaml").read_text(encoding="utf-8"))
    joint = yaml.safe_load(
        (INPUT / "v2_visual_kinematic" / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    visual_save = str(visual.get("train", {}).get("save_dir", ""))
    joint_save = str(joint.get("train", {}).get("save_dir", ""))
    visual_is_accelerated = "accelerated_20260719_143555/visual_seed_42" in visual_save
    joint_is_accelerated = "accelerated_20260719_143555/joint_seed_42" in joint_save
    return {
        "v2_visual_saved_config_save_dir": visual_save,
        "v2_joint_saved_config_save_dir": joint_save,
        "v2_visual_is_current_accelerated_checkpoint": visual_is_accelerated,
        "v2_joint_is_current_accelerated_checkpoint": joint_is_accelerated,
        "current_visual_to_joint_comparison_valid": visual_is_accelerated and joint_is_accelerated,
        "conclusion": (
            "The joint arm is the current accelerated checkpoint, but the visual arm is an older "
            "checkpoint. Current visual-to-joint claims are quarantined."
        ),
    }


def clean_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        subset = frame[(frame.algorithm == algorithm) & (frame.variant == "clean")]
        scene = scene_errors(frame, algorithm, "clean").to_numpy()
        low, high = bootstrap_ci(scene)
        rows.append(
            {
                "algorithm": algorithm,
                "samples": len(subset),
                "scenes": subset.scene_name.nunique(),
                "endpoint_epe_mps": subset.endpoint_error_mps.mean(),
                "scene_mean_epe_mps": scene.mean(),
                "scene_ci95_low_mps": low,
                "scene_ci95_high_mps": high,
                "mae_vx_mps": subset.error_vx_mps.abs().mean(),
                "mae_vy_mps": subset.error_vy_mps.abs().mean(),
                "p95_epe_mps": subset.endpoint_error_mps.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def stress_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        clean_scene = scene_errors(frame, algorithm, "clean")
        clean_sample = frame[(frame.algorithm == algorithm) & (frame.variant == "clean")]
        clean_epe = clean_sample.endpoint_error_mps.mean()
        for variant in VARIANTS[1:]:
            variant_scene = scene_errors(frame, algorithm, variant)
            degradation = (variant_scene - clean_scene).to_numpy()
            low, high = bootstrap_ci(degradation)
            subset = frame[(frame.algorithm == algorithm) & (frame.variant == variant)]
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "endpoint_epe_mps": subset.endpoint_error_mps.mean(),
                    "degradation_mps": subset.endpoint_error_mps.mean() - clean_epe,
                    "relative_degradation_percent": 100.0
                    * (subset.endpoint_error_mps.mean() - clean_epe)
                    / clean_epe,
                    "scene_mean_degradation_mps": degradation.mean(),
                    "scene_ci95_low_mps": low,
                    "scene_ci95_high_mps": high,
                }
            )
    return pd.DataFrame(rows)


def paired_fault_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    clean = scene_errors(frame, PRIMARY, "clean")
    rows = []
    p_values = []
    for variant in VARIANTS[1:]:
        variant_scene = scene_errors(frame, PRIMARY, variant)
        delta = (variant_scene - clean).to_numpy()
        low, high = bootstrap_ci(delta)
        if np.allclose(delta, 0.0, rtol=0.0, atol=1e-12):
            shapiro_w, shapiro_p = 1.0, 1.0
            wilcoxon_w, wilcoxon_p = 0.0, 1.0
        else:
            shapiro = stats.shapiro(delta)
            shapiro_w, shapiro_p = float(shapiro.statistic), float(shapiro.pvalue)
            wilcoxon = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            wilcoxon_w, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        p_values.append(wilcoxon_p)
        rows.append(
            {
                "comparison": f"{PRIMARY}:{variant}-clean",
                "variant": variant,
                "scene_count": len(delta),
                "scene_mean_degradation_mps": delta.mean(),
                "scene_median_degradation_mps": np.median(delta),
                "ci95_low_mps": low,
                "ci95_high_mps": high,
                "scene_worsening_rate": float((delta > 0).mean()),
                "shapiro_w": shapiro_w,
                "shapiro_p": shapiro_p,
                "wilcoxon_w": wilcoxon_w,
                "wilcoxon_p_raw": wilcoxon_p,
                "rank_biserial": rank_biserial(delta),
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = value
    return pd.DataFrame(rows)


def save_figures(frame: pd.DataFrame, clean: pd.DataFrame, stress: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8})

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(clean))
    values = clean.scene_mean_epe_mps.to_numpy()
    lower = values - clean.scene_ci95_low_mps.to_numpy()
    upper = clean.scene_ci95_high_mps.to_numpy() - values
    ax.bar(x, values, color=[COLORS[name] for name in clean.algorithm], width=0.68)
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=4)
    ax.set_xticks(x, ["Base", "Legacy KD", "Legacy VKD", "V2 visual*", "V2 joint"])
    ax.set_ylabel("Scene-level mean EPE (m/s)")
    ax.grid(axis="y", alpha=0.25)
    ax.text(3, values[3], "old checkpoint", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-01-clean-comparison.pdf")
    fig.savefig(FIGURES / "figure-01-clean-comparison.png", dpi=300)
    plt.close(fig)

    selected = stress[stress.algorithm == PRIMARY].set_index("variant")
    groups = [["can_missing", "yaw_bias_pos", "wheel_scale_low"], ["camera_missing"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [3, 1]})
    for ax, variants in zip(axes, groups):
        part = selected.loc[variants]
        x = np.arange(len(part))
        values = part.scene_mean_degradation_mps.to_numpy()
        lower = values - part.scene_ci95_low_mps.to_numpy()
        upper = part.scene_ci95_high_mps.to_numpy() - values
        ax.bar(x, values, color="#CC79A7", width=0.65)
        ax.errorbar(
            x,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color="black",
            capsize=4,
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [name.replace("_", "\n") for name in part.index])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("V2 joint scene degradation vs clean (m/s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-02-joint-fault-degradation.pdf")
    fig.savefig(FIGURES / "figure-02-joint-fault-degradation.png", dpi=300)
    plt.close(fig)

    clean_scene = scene_errors(frame, PRIMARY, "clean")
    deltas = [
        (scene_errors(frame, PRIMARY, variant) - clean_scene).to_numpy()
        for variant in VARIANTS[1:]
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plot = ax.boxplot(
        deltas,
        tick_labels=[name.replace("_", "\n") for name in VARIANTS[1:]],
        patch_artist=True,
        showfliers=False,
    )
    for box in plot["boxes"]:
        box.set_facecolor("#56B4E9")
        box.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Paired scene EPE degradation (m/s)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-03-joint-scene-degradation.pdf")
    fig.savefig(FIGURES / "figure-03-joint-scene-degradation.png", dpi=300)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(
                    f"{value:.3e}"
                    if value != 0 and abs(value) < 10 ** -digits
                    else f"{value:.{digits}f}"
                )
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_reports(
    clean: pd.DataFrame,
    stress: pd.DataFrame,
    paired: pd.DataFrame,
    audit: dict[str, object],
    checkpoints: dict[str, object],
    fault_paths: dict[str, object],
) -> None:
    joint = clean.set_index("algorithm").loc[PRIMARY]
    joint_stress = stress[stress.algorithm == PRIMARY]
    stress_table = markdown_table(
        joint_stress,
        ["variant", "endpoint_epe_mps", "degradation_mps", "relative_degradation_percent"],
    )
    clean_table = markdown_table(
        clean,
        ["algorithm", "samples", "scenes", "endpoint_epe_mps", "mae_vx_mps", "mae_vy_mps", "p95_epe_mps"],
    )
    paired_table = markdown_table(
        paired,
        [
            "variant",
            "scene_count",
            "scene_mean_degradation_mps",
            "ci95_low_mps",
            "ci95_high_mps",
            "scene_worsening_rate",
            "wilcoxon_p_raw",
            "wilcoxon_p_holm",
            "rank_biserial",
        ],
    )
    camera = joint_stress.set_index("variant").loc["camera_missing"]
    can = joint_stress.set_index("variant").loc["can_missing"]
    yaw = joint_stress.set_index("variant").loc["yaw_bias_pos"]
    wheel = joint_stress.set_index("variant").loc["wheel_scale_low"]

    analysis_report = f"""# New-Code quick_512 Strict Analysis

## Analysis Question

How does the current accelerated seed-42 V2 visual-kinematic checkpoint behave under the four synthetic sensor perturbations already evaluated on 512 exactly aligned nuScenes windows?

Primary metric: endpoint velocity error (EPE, lower is better). Statistical unit: scene. The 512 windows come from 75 scenes and are not independent replicates.

## Integrity Decision

- Sample keys and velocity labels align across all 25 algorithm/variant cells.
- Current V2 joint checkpoint: valid accelerated checkpoint.
- V2 visual checkpoint: older non-accelerated checkpoint; current visual-to-joint claims are quarantined.
- Scenario metadata invariant across adapters: **{audit['scenario_metadata_invariant']}**. Cross-algorithm turning and other CAN-derived subgroup comparisons are invalid.

## Clean Descriptive Results

{clean_table}

The V2 joint clean EPE on this deterministic 512-window subset is {joint.endpoint_epe_mps:.5f} m/s. This subset value must not replace the full 17,091-sample Stage-1 result of 0.64464 m/s.

## V2 Joint Fault Results

{stress_table}

- Complete camera removal is catastrophic: EPE increases by {camera.degradation_mps:.5f} m/s ({camera.relative_degradation_percent:.1f}%). All {fault_paths['fallback_flag_rows']} samples are marked `fallback_used`, but mean predicted vx is {fault_paths['camera_missing_mean_pred_vx_mps']:.3f} m/s instead of the available base-velocity mean {fault_paths['camera_missing_mean_base_vx_mps']:.3f} m/s. The fallback flag and final output path are inconsistent.
- Complete CAN removal remains finite but raises EPE by {can.degradation_mps:.5f} m/s ({can.relative_degradation_percent:.1f}%). Finite output is not acceptable fault tolerance.
- Positive yaw bias changes EPE by only {yaw.degradation_mps:.5f} m/s ({yaw.relative_degradation_percent:.2f}%). Combined with the earlier no-yaw ablation, this is evidence that the yaw path has little measured effect, not proof of robust compensation.
- A 0.9 wheel-speed scale causes {wheel.degradation_mps:.5f} m/s measured change. Because most full-holdout wheel scales already saturate at the 0.85 lower bound, this perturbation is masked and cannot establish wheel-speed robustness.

## Claim Candidates

- Claim:
  - Source evidence: Current joint camera-missing EPE {camera.endpoint_epe_mps:.5f} vs clean {joint.endpoint_epe_mps:.5f} m/s.
  - Allowed wording: "The current V2 joint checkpoint fails severely under complete synthetic camera removal."
  - Forbidden stronger wording: "The estimator safely falls back when vision fails."
  - Uncertainty: Synthetic blank-camera injection; no real sensor-disconnect recording.
  - Next check: Fix the existing fallback output path so the fallback flag actually selects the deterministic base velocity.
  - Decision: keep

- Claim:
  - Source evidence: Current joint CAN-missing degradation {can.degradation_mps:.5f} m/s.
  - Allowed wording: "The model remains numerically finite under complete CAN removal but loses substantial accuracy."
  - Forbidden stronger wording: "The model is robust to missing CAN."
  - Uncertainty: Only complete removal was run; staleness and burst dropout were not evaluated.
  - Next check: Run graded missingness after the test protocol is corrected.
  - Decision: keep

- Claim:
  - Source evidence: yaw-bias and wheel-scale-low aggregate deltas are near zero.
  - Allowed wording: "These two injections produced little measured change in this run."
  - Forbidden stronger wording: "The estimator is robust to yaw and wheel-speed faults."
  - Uncertainty: yaw pathway has weak measured contribution and wheel scaling is saturated.
  - Next check: Remove saturation masking and verify perturbations change the intended intermediate tensors.
  - Decision: weaken

## Main Caveats

- One seed and one retained checkpoint per algorithm; no training-seed uncertainty.
- Deterministic 512-window screening subset, not the full holdout.
- V2 visual arm is the wrong checkpoint for the current model generation.
- Adapter metadata mismatch invalidates cross-algorithm subgroup comparisons.
- CAN stale/dropout, negative yaw bias, wheel-scale-high, and frozen-camera variants were implemented but not run.
- Synthetic open-loop perturbations do not validate physical parameter mismatch or closed-loop stability.
"""

    stats_appendix = f"""# Statistical Appendix

## Integrity Audit

- Rows: {audit['total_rows']:,} = 5 algorithms x 5 variants x 512 samples.
- Unique scenes: {audit['scene_count']}.
- Pairing key: `scene_name + center_token`.
- Sample keys and target velocity labels: exactly aligned.
- Scenario metadata aligned across adapters: {audit['scenario_metadata_invariant']}.
- Current visual-to-joint checkpoint comparison valid: {checkpoints['current_visual_to_joint_comparison_valid']}.
- Camera-missing geometry-invalid rows: {fault_paths['geometry_invalid_rows']}/{fault_paths['camera_missing_rows']}.
- Camera-missing fallback-flag rows: {fault_paths['fallback_flag_rows']}/{fault_paths['camera_missing_rows']}.
- Fallback prediction consistent with base velocity: {fault_paths['fallback_output_consistent_with_base_velocity']}.

## Methods

For each V2 joint fault, EPE is first averaged within each scene. Fault-minus-clean differences are then computed over 75 paired scene means. Two-sided Wilcoxon signed-rank tests use one family of four fault contrasts with Holm adjustment. Rank-biserial correlation is positive when the fault worsens EPE. Bootstrap intervals resample paired scene differences 10,000 times with seed 42.

The scene bootstrap quantifies scene-sampling uncertainty conditional on seed 42. It does not quantify training randomness. Tests on the 512 windows as independent observations are prohibited.

## V2 Joint Paired Fault Statistics

{paired_table}

## Quarantined Statistics

- Current V2 visual versus joint: wrong visual checkpoint.
- Cross-algorithm turning/high-steering/high-acceleration groups: adapter CAN metadata differs for identical tokens.
- Wheel-scale-low robustness: perturbation is masked by the model's lower scale saturation.
"""

    figure_catalog = """# Figure Catalog

## Figure 01: Clean Existing-Checkpoint Comparison

- Files: `figures/figure-01-clean-comparison.pdf` and `.png`.
- Purpose: Describe clean EPE for all checkpoints present in the server run.
- Data: Mean scene EPE over 75 scenes; bars show scene-bootstrap 95% intervals.
- Observation: The old V2 visual checkpoint and current V2 joint checkpoint are not a valid generation-matched pair.
- Implication: Use this figure descriptively; do not claim a fair architecture ranking.

## Figure 02: Current Joint Fault Degradation

- Files: `figures/figure-02-joint-fault-degradation.pdf` and `.png`.
- Purpose: Compare each synthetic fault against the current joint clean baseline.
- Data: Paired scene fault-minus-clean EPE with 95% scene-bootstrap intervals.
- Observation: Camera removal dominates; CAN removal is also materially harmful; yaw and wheel-low changes are near zero.
- Implication: Camera fallback is the immediate design priority. Near-zero yaw/wheel effects require pathway audits, not robustness claims.

## Figure 03: Scene-Level Joint Degradation

- Files: `figures/figure-03-joint-scene-degradation.pdf` and `.png`.
- Purpose: Show heterogeneity of fault effects across 75 scenes.
- Data: Paired scene fault-minus-clean EPE distributions.
- Observation: Camera and CAN failures worsen many scenes; mild perturbations cluster near zero.
- Implication: Aggregate camera/CAN degradation is not caused by a single isolated scene.
"""

    (OUTPUT / "analysis-report.md").write_text(analysis_report, encoding="utf-8")
    (OUTPUT / "stats-appendix.md").write_text(stats_appendix, encoding="utf-8")
    (OUTPUT / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT / "all_predictions.csv")
    audit = audit_alignment(frame)
    checkpoints = checkpoint_audit()
    fault_paths = fault_path_audit(frame)
    clean = clean_summary(frame)
    stress = stress_summary(frame)
    paired = paired_fault_statistics(frame)

    clean.to_csv(OUTPUT / "clean-summary.csv", index=False)
    stress.to_csv(OUTPUT / "stress-summary.csv", index=False)
    paired.to_csv(OUTPUT / "paired-scene-statistics.csv", index=False)
    (OUTPUT / "alignment-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUTPUT / "checkpoint-audit.json").write_text(
        json.dumps(checkpoints, indent=2), encoding="utf-8"
    )
    (OUTPUT / "fault-path-audit.json").write_text(
        json.dumps(fault_paths, indent=2), encoding="utf-8"
    )
    save_figures(frame, clean, stress)
    write_reports(clean, stress, paired, audit, checkpoints, fault_paths)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "alignment": audit,
                "checkpoints": checkpoints,
                "fault_paths": fault_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
