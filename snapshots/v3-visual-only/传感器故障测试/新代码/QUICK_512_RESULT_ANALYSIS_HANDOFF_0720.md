# nuScenes New-Code Quick-512 Result Analysis Handoff

## 1. Purpose

This package hands off the completed new-code `quick_seed42_20260720_072105` evaluation and its strict local analysis.

The primary valid question is:

> How does the current accelerated seed-42 V2 visual-kinematic checkpoint behave under the four synthetic sensor faults that were actually evaluated on 512 aligned nuScenes samples?

The package also contains existing-checkpoint results for base velocity, legacy KD, legacy VKD, and V2 visual. Those arms are descriptive context. They do not form a fully retrained, generation-matched comparison.

## 2. Deliver These Three Items Together

1. This document: `QUICK_512_RESULT_ANALYSIS_HANDOFF.md`.
2. Raw server results: `quick_512_manual_server/`.
3. Strict local analysis: `quick_512_manual_analysis/`.

Recommended reading order:

1. This handoff.
2. `quick_512_manual_server/quick_report.json`.
3. `quick_512_manual_analysis/alignment-audit.json`.
4. `quick_512_manual_analysis/checkpoint-audit.json`.
5. `quick_512_manual_analysis/fault-path-audit.json`.
6. `quick_512_manual_analysis/analysis-report.md`.
7. `quick_512_manual_analysis/stats-appendix.md`.
8. `quick_512_manual_server/all_predictions.csv` for new analyses.

## 3. Artifact Inventory

### Raw server results

`quick_512_manual_server/` contains:

- `all_predictions.csv`: 12,800 prediction rows, excluding the header.
- `quick_report.json` and `quick_report.md`: server-generated quick summary.
- `split_manifest.json`: the fixed V2 split manifest copied from server `logs/`.
- Five algorithm directories:
  - `base_velocity/`
  - `legacy_kd/`
  - `legacy_vkd/`
  - `v2_visual/`
  - `v2_visual_kinematic/`
- Each algorithm directory contains:
  - `predictions.csv`: 2,560 rows (`512 samples x 5 variants`).
  - `metrics.json`: aggregate metrics by variant.
  - `resolved_config.yaml`: checkpoint-native evaluation configuration.

### Strict local analysis

`quick_512_manual_analysis/` contains:

- `analysis-report.md`: evidence-first conclusions and claim boundaries.
- `stats-appendix.md`: statistical unit, tests, confidence intervals, and effect sizes.
- `figure-catalog.md`: purpose and interpretation of each figure.
- `alignment-audit.json`: key, label, and adapter-metadata audit.
- `checkpoint-audit.json`: V2 checkpoint generation audit.
- `fault-path-audit.json`: camera-missing fallback-path audit.
- `clean-summary.csv`, `stress-summary.csv`, and `paired-scene-statistics.csv`.
- `figures/`: three PDF figures and matching 300-DPI PNG files.
- `reproduce_analysis.py`: reproduces all generated analysis artifacts.

## 4. Provenance

- Dataset: nuScenes `v1.0-trainval` with CAN bus expansion.
- Evaluation partition: `stage1_test`.
- Evaluation seed: 42.
- Runtime Git commit: `df43ed489f73a2d0dda77e42382aaf94e725d96e`.
- Experiment-package Git history: latest relevant commit `361b583`.
- Server run ID: `quick_seed42_20260720_072105`.
- Experimental unit for inference: scene.
- Selected common set: 512 windows from 75 scenes.
- Selection: deterministic scene round-robin over the common token intersection, not random sampling.
- Pairing key: `scene_name + center_token`.
- Split-manifest SHA-256: `d48ae71d98534c72d51d4629a6c1bcf775e1f5896ae5f324e2a262245e49e931`.

Actual checkpoint SHA-256 values:

- Legacy KD: `ef89913d9e7a10725af6a4236b372ae05a1956fc12eb273ebb1b886a52a6b16b`.
- Legacy VKD: `1c9dd173b4b564a285ab643ff69e70d6eee0fb58edb7f25f858442e5845a30fd`.
- V2 visual: `f2c79a59a47afa0cbef84f9faac34282adb56711a0843ee6dde3123d327a8398`.
- V2 visual-kinematic joint: `fa553f5ad87cc6309736431475054a3be88cb3f4293d5f114ce516eebd44e441`.

## 5. Integrity Status

The raw results passed these checks:

- 5 algorithms x 5 variants x 512 samples = 12,800 rows.
- Every algorithm/variant cell contains exactly 512 rows.
- No duplicate `scene_name + center_token` keys within any cell.
- Sample keys are aligned across all cells.
- `target_vx_mps` and `target_vy_mps` are invariant across algorithms and faults.
- All 512 current-joint clean predictions exactly match the same tokens in the earlier 17,091-sample Stage-1 holdout.

Two material integrity limitations were found:

1. The quick run loaded the current accelerated joint checkpoint, but its `v2_visual` arm used the older `checkpoints/velocity_xy_v2_visual/best.pth`. Current visual-to-joint comparisons are quarantined.
2. CAN-derived scenario metadata is not invariant across adapters. V2 yaw rate differs from legacy KD for 160 of 512 identical tokens, with maximum absolute difference `0.47822 rad/s`. Cross-algorithm turning and other CAN-derived subgroup comparisons are invalid.

Do not treat 512 windows as independent statistical replicates. Windows within a scene are correlated.

## 6. Variants Actually Evaluated

- `clean`: no injected fault.
- `can_missing`: complete CAN removal.
- `yaw_bias_pos`: `+0.02 rad/s` yaw-rate bias.
- `wheel_scale_low`: wheel-speed scale `0.9`.
- `camera_missing`: camera input removed.

The code implements but this run did not evaluate:

- `can_stale_500ms`
- `can_dropout_250ms`
- `yaw_bias_neg`
- `wheel_scale_high`
- `camera_frozen`
- V2 calibration perturbations

## 7. Current V2 Joint Results

Clean metrics on the deterministic 512-window subset:

| Metric | Value |
| --- | ---: |
| EPE | 0.88567 m/s |
| MAE vx | 0.57877 m/s |
| MAE vy | 0.42965 m/s |
| P95 EPE | 3.75225 m/s |

The subset clean EPE must not replace the full Stage-1 result of `0.64464 m/s` over 17,091 samples.

Fault results:

| Variant | EPE | Degradation | Relative degradation |
| --- | ---: | ---: | ---: |
| can_missing | 1.37482 | +0.48915 | +55.23% |
| yaw_bias_pos | 0.88800 | +0.00233 | +0.26% |
| wheel_scale_low | 0.88567 | 0.00000 | 0.00% |
| camera_missing | 8.19996 | +7.31428 | +825.84% |

Paired scene-level results:

| Variant | Mean scene degradation | Bootstrap 95% CI | Scene worsening rate |
| --- | ---: | ---: | ---: |
| can_missing | +0.48762 | [0.33367, 0.65216] | 85.3% |
| yaw_bias_pos | +0.00228 | [0.00084, 0.00403] | 68.0% |
| wheel_scale_low | 0.00000 | [0.00000, 0.00000] | 0.0% |
| camera_missing | +7.28556 | [6.55545, 8.05051] | 100.0% |

## 8. Critical Fault-Path Finding

Under `camera_missing`:

- Geometry is marked invalid for 512/512 samples.
- `fallback_used` is marked for 512/512 samples.
- Mean available `base_vx_mps` is `5.15555 m/s`.
- Mean final `pred_vx_mps` is `-2.75285 m/s`.
- Mean EPE is `8.19996 m/s`.

The failure detector and fallback flag activate, but the final output does not use the available deterministic base velocity. This is a functional fallback-path defect, not merely weak learned performance.

## 9. Supported and Unsupported Claims

Supported wording:

- The current joint checkpoint remains numerically finite under complete CAN removal but loses substantial accuracy.
- Complete camera removal causes catastrophic error despite geometry-invalid and fallback flags being set.
- The existing fallback flag is inconsistent with the final prediction path under camera removal.
- The tested positive yaw bias produces a very small measured change.
- The wheel-scale-low perturbation produces no measured change because the existing scale path is saturated or clipped.

Do not claim:

- The current model is robust to missing CAN or missing camera.
- Camera failure safely falls back to wheel speed.
- Near-zero yaw sensitivity proves effective yaw-bias robustness.
- Near-zero wheel-scale sensitivity proves wheel-speed robustness.
- Current V2 joint is fairly compared against the V2 visual arm in this quick run.
- Cross-algorithm turning or high-excitation subgroup results are valid.
- One seed establishes training stability or statistical significance across training randomness.

## 10. Statistical Contract

- Primary metric: endpoint velocity error, lower is better.
- Primary valid comparisons: current V2 joint fault minus current V2 joint clean.
- Primary inferential unit: scene, not window.
- Average EPE within each scene before computing paired fault-minus-clean differences.
- Use two-sided Wilcoxon signed-rank tests over 75 paired scene means.
- Treat the four evaluated fault contrasts as one Holm-adjusted family.
- Bootstrap 95% CIs resample the 75 paired scene differences 10,000 times using seed 42.
- Report practical magnitude with p-values; a tiny p-value does not make a `0.002 m/s` yaw effect operationally important.
- Scene bootstrap measures evaluation-scene uncertainty conditional on seed 42, not training-seed uncertainty.

## 11. Evidence Still Missing

- Correct accelerated V2 visual checkpoint in the quick comparison.
- Aligned CAN-derived scenario metadata across legacy and V2 adapters.
- CAN staleness and burst-dropout results.
- Negative yaw bias, positive wheel-scale, and frozen-camera results.
- A repaired and verified deterministic camera-failure fallback.
- Multiple training seeds.
- Real sensor-fault recordings.
- Vehicle center-of-gravity velocity validation.
- Closed-loop estimator-controller testing.

## 12. Reproducing the Local Analysis

The reproduction script expects `quick_512_manual_server/` and `quick_512_manual_analysis/` to be siblings under `new code/`.

```powershell
Set-Location '<repository>/sensor fault testing/new code'
& 'D:\Miniconda3\envs\vision_est\python.exe' `
  quick_512_manual_analysis\reproduce_analysis.py
```

Packages used:

- pandas 2.3.3
- numpy 2.2.6
- scipy 1.15.3
- matplotlib 3.10.8
- PyYAML

The script overwrites only generated files inside `quick_512_manual_analysis/`.

## 13. Start Prompt for the Next Codex

> Read `QUICK_512_RESULT_ANALYSIS_HANDOFF.md` first. Audit the raw artifacts, alignment audit, checkpoint audit, and fault-path audit before making claims. Treat current V2 joint fault-minus-clean at scene level as the primary valid analysis. Quarantine the V2 visual comparison because it used an old checkpoint. Do not use cross-algorithm CAN-derived subgroup comparisons because adapter metadata differs. Focus next on fixing the camera fallback output path, removing wheel-scale saturation masking, and aligning CAN metadata. Do not treat 512 windows as independent samples and do not infer multi-seed stability from seed 42.
