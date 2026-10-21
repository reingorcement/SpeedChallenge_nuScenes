# nuScenes KD/VKD Quick-512 Result Analysis Handoff

## 1. Purpose

This package hands off the completed `quick_512_manual` evaluation for continued analysis by a fresh Codex instance with no access to the original conversation.

The comparison is:

- `base_velocity`: `[base_vx, 0]` reference.
- `legacy_kd`: kinematic + dynamics-proxy branches.
- `legacy_vkd`: visual + kinematic + dynamics-proxy branches.

Primary question: whether the retained VKD checkpoint improves causal open-loop longitudinal/lateral velocity estimation and synthetic sensor-stress behavior relative to the retained KD checkpoint on aligned nuScenes samples.

## 2. Deliver These Three Items Together

1. This document: `QUICK_512_RESULT_ANALYSIS_HANDOFF.md`.
2. Raw server results: `quick_512_manual_server/`.
3. Existing strict analysis: `quick_512_manual_analysis/`.

The two directories are sufficient only when accompanied by this handoff. The raw directory now includes the split manifest that was originally generated under the server `logs/` directory.

Recommended reading order:

1. This handoff.
2. `quick_512_manual_server/quick_report.json`.
3. `quick_512_manual_analysis/alignment-audit.json`.
4. `quick_512_manual_analysis/analysis-report.md`.
5. `quick_512_manual_analysis/stats-appendix.md`.
6. `quick_512_manual_server/all_predictions.csv` for new analysis.

## 3. Artifact Inventory

### Raw results

`quick_512_manual_server/` contains:

- `all_predictions.csv`: 7,680 prediction rows, excluding the header.
- `quick_report.json` and `quick_report.md`: official quick-evaluation summary.
- `split_manifest.json`: fixed scene-level split manifest, seed 42.
- `base_velocity/`, `legacy_kd/`, `legacy_vkd/`:
  - `predictions.csv`: 2,560 rows per algorithm (`512 samples x 5 variants`).
  - `metrics.json`: aggregate metrics by variant.
  - `resolved_config.yaml`: checkpoint-native evaluation configuration.

`split_manifest.json` also contains per-scene `location`, `lighting`, and `weather` metadata, so domain analysis does not require a live nuScenes installation. `all_predictions.csv` includes `yaw_rate_radps`, `steering_feedback`, `longitudinal_accel_mps2`, `lateral_accel_mps2`, `throttle`, `brake`, and `wheel_relative_std`; it also contains branch/gate fields including `gate_vision`, `gate_kinematic`, `gate_dynamics`, `visual_vx_mps`, `visual_vy_mps`, and kinematic-gate/correction columns. Empty values are expected when an algorithm does not have the corresponding branch.

### Existing analysis

`quick_512_manual_analysis/` contains:

- `analysis-report.md`: evidence-first conclusions and claim boundaries.
- `stats-appendix.md`: statistical unit, tests, confidence intervals, and effect sizes.
- `figure-catalog.md`: purpose and interpretation of each figure.
- `alignment-audit.json`: sample-key and label-integrity audit.
- `clean-summary.csv`, `stress-summary.csv`, `paired-scene-statistics.csv`: exact numeric summaries.
- `figures/`: PDF and 300-DPI PNG figures.
- `reproduce_analysis.py`: reproduces the audit, statistics, reports, and figures.

## 4. Provenance

- Dataset: nuScenes `v1.0-trainval` with CAN bus expansion.
- Evaluation partition: `stage1_test`.
- Split seed: 42.
- Experimental unit: scene.
- Split counts:
  - `train_core`: 582 scenes.
  - `dev`: 103 scenes.
  - `stage1_test`: 75 scenes.
  - `final_system_test`: 75 sealed scenes.
  - 15 scenes excluded because CAN data was unavailable.
- Selected common evaluation set: 512 windows from all 75 `stage1_test` scenes.
- Window selection is deterministic, not random: intersect algorithm key sets, preserve reference-dataset order within each scene, sort scene names, then take one remaining window per scene in round-robin passes until 512 windows are selected.
- Pairing key: `scene_name + center_token`.
- Runtime Git commit: `75ae7e706019ea5481f8f50219934856760fd293`.
- Legacy reference commit recorded in configs: `7de2aad`.
- KD checkpoint SHA-256: `ef89913d9e7a10725af6a4236b372ae05a1956fc12eb273ebb1b886a52a6b16b`.
- VKD checkpoint SHA-256: `1c9dd173b4b564a285ab643ff69e70d6eee0fb58edb7f25f858442e5845a30fd`.
- Split-manifest internal checksum: `3409681603aab0fb0e5ca7b60394d273220432069b0a8d86e41734a560435736`.

Checkpoint-native input configuration:

- Sequence length: 13.
- Image input: `CAM_FRONT`, 64 x 64 grayscale.
- Chassis window: 1.0 s at 20 Hz.
- Online mode: enabled; causal current frame is the sequence end.
- Base velocity source: `zero_lateral`.
- KD disables the vision branch; VKD enables it.
- Both use kinematic and dynamics-proxy branches.

## 5. Integrity Status

The downloaded raw results passed the following checks:

- 3 algorithms x 5 variants x 512 samples = 7,680 rows.
- Every algorithm/variant cell contains exactly 512 rows.
- No duplicate `scene_name + center_token` keys within a cell.
- Key sets are exactly aligned across every algorithm and variant.
- `target_vx_mps` and `target_vy_mps` are invariant across algorithms and fault injections.
- No missing algorithm/variant combinations.

Do not repeat significance tests at the 512-window level. Windows within a scene are correlated and are not independent experimental replicates.

## 6. Variants

- `clean`: no injected perturbation.
- `can_missing`: complete CAN removal.
- `yaw_bias_pos`: positive yaw-rate bias.
- `wheel_scale_low`: low wheel-speed scale.
- `camera_missing`: camera input removed.

These are synthetic stress tests. They are not evidence of real vehicle parameter mismatch or closed-loop estimator-controller stability.

## 7. Current Numeric Results

Clean sample-weighted metrics:

| Algorithm | EPE (m/s) | MAE vx (m/s) | MAE vy (m/s) | P95 EPE (m/s) |
| --- | ---: | ---: | ---: | ---: |
| base_velocity | 0.15208 | 0.14243 | 0.03549 | 0.48161 |
| legacy_kd | 0.07958 | 0.07488 | 0.01493 | 0.22693 |
| legacy_vkd | 0.07759 | 0.07287 | 0.01525 | 0.22606 |

Observed clean effects:

- VKD has 2.50% lower mean EPE than KD.
- VKD has 2.69% lower MAE vx than KD.
- VKD has 2.15% higher MAE vy than KD; lateral MAE does not improve.
- VKD wins only 49.3% of clean scenes.
- Scene-level mean VKD-KD EPE delta: `-0.00193 m/s`.
- Scene-bootstrap 95% CI: `[-0.00438, 0.00062] m/s`; it crosses zero.
- Scene deltas are non-normal: Shapiro-Wilk `p=4.33e-06`.
- Wilcoxon signed-rank: raw `p=0.136`, Holm-adjusted `p=0.448`.

Stress summary:

| Variant | KD EPE (m/s) | VKD EPE (m/s) | KD degradation | VKD degradation |
| --- | ---: | ---: | ---: | ---: |
| can_missing | 4.52313 | 4.52157 | 4.44355 | 4.44398 |
| yaw_bias_pos | 0.07981 | 0.07751 | 0.00023 | -0.00008 |
| wheel_scale_low | 0.09344 | 0.09089 | 0.01386 | 0.01330 |
| camera_missing | 0.07958 | 0.07918 | 0.00000 | 0.00159 |

Important interpretation: CAN removal produces a statistically consistent but practically negligible VKD-KD difference while both models fail catastrophically. Do not describe this as meaningful fault tolerance.

## 8. Supported and Unsupported Claims

Supported wording:

- The retained VKD checkpoint shows a small clean-set EPE improvement trend over the retained KD checkpoint.
- The observed aggregate gain is longitudinal; lateral MAE does not improve.
- Neither retained checkpoint is robust to complete CAN removal under this injection.
- Wheel-speed scaling degrades both retained checkpoints.
- Camera removal moves VKD back toward KD, indicating a small visual contribution in this checkpoint pair.

Do not claim:

- VKD is statistically or generally superior to KD.
- Vision improves both longitudinal and lateral velocity estimation.
- VKD provides effective CAN-fault tolerance.
- The experiment validates real physical parameter mismatch.
- The experiment validates cross-vehicle dynamics generalization.
- The experiment validates estimator-controller closed-loop coupling.

## 9. Statistical Contract for Continued Analysis

- Primary metric: endpoint velocity error, lower is better.
- Per-window EPE is `sqrt((pred_vx - target_vx)^2 + (pred_vy - target_vy)^2)` in m/s.
- Primary comparison: `legacy_vkd - legacy_kd`.
- Primary inferential unit: scene, not window.
- Use paired scene-level statistics because all methods share exact samples.
- For the delivered strict analysis, first average window EPE within each scene and algorithm/variant. Then compute 75 paired deltas as `scene_mean_EPE_VKD - scene_mean_EPE_KD`. Each scene has equal weight regardless of its selected-window count.
- Use non-parametric paired tests when scene-delta normality fails.
- The delivered Holm family contains exactly five VKD-KD contrasts: `clean`, `can_missing`, `yaw_bias_pos`, `wheel_scale_low`, and `camera_missing`.
- Delivered 95% CIs resample the 75 paired scene deltas with replacement 10,000 times using seed 42 and take the 2.5th/97.5th percentiles of bootstrap mean deltas.
- Always report effect size, confidence interval, scene count, and practical magnitude with p-values.
- Scene bootstrap measures evaluation-scene uncertainty only; it does not measure training-seed variance.
- Scenario groups overlap and some contain very few scenes. Treat them as exploratory.

## 10. High-Value Next Analyses

1. Join `all_predictions.csv` with `split_manifest.json` scene metadata and compare location, lighting, and weather domains at scene level.
2. Analyze longitudinal and lateral errors separately, especially in turning, high-lateral, and high-lateral-acceleration subsets.
3. Use `gate_vision`, `gate_kinematic`, `gate_dynamics`, and branch prediction columns to test when vision helps or hurts. Missing gate values for algorithms without that branch are expected.
4. Produce scene-ranked VKD improvements and regressions, then inspect bad cases rather than reporting only aggregate means.
5. Quantify whether visual benefit changes with yaw rate, steering, speed, acceleration, lighting, and weather using scene-aware uncertainty.
6. Audit tail behavior (`P90/P95/P99`) because mean EPE gains are small.
7. Treat `can_missing` separately from mild perturbations because its scale dominates all other conditions.

## 11. Missing Evidence

This package contains one retained checkpoint per model, not three independently retrained seeds. It cannot estimate training randomness.

The complete protocol still requires training and evaluating:

- Seeds: 42, 3407, 2026.
- Algorithms: `legacy_kd`, `legacy_vkd`.
- Total training runs: 6.

After the full run, add seed-level checkpoint metrics, training histories, stage-1 predictions, scene-level paired bootstrap results, and a grouped report to this handoff package. Do not combine quick-checkpoint and retrained results as if they were the same comparison family.

## 12. Reproducing Existing Local Analysis

The reproduction script expects `quick_512_manual_server/` and `quick_512_manual_analysis/` to be siblings under the same parent directory.

Example on the current Windows environment:

```powershell
Set-Location '<parent directory containing both quick_512 folders>'
& 'D:\Miniconda3\envs\vision_est\python.exe' `
  quick_512_manual_analysis\reproduce_analysis.py
```

Required Python packages:

- pandas
- numpy
- scipy
- matplotlib

Versions used for the delivered analysis:

- pandas 2.3.3
- numpy 2.2.6
- scipy 1.15.3
- matplotlib 3.10.8

Different compatible versions should reproduce the numeric CSVs closely, but byte-identical PDF/PNG rendering is not guaranteed.

The script overwrites only generated files inside `quick_512_manual_analysis/`.

## 13. Start Prompt for the Next Codex

Use this prompt with the three delivered items:

> Read `QUICK_512_RESULT_ANALYSIS_HANDOFF.md` first. Audit the raw and existing analysis artifacts before making claims. Continue the result analysis using scene as the inferential unit and `legacy_vkd - legacy_kd` as the primary paired contrast. Focus next on domain breakdown, longitudinal versus lateral behavior, vision-gate behavior, and scene-level failure cases. Do not treat 512 windows as independent samples, do not infer training-seed stability from one checkpoint pair, and do not claim validation of real model mismatch or closed-loop estimator-controller coupling. Update the analysis bundle with exact statistics, real figures, claim boundaries, and a list of evidence still missing.
