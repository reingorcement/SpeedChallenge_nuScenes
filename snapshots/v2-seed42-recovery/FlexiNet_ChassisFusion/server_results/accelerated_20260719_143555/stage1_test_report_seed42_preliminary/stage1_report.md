# nuScenes Stage-1 Seed-42 Preliminary Test Report

- Overall result: **PASS**
- Analysis mode: `preliminary_single_seed`
- Experimental unit: scene
- Paired seeds: 42
- Holdout scenes: 75
- `final_system_test` remained sealed and was not evaluated.

## Primary Metrics

| Model | EPE mean +/- std (m/s) | vx MAE (m/s) | vy MAE (m/s) |
|---|---:|---:|---:|
| geometry | 0.8830 +/- 0.0000 | 0.5837 | 0.4463 |
| visual | 0.7228 +/- 0.0000 | 0.5228 | 0.3148 |
| joint | 0.6446 +/- 0.0000 | 0.4714 | 0.2688 |

## Visual to Joint Changes

| Metric | Visual | Joint | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| endpoint_error_mps | 0.7228 | 0.6446 | -0.0781 | -10.81% |
| mae_vx_mps | 0.5228 | 0.4714 | -0.0514 | -9.83% |
| mae_vy_mps | 0.3148 | 0.2688 | -0.0459 | -14.59% |
| turning_mae_vy_mps | 1.1718 | 1.0372 | -0.1346 | -11.48% |
| high_lateral_mae_vy_mps | 0.3809 | 0.3168 | -0.0641 | -16.82% |

## Acceptance

| Check | Result |
|---|---:|
| geometry_valid_rate | PASS |
| geometry_condition | PASS |
| geometry_vx_sign | PASS |
| visual_epe_within_2_percent | PASS |
| visual_axis_mae_improved | PASS |
| joint_epe_within_2_percent | PASS |
| joint_outputs_finite | PASS |
| joint_correction_bounds | PASS |
| joint_group_improved_seed42 | PASS |
| missing_can_outputs_finite | PASS |
| kinematic_ablation_changes_predictions | PASS |

## Paired Scene Bootstrap

- joint_minus_visual_all_endpoint_error_mps: delta=-0.0783, 95% CI [-0.0957, -0.0620], n=75 scenes
- joint_minus_visual_turning_mae_vy_mps: delta=-0.1343, 95% CI [-0.1591, -0.1092], n=40 scenes
- joint_minus_visual_high_lateral_mae_vy_mps: delta=-0.0821, 95% CI [-0.1018, -0.0636], n=59 scenes

## Interpretation

Negative paired deltas favor the joint model. Acceptance thresholds are engineering gates; the bootstrap intervals quantify scene-to-scene uncertainty and are not computed from adjacent frames as independent samples.
This single-seed bootstrap is conditional on seed 42 and does not quantify training-seed variability or establish statistical significance across random initializations.

Ablation results, per-scene metrics, and ranked bad cases are stored beside this report.
