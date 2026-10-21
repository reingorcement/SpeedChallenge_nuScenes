# quick_512_manual Strict Analysis

## Analysis Question

Does the retained visual-kinematic-dynamic checkpoint (`legacy_vkd`) improve endpoint velocity error and synthetic sensor-stress behavior relative to the retained kinematic-dynamic checkpoint (`legacy_kd`) on exactly aligned causal nuScenes samples?

Primary metric: endpoint velocity error (EPE, lower is better). Statistical unit: scene. The 512 windows belong to 75 scenes and are not treated as independent experimental replicates.

## Key Findings

| algorithm | samples | scenes | endpoint_epe_mps | mae_vx_mps | mae_vy_mps | p95_epe_mps |
| --- | --- | --- | --- | --- | --- | --- |
| base_velocity | 512 | 75 | 0.15208 | 0.14243 | 0.03549 | 0.48161 |
| legacy_kd | 512 | 75 | 0.07958 | 0.07488 | 0.01493 | 0.22693 |
| legacy_vkd | 512 | 75 | 0.07759 | 0.07287 | 0.01525 | 0.22606 |

- On clean samples, VKD lowers sample-weighted EPE from 0.07958 to 0.07759 m/s (2.50% relative), while remaining 48.98% below the base-velocity reference.
- The gain is concentrated in longitudinal error: MAE vx improves 2.69%, but MAE vy changes by -2.15% (negative means worse). Thus this result does not support a claim of improved lateral-velocity accuracy.
- Scene-level VKD-KD clean delta is -0.00193 m/s with bootstrap 95% CI [-0.00438, 0.00062]. The interval crosses zero. The existing-checkpoint clean comparison is a trend, not a stable winner claim.
- VKD wins only 49.3% of clean scenes and the median scene delta is 0.00045 m/s (positive favors KD). Scene deltas are non-normal (Shapiro-Wilk p=4.335e-06); the two-sided Wilcoxon test is not significant before or after Holm correction (raw p=0.136, adjusted p=0.448).
- CAN removal is catastrophic for both models: degradation exceeds 4.44 m/s. The statistically narrow VKD-KD absolute difference is practically negligible and does not demonstrate fault tolerance.
- Wheel-speed scaling raises error for both models by about 0.013-0.014 m/s. Camera removal only affects VKD and moves it back toward KD, consistent with a small visual contribution rather than robust visual compensation.

## Stress Results

| algorithm | variant | endpoint_epe_mps | degradation_mps |
| --- | --- | --- | --- |
| legacy_kd | can_missing | 4.52313 | 4.44355 |
| legacy_kd | yaw_bias_pos | 0.07981 | 0.00023 |
| legacy_kd | wheel_scale_low | 0.09344 | 0.01386 |
| legacy_kd | camera_missing | 0.07958 | 0.00000 |
| legacy_vkd | can_missing | 4.52157 | 4.44398 |
| legacy_vkd | yaw_bias_pos | 0.07751 | -0.00008 |
| legacy_vkd | wheel_scale_low | 0.09089 | 0.01330 |
| legacy_vkd | camera_missing | 0.07918 | 0.00159 |

## Claim Candidates

- Claim:
  - Source evidence: Clean EPE 0.07759 vs 0.07958 m/s on 512 aligned samples from 75 scenes.
  - Allowed wording: "The retained VKD checkpoint showed a small clean-set improvement trend over the retained KD checkpoint."
  - Forbidden stronger wording: "VKD is significantly or generally superior to KD."
  - Uncertainty: One checkpoint per method; scene-level clean CI crosses zero; no training-seed variance.
  - Next check: Complete the three-seed retraining protocol and compare seed-level paired scene estimates.
  - Decision: weaken

- Claim:
  - Source evidence: MAE vx decreases from 0.07488 to 0.07287 m/s, whereas MAE vy changes from 0.01493 to 0.01525 m/s.
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
