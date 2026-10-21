# New-Code quick_512 Strict Analysis

## Analysis Question

How does the current accelerated seed-42 V2 visual-kinematic checkpoint behave under the four synthetic sensor perturbations already evaluated on 512 exactly aligned nuScenes windows?

Primary metric: endpoint velocity error (EPE, lower is better). Statistical unit: scene. The 512 windows come from 75 scenes and are not independent replicates.

## Integrity Decision

- Sample keys and velocity labels align across all 25 algorithm/variant cells.
- Current V2 joint checkpoint: valid accelerated checkpoint.
- V2 visual checkpoint: older non-accelerated checkpoint; current visual-to-joint claims are quarantined.
- Scenario metadata invariant across adapters: **False**. Cross-algorithm turning and other CAN-derived subgroup comparisons are invalid.

## Clean Descriptive Results

| algorithm | samples | scenes | endpoint_epe_mps | mae_vx_mps | mae_vy_mps | p95_epe_mps |
| --- | --- | --- | --- | --- | --- | --- |
| base_velocity | 512 | 75 | 0.17074 | 0.16066 | 0.03822 | 0.44656 |
| legacy_kd | 512 | 75 | 0.18399 | 0.18046 | 0.01978 | 0.43862 |
| legacy_vkd | 512 | 75 | 0.19123 | 0.18754 | 0.01993 | 0.54425 |
| v2_visual | 512 | 75 | 1.17965 | 0.68419 | 0.68293 | 4.30338 |
| v2_visual_kinematic | 512 | 75 | 0.88567 | 0.57877 | 0.42965 | 3.75225 |

The V2 joint clean EPE on this deterministic 512-window subset is 0.88567 m/s. This subset value must not replace the full 17,091-sample Stage-1 result of 0.64464 m/s.

## V2 Joint Fault Results

| variant | endpoint_epe_mps | degradation_mps | relative_degradation_percent |
| --- | --- | --- | --- |
| can_missing | 1.37482 | 0.48915 | 55.22877 |
| yaw_bias_pos | 0.88800 | 0.00233 | 0.26273 |
| wheel_scale_low | 0.88567 | 0.00000 | 0.00000 |
| camera_missing | 8.19996 | 7.31428 | 825.84389 |

- Complete camera removal is catastrophic: EPE increases by 7.31428 m/s (825.8%). All 512 samples are marked `fallback_used`, but mean predicted vx is -2.753 m/s instead of the available base-velocity mean 5.156 m/s. The fallback flag and final output path are inconsistent.
- Complete CAN removal remains finite but raises EPE by 0.48915 m/s (55.2%). Finite output is not acceptable fault tolerance.
- Positive yaw bias changes EPE by only 0.00233 m/s (0.26%). Combined with the earlier no-yaw ablation, this is evidence that the yaw path has little measured effect, not proof of robust compensation.
- A 0.9 wheel-speed scale causes 0.00000 m/s measured change. Because most full-holdout wheel scales already saturate at the 0.85 lower bound, this perturbation is masked and cannot establish wheel-speed robustness.

## Claim Candidates

- Claim:
  - Source evidence: Current joint camera-missing EPE 8.19996 vs clean 0.88567 m/s.
  - Allowed wording: "The current V2 joint checkpoint fails severely under complete synthetic camera removal."
  - Forbidden stronger wording: "The estimator safely falls back when vision fails."
  - Uncertainty: Synthetic blank-camera injection; no real sensor-disconnect recording.
  - Next check: Fix the existing fallback output path so the fallback flag actually selects the deterministic base velocity.
  - Decision: keep

- Claim:
  - Source evidence: Current joint CAN-missing degradation 0.48915 m/s.
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
