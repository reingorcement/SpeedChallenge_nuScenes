# Quick Existing-Checkpoint nuScenes Evaluation

This report reuses existing checkpoints and does not retrain models.

## Clean Accuracy

| Algorithm | Samples | Scenes | Endpoint EE | MAE vx | MAE vy | P95 EE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_velocity | 512 | 75 | 0.15208 | 0.14243 | 0.03549 | 0.48161 |
| legacy_kd | 512 | 75 | 0.07958 | 0.07488 | 0.01493 | 0.22693 |
| legacy_vkd | 512 | 75 | 0.07759 | 0.07287 | 0.01525 | 0.22606 |

## Paired Deltas

- `legacy_vkd_minus_legacy_kd:clean`: delta -0.00193 m/s, 95% CI [-0.00437, 0.00059], trend_only.

## Stress Summary

- `legacy_vkd_minus_legacy_kd:can_missing`: absolute delta -0.00156 m/s; degradation 4.44398 vs 4.44355 m/s.
- `legacy_vkd_minus_legacy_kd:yaw_bias_pos`: absolute delta -0.00223 m/s; degradation -0.00008 vs 0.00023 m/s.
- `legacy_vkd_minus_legacy_kd:wheel_scale_low`: absolute delta -0.00242 m/s; degradation 0.01330 vs 0.01386 m/s.
- `legacy_vkd_minus_legacy_kd:camera_missing`: absolute delta -0.00041 m/s; degradation 0.00159 vs 0.00000 m/s.

## Boundary

- Verified: Existing-checkpoint open-loop accuracy on aligned nuScenes samples, synthetic sensor perturbation trends, and high-excitation subgroup trends.
- Trend-only: This is a fast screening test, not a retrained fair protocol with repeated seeds. Checkpoint-native architectures are used, while evaluation inputs are forced causal.
- Not verified: True physical parameter mismatch, cross-vehicle dynamics generalization, and closed-loop estimator-controller coupling.
- Current V2 note: Use v2_visual_kinematic only when a real checkpoint is supplied; v2_visual has no kinematic or dynamics branch.
