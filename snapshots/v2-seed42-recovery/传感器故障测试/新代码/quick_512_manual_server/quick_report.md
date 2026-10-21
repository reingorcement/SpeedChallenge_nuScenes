# Quick Existing-Checkpoint nuScenes Evaluation

This report reuses existing checkpoints and does not retrain models.

## Clean Accuracy

| Algorithm | Samples | Scenes | Endpoint EE | MAE vx | MAE vy | P95 EE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_velocity | 512 | 75 | 0.17074 | 0.16066 | 0.03822 | 0.44656 |
| legacy_kd | 512 | 75 | 0.18399 | 0.18046 | 0.01978 | 0.43862 |
| legacy_vkd | 512 | 75 | 0.19123 | 0.18754 | 0.01993 | 0.54425 |
| v2_visual | 512 | 75 | 1.17965 | 0.68419 | 0.68293 | 4.30337 |
| v2_visual_kinematic | 512 | 75 | 0.88567 | 0.57877 | 0.42965 | 3.75225 |

## Paired Deltas

- `v2_visual_kinematic_minus_legacy_vkd:clean`: delta 0.70121 m/s, 95% CI [0.40193, 1.03565], trend_only.
- `v2_visual_kinematic_minus_legacy_kd:clean`: delta 0.70840 m/s, 95% CI [0.41633, 1.03274], trend_only.
- `v2_visual_minus_legacy_vkd:clean`: delta 0.99280 m/s, 95% CI [0.67913, 1.32126], trend_only.
- `v2_visual_minus_legacy_kd:clean`: delta 1.00000 m/s, 95% CI [0.68853, 1.32944], trend_only.
- `legacy_vkd_minus_legacy_kd:clean`: delta 0.00720 m/s, 95% CI [-0.01380, 0.03004], trend_only.

## Stress Summary

- `v2_visual_kinematic_minus_legacy_vkd:can_missing`: absolute delta -3.87635 m/s; degradation 0.48915 vs 5.09021 m/s.
- `v2_visual_kinematic_minus_legacy_vkd:yaw_bias_pos`: absolute delta 0.70377 m/s; degradation 0.00233 vs -0.00031 m/s.
- `v2_visual_kinematic_minus_legacy_vkd:wheel_scale_low`: absolute delta 0.68849 m/s; degradation 0.00000 vs 0.01275 m/s.
- `v2_visual_kinematic_minus_legacy_vkd:camera_missing`: absolute delta 7.99189 m/s; degradation 7.31428 vs -0.00518 m/s.
- `v2_visual_kinematic_minus_legacy_kd:can_missing`: absolute delta -3.87832 m/s; degradation 0.48915 vs 5.09942 m/s.
- `v2_visual_kinematic_minus_legacy_kd:yaw_bias_pos`: absolute delta 0.71038 m/s; degradation 0.00233 vs 0.00032 m/s.
- `v2_visual_kinematic_minus_legacy_kd:wheel_scale_low`: absolute delta 0.69629 m/s; degradation 0.00000 vs 0.01230 m/s.
- `v2_visual_kinematic_minus_legacy_kd:camera_missing`: absolute delta 7.99397 m/s; degradation 7.31428 vs 0.00000 m/s.
- `v2_visual_minus_legacy_vkd:can_missing`: absolute delta -3.73697 m/s; degradation 0.33472 vs 5.09021 m/s.
- `v2_visual_minus_legacy_vkd:yaw_bias_pos`: absolute delta 0.99307 m/s; degradation -0.00000 vs -0.00031 m/s.
- `v2_visual_minus_legacy_vkd:wheel_scale_low`: absolute delta 0.98008 m/s; degradation 0.00000 vs 0.01275 m/s.
- `v2_visual_minus_legacy_vkd:camera_missing`: absolute delta 4.90832 m/s; degradation 3.93902 vs -0.00518 m/s.
- `v2_visual_minus_legacy_kd:can_missing`: absolute delta -3.73894 m/s; degradation 0.33472 vs 5.09942 m/s.
- `v2_visual_minus_legacy_kd:yaw_bias_pos`: absolute delta 0.99969 m/s; degradation -0.00000 vs 0.00032 m/s.
- `v2_visual_minus_legacy_kd:wheel_scale_low`: absolute delta 0.98788 m/s; degradation 0.00000 vs 0.01230 m/s.
- `v2_visual_minus_legacy_kd:camera_missing`: absolute delta 4.91040 m/s; degradation 3.93902 vs 0.00000 m/s.
- `legacy_vkd_minus_legacy_kd:can_missing`: absolute delta -0.00197 m/s; degradation 5.09021 vs 5.09942 m/s.
- `legacy_vkd_minus_legacy_kd:yaw_bias_pos`: absolute delta 0.00661 m/s; degradation -0.00031 vs 0.00032 m/s.
- `legacy_vkd_minus_legacy_kd:wheel_scale_low`: absolute delta 0.00780 m/s; degradation 0.01275 vs 0.01230 m/s.
- `legacy_vkd_minus_legacy_kd:camera_missing`: absolute delta 0.00208 m/s; degradation -0.00518 vs 0.00000 m/s.

## Boundary

- Verified: Existing-checkpoint open-loop accuracy on aligned nuScenes samples, synthetic sensor perturbation trends, and high-excitation subgroup trends.
- Trend-only: This is a fast screening test, not a retrained fair protocol with repeated seeds. Checkpoint-native architectures are used, while evaluation inputs are forced causal.
- Not verified: True physical parameter mismatch, cross-vehicle dynamics generalization, and closed-loop estimator-controller coupling.
- Current V2 note: Use v2_visual_kinematic only when a real checkpoint is supplied; v2_visual has no kinematic or dynamics branch.
