# Quick Existing-Checkpoint Evaluation

This mode validates existing KD/VKD/V2 checkpoints without retraining. It is a
fast screening test, not the formal three-seed comparison protocol.

## What It Tests

- Clean open-loop velocity accuracy on aligned `scene_name + center_token`
  samples.
- Synthetic sensor-fault trends: CAN missing, yaw-rate bias, wheel-speed scale
  error, and camera missing.
- Logged high-excitation subgroups: turning, high lateral velocity, hard
  acceleration, high steering, throttle, braking, and high lateral acceleration.

The script reuses checkpoint-native model architectures, but forces causal
online inputs at evaluation time. The server wrapper also disables the V2
geometry cache so camera-missing and other visual perturbations can modify the
actual online inputs.

It does not verify true physical vehicle-parameter mismatch or closed-loop
estimator-controller coupling. nuScenes is logged open-loop data, so control
coupling can only be reported as subgroup behavior during strong driver inputs.

## Server Command

From `FlexiNet_ChassisFusion`:

```bash
COMPARISON_RUN_ID=quick_existing_$(date +%Y%m%d_%H%M%S) \
NUSCENES_ROOT=/path/to/nuscenes \
QUICK_MAX_SAMPLES=512 \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

Optional checkpoint overrides:

```bash
LEGACY_KD_CHECKPOINT=checkpoints/velocity_xy_kd_only/best.pth
LEGACY_VKD_CHECKPOINT=checkpoints/velocity_xy_vkd_gated/best.pth
V2_VISUAL_CHECKPOINT=checkpoints/velocity_xy_v2_visual/best.pth
V2_VISUAL_KINEMATIC_CHECKPOINT=checkpoints/velocity_xy_v2_visual_kinematic/best.pth
```

If `V2_VISUAL_KINEMATIC_CHECKPOINT` does not exist, the report skips that arm
and still compares `v2_visual` against `legacy_kd` and `legacy_vkd`.

## Runtime

With existing frame, flow, geometry, and split/statistics files:

- `QUICK_MAX_SAMPLES=256`: usually 10-30 minutes.
- `QUICK_MAX_SAMPLES=512`: usually 20-60 minutes.
- `QUICK_MAX_SAMPLES=2048`: often close to 1-2 hours.

The first two settings are best for debugging the claim and checking whether
the trend is worth a longer run.

## Outputs

```text
outputs/nuscenes_version_comparison/<run_id>/quick_existing_eval/
  quick_report.md
  quick_report.json
  all_predictions.csv
  <algorithm>/predictions.csv
  <algorithm>/metrics.json
```

Delta definitions follow the formal protocol: `left error - right error`.
Negative values mean the left algorithm is better.
