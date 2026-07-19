# Notes: FlexiNet ChassisFusion Algorithm Handoff

## Scope
- Target directory: `FlexiNet_ChassisFusion`
- Deliverable: `FlexiNet_ChassisFusion/ALGORITHM_HANDOFF.md`
- Purpose: detailed technical handoff document for another model or engineer.

## Findings

### High-Level Algorithm
- Current target is 2D ego/body-frame velocity, not scalar speed:
  `target_velocity = [v_x, v_y]`, `pred_velocity = base_velocity + delta_velocity`.
- Main implementation directory is `FlexiNet_ChassisFusion`.
- The current VKD configuration is `configs/velocity_xy_vkd_gated.yaml`.
- The current main method uses `fusion_mode: residual_gated` with vision, kinematic, and dynamics branches enabled.

### Data Pipeline
- Dataset file: `FlexiNet_ChassisFusion/dataset/nuscenes_chassis.py`.
- Dataset class: `NuScenesChassisFusionDataset`.
- Each sample returns `images`, `chassis`, `base_velocity`, `velocity`, `flow`, `geometry`, legacy `base_speed`/`speed`, and metadata.
- Images are a sequence of `seq_len` camera frames, default `[13, 1, 64, 64]` for gray `CAM_FRONT`.
- Velocity label uses centered ego-pose difference:
  `v_global = (p_next - p_prev) / dt`, then rotates to center ego frame.
- Boundary frames without previous or next sample_data are skipped.
- `fair_chassis` has 16 deployable CAN/chassis features.
- `pose_assisted` adds 6 pose-derived features and should be treated as an upper-bound experiment because pose-derived inputs share source with labels.
- Default chassis sequence is a 1.0 s window at 20 Hz, encoded as `[K, C]`.
- Default base velocity source is `zero_lateral`: `[base_vx, 0]`.
- Optional EOVM-EKF base velocity uses CSV lookup via `base_velocity_source: eovm_ekf` and `eovm_base_root`.

### Model Pipeline
- Model file: `FlexiNet_ChassisFusion/model/fusion_model.py`.
- Backbone: `FlexiNetBackbone` in `model/flexinet_backbone.py`, output feature dimension 256.
- Chassis sequence encoder: GRU or mean/none through `ChassisSequenceEncoder`.
- Optional encoders: `FlowEncoder`, `GeometryEncoder`.
- VKD additions:
  - `ProjectionBlock` maps vision/kinematic branches to common branch dimension.
  - `DynamicsProxyEncoder` encodes 8 dynamics proxy values.
  - `ConfidenceGatedFusionHead` softmaxes active branch weights and predicts `[delta_vx, delta_vy]`.
- Supported fusion modes:
  - `residual`: concat active features, predict delta, return base + delta.
  - `direct`: concat active features, directly predict velocity.
  - `residual_gated`: active branches -> projections -> confidence gate -> delta -> base + delta.
- Dynamics proxy order is `[base_vx, base_vy, yaw_rate, steering, steering_feedback, longitudinal_accel, transversal_accel, wheel_speed_std]`.
- Gate weights are saved only if `save_gate_weights: true`; names are active branch names only.

### Training and Evaluation
- Training entry: `FlexiNet_ChassisFusion/train.py`.
- Eval entry: `FlexiNet_ChassisFusion/test.py`.
- Analysis entry: `FlexiNet_ChassisFusion/analysis/evaluate_checkpoint.py`.
- Loss:
  - Default `VelocitySmoothL1Loss` normalizes axes by `sigma_vx` and `sigma_vy`.
  - Optional `KnowledgeDrivenVelocityLoss` adds beta, prior, smoothness, and consistency switches; defaults are zero in configs.
- Optimizer: AdamW with separate head and backbone learning rates.
- Default backbone is frozen.
- Best checkpoint selection metric is `mae_vx_mps + mae_vy_mps` via `selection_metric_mps`.
- Outputs include checkpoints, training history, eval metrics JSON, prediction CSV, and diagnostics.
- `analysis/inference.py` passes `frame_cache_root`; `test.py` currently does not pass it, so standalone `test.py` may use raw images instead of cache.

### Metrics and Result Evidence
- Evaluation file: `FlexiNet_ChassisFusion/utils/evaluation.py`.
- Main metrics: `mae_vx_mps`, `mae_vy_mps`, `vector_endpoint_error_mps`, speed magnitude error, R2 per axis, base error, gain vs base.
- Scenario groups include low/mid/high speed, high lateral, turning/straight, wheel inconsistent, hard accel, high lateral accel.
- Prediction CSV includes target/pred/base vx/vy, delta vx/vy, proxy features, and optional gate columns.
- Current `vkd_0704` comparison aligns 33,148 validation samples.
- `vkd_gated` is best in the single-seed comparison:
  - vector endpoint error `0.07438 m/s`
  - MAE vx `0.06875 m/s`
  - MAE vy `0.01782 m/s`
  - gain vs base vector `0.11898 m/s`
- Single-seed warning: do not claim statistical significance.

### Verification Evidence
- `tests/test_velocity_xy_pipeline.py` covers label rotation, fair chassis exclusion of pose-derived values, 2D model output, gated branch weights, KD-only branch disabling, metric columns, knowledge loss, frame cache behavior, and script help.
- Existing docs report `14 passed` previously for `tests/test_velocity_xy_pipeline.py eovm_ekf/tests`.

### Risks / Handoff Warnings
- Do not interpret this branch as old scalar speed estimation.
- Do not change `output_dim` back to 1 unless explicitly doing a legacy scalar experiment.
- Keep dataset batch contract stable.
- Use `fair_chassis` for deployable conclusions; keep `pose_assisted` as upper bound only.
- Current dynamics branch is a proxy MLP, not the full EOVM-EKF estimator.
- EOVM-EKF must be audited for wheel speed scale, steering scale/sign, yaw rate unit, acceleration directions, and vehicle geometry before being used as a main physics prior.
- `test.py` and `scripts/path_sanity_check.py` do not currently pass `frame_cache_root` into the dataset; `train.py` and `analysis/inference.py` do.
