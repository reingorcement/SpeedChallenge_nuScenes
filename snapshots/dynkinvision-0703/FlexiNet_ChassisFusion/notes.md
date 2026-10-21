# Notes: VKD Fusion Refactor

## Sources

### Refactor Plan
- Goal: upgrade the existing velocity-XY residual pipeline to `VisionBranch + KinematicBranch + DynamicsBranch + ConfidenceGatedFusion`.
- Compatibility boundary: keep dataset batch contract and default `fusion_mode: residual` working.
- Minimal viable scope: configuration switches, `residual_gated`, four ablation YAMLs, and scenario evaluation.
- Verification: compileall, pytest velocity-XY tests, `smoke_model.py`, optional train smoke config.

### `AI_CODE_CONTEXT.md`
- Current task is ego/body-frame 2D velocity estimation, not scalar speed.
- Main contract: `pred_velocity = base_velocity + delta_velocity`, both `[vx, vy]`.
- `fair_chassis` is deployable main mode; `pose_assisted` is upper-bound only.
- EOVM-EKF can supply `[base_vx, base_vy]`, but should not be a main conclusion before input audit.

### `速度XY改造使用说明.md`
- Existing code already supports velocity labels, chassis sequences, flow, geometry, and EOVM base velocity.
- Local validation previously reported `14 passed` for velocity-XY/eovm tests.
- Full training likely requires server data; local validation should emphasize compile, unit tests, and model smoke.

## Current Code Findings

### Dataset
- `FAIR_CHASSIS_FEATURES` has 16 entries: vehicle/odom speed, yaw rate, controls, accelerations, four wheel speeds, wheel mean/std, and `base_vx`.
- `__getitem__` returns `images`, `chassis`, `base_velocity`, `velocity`, `flow`, `geometry`, `base_speed`, `speed`, and metadata.
- No new `dynamics` batch field is needed for the first phase; the model can build dynamics proxy from `chassis` and `base_velocity`.

### Model
- `FlexiNetChassisFusion` currently supports only `residual` and fallback direct behavior.
- Existing fusion is concatenation of visual, chassis, optional flow, and optional geometry features followed by `FusionHead`.
- There are no branch enable flags, no `residual_gated`, no dynamics proxy encoder, and no saved gate weights.

### Training / Eval Entrypoints
- Model construction is duplicated in `train.py`, `test.py`, `analysis/inference.py`, and `scripts/path_sanity_check.py`.
- Any new model constructor parameters must be threaded through all of these.
- `VelocitySmoothL1Loss` currently accepts only `pred` and `target`; a knowledge loss would require optional `base_velocity` and `chassis` handling.

### Evaluation / Analysis
- `EvaluationResult` currently stores yaw rates only.
- `compute_metrics` already has speed bins, high lateral, turning, and straight metrics, but grouped metrics only include endpoint error and MAE vy.
- Predictions CSV does not include chassis proxy columns or gate weights.
- `analysis/postprocess.py` has per-scene and per-speed-bin aggregation but no scenario aggregation yet.

## Risks
- Branch flags could accidentally disable all branches; model should fail clearly.
- `vision_only` must not leak `base_velocity` through residual mode; direct mode is appropriate.
- `residual_gated` should only include active branches in softmax, not zero placeholders.
- Dynamics proxy indices must use `FAIR_CHASSIS_FEATURES` rather than hard-coded positions where possible.
- `base_velocity_source: eovm_ekf` requires audited paths/data; configs with placeholder paths should not be run as main evidence.
- Checkpoint compatibility for old models should be preserved by leaving `residual` architecture unchanged.

## Proposed Safe First Scope
- Keep dataset contract unchanged.
- Add config fields with defaults that preserve current behavior.
- Add gated model path behind `fusion_mode: residual_gated`.
- Add tests for direct vision-only, gated kd-only/vkd, gate shape/sum, and old residual behavior.
- Defer nonzero knowledge losses and EOVM-dependent conclusions until after structural smoke is green.
