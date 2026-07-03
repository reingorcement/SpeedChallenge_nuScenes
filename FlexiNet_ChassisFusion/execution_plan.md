# Execution Plan: VKD Fusion Refactor

## Extracted Goal
Upgrade the existing velocity-XY pipeline from concatenation fusion to a knowledge-driven vision, kinematic, and dynamics branch design:

```text
VisionBranch + KinematicBranch + DynamicsBranch + ConfidenceGatedFusion
```

The implementation must preserve the current default behavior:

```text
pred_velocity = base_velocity + delta_velocity
target_velocity = [v_x, v_y]
base_velocity = [base_vx, base_vy]
```

## Involved Modules
- `configs/*.yaml`: defaults and ablation experiment configs.
- `dataset/nuscenes_chassis.py`: chassis feature names and batch contract reference.
- `model/fusion_model.py`: new branch projections, dynamics proxy encoder, and gated fusion path.
- `train.py`: model construction and optional future knowledge loss.
- `test.py`: evaluation model construction.
- `analysis/inference.py`: checkpoint evaluation model construction.
- `utils/evaluation.py`: scenario metrics, proxy collection, prediction CSV extensions.
- `analysis/postprocess.py`: offline scenario report extensions.
- `scripts/path_sanity_check.py`: model construction and skip-dataset config validation.
- `smoke_model.py` and `tests/test_velocity_xy_pipeline.py`: smoke and regression coverage.

## Expected Code Changes
- Add config fields while preserving existing defaults:
  - `fusion_type`
  - `use_vision_branch`
  - `use_kinematic_branch`
  - `use_dynamics_branch`
  - `branch_feature_dim`
  - `dynamics_feature_dim`
  - `save_gate_weights`
  - default-zero knowledge loss weights.
- Add model components:
  - `ProjectionBlock`
  - `DynamicsProxyEncoder`
  - `ConfidenceGatedFusionHead`
  - branch-gated path for `fusion_mode: residual_gated`.
- Keep current `residual` and `direct` concatenation paths compatible.
- Build dynamics proxy internally from `base_velocity` plus center-frame chassis columns.
- Add experiment YAMLs:
  - `velocity_xy_vision_only.yaml`
  - `velocity_xy_kd_only.yaml`
  - `velocity_xy_vision_kinematics.yaml`
  - `velocity_xy_vkd_gated.yaml`
  - `smoke_velocity_xy_vkd_gated.yaml`
- Extend tests for branch flags, gate shape/sum, old residual compatibility, and config build.
- Extend evaluation to collect chassis proxy fields and gate weights where available.

## Risks and Controls
- All branch flags disabled: raise a clear `ValueError`.
- Vision-only leakage: use `fusion_mode: direct`, not residual base-velocity correction.
- Gated softmax with inactive branches: include only active branch tensors.
- Chassis index mismatch: use `FAIR_CHASSIS_FEATURES` lookup rather than magic indices where practical.
- EOVM uncertainty: keep EOVM configs opt-in and do not change default `zero_lateral`.
- Old checkpoint behavior: do not alter modules used by default `fusion_mode: residual` more than necessary.
- Repeated model construction: update `train.py`, `test.py`, `analysis/inference.py`, and `scripts/path_sanity_check.py` together.

## Acceptance Criteria
- Old residual forward pass still outputs `[B, 2]`.
- `vision_only`, `kd_only`, `vision_kinematics`, and `vkd_gated` forward passes output `[B, 2]`.
- `residual_gated` produces gate weights with shape `[B, active_branch_count]`.
- Gate rows sum to 1 within tolerance.
- New configs can construct models via `scripts/path_sanity_check.py --skip-dataset`.
- Metrics JSON includes added scenario metrics when proxy inputs exist.
- Predictions CSV includes proxy/gate columns when available and leaves missing values blank.

## Recommended Staged Plan

### Stage 1: Model Configuration and Branch Switches
- Update config defaults with new fields that preserve old behavior.
- Extend model constructor and all model builders with branch flags.
- Add tests for old residual, direct vision-only, and disabled-branch validation.
- Verify with targeted pytest and `smoke_model.py`.

### Stage 2: Dynamics Proxy and Confidence-Gated Fusion
- Add `ProjectionBlock`, `DynamicsProxyEncoder`, and `ConfidenceGatedFusionHead`.
- Implement `fusion_mode: residual_gated`.
- Save `last_gate_weights` and `last_gate_names` on the model.
- Add tests for kd-only and vkd-gated output/gate behavior.
- Verify with compileall, targeted pytest, and smoke.

### Stage 3: Experiment YAMLs and Sanity Checks
- Add the four ablation configs plus gated smoke config.
- Update `scripts/path_sanity_check.py` if needed for new constructor fields.
- Verify all new configs can build a model with `--skip-dataset`.

### Stage 4: Evaluation and Analysis Extensions
- Extend `EvaluationResult`, `run_eval`, `compute_metrics`, and prediction CSV writing.
- Collect yaw rate, wheel speed std, longitudinal/transversal accelerations, and gate weights.
- Extend `analysis/postprocess.py` with scenario metrics.
- Add tests for new metric groups and CSV columns.

### Stage 5: Knowledge Loss Scaffold
- Add `KnowledgeDrivenVelocityLoss` with default-zero beta, prior, smoothness, and consistency weights.
- Keep old `velocity_smoothl1` unchanged.
- Wire optional `base_velocity` and `chassis` into training only when the new loss is selected.
- Add finite-loss and backward-pass tests.

## Recommended Scope for First Implementation Pass
Implement Stages 1-4 first, keeping knowledge loss weights and EOVM-dependent conclusions deferred. Stage 5 can be implemented afterward once structural gated fusion and evaluation are verified.
