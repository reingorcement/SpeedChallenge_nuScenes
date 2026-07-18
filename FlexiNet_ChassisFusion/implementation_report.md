# Implementation Report

## Implemented
- Added a 25-field canonical physics input schema, per-field validity mask, causal time steps, and optional full-signal CSV lookup.
- Added a differentiable causal physics estimator with robust wheel observation, adaptive longitudinal Kalman update, kinematic/dynamic lateral estimation, and the Simulink Brush tire formula for full inputs.
- Added `base`, `physics`, and `physics_residual_gated` model modes without changing legacy mode behavior.
- Added physics prior, confidence, per-axis correction gates, hybrid losses, metrics, CSV diagnostics, and six causal ablation configurations.
- Added shared CAN-unit audit enforcement and updated training, test, analysis, profile, path-check, and smoke entry points.

## Verification
- `python -m compileall`: passed.
- `python -m pytest tests eovm_ekf/tests -q`: 48 passed.
- `python smoke_model.py --batch_size 2 --seq_len 3 --with_physics`: output shape `(2, 2)`.
- Synthetic tiny-batch overfit: loss decreased from `0.045129` to `0.000010` in 80 steps.
- Eight relevant YAML configurations parsed and built successfully with causal windows enabled.

## Review Fixes
- Online image windows now end at the supervised current frame; future images are not included.
- Dataset returns an explicit `chassis_current_index`, used by dynamics proxy features, consistency loss, geometry, and evaluation diagnostics.
- CAN validity is field-level and finite-value-aware; missing wheel channels no longer enter wheel mean/std statistics.
- CAN and full-physics CSV lookups reject stale records using configurable maximum ages.
- `physics_dt[:,0]` is zero, eliminating the extra first integration interval; negative, non-finite, and malformed time axes are rejected.
- Added `velocity_xy_physics_feature_only.yaml` to train the physics estimator, 18-feature GRU, and residual head without vision or raw-chassis branches.
- `rear_drive_enable` now affects full-signal effort selection. The unused new-branch tire-radius parameter was removed because wheel inputs are already linear m/s.
- Corrected physics-feature-only overfit with observable lateral inputs: loss decreased from `0.169108` to `0.000020`.

## Remaining External Validation
- Real nuScenes unit and sign auditing is still required. Physics configs intentionally keep `physics_units_audited: false` until this is completed.
- Solver_SF and CarSim are unavailable, so full Simulink closed-loop numerical parity was not run.
- Real-data accuracy, parameter identifiability, and ablation conclusions require server-side training with valid dataset paths.
