# Task Plan: Unattended nuScenes Stage-1 Training and Evaluation

## Goal
Provide one server command that trains the nuScenes-only visual/kinematic system, evaluates it on an independent scene holdout, runs ablations and robustness checks, and writes an auditable statistical report before returning success or failure.

## Assumptions
- Phase 1 remains nuScenes-only; CarSim and the dynamics teacher/student are out of scope.
- The independent experimental unit is a nuScenes scene, not an adjacent frame.
- Official train may be used for training/development; official val is split into a stage-1 holdout and a sealed phase-2 final test.
- Existing RAFT/FlexiNet weights are initialization only and do not change the data split rules.
- Full dataset, RAFT, CUDA smoke, training, and holdout evaluation run on the server only.

## Decisions
- Split scenes deterministically with seed 42: official train -> 85% train_core / 15% dev; official val -> 50% stage1_test / 50% final_system_test.
- Persist exact scene assignments and a checksum in a manifest; stage-1 code must reject final_system_test.
- Select checkpoints only on dev; evaluate once on stage1_test after all training decisions are complete.
- Keep final_system_test sealed until the CarSim phase-2 system is frozen.
- Use paired scene-level bootstrap confidence intervals and three paired seeds (42, 3407, 2026).
- Always write JSON/CSV/Markdown outputs before applying the final acceptance exit code.

## Phases
- [x] Phase 1: Confirm goals, experimental unit, split policy, and acceptance criteria
- [x] Phase 2: Implement deterministic scene manifests and dataset/train split enforcement
- [x] Phase 3: Implement holdout prediction export, ablations, and robustness evaluations
- [x] Phase 4: Implement scene-level bootstrap analysis and acceptance report
- [x] Phase 5: Integrate one-command server orchestration and documentation
- [x] Phase 6: Add synthetic tests and run local compile/regression verification

## Acceptance Criteria
- Geometry: valid rate >= 60%, median condition number < 1e5, straight-driving vx sign agreement >= 95%.
- Visual: EPE <= 1.02 * geometry EPE and at least one of vx/vy MAE improves.
- Joint: mean EPE across seeds <= 1.02 * visual; at least 2/3 seeds improve turning or high-lateral vy; all values finite and corrections remain bounded.
- Split sets are pairwise disjoint and stage-1 execution never reads final_system_test.
- Reports contain per-seed, aggregate, subgroup, ablation, bad-case, and 95% scene-bootstrap results.

## Errors Encountered
- The environment-declared backup worktree no longer exists; implementation continues in the current `dynkinvision-0718` worktree.
- The system-default Python 3.14 has no pytest installation; local verification uses the user-provided `speedXY_py310` environment instead.
- One patch included an unrelated context anchor and was rejected without changing files; it was immediately reapplied with the correct narrow context.

## Status
**Complete** - unattended server orchestration, independent holdout testing, ablations, scene-level statistics, reporting, documentation, and 97-test local regression verification are complete. Full nuScenes/RAFT/CUDA execution remains server-only.
