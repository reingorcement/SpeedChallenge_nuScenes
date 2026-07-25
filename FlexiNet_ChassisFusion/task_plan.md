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

---

# Task Plan: Seed-42 Recovery and Preliminary Evaluation

## Goal
Resume from the existing seed-42 kinematic best checkpoint, train only joint seed 42, and automatically produce online holdout, ablation, and explicitly preliminary single-seed reports without rerunning completed stages.

## Phases
- [x] Phase 1: Inspect returned artifacts, current trainer, analyzer, tests, and server pipeline
- [x] Phase 2: Implement post-checkpoint gradient-clip health handling and resume state
- [x] Phase 3: Implement preliminary single-seed analysis and the continuation pipeline
- [x] Phase 4: Add focused tests and run local regression, compile, and shell checks
- [x] Phase 5: Update the implementation report and provide the exact server handoff

## Decisions Made
- Existing visual and kinematic checkpoints remain inputs; no completed training stage is rerun.
- Keep model, data, loss, optimizer, and joint hyperparameters unchanged.
- Treat one clip-warning epoch as recoverable, two consecutive warning epochs or one epoch above 30% as fatal only after validation and checkpoint persistence.
- Single-seed scene bootstrap is conditional on seed 42 and cannot establish training-seed stability.

## Errors Encountered
- The environment-declared backup worktree no longer exists; implementation uses `dynkinvision-0718`, which contains the returned server artifacts.
- `FlexiNet_ChassisFusion/.gitignore` does not exist; generated-artifact handling must use the repository-level ignore rules and explicit script paths.
- The expected `dataset/v2_frame_cache.py` helper does not exist; frame cache validation is implemented by the existing `NuScenesChassisFusionDataset` manifest contract.
- The first completion-marker test serialized the YAML config through JSON, changing `backbone_lr` from the YAML string `5e-06` to a float; the fixture now uses the same `yaml.safe_dump` path as the trainer.
- Windows `bash.exe` points to an unavailable WSL installation; Git Bash at `D:/Program Files/Git/bin/bash.exe` provides the local syntax check instead.
- Parallel `conda run` calls contended on a Windows temporary activation file; static tools are run sequentially from the environment executable path.
- Ruff reports one pre-existing unused `math` import in `tests/test_v2_velocity.py`; it is outside this change and was not removed.

## Status
**Complete** - seed-42 recovery, artifact validation, resumable joint training, online holdout ablations, preliminary reporting, and local verification are complete. Full nuScenes/CUDA execution remains server-only.

---

# Task Plan: Unattended V3 Visual-Only Pipeline

## Goal
Provide one server command that tests V3, trains direct and fusion models for three seeds, evaluates frozen checkpoints on `stage1_test`, and produces scene-level paired analysis artifacts without CAN or manual intervention.

## Phases
- [x] Phase 1: Audit current V3 training, evaluation, and bootstrap interfaces
- [x] Phase 2: Add frozen-checkpoint stage1 holdout evaluation
- [x] Phase 3: Add strict multi-seed result analysis bundle
- [x] Phase 4: Add resumable one-command server orchestration and documentation
- [x] Phase 5: Add tests and run local verification

## Decisions Made
- Direct and fusion are paired by seed: 42, 3407, and 2026.
- Checkpoint selection remains on dev; `stage1_test` is read only after training.
- The primary comparison is final fusion versus strict direct visual baseline.
- Statistical resampling uses scenes, not adjacent frames, as independent units.
- Existing completed checkpoints are skipped and interrupted runs resume from `last.pth`.

## Errors Encountered
- No V3 stage1 checkpoint evaluator or end-to-end orchestration script exists yet; both are required before a scientifically valid one-command run can be given.
- Full nuScenes/CUDA training cannot run locally; server geometry/direct/fusion smoke results and local synthetic regression tests cover the executable paths before the formal run.

## Status
**Complete** - the unattended three-seed V3 training, independent holdout evaluation, and strict result-analysis pipeline is implemented and locally verified.
