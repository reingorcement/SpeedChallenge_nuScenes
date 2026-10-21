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

---

# Task Plan: V3 Training Throughput Optimization

## Goal
Remove unnecessary flow I/O and inactive loss computation from direct pretraining, and expose safe runtime controls so the RTX 5090 is used efficiently without changing V3 predictions or experiment semantics.

## Phases
- [x] Phase 1: Diagnose the server throughput bottleneck
- [x] Phase 2: Implement a true RGB-only direct data/model/loss path
- [x] Phase 3: Add batch and DataLoader runtime controls to the V3 pipeline
- [x] Phase 4: Add regression tests and run V3-only verification

## Decisions Made
- Direct pretraining must not load or transfer RAFT packages.
- Zero-weight loss terms must not execute.
- Fusion retains the full visual flow and geometry path.
- Default server runtime starts at batch size 16 and 8 workers, with environment overrides.

## Errors Encountered
- The initial formal run used batch size 4 and loaded three unused high-resolution flow packages per direct sample, yielding about 2.35 iterations/s and an impractical multi-stage ETA.

## Status
**Complete** - 22 V3-only tests, Ruff, Python compilation, Bash syntax, forbidden-command scan, and diff validation all pass.

---

# Task Plan: Windows V3 Seed-42 Validation

## Goal
Run the existing V3 direct-then-fusion experiment for seed 42 from one resumable PowerShell command on the Windows RTX 5090 server, followed by clearly preliminary holdout analysis.

## Phases
- [x] Phase 1: Confirm the server GPU, Conda environment, dataset path, repository path, branch, and seed scope
- [x] Phase 2: Add an explicit preliminary single-seed analysis mode
- [x] Phase 3: Add a Windows-native resumable seed-42 pipeline
- [x] Phase 4: Document the exact GitHub-to-server setup and launch commands
- [x] Phase 5: Run focused tests, PowerShell syntax checks, and regression verification
- [x] Phase 6: Replace PowerShell 5.1-incompatible inline Python and verify the server hotfix
- [x] Phase 7: Make the RGB dataset transform pickleable by Windows DataLoader workers
- [x] Phase 8: Bound smoke dataset construction and expose fusion metadata progress

## Decisions Made
- Use the existing `feature/v3-visual-only` branch and preserve the Linux pipeline.
- Run direct pretraining before fusion because fusion initializes from the paired direct checkpoint.
- Treat seed 42 as preliminary validation; retain the three-seed requirement for formal claims.
- Pass `D:\Workspaces\zy\nuscenes` at launch time rather than hard-coding it in source.
- Save checkpoints after every epoch and resume only when the same run tag is supplied.

## Errors Encountered
- The first local V3 pytest command ran from the repository root, where the project-local `dataset` package is not importable. Rerun from `FlexiNet_ChassisFusion`, matching the PowerShell pipeline.
- Windows PowerShell 5.1 strips embedded double quotes from multiline arguments passed through `python -c`, producing an unterminated Python string during preflight. The same mechanism also affects checkpoint inspection; both checks must use a Python file instead of inline code.
- The local regression environment lacks `pandas`; V3 does not import `pandas` or `sklearn`, so both unrelated packages were removed from the server preflight requirement.
- Windows DataLoader `spawn` cannot pickle the local RGB conversion lambda stored in `NuScenesChassisFusionDataset.transform`; replace only that stored lambda with a module-level function.
- V3 smoke limits the training loop to two batches but still builds fusion metadata for the full train/dev sample set before showing a progress bar. Limit smoke scene construction and make full metadata preparation observable.

## Status
**Complete** - smoke construction is limited to two scenes per partition, full fusion metadata progress is visible, and 28 V3 tests plus static checks pass.

---

# Task Plan: Compact V3 Fusion Metadata

## Goal
Remove the Windows DataLoader spawn bottleneck by storing full-split fusion metadata in a fixed number of contiguous tensors without changing per-sample values or the direct-pretraining contract.

## Phases
- [x] Phase 1: Confirm the worker startup failure mode and existing V3 data contract
- [x] Phase 2: Replace per-sample tensor dictionaries with preallocated split-level tensors
- [x] Phase 3: Add value, layout, and Windows serialization regression tests
- [x] Phase 4: Run focused tests, static checks, and diff review

## Decisions Made
- Derive flow token pairs from the existing causal sample tokens instead of storing them twice.
- Preserve every public sample key, tensor shape, dtype, and computed value.
- Leave direct pretraining and checkpoint formats unchanged.

## Errors Encountered
- A parallel inspection command stopped when the optional project-config search returned no matches; rerun required checks independently.

## Status
**Complete** - fusion metadata now uses seven contiguous split-level tensors; 30 V3 tests, a real Windows spawn-worker test, Ruff, formatting, compilation, and diff checks pass.

---

# Task Plan: V3 Early Stopping

## Goal
Stop V3 training after sustained dev EPE stagnation while preserving checkpoint resume behavior, the fusion unfreeze phase, and unattended pipeline completion detection.

## Phases
- [x] Phase 1: Inspect V3 training/checkpoint contracts and the existing V2 early-stopping policy
- [x] Phase 2: Implement resumable dev-EPE early stopping in V3
- [x] Phase 3: Expose configuration and PowerShell runtime controls and update documentation
- [x] Phase 4: Add regression tests and run focused verification

## Decisions Made
- Monitor dev `epe_mps`; lower is better, with the repository-standard `1e-4` minimum improvement.
- Default patience to five epochs and allow zero to disable stopping.
- Do not accumulate fusion patience while the visual backbone is frozen.
- Persist the patience counter and terminal early-stop flag in `last.pth`.

## Errors Encountered
- None.

## Status
**Complete** - resumable V3 early stopping, pipeline completion handling, Windows runtime controls, and documentation are implemented; 32 V3 tests and all static checks pass.
