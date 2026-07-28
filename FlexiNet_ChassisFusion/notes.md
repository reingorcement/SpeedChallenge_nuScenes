# Notes: V3 Unattended Pipeline

## Current Evidence
- V3 unit tests pass under the project PyTorch environment.
- Geometry, direct, and fusion smoke paths pass on the server.
- `train_v3.py` selects checkpoints on `dev` and exports token-aligned predictions.
- `scripts/evaluate_v3_paired_bootstrap.py` supports scene-paired comparison but does not run checkpoint inference or build a full report.

## Required Closure
- Implemented frozen direct, fusion, and geometry evaluation on `stage1_test`.
- Paired seeds are fixed to 42, 3407, and 2026.
- Analysis produces descriptive metrics, scene-bootstrap confidence intervals, paired effect sizes, Markdown reports, CSV/JSON, and figures.
- The Bash entry point validates dependencies and paths, runs only V3 regression tests, skips completed stages, resumes incomplete stages, and logs all output.

## Interpretation Guardrails
- The primary comparison is fusion versus the direct pure-visual expert.
- A performance-improvement claim is emitted only when the scene-bootstrap 95% interval is below zero.
- Three seeds are reported as limited seed evidence; frames are never treated as independent experimental units.
- Smoke metrics are excluded from formal analysis.

## Throughput Diagnosis
- The first formal direct run used batch size 4, four workers, and about 3.9 GB of 32 GB GPU memory.
- At 2.35 iterations/s, one epoch was projected at roughly 3.9 hours.
- Direct pretraining currently loads three RAFT packages per sample even though the direct expert consumes RGB only.
- The loss implementation computes disabled geometry terms before multiplying them by zero.
- Optimization must preserve the full flow path for fusion and change execution only, not the model comparison.

## Throughput Changes
- Direct dataset samples now contain only RGB images and velocity labels.
- Direct model forward accepts the RGB-only contract and emits compatible diagnostic placeholders.
- Disabled loss terms are short-circuited before accessing flow or geometry tensors.
- Runtime controls separate direct batch size from fusion batch size; defaults are 16 and 4 respectively, with 8 DataLoader workers.

## Windows Seed-42 Server Target
- Target OS: native Windows, no WSL dependency.
- GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM; driver 572.83, CUDA capability reported as 12.8.
- Conda environment: `zy_speedest`.
- Dataset root: `D:\Workspaces\zy\nuscenes`.
- Repository target: `D:\Workspaces\zy\code\zhangyu\09_SpeedChallenge_nuScenes_worktrees\v3-visual-only`.
- Branch: `feature/v3-visual-only`.
- Validation scope: paired direct and fusion training for seed 42 only.
- Statistical scope: scene bootstrap conditional on seed 42; no claim of training-seed stability.

## Windows PowerShell 5.1 Quoting Failure
- Passing multiline Python source with embedded double quotes via `python -c $code` corrupts the quotes before `python.exe` receives the argument.
- The observed server syntax error was reproduced locally with the same PowerShell 5.1 call shape.
- Environment preflight and checkpoint completion checks must execute a normal Python helper file.
