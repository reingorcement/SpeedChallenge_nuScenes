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
