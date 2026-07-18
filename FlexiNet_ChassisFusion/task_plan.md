# Task Plan: nuScenes-only V2 Training Corrections

## Goal
Make the existing plain-PyTorch V2 stages reproducible and stage-correct while keeping all optimization data nuScenes-only.

## Decisions
- Keep plain PyTorch; do not migrate to Lightning.
- Install only K-Dense `experimental-design` and `statistical-analysis`, pinned to `3f825ca`.
- Use effective batch size 8 through micro-batch 8 or micro-batch 4 with accumulation 2.
- Use seed 42 for screening and `42/3407/2026` only for final visual and joint reproduction.
- Preserve `ego_reference`; do not add CarSim, dynamics, slip labels, or pose-derived model inputs.

## Phases
- [x] Correct stage train/eval modes and neutral branch initialization
- [x] Add optimizer grouping, warmup-cosine scheduling, accumulation, resume, and baseline-safe checkpointing
- [x] Add train-only kinematic statistics, field ages, sequence-consistent augmentation, and scenario sampling
- [x] Update configs, data audit, server runner, multi-seed acceptance, and implementation report
- [x] Add unit/synthetic tests and run complete local verification

## Status
**Complete** - code, configs, documentation, CLI checks, compilation, and 92-test local regression verification passed. Real nuScenes, RAFT, CUDA smoke, and formal training remain server-only.

## Errors Encountered
- GitHub API directory listing was rate-limited; raw pinned skill files and the repository HEAD were still verified successfully.
- The first config patch assumed a geometry-only key that was absent; configs were then replaced from their actual contents.
- The first training-statistics CLI check exposed a missing project-root import path; the script now follows the existing script entrypoint pattern.
- One local YAML inspection one-liner had incorrect PowerShell quoting; repository files were unaffected.
- Windows exposes only an unconfigured WSL `bash.exe`, so local `bash -n` could not run; shell execution remains a server preflight check.
- PyTorch 2.6 rejected a native NumPy RNG tuple under safe checkpoint loading; NumPy RNG is now stored as tensors and primitive values.
