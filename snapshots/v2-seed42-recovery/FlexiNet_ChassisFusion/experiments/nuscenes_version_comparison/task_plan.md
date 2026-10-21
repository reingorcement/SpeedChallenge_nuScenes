# Task Plan: nuScenes Version Comparison

## Goal
Build an isolated, runnable experiment package that compares the legacy KD/VKD models with the current V2 visual and visual-kinematic models on one causal nuScenes protocol.

## Phases
- [x] Phase 1: Inspect model, dataset, training, and evaluation contracts
- [x] Phase 2: Implement configs, adapters, perturbations, and runners
- [x] Phase 3: Add focused tests and server pipeline
- [x] Phase 4: Run verification and write the implementation report
- [x] Phase 5: Harden execution when hosted by the legacy checkout

## Key Questions
1. Which existing constructors and forward contracts can be reused without changing source modules?
2. How can legacy and V2 samples be aligned exactly by scene and center token?
3. Which perturbations can be applied consistently at the shared batch boundary?

## Decisions Made
- All experiment-specific source, configs, tests, and documentation stay in this directory.
- Existing legacy and V2 modules remain read-only dependencies.
- `final_system_test` remains inaccessible from comparison commands.
- V2 training delegates to `train_v2.py` so its staged constraints remain intact.
- Legacy training is implemented locally because the original CLI cannot consume the V2 manifest.
- Calibration perturbations are V2-only diagnostics; shared sensor variants operate on semantic fields.
- Legacy-host execution uses an explicit current-project source root because `7de2aad` predates V2.

## Errors Encountered
- Default `C:\\Python314\\python.exe` has no pytest: locate a repository environment or use `uv`.
- `bash -n` is unavailable because WSL is not installed on this Windows host: validate the script structurally and defer execution to the Linux server.
- Ruff and Flake8 are not installed in `speedXY_py310`; use compileall, pytest, CLI import checks, and manual review.
- A computed recursive cache cleanup was blocked by local policy; verified each cache path and removed only explicit experiment-local targets.

## Status
**Complete** - Compatibility hardening, runtime estimate, tests, and server handoff are ready.
