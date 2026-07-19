# Task Plan: Self-Contained nuScenes KD/VKD Comparison

## Goal
Make the current `7de2aad` checkout directly runnable on a Linux server for causal KD/VKD comparison after a normal Git upload, without a second V2 source tree.

## Phases
- [x] Phase 1: Inspect current legacy/V2 dependencies and server entrypoints
- [x] Phase 2: Add backward-compatible causal dataset/model interfaces
- [x] Phase 3: Make the experiment package legacy-first and self-contained
- [x] Phase 4: Add preflight, server wrappers, and focused tests
- [x] Phase 5: Run verification and update the implementation report

## Key Questions
1. Which existing constructors and forward contracts can be reused without changing source modules?
2. How can legacy and V2 samples be aligned exactly by scene and center token?
3. Which perturbations can be applied consistently at the shared batch boundary?

## Decisions Made
- Legacy `base_velocity`, `legacy_kd`, and `legacy_vkd` are the default comparison arms.
- The current checkout is the only required source tree; no source-root environment variable is required.
- V2 remains optional and is imported only when a full V2 source tree is actually present.
- Frame caches are optional for the quick legacy path; raw nuScenes images are the fallback.
- `final_system_test` remains inaccessible from comparison commands.
- Legacy training is implemented locally because the original CLI cannot consume the V2 manifest.
- Shared sensor variants operate on semantic fields and must never modify targets.

## Errors Encountered
- `npx -p bash-parser` did not expose the package to `require()` on this Windows npm setup: retain compile/tests plus structural review and defer native `bash -n` to Linux.
- PowerShell blocked `npx.ps1` while checking Bash syntax: use `npx.cmd` without changing execution policy.
- `bashlex` cannot parse Bash's `${BASH_SOURCE[0]}` syntax: parse the scripts in-memory with that expression simplified, keep native `bash -n` as the Linux acceptance check.
- The first inline checkpoint forward-smoke command had mismatched parentheses: replaced it with a readable stdin Python script.
- Legacy adapter and criterion construction still passed newer V2/physics arguments that do not exist in `7de2aad`: removed them from the legacy execution path.
- The experiment trainer targeted a newer `_loss_from_batch` signature and passed one extra model argument: aligned both train and evaluation calls with the current checkout.
- Full legacy training reused an optimizer helper that assumed every model had a visual backbone: made parameter groups conditional and added a KD-only regression test.
- Geometry-prior unit test constructs the dataset without `__init__`, so `online_mode` was absent: defaulted the helper to offline mode when the attribute is missing.
- Default `C:\\Python314\\python.exe` has no pytest: locate a repository environment or use `uv`.
- `bash -n` is unavailable because WSL is not installed on this Windows host: validate the script structurally and defer execution to the Linux server.
- Ruff and Flake8 are not installed in `speedXY_py310`; use compileall, pytest, CLI import checks, and manual review.
- A computed recursive cache cleanup was blocked by local policy; verified each cache path and removed only explicit experiment-local targets.

## Status
**Complete** - Self-contained legacy execution, server handoff, and local verification are finished.
