# Implementation Report

## Status

The `7de2aad` checkout now supports a self-contained legacy KD/VKD comparison.
No V2 source tree or `NUSCENES_COMPARISON_PROJECT_ROOT` is required for the
default profile.

## Delivered

- Backward-compatible causal image/chassis indexing and explicit scene selection.
- Optional per-batch chassis current index with unchanged checkpoint parameters.
- Legacy-first imports; V2 modules load only for an explicit full profile.
- Experiment-local deterministic scene split and sealed final-system partition.
- Automatic manifest creation and server preflight checks.
- Raw-image fallback when a compatible frame cache is absent.
- Existing-checkpoint quick evaluation for base, KD, and VKD.
- Resumable three-seed KD/VKD train/evaluate/analyze pipeline.
- Focused tests for causal indexing, scene selection, perturbation invariants,
  compatibility profiles, optimizer construction, and real checkpoint loading.

## Verification

Local verification used `D:/Miniconda3/envs/speedXY_py310/python.exe`:

```text
experiment + legacy focused tests: 45 passed
Python compileall:                 passed
legacy compatibility profile:     passed
full compatibility profile:       rejected as expected (V2 absent)
KD/VKD strict checkpoint load:     passed
KD/VKD synthetic forward/loss:     passed
CLI import/help checks:            passed
Shell line endings:                LF-only
```

Native `bash -n` and a real 32-sample nuScenes/CUDA smoke run could not be
executed on this Windows host because WSL, the dataset, and a visible server GPU
are unavailable. Both are explicit first-run server checks.

## Git Handoff

The experiment package is currently untracked and must be included in the
eventual commit:

```bash
git add FlexiNet_ChassisFusion/experiments/nuscenes_version_comparison
```

No commit or push was performed by this implementation.

## Result Boundary

The generated report supports open-loop nuScenes accuracy, synthetic sensor
fault trends, and logged high-excitation subgroup analysis. It does not verify
physical vehicle-parameter mismatch or closed-loop estimator-controller behavior.
