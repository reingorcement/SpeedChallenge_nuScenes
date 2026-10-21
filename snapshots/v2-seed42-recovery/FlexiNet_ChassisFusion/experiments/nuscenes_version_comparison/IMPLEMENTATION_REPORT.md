# Implementation Report

## Status

The isolated comparison package is implemented. No existing legacy or V2 source
file was modified.

## Delivered

- Shared causal five-frame RGB and one-second/20 Hz CAN protocol.
- Manifest-aware legacy and V2 dataset adapters with a hard sealed-partition gate.
- Legacy KD/VKD training and delegation to the existing staged V2 trainer.
- Unified evaluation rows for base, geometry, KD, VKD, V2 visual, and V2 joint models.
- Shared sensor faults and V2-only focal/height/pitch calibration diagnostics.
- Boston-only training statistics and Singapore-only domain-shift evaluation.
- Train-derived scenario thresholds, exact token/target alignment, three-seed
  aggregation, scene-level paired bootstrap, stress degradation, and bad cases.
- Resumable Linux server pipeline and focused synthetic tests.
- Legacy-host bootstrap: the experiment folder may live under `7de2aad` while
  `NUSCENES_COMPARISON_PROJECT_ROOT` supplies the current V2 source tree.
- Compatibility validation checks required files and causal/V2 interface markers
  before data preparation or training starts.

## Verification

Local verification used `D:/Miniconda3/envs/speedXY_py310/python.exe`:

```text
experiment package tests: 21 passed
legacy repository tests:   22 passed
V2 repository tests:       61 passed
total:                     104 passed
```

All Python files compile, all three CLIs load, and all four learned adapters build.
The Linux shell syntax check and nuScenes smoke/full runs were not executable on
this Windows host because WSL, the nuScenes data, and the server GPU are absent.

The legacy-host test copies the package into a temporary legacy-shaped directory
and launches all three CLIs while sourcing model dependencies from the current
project root.

## Runtime

The existing server log projects one V2 epoch to approximately 1.5-1.8 hours
including dev evaluation. The three-seed main experiment is approximately 5-7
days on one GPU; the default pipeline including Boston-to-Singapore training is
approximately 8-12 days. See `RUNTIME_ESTIMATE.md` for assumptions and parallel
GPU ranges.

## Result Boundary

The generated report supports open-loop nuScenes accuracy, sensor perturbation,
calibration sensitivity, logged high-excitation scenarios, and location shift.
It does not establish physical vehicle-parameter mismatch robustness or closed-loop
estimator-controller behavior. Current V2 has no dynamics-student branch.
