# Task Plan: Pure Vision nuScenes Evaluation

## Goal
Add a clean validation-only nuScenes baseline pipeline in `FlexiNet-main` that evaluates the original pure-vision FlexiNet checkpoint on the same ordered validation samples used by `FlexiNet_ChassisFusion`.

## Phases
- [x] Phase 1: Inspect current project structure
- [x] Phase 2: Add config, dataset, and utility modules
- [x] Phase 3: Update test entrypoint for nuScenes baseline evaluation
- [x] Phase 4: Verify imports, syntax, and sample alignment logic

## Key Questions
1. Can pure-vision evaluation use the same ordered sample construction as the fusion project?
2. Can the original checkpoint be loaded without changing the original FlexiNet architecture?

## Decisions Made
- Keep `FlexiNet` as a pure image-only model and load `checkpoint_epoch_100_nuimage_L1_best.pth`.
- Filter scenes by CAN bus availability only to align sample order with `FlexiNet_ChassisFusion`; do not feed CAN data to the model.
- Use validation sample index as the x-axis for comparison plots.
- Match `FlexiNet_ChassisFusion/configs/default.yaml` for nuScenes data roots: `/mnt/sda2/zy/nuscenes` and `/mnt/sda2/zy/nuscenes/can_bus`.

## Errors Encountered
- Runtime import check failed because the current Python interpreter has no `torch` module installed. Syntax checks still passed with `py_compile`.

## Status
**Completed** - Static syntax verification passed; full runtime validation needs a Python environment with PyTorch and nuScenes data.
