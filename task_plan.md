# Task Plan: FlexiNet Chassis Fusion Project

## Goal
Create a standalone FlexiNet-based project that fine-tunes ego-speed estimation on nuScenes with CAN bus chassis kinematic fusion.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect source project and define reusable components
- [x] Phase 3: Build standalone project files
- [x] Phase 4: Verify imports, syntax, and basic dataset/model smoke paths
- [x] Phase 5: Document usage and deliver summary

## Key Questions
1. How should CAN bus signals align to CAM_FRONT image timestamps?
2. How should the official FlexiNet checkpoint load while replacing the final head?
3. How can the project run in the user's `vision_est` environment with minimal assumptions?

## Decisions Made
- Project directory: `FlexiNet_ChassisFusion`, to avoid modifying the downloaded `FlexiNet` source.
- Fusion strategy: late feature fusion plus optional residual correction, using official FlexiNet visual blocks as the backbone.
- Input setting: default to 13 CAM_FRONT frames to match official FlexiNet usage.
- CAN assumption: `can_bus` exists under the nuScenes dataroot as recommended by the official devkit.

## Errors Encountered
- Smoke test in the current workspace failed because the base Python environment does not have `torch` installed. This should be re-run inside `vision_est`.

## Status
**Completed** - Standalone project generated, syntax checked, and run instructions documented.

## Current Update: Adapt FlexiNet_ChassisFusion to full nuScenes trainval

### Goal
Update the dataset configuration and loader so `FlexiNet_ChassisFusion` uses `/mnt/sda2/zy/nuscenes`, official nuScenes train/val splits, and the existing root-level `can_bus` layout.

### Phases
- [x] Phase 1: Inspect current config and dataset assumptions
- [x] Phase 2: Update config paths and split-aware dataset construction
- [x] Phase 3: Add dataset integrity checks and train/test sample-count reporting
- [x] Phase 4: Run syntax verification and summarize changes

### Decisions Made
- Keep image loading through `NuScenes.get_sample_data_path(token)` instead of manually scanning `samples/CAM_FRONT`.
- Keep the existing 16-dimensional CAN/chassis feature vector for this update.
- Use `nuscenes.utils.splits.create_splits_scenes()` for `v1.0-trainval` and `v1.0-mini`.

### Status
**Completed** - Config, dataset split logic, and entrypoint diagnostics updated for full nuScenes trainval.

## Current Update: Add Evaluation Metrics and Visualization

### Goal
Add training/evaluation artifacts for fine-tuning the pretrained FlexiNet backbone: metric histories, prediction tables, and diagnostic plots.

### Phases
- [x] Phase 1: Inspect current training, testing, metrics, and dependency setup
- [x] Phase 2: Add reusable evaluation and plotting utilities
- [x] Phase 3: Wire train/test entrypoints to save metrics, predictions, and figures
- [x] Phase 4: Run syntax verification and summarize generated artifacts

### Decisions Made
- Keep using the existing pretrained FlexiNet checkpoint path from `FlexiNet-main/pretrained_models`.
- Report both model error and `base_speed` error because this is residual chassis fusion.
- Save artifacts under configurable run directories instead of only printing terminal metrics.

### Status
**Completed** - Training and testing now save metric tables, prediction CSVs, and diagnostic plots.
