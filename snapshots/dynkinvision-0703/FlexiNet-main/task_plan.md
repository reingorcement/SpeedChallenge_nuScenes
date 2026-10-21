# Task Plan: Pure Vision nuScenes Training and Evaluation

## Goal
Train a pure-vision `FlexiNet-main` model initialized from the open-source nuImage L1 checkpoint on the same ordered nuScenes samples used by `FlexiNet_ChassisFusion`, then evaluate the trained checkpoint.

## Phases
- [x] Phase 1: Inspect current project structure
- [x] Phase 2: Add config, dataset, and utility modules
- [x] Phase 3: Update test entrypoint for nuScenes baseline evaluation
- [x] Phase 4: Verify imports, syntax, and sample alignment logic
- [x] Phase 5: Add ChassisFusion-aligned nuScenes training entrypoint
- [ ] Phase 6: Run training to produce `checkpoints_chassis_aligned/best.pth`
- [ ] Phase 7: Run evaluation with the trained checkpoint

## Key Questions
1. Can pure-vision evaluation use the same ordered sample construction as the fusion project?
2. Can the original checkpoint be loaded without changing the original FlexiNet architecture?
3. Can `FlexiNet-main` train on ChassisFusion-aligned nuScenes train/val samples while staying pure vision?

## Decisions Made
- Keep `FlexiNet` as a pure image-only model and load `checkpoint_epoch_100_nuimage_L1_best.pth`.
- Filter scenes by CAN bus availability only to align sample order with `FlexiNet_ChassisFusion`; do not feed CAN data to the model.
- Use validation sample index as the x-axis for comparison plots.
- Match `FlexiNet_ChassisFusion/configs/default.yaml` for nuScenes data roots: `/mnt/sda2/zy/nuscenes` and `/mnt/sda2/zy/nuscenes/can_bus`.
- Add `python train.py --mode nuscenes --config configs/train_chassis_aligned.yaml` for the new training path and keep the original train path as legacy behavior.
- Match ChassisFusion training knobs where practical: 10 epochs, batch size 16, `num_workers=1`, seed 42, SmoothL1 loss, AMP, gradient clipping, cosine LR.
- Use a low full-model fine-tuning LR of `3e-5`, matching ChassisFusion's visual backbone LR, because this pure-vision model starts from the nuImage L1 checkpoint.

## Errors Encountered
- Runtime import check failed because the current Python interpreter has no `torch` module installed. Syntax checks still passed with `py_compile`.
- Local Windows training cannot access the server-only `/mnt/sda2/zy/nuscenes` dataset path. This is expected because the code will run on the server.

## Status
**Currently in Phase 6** - Training code and configs are in place; next step is to run the aligned nuScenes training command on the server where `/mnt/sda2/zy/nuscenes` is mounted.
