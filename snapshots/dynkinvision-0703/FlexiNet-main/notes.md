# Notes: Pure Vision nuScenes Evaluation

## Findings
- `FlexiNet-main` originally only supported KITTI and nuImages testing in `test.py`.
- `FlexiNet-main` originally only supported KITTI and nuImages training in `train.py`; it could not train on the ChassisFusion-aligned nuScenes samples.
- The pure-vision comparison must load `pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth` into the original `FlexiNet` architecture.
- The validation x-axis must be the ordered validation sample index from `DataLoader(shuffle=False)`.
- To align with `FlexiNet_ChassisFusion`, the pure-vision dataset filters validation scenes by CAN bus availability but does not use CAN features as model input.
- The default local Python interpreter does not currently have `torch` installed, so runtime loading could not be verified in this environment. Syntax checks passed.
- `configs/test.yaml` uses the same nuScenes root as `FlexiNet_ChassisFusion/configs/default.yaml`: `/mnt/sda2/zy/nuscenes` and `/mnt/sda2/zy/nuscenes/can_bus`. The split remains `val` because this project is validation-only.
- Added `configs/train_chassis_aligned.yaml` for pure-vision fine-tuning on ChassisFusion-aligned nuScenes train/val samples.
- Added `configs/test_chassis_aligned.yaml` to evaluate `checkpoints_chassis_aligned/best.pth`.
- The new training command is:
  `python train.py --mode nuscenes --config configs/train_chassis_aligned.yaml`
- The trained checkpoint evaluation command is:
  `python test.py --mode nuscenes --config configs/test_chassis_aligned.yaml`
- New checkpoints are saved as `checkpoints_chassis_aligned/last.pth` and `checkpoints_chassis_aligned/best.pth`.
- New training artifacts are saved under `runs/FlexiNet-main-chassis-aligned` and copied to `results_git/FlexiNet-main/chassis_aligned_train`.
- Local smoke test in `vision_est` verified the train CLI, strict pretrained checkpoint loading, and a random forward pass. Full training is expected to run only on the server because `/mnt/sda2/zy/nuscenes` is not mounted locally.

## Expected Alignment Check
- Run pure-vision eval and confirm the sample count matches the fusion validation result, expected `33148` for the current fusion output.
- Compare the first `scene_name`, `center_token`, and `center_timestamp` rows in both prediction CSVs.
