# Notes: FlexiNet Chassis Fusion Project

## Source Project Findings

- Original FlexiNet visual model lives in `FlexiNet/model/FlexiNet.py`.
- Original modules live in `FlexiNet/model/modules.py`.
- The original model converts input `[B, T, C, H, W]` to `[B, C, T, H, W]`.
- The feature immediately before the original `fc` head is a 256-dimensional pooled visual feature.
- Existing nuScenes-mini dataset uses CAM_FRONT sequences and derives labels from ego pose displacement.

## Design Notes

- The new model should preserve the visual blocks and replace only the regression stage.
- CAN bus feature lookup should be timestamp-aligned and should not use future samples by default.
- The residual output should be:
  - `prediction = base_speed + delta`
  - `base_speed` is selected from CAN vehicle speed / odometry / wheel speed or fallback history.
- The non-residual output should be:
  - `prediction = fusion_head(concat(visual_feature, chassis_feature))`

## Verification Targets

- Python files compile with `python -m compileall FlexiNet_ChassisFusion`.
- Model dummy forward is implemented in `smoke_model.py`, but could not be executed in the current base Python environment because `torch` is unavailable.
- Dataset construction should fail clearly if CAN bus files are missing, unless pose fallback is enabled.

## Implementation Summary

- `model/flexinet_backbone.py`: FlexiNet visual blocks without the original `fc` head.
- `model/fusion_model.py`: residual/direct chassis fusion model.
- `dataset/nuscenes_chassis.py`: CAM_FRONT sequence dataset plus official CAN bus JSON-file reader.
- `train.py`: two-parameter-group fine-tuning script with frozen-backbone support.
- `test.py`: evaluation script with MAE/RMSE.
- `smoke_model.py`: no-dataset model forward smoke test.

## Full nuScenes Trainval Adaptation

- Data root is now expected at `/mnt/sda2/zy/nuscenes`.
- CAN bus JSON files are expected at `/mnt/sda2/zy/nuscenes/can_bus`, using names such as `scene-0001_pose.json`.
- `v1.0-trainval` now uses official nuScenes `train` and `val` scene-name splits from `create_splits_scenes()`.
- Image loading remains delegated to `NuScenes.get_sample_data_path(token)`, so `samples/CAM_FRONT/*.jpg` paths come from nuScenes metadata rather than manual directory scanning.
- Dataset construction skips scenes without CAN files when CAN bus is configured, and raises a clear error if no samples remain.
- Syntax verification passed with `python -m py_compile` on the dataset, train, and test entrypoints.

## Evaluation and Visualization Update

- Training now records per-epoch `train_loss`, `val_loss`, `val_mae_mps`, `val_rmse_mps`, `base_mae_mps`, and gain against the CAN `base_speed` baseline.
- Training artifacts are written under `train.log_dir`, currently `runs/FlexiNet_ChassisFusion`.
- Best and last validation predictions are exported with metrics JSON, prediction CSV, pred-vs-target, residual, error histogram, delta-v histogram, and speed-bin MAE plots.
- `test.py` now accepts `--output_dir` and writes the same evaluation artifact bundle for any checkpoint.
