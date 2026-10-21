# Notes: FlexiNet ChassisFusion Report

## Sources

### README
- Project is a nuScenes ego-speed estimation fusion project.
- Main idea: FlexiNet visual temporal feature + CAN bus / chassis kinematic features.
- Default fusion: `predicted_speed = base_speed + learned_delta_v`.
- Report should emphasize fusion, not FlexiNet internal architecture.

## Findings

### Config: `FlexiNet_ChassisFusion/configs/default.yaml`
- Data: `version=v1.0-trainval`, `split=train`, `seq_len=13`, image size `64x64`, camera `CAM_FRONT`, CAN root `/mnt/sda2/zy/nuscenes/can_bus`, `allow_pose_fallback=false`, `image_cache_size=4096`.
- Model: pretrained FlexiNet checkpoint `../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth`, `fusion_mode=residual`, `chassis_feature_dim=16`, `hidden_dim=128`, `dropout=0.3`, `freeze_backbone=true`.
- Train: batch size 16, workers 1, epochs 10, `learning_rate_head=0.001`, `learning_rate_backbone=0.00003`, weight decay 0.0001, AMP true, seed 42, SmoothL1 loss, grad clip 2.0.

### Dataset: `dataset/nuscenes_chassis.py`
- `NuScenesChassisFusionDataset` builds sliding windows of camera sample_data tokens per scene. Each sample uses 13 front-camera frames; center index is `seq_len // 2`.
- Visual tensor after transform: `[T, 1, H, W]`, default `[13, 1, 64, 64]`. Images are resized, converted to grayscale, tensorized, and normalized with mean/std 0.5.
- Label speed is computed from ego pose displacement between center camera frame and its next camera frame: distance in XY divided by timestamp difference.
- CAN/chassis data alignment uses `_select_latest_before`: chooses the latest CAN record with `utime <= center_timestamp`, so it does not use future CAN information.
- Returned sample keys: `images`, `chassis`, `base_speed`, `speed`, `scene_name`, `center_token`, `center_timestamp`.
- Chassis feature vector has 16 dimensions:
  0 vehicle_speed, 1 odom_speed, 2 pose_speed, 3 yaw_rate, 4 throttle, 5 brake, 6 steering, 7 steering_feedback, 8 longitudinal_accel, 9 transversal_accel, 10 wheel_speed_mean, 11 wheel_speed_std, 12 accel_x, 13 accel_y, 14 rotation_rate_z, 15 base_speed.
- Unit/normalization details: vehicle_speed and odom_speed divided by 3.6; yaw_rate converted to radians; throttle divided by 1000; brake divided by 126; steering divided by 780; steering_feedback divided by 7.7; wheel speeds divided by 1000.
- Base speed priority in code: if `vehicle_monitor` exists, use vehicle_speed; if still zero and wheel message exists, use odom_speed; if still zero and pose exists, use pose_speed; otherwise 0.
- Pose fallback exists but default config disables it.

### Model: `model/flexinet_backbone.py` and `model/fusion_model.py`
- `FlexiNetBackbone.forward_features` accepts `[B,T,C,H,W]`, permutes to `[B,C,T,H,W]`, passes through the FlexiNet visual modules, then global-pools to `[B,256]`.
- `FlexiNetChassisFusion` contains:
  - `backbone`: FlexiNet feature extractor.
  - `chassis_encoder`: MLP `16 -> hidden_dim -> chassis_feature_dim`, default `16 -> 128 -> 16`.
  - `residual_speed_head`: MLP over concatenated `[256 + chassis_feature_dim]`, predicts `delta_v`.
  - `direct_speed_head`: same form, predicts speed directly.
- Forward logic:
  - visual feature `[B,256]`.
  - chassis embedding `[B,16]`.
  - concat `[B,272]`.
  - residual mode: `pred = base_speed + residual_speed_head(fused)`.
  - direct mode: `pred = direct_speed_head(fused)`.
- `load_flexinet_backbone` filters pretrained checkpoint keys to visual backbone modules and ignores old regression heads.
- `freeze_backbone` disables gradients for the visual backbone; default config uses this.

### Training and Test
- `train.py` builds train/val datasets using splits `train` and `val`, loads pretrained FlexiNet weights, optionally freezes backbone, optimizes head/backbone parameter groups with AdamW and CosineAnnealingLR.
- Loss options: L1, MSE, SmoothL1/Huber; default is SmoothL1 with beta=1.0.
- Uses AMP only on CUDA; gradient clipping uses max norm 2.0.
- Saves `last.pth` every epoch and `best.pth` when validation loss improves.
- Writes `training_history.csv/json`, training curves, last/best evaluation artifacts, and compact git-result copies.
- `test.py` evaluates `checkpoints/best.pth` on validation split and writes outputs to `outputs/FlexiNet_ChassisFusion/eval` and `results_git/FlexiNet_ChassisFusion/test_eval`.

### Metrics and Result Artifacts
- `compute_metrics` returns MAE, RMSE, R2, km/h conversions, base_speed baseline metrics, and gains vs baseline.
- Predictions CSV columns include `target_speed_mps`, `pred_speed_mps`, `base_speed_mps`, model/base errors, and `delta_v_mps`.
- Test metric JSON from `outputs/.../val_metrics.json` and `results_git/.../test_eval/val_metrics.json` match exactly.

### Current Test Results
- Samples in `outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv`: 33,148.
- Target speed range: about 0 to 13.3066 m/s; mean 5.3050 m/s.
- Model: MAE 0.089532 m/s = 0.322316 km/h; RMSE 0.124575 m/s = 0.448469 km/h; R2 0.998744.
- Base speed baseline: MAE 0.199453 m/s = 0.718032 km/h; RMSE 0.281271 m/s = 1.012576 km/h.
- Improvement: MAE gain 0.109921 m/s = 0.395716 km/h; RMSE gain 0.156696 m/s = 0.564107 km/h.
- Relative reduction: MAE reduced by about 55.1%; RMSE reduced by about 55.7%.
- Baseline R2 recomputed from prediction CSV: about 0.993598. This is not written by `compute_metrics` for the baseline, so label it as supplemental analysis if used.
- Median abs error: model 0.067112 m/s; base 0.151197 m/s.
- Delta correction distribution from CSV: mean 0.094710 m/s, std 0.222557 m/s, min -1.220696 m/s, max 1.164439 m/s.
- Speed-bin MAE:
  - 0-2 m/s: n=7709, model 0.03394, base 0.08905, gain 0.05511.
  - 2-5 m/s: n=6808, model 0.10769, base 0.26379, gain 0.15611.
  - 5-10 m/s: n=15771, model 0.10563, base 0.21932, gain 0.11369.
  - 10-15 m/s: n=2860, model 0.10739, base 0.23431, gain 0.12692.
  - 15+ m/s: n=0 in current validation/test output.

### Figures Available for Report
- `results_git/FlexiNet_ChassisFusion/test_eval/val_true_vs_estimated_comparison.svg`: true vs estimated overview.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_true_vs_estimated_zoom_first1000.svg`: first 1000-sample zoom.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_pred_vs_target.png`: target-speed vs predicted/base scatter with ideal line.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_residual_vs_speed.png`: prediction residual vs target speed.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_abs_error_histogram.png`: model absolute error histogram.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_delta_v_histogram.png`: learned correction `delta_v` distribution.
- `results_git/FlexiNet_ChassisFusion/test_eval/val_speed_bin_mae.png`: model vs base MAE by target speed bin.
- `results_git/FlexiNet_ChassisFusion/train_val_loss_curve.png`: train/val loss over epochs.
- `results_git/FlexiNet_ChassisFusion/val_mae_rmse_curve.png`: validation MAE/RMSE and base MAE over epochs.

### README vs Code Consistency
- Main README claims match code for default residual fusion, feature dimension, sequence length, tensor shape, label construction, no-future CAN alignment, training/test outputs, and metric values.
- Minor caveat: README mentions `base_speed` priority as `vehicle_speed -> odom_speed -> pose_speed -> 0`; code implements that priority only when the current candidate value is exactly 0.0, so a real zero vehicle speed can trigger fallback to odom/pose if those are nonzero.
- Important reporting caveat: CAN bus feature selection is causal relative to the center timestamp, but the visual sequence is a centered 13-frame window with 6 frames after the center frame. Therefore the current setup should be described as "center-frame-neighborhood speed estimation" rather than strictly online causal inference.
