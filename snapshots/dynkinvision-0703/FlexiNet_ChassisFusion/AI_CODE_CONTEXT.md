# FlexiNet_ChassisFusion AI Code Context

这份文档是给另一个 AI 阅读本项目代码时使用的上下文说明。请优先依据本文档理解当前分支，不要只依据旧 `README.md` 判断任务形态。

## 一句话概述

`FlexiNet_ChassisFusion` 是一个面向 nuScenes ego vehicle velocity estimation 的多模态融合项目。当前分支的主任务已经从旧版“标量车速 speed 估计”改造成 ego/body frame 下的二维速度估计：

```text
target_velocity = [v_x, v_y]
base_velocity   = [base_vx, base_vy]
pred_velocity   = base_velocity + delta_velocity
```

其中：

- `v_x`：车辆前向速度，单位 m/s。
- `v_y`：车辆左向速度，单位 m/s。
- 默认 `base_velocity_source: zero_lateral`，即 `base_velocity = [base_vx, 0]`。
- 可选 `base_velocity_source: eovm_ekf`，使用预计算 EOVM-EKF 结果作为二维底盘先验。

当前代码主线是二维速度，不是只预测一个标量 `speed`。代码中仍保留 `speed` 和 `base_speed` 字段，主要用于兼容旧逻辑和计算派生指标。

## 当前默认配置

主要配置文件：

```text
configs/default.yaml
```

关键默认项：

```yaml
data:
  target_mode: velocity_xy
  chassis_mode: fair_chassis
  chassis_window_sec: 1.0
  chassis_hz: 20.0
  online_mode: false
  image_mode: gray
  flow_mode: none
  geometry_mode: none
  base_velocity_source: zero_lateral
  frame_cache_root: /media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray

model:
  fusion_mode: residual
  output_dim: 2
  chassis_feature_dim: 8
  chassis_sequence_model: gru
  hidden_dim: 96
  freeze_backbone: true

train:
  loss: velocity_smoothl1
  sigma_vx: 5.0
  sigma_vy: 0.5
  speed_loss_weight: 0.1
```

默认输入是 `CAM_FRONT` 的 13 帧图像序列，灰度、64x64。底盘输入默认是 `fair_chassis` 的 CAN/底盘时间序列。模型默认冻结 FlexiNet 视觉 backbone，只训练底盘序列编码器和融合回归头。

## 推荐阅读顺序

如果你要理解或修改代码，建议按这个顺序阅读：

```text
1. AI_CODE_CONTEXT.md
2. 速度XY改造使用说明.md
3. configs/default.yaml
4. train.py
5. dataset/nuscenes_chassis.py
6. model/fusion_model.py
7. utils/evaluation.py
8. test.py
9. analysis/README.md
10. eovm_ekf/README.md
```

注意：根目录 `README.md` 结构说明很完整，但其中大量内容来自旧版标量速度任务。当前分支应以 `速度XY改造使用说明.md` 和本文件为准。

## 项目结构

```text
FlexiNet_ChassisFusion/
  configs/
    default.yaml
    velocity_xy_fair_gray.yaml
    velocity_xy_fair_flow.yaml
    velocity_xy_fair_rgb_flow_geom.yaml
    velocity_xy_pose_assisted.yaml
    smoke_velocity_xy_fair_gray.yaml

  dataset/
    nuscenes_chassis.py

  model/
    modules.py
    flexinet_backbone.py
    fusion_model.py

  utils/
    evaluation.py
    metrics.py
    seed.py

  scripts/
    precompute_frames.py
    precompute_optical_flow.py
    audit_eovm_inputs.py
    precompute_eovm_base_velocity.py
    profile_training_step.py

  analysis/
    inference.py
    postprocess.py
    evaluate_checkpoint.py
    compare_checkpoints.py

  eovm_ekf/
    eovm_ekf/
    tests/

  train.py
  test.py
  smoke_model.py
```

## 数据流

核心数据集文件：

```text
dataset/nuscenes_chassis.py
```

核心类：

```text
NuScenesCanBusAdapter
NuScenesChassisFusionDataset
```

单个样本当前返回的关键字段：

```python
{
    "images": Tensor,          # [T, C, H, W]
    "chassis": Tensor,         # [K, C_chassis]
    "base_velocity": Tensor,   # [2]
    "velocity": Tensor,        # [2]
    "flow": Tensor,            # [T-1, 2, H, W] 或 empty Tensor
    "geometry": Tensor,        # geometry prior 或 empty Tensor

    # 兼容旧代码和派生指标
    "base_speed": Tensor,      # [1]
    "speed": Tensor,           # [1]

    "scene_name": str,
    "center_token": str,
    "center_timestamp": Tensor,
}
```

### 二维速度标签

`velocity` 标签由 nuScenes `ego_pose` 生成。当前实现使用中心差分：

```text
p_prev = previous ego pose position
p_next = next ego pose position
dt = t_next - t_prev
v_global = (p_next - p_prev) / dt
v_ego = rotate_global_velocity_to_center_ego_frame(v_global)
target_velocity = [v_ego_x, v_ego_y]
```

没有 `prev` 或 `next` 的边界帧会被跳过或报错，不应把边界帧速度填成 0 作为监督。

### 底盘特征模式

当前支持两种 `chassis_mode`：

```text
fair_chassis
pose_assisted
```

`fair_chassis` 是主实验模式，只使用更接近真实车辆可部署的 CAN/底盘信号，例如：

```text
vehicle_speed
odom_speed
yaw_rate
throttle
brake
steering
steering_feedback
longitudinal_accel
transversal_accel
four wheel speeds
wheel_speed_mean
wheel_speed_std
base_vx
```

`pose_assisted` 会额外加入 nuScenes pose 派生量，例如 `pose_speed`、`pose_vel_x`、`pose_vel_y`、`pose_accel_x`、`pose_accel_y`、`pose_rotation_rate_z`。它只能作为上限实验，不建议作为可部署主结论，因为标签本身也来自 ego pose。

### 底盘时间序列

当前数据集可以围绕中心图像时间构造 CAN 时间窗口：

```yaml
data:
  chassis_window_sec: 1.0
  chassis_hz: 20.0
  online_mode: false
```

模型侧由 `ChassisSequenceEncoder` 编码 `[K, C]` 的底盘序列。默认 `chassis_sequence_model: gru`。

`online_mode: false` 表示离线训练，窗口可覆盖中心时间前后；`online_mode: true` 表示更接近在线/因果推理，只使用历史窗口。

## 模型结构

核心模型文件：

```text
model/fusion_model.py
```

核心类：

```text
FlexiNetChassisFusion
ChassisSequenceEncoder
FlowEncoder
GeometryEncoder
FusionHead
```

默认 forward 逻辑：

```text
images
  -> FlexiNetBackbone.forward_features()
  -> visual_feat [B, 256]

chassis
  -> ChassisSequenceEncoder
  -> chassis_feat [B, chassis_feature_dim]

optional flow
  -> FlowEncoder
  -> flow_feat

optional geometry
  -> GeometryEncoder
  -> geometry_feat

concat all features
  -> FusionHead
  -> delta_velocity [B, 2]

residual mode:
  pred_velocity = base_velocity + delta_velocity

direct mode:
  pred_velocity = direct_speed_head(fused_feature)
```

默认推荐使用 `fusion_mode: residual`，因为 `base_velocity` 是强底盘先验，网络主要学习相对先验的修正量。`direct` 模式可用于消融实验。

视觉 backbone 来自 FlexiNet：

```text
model/flexinet_backbone.py
model/modules.py
```

当前项目把 FlexiNet 当作视觉时序特征提取器，输出 256 维视觉特征。原始 FlexiNet 的旧速度回归头不是当前主线。

## 训练逻辑

训练入口：

```text
train.py
```

训练主线：

```text
configs/*.yaml
  -> train.py
  -> NuScenesChassisFusionDataset
  -> DataLoader batch
  -> FlexiNetChassisFusion
  -> VelocitySmoothL1Loss(pred_velocity, target_velocity)
  -> optimizer.step()
  -> run_eval()
  -> checkpoints + runs + results_git
```

当前二维损失：

```python
loss = VelocitySmoothL1Loss(pred_velocity, target_velocity)
```

损失支持按轴归一化：

```yaml
train:
  sigma_vx: 5.0
  sigma_vy: 0.5
  speed_loss_weight: 0.1
```

这样做是因为 `v_y` 的数值通常远小于 `v_x`，如果直接等权用原始数值训练，侧向速度误差容易被纵向速度误差淹没。

## 评估逻辑

主要评估文件：

```text
utils/evaluation.py
test.py
analysis/evaluate_checkpoint.py
analysis/compare_checkpoints.py
```

二维速度主指标应优先看：

```text
mae_vx_mps
rmse_vx_mps
mae_vy_mps
rmse_vy_mps
vector_endpoint_error_mps
speed_magnitude_mae_mps
base_mae_vx_mps
base_mae_vy_mps
gain_vs_base_vx_mps
gain_vs_base_vy_mps
turning_vector_endpoint_error_mps
high_lateral_mae_vy_mps
```

预测 CSV 中的关键字段通常包括：

```text
target_vx_mps
target_vy_mps
pred_vx_mps
pred_vy_mps
base_vx_mps
base_vy_mps
delta_vx_mps
delta_vy_mps
endpoint_error_mps
```

`analysis/` 目录用于训练后分析，不重新训练模型。它会读取 checkpoint 或已有预测文件，生成 bad cases、per-scene、per-speed-bin 和 Markdown 报告。

## 常用命令

在本地 Windows 环境如果使用 `speedXY_py310`：

```powershell
conda run -n speedXY_py310 python -m pytest tests/test_velocity_xy_pipeline.py eovm_ekf/tests -q
conda run -n speedXY_py310 python -m compileall dataset model utils train.py test.py smoke_model.py scripts
conda run -n speedXY_py310 python smoke_model.py
conda run -n speedXY_py310 python smoke_model.py --with_flow
```

服务器训练示例：

```bash
python train.py --config configs/velocity_xy_fair_gray.yaml
```

评估 checkpoint：

```bash
python test.py \
  --config configs/velocity_xy_fair_gray.yaml \
  --checkpoint checkpoints/velocity_xy_fair_gray/best.pth
```

训练后详细分析：

```bash
python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_fair_gray.yaml \
  --checkpoint checkpoints/velocity_xy_fair_gray/best.pth \
  --output_dir outputs/velocity_xy_fair_gray/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/fair_gray/eval_best
```

比较 best 和 last：

```bash
python analysis/compare_checkpoints.py \
  --config configs/velocity_xy_fair_gray.yaml \
  --checkpoint_a checkpoints/velocity_xy_fair_gray/best.pth \
  --checkpoint_b checkpoints/velocity_xy_fair_gray/last.pth \
  --output_dir_a outputs/velocity_xy_fair_gray/eval_best \
  --output_dir_b outputs/velocity_xy_fair_gray/eval_last \
  --report_dir outputs/velocity_xy_fair_gray/eval_compare
```

## 预计算缓存

完整 nuScenes 训练直接读原始 JPEG 会很慢。推荐先预计算图像帧缓存：

```bash
python scripts/precompute_frames.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split trainval \
  --cam_channel CAM_FRONT \
  --height 64 --width 64 \
  --image_mode gray \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

然后在配置中设置：

```yaml
data:
  frame_cache_root: /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

RGB 实验需要单独生成 RGB 缓存，缓存的 `image_mode`、尺寸、相机通道、nuScenes 版本必须和配置一致。

光流实验需要先预计算光流：

```bash
python scripts/precompute_optical_flow.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split train \
  --cam_channel CAM_FRONT \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_flow/CAM_FRONT \
  --width 64 \
  --height 64
```

## 实验配置含义

推荐实验顺序：

```text
configs/velocity_xy_fair_gray.yaml
  灰度图像 + fair_chassis，主 baseline。

configs/velocity_xy_fair_flow.yaml
  灰度图像 + 预计算光流 + fair_chassis，用于判断光流是否改善 v_y 和转弯场景。

configs/velocity_xy_fair_rgb_flow_geom.yaml
  RGB + 光流 + geometry + fair_chassis，算力要求更高，用于测试更强视觉运动信息。

configs/velocity_xy_pose_assisted.yaml
  pose_assisted 上限实验，不作为可部署结论。

configs/smoke_velocity_xy_fair_gray.yaml
  小规模流程验证配置，用于快速检查缓存、数据集、模型、评估和存档是否跑通。
```

## EOVM-EKF 子模块

相关路径：

```text
eovm_ekf/
scripts/audit_eovm_inputs.py
scripts/precompute_eovm_base_velocity.py
```

`eovm_ekf/` 复现了一个基于 IMU、steering angle 和 wheel speed 的 EKF 车辆速度估计核心，输出二维 `v_x, v_y`。当前它更适合作为物理先验消融，不建议在未完成单位和参数审计前作为默认主模型。

如果启用 EOVM 先验：

```yaml
data:
  base_velocity_source: eovm_ekf
  eovm_base_root: /path/to/eovm_base_velocity
```

使用前应审计：

```text
wheel speed 单位和比例
steering 字段符号和比例
yaw_rate 单位
longitudinal_accel / transversal_accel 方向
车辆几何参数和 tire_radius
```

## AI 修改代码时的注意事项

1. 不要把当前分支误认为标量速度估计项目。当前主输出是 `[v_x, v_y]`。
2. 修改数据集时，必须保持 `velocity`、`base_velocity`、`chassis`、`images` 的 batch contract。
3. 修改模型时，默认输出维度是 2；不要把 `FusionHead` 改回 1 维，除非明确做旧版标量实验。
4. `base_speed` 和 `speed` 是兼容字段，不是当前训练主目标。
5. `fair_chassis` 是主实验模式；`pose_assisted` 只用于上限实验，可能存在 pose 同源信息泄漏。
6. 如果使用光流，Dataset 返回的 `flow` 应该是 `[T-1, 2, H, W]`，batch 后是 `[B, T-1, 2, H, W]`。
7. 如果使用 geometry，必须同时配置 `geometry_input_dim` 和 `geometry_feature_dim`，否则模型不会启用 geometry encoder。
8. 如果修改评估指标，需要同步检查 `utils/evaluation.py`、`test.py`、`analysis/postprocess.py`。
9. 如果修改 YAML 路径，优先改 `dataroot`、`can_bus_root`、`frame_cache_root`、`pretrained_path`、`flow_root`。
10. 如果要证明视觉模态有效，不能只看 fusion 结果，应该补 `chassis-only`、`vision-only`、`vision+chassis` 消融。

## 给 AI 的任务提示模板

可以把下面这段直接发给另一个 AI：

```text
请阅读 FlexiNet_ChassisFusion/AI_CODE_CONTEXT.md 和 FlexiNet_ChassisFusion/速度XY改造使用说明.md。

这个项目当前不是旧版标量 speed 估计，而是 nuScenes ego/body frame 下的二维速度估计：
pred_velocity = base_velocity + delta_velocity，其中 velocity = [v_x, v_y]。

请以 configs/default.yaml、dataset/nuscenes_chassis.py、model/fusion_model.py、train.py、utils/evaluation.py 为当前主线代码。

分析或修改代码时，请优先保持以下 contract：
- Dataset 返回 images, chassis, base_velocity, velocity, flow, geometry。
- Model 输出 [B, 2]。
- 训练主目标是 VelocitySmoothL1Loss(pred_velocity, target_velocity)。
- fair_chassis 是主实验模式，pose_assisted 只作为上限实验。
- base_speed/speed 是兼容字段，不是当前主训练目标。

如果你发现 README.md 和当前代码不一致，请以 AI_CODE_CONTEXT.md、速度XY改造使用说明.md 和实际代码为准。
```
