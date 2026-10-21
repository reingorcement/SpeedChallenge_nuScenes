# 服务器全量 nuScenes 下 VKD 融合速度估计运行说明

本文面向服务器上的完整 `nuScenes v1.0-trainval` 数据集，说明 `FlexiNet_ChassisFusion` 当前代码各部分原理、数据准备、训练评估流程，以及“基于动力学的速度估计”在服务器上如何运行。

当前项目不是旧版标量 `speed` 回归，而是 ego/body frame 下二维速度估计：

```text
target_velocity = [v_x, v_y]
base_velocity   = [base_vx, base_vy]
pred_velocity   = base_velocity + delta_velocity
```

其中：

- `v_x`：车体前向速度，单位 m/s。
- `v_y`：车体左向速度，单位 m/s。
- 默认 `base_velocity_source: zero_lateral`，即 `base_velocity = [base_vx, 0]`。
- 可选 `base_velocity_source: eovm_ekf`，使用预计算 EOVM-EKF 输出作为二维物理先验。

## 1. 当前主方法能否在服务器上运行

可以。分两层理解：

1. **当前 proxy-dynamics gated 模型可以直接跑全量 nuScenes**
   使用 `configs/velocity_xy_vkd_gated.yaml`。它启用：

   ```text
   VisionBranch + KinematicBranch + DynamicsBranch + ConfidenceGatedFusion
   ```

   这里的 `DynamicsBranch` 第一版不依赖外部 EOVM 文件，而是在模型内部从 `base_velocity` 和 chassis proxy 构造 8 维动力学代理特征；默认 `zero_lateral` 时不能把它表述为已接入 EOVM 的 VKD 主方法。因此只要服务器上有 nuScenes、CAN bus、图像帧缓存和 FlexiNet 预训练权重，就能训练。

2. **真正 EOVM-EKF 物理先验也可以跑，但需要先审计和预计算**
   `eovm_ekf/` 已经实现 EOVM-EKF 估计器核心，并能输出 `[vx, vy]`。但 nuScenes CAN 字段的轮速单位、转向比例、符号和车辆几何参数需要审计。审计通过后，先为每个 scene 预计算 EOVM CSV，再把配置改为：

   ```yaml
   data:
     base_velocity_source: eovm_ekf
     eovm_base_root: /mnt/sda2/zy/nuscenes/eovm_base_velocity
   ```

   在未审计前，建议只把 EOVM 作为“待验证物理 baseline”，不要直接作为论文主结论。

## 2. 代码模块原理

### 2.1 Dataset 数据流

核心文件：

```text
dataset/nuscenes_chassis.py
```

单个样本返回：

```python
{
    "images": Tensor,          # [T, C, H, W]
    "chassis": Tensor,         # [K, C_chassis]
    "base_velocity": Tensor,   # [2]
    "velocity": Tensor,        # [2]
    "flow": Tensor,            # [T-1, 2, H, W] 或 empty Tensor
    "geometry": Tensor,        # geometry prior 或 empty Tensor
    "base_speed": Tensor,      # [1]，兼容旧逻辑
    "speed": Tensor,           # [1]，兼容旧逻辑
    "scene_name": str,
    "center_token": str,
    "center_timestamp": Tensor,
}
```

### 2.2 二维速度标签

`velocity` 标签来自 nuScenes `ego_pose`，用中心差分计算：

```text
p_prev = previous ego pose position
p_next = next ego pose position
dt = t_next - t_prev
v_global = (p_next - p_prev) / dt
v_ego = rotate_global_velocity_to_center_ego_frame(v_global)
target_velocity = [v_ego_x, v_ego_y]
```

注意：

- 标签来自 ego pose，不来自 CAN 速度。
- 边界帧没有 `prev` 或 `next` 时不能作为监督样本。
- `pose_assisted` 输入会泄漏 pose 派生信息，只能作为上限实验。

### 2.3 fair_chassis 底盘特征

主实验使用 `fair_chassis`，只包含可部署信号：

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
fl_wheel_speed
fr_wheel_speed
rl_wheel_speed
rr_wheel_speed
wheel_speed_mean
wheel_speed_std
base_vx
```

模型侧用 `ChassisSequenceEncoder` 编码 `[K, C]` 时间序列。默认：

```yaml
data:
  chassis_window_sec: 1.0
  chassis_hz: 20.0
  online_mode: false

model:
  chassis_sequence_model: gru
```

即中心图像时间附近 1 秒 CAN 窗口，约 21 个时间点，交给 GRU 编码。

### 2.4 VKD 三分支模型

核心文件：

```text
model/fusion_model.py
```

当前支持三种融合模式：

```text
residual:
  旧默认逻辑，concat 后预测 delta_velocity，再加 base_velocity。

direct:
  直接输出 pred_velocity，不加 base_velocity。适合 vision_only。

residual_gated:
  新增 VKD 门控融合，按 active branches 动态加权。
```

#### VisionBranch

输入：

```text
images: [B, T, C, H, W]
```

经过 FlexiNet backbone 输出视觉时序特征：

```text
visual_feat: [B, 256]
```

如果启用 flow/geometry，会拼接到视觉分支中。

#### KinematicBranch

输入：

```text
chassis: [B, K, C]
```

经过 `ChassisSequenceEncoder`：

```text
kinematic_feat: [B, chassis_feature_dim]
```

它主要表达车速、轮速、横摆角速度、转向、加速度等运动学状态。

#### DynamicsBranch

当前第一阶段使用 8 维动力学代理输入：

```text
[
  base_vx,
  base_vy,
  yaw_rate,
  steering,
  steering_feedback,
  longitudinal_accel,
  transversal_accel,
  wheel_speed_std,
]
```

其中：

- `base_vx/base_vy` 来自 `base_velocity`。
- `yaw_rate/steering/accel/wheel_speed_std` 从 chassis 中心时刻抽取。
- 如果 `base_velocity_source: zero_lateral`，则 `base_vy = 0`。
- 如果 `base_velocity_source: eovm_ekf`，则 `base_velocity = [EOVM_vx, EOVM_vy]`。

这个分支不是完整车辆动力学模型，而是“物理/动力学代理特征分支”。它让模型显式学习动力学先验何时可信。

#### ConfidenceGatedFusion

对启用的分支做 softmax gate：

```text
branch_features -> gate_logits -> softmax gate
fused = sum_i gate_i * branch_i
delta_velocity = pred_head(fused)
pred_velocity = base_velocity + delta_velocity
```

特点：

- 只对 active branches 做 softmax。
- `vision_only` 不会把关闭的分支用零特征参与 gate。
- 如果 `save_gate_weights: true`，评估输出中可以保存 `gate_vision/gate_kinematic/gate_dynamics`。

## 3. 服务器目录和环境准备

假设服务器目录类似：

```text
/mnt/sda2/zy/nuscenes/
  maps/
  samples/
  sweeps/
  v1.0-trainval/
  can_bus/

/mnt/sda2/zy/nuscenes/precomputed_frames/
/mnt/sda2/zy/nuscenes/eovm_base_velocity/
```

进入项目：

```bash
cd /path/to/FlexiNet_ChassisFusion
```

建议环境：

```bash
conda activate speedXY_py310
```

或：

```bash
conda run -n speedXY_py310 python ...
```

需要确认配置中的路径：

```yaml
data:
  dataroot: /mnt/sda2/zy/nuscenes
  can_bus_root: /mnt/sda2/zy/nuscenes/can_bus
  frame_cache_root: /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray

model:
  pretrained_path: ../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

当前 YAML 里的默认路径是本地工作站路径 `/media/ps/...`，服务器上必须改成实际路径。

## 4. 全量 nuScenes 图像帧缓存

强烈建议先预计算图像帧缓存。全量 `v1.0-trainval` 如果直接读原始 JPEG，每个 epoch 会重复解码大量 1600x900 图像，GPU 很容易等 DataLoader。

灰度 64x64 缓存：

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

输出：

```text
CAM_FRONT_64x64_gray/
  manifest.json
  labels.json
  scene-xxxx.npy
```

注意：

- `manifest.json` 记录缓存配置。
- `labels.json` 缓存二维速度标签 `[vx, vy]`。
- 配置中的 `img_height/img_width/image_mode/version/cam_channel` 必须与缓存一致。
- RGB 实验要单独生成 RGB 缓存。

## 5. 路径和流程 sanity check

先检查配置路径和模型构建：

```bash
python scripts/path_sanity_check.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --skip-dataset \
  --skip-train-step
```

有完整数据和缓存后，可以检查 dataset + sample：

```bash
python scripts/path_sanity_check.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --skip-train-step
```

如果希望跑一个 tiny train step：

```bash
python scripts/path_sanity_check.py \
  --config configs/smoke_velocity_xy_vkd_gated.yaml
```

## 6. 先跑 smoke 训练

`configs/smoke_velocity_xy_vkd_gated.yaml` 使用少量 scene、少量 batch，用来确认：

- 帧缓存能读。
- CAN bus 能对齐。
- 模型 forward/backward 能跑。
- 评估、checkpoint、结果输出链路正常。

命令：

```bash
python train.py --config configs/smoke_velocity_xy_vkd_gated.yaml
```

如果顺利，应看到：

```text
train_loss=...
val_loss=...
val_mae_vx=...
val_mae_vy=...
Saved new best checkpoint ...
```

输出目录：

```text
runs/smoke_vkd_gated
checkpoints/smoke_vkd_gated
results_git/smoke_vkd_gated
```

## 7. 训练主方法：Vision + Kinematics + Dynamics gated

主方法配置：

```text
configs/velocity_xy_vkd_gated.yaml
```

核心配置：

```yaml
model:
  fusion_mode: residual_gated
  use_vision_branch: true
  use_kinematic_branch: true
  use_dynamics_branch: true
  save_gate_weights: true

data:
  base_velocity_source: zero_lateral
```

运行：

```bash
python train.py --config configs/velocity_xy_vkd_gated.yaml
```

输出：

```text
runs/FlexiNet_ChassisFusion_velocity_xy/vkd_gated
checkpoints/velocity_xy_vkd_gated
results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated
```

## 8. 消融实验如何跑

### 8.1 VisionOnly

```bash
python train.py --config configs/velocity_xy_vision_only.yaml
```

特点：

- `fusion_mode: direct`
- 只启用视觉分支。
- 不使用 `base_velocity + delta`，避免底盘先验污染纯视觉结论。

### 8.2 Kinematics + Dynamics Only

```bash
python train.py --config configs/velocity_xy_kd_only.yaml
```

特点：

- 关闭视觉分支。
- 启用 kinematic + dynamics。
- 默认仍使用 `zero_lateral` base。
- 如果要测试 EOVM 物理先验，需先预计算 EOVM CSV，再改 `base_velocity_source`。

### 8.3 Vision + Kinematics

```bash
python train.py --config configs/velocity_xy_vision_kinematics.yaml
```

特点：

- 启用视觉和运动学。
- 不启用 dynamics branch。
- 用于证明 dynamics branch 是否额外有贡献。

### 8.4 Vision + Kinematics + Dynamics

```bash
python train.py --config configs/velocity_xy_vkd_gated.yaml
```

主方法，用于与前三个消融对比。

## 9. GPU 训练加速建议

先用 profiling 脚本判断瓶颈：

```bash
python scripts/profile_training_step.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --batches 50 \
  --batch_size 96 \
  --num_workers 12
```

解释：

- `load` 高：DataLoader/缓存/CPU 瓶颈。
- `transfer` 高：CPU 到 GPU 搬运瓶颈。
- `compute` 高且显存未满：可以继续加 batch。

推荐服务器起步参数：

```yaml
train:
  batch_size: 96
  num_workers: 12
  prefetch_factor: 4
  eval_every: 3
  log_interval: 50
  use_amp: true
```

显存还没拉满再试：

```yaml
train:
  batch_size: 128
  num_workers: 16
  prefetch_factor: 4
  eval_every: 5
  log_interval: 100
```

不要先关掉 frame cache。没有缓存时 JPEG 解码通常会让 GPU 吃不饱。

## 10. 评估 checkpoint

训练后评估 best checkpoint：

```bash
python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --checkpoint checkpoints/velocity_xy_vkd_gated/best.pth \
  --output_dir outputs/velocity_xy_vkd_gated/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated/eval_best
```

它会生成：

```text
outputs/velocity_xy_vkd_gated/eval_best/
  val_metrics.json
  val_predictions.csv
  analysis/
    val_bad_cases_top50.csv
    val_per_scene_metrics.csv
    val_per_speed_bin_metrics.csv
    val_per_scenario_metrics.csv
    val_report.md
```

重点看：

```text
mae_vx_mps
mae_vy_mps
vector_endpoint_error_mps
base_mae_vx_mps
base_mae_vy_mps
gain_vs_base_vx_mps
gain_vs_base_vy_mps
turning_vector_endpoint_error_mps
high_lateral_mae_vy_mps
wheel_inconsistent_vector_endpoint_error_mps
hard_accel_vector_endpoint_error_mps
high_lat_accel_vector_endpoint_error_mps
```

如果 `save_gate_weights: true`，预测 CSV 还会包含：

```text
gate_vision
gate_kinematic
gate_dynamics
```

这些列可用于分析不同场景下模型更信任哪个分支。

## 11. EOVM-EKF 动力学先验运行流程

### 11.1 先审计单个 scene

```bash
python scripts/audit_eovm_inputs.py \
  --can_bus_root /mnt/sda2/zy/nuscenes/can_bus \
  --scene_name scene-0061
```

需要确认：

- `FL/FR/RL/RR_wheel_speed` 单位和比例。
- `vehicle_monitor.steering` 符号和比例。
- `steeranglefeedback.value` 是否更适合做转角输入。
- `yaw_rate` 单位是否按代码转为 rad/s。
- `longitudinal_accel/transversal_accel` 方向是否与车体系一致。
- 默认车辆几何和 tire radius 是否合理。

### 11.2 为单个 scene 预计算 EOVM base_velocity

示例：

```bash
python scripts/precompute_eovm_base_velocity.py \
  --can_bus_root /mnt/sda2/zy/nuscenes/can_bus \
  --scene_name scene-0061 \
  --output_csv /mnt/sda2/zy/nuscenes/eovm_base_velocity/scene-0061.csv \
  --wheel_speed_scale 1.0 \
  --steering_scale 0.001282051282
```

输出 CSV：

```text
timestamp_us,vx,vy,gamma,delta,delta_dot
```

### 11.3 为全量 trainval 批量预计算

服务器 Linux/bash 示例：

```bash
mkdir -p /mnt/sda2/zy/nuscenes/eovm_base_velocity

for f in /mnt/sda2/zy/nuscenes/can_bus/scene-*_vehicle_monitor.json; do
  scene=$(basename "$f" _vehicle_monitor.json)
  python scripts/precompute_eovm_base_velocity.py \
    --can_bus_root /mnt/sda2/zy/nuscenes/can_bus \
    --scene_name "$scene" \
    --output_csv /mnt/sda2/zy/nuscenes/eovm_base_velocity/${scene}.csv \
    --wheel_speed_scale 1.0 \
    --steering_scale 0.001282051282
done
```

如果审计后发现比例不同，必须改 `--wheel_speed_scale` 和 `--steering_scale`。

### 11.4 启用 EOVM base_velocity

复制一份配置，例如：

```bash
cp configs/velocity_xy_vkd_gated.yaml configs/velocity_xy_vkd_gated_eovm.yaml
```

修改：

```yaml
data:
  base_velocity_source: eovm_ekf
  eovm_base_root: /mnt/sda2/zy/nuscenes/eovm_base_velocity
```

然后先 sanity check：

```bash
python scripts/path_sanity_check.py \
  --config configs/velocity_xy_vkd_gated_eovm.yaml \
  --skip-train-step
```

再训练：

```bash
python train.py --config configs/velocity_xy_vkd_gated_eovm.yaml
```

## 12. EOVM 结果怎么判断是否可靠

建议先比较三类结果：

1. `zero_lateral` base：

   ```text
   base_velocity = [base_vx, 0]
   ```

2. EOVM base：

   ```text
   base_velocity = [eovm_vx, eovm_vy]
   ```

3. 模型输出：

   ```text
   pred_velocity = base_velocity + delta_velocity
   ```

重点看：

```text
base_mae_vx_mps
base_mae_vy_mps
gain_vs_base_vx_mps
gain_vs_base_vy_mps
turning_vector_endpoint_error_mps
high_lateral_mae_vy_mps
wheel_inconsistent_vector_endpoint_error_mps
```

如果 EOVM 的 `base_mae_vy_mps` 明显比 `zero_lateral` 更差，说明侧向物理先验还不可靠，应暂时不要把 EOVM 作为主方法结论，只作为待审计 baseline。

如果 EOVM 在 turning/high_lateral/wheel_inconsistent 场景改善明显，说明动力学先验有论文价值。

## 13. 推荐服务器实验顺序

建议按下面顺序跑：

```text
1. path_sanity_check + smoke_velocity_xy_vkd_gated
2. velocity_xy_kd_only
3. velocity_xy_vision_only
4. velocity_xy_vision_kinematics
5. velocity_xy_vkd_gated
6. audit_eovm_inputs
7. precompute_eovm_base_velocity for all scenes
8. velocity_xy_kd_only_eovm
9. velocity_xy_vkd_gated_eovm
```

这样可以分清：

- 视觉是否单独有效。
- 运动学/动力学底盘信号能达到什么水平。
- 视觉 + 运动学是否互补。
- 加 dynamics branch 是否提升。
- EOVM-EKF 是否是可靠物理先验。

## 14. 常见问题

### Q1：当前 dynamics branch 等于 EOVM 吗？

不等于。当前 dynamics branch 是模型内部的 8 维动力学代理 MLP 分支。EOVM 是外部 EKF 物理估计器，需要先预计算 CSV，然后作为 `base_velocity` 输入。

### Q2：不预计算 EOVM 能不能跑 VKD？

能。`velocity_xy_vkd_gated.yaml` 默认使用 `zero_lateral`，可以直接跑。

### Q3：为什么 vision_only 用 direct？

因为 `vision_only` 如果用 residual，就会用到底盘 `base_velocity`，污染“纯视觉”消融结论。

### Q4：为什么 full nuScenes 必须做 frame cache？

全量样本量大，直接读原 JPEG 会严重拖慢 DataLoader。frame cache 把图像缩放和速度标签提前落盘，训练时直接读小数组。

### Q5：gate 权重有什么用？

`gate_vision/gate_kinematic/gate_dynamics` 可以解释模型在不同场景更信任哪个分支。例如转弯、高横向速度、轮速不一致时，是否提高视觉或动力学分支权重。

## 15. 最小可运行命令清单

```bash
cd /path/to/FlexiNet_ChassisFusion

python scripts/precompute_frames.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split trainval \
  --cam_channel CAM_FRONT \
  --height 64 --width 64 \
  --image_mode gray \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray

python scripts/path_sanity_check.py \
  --config configs/smoke_velocity_xy_vkd_gated.yaml

python train.py \
  --config configs/smoke_velocity_xy_vkd_gated.yaml

python train.py \
  --config configs/velocity_xy_vkd_gated.yaml

python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --checkpoint checkpoints/velocity_xy_vkd_gated/best.pth \
  --output_dir outputs/velocity_xy_vkd_gated/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated/eval_best
```
