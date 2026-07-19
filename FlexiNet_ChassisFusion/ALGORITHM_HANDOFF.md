# FlexiNet_ChassisFusion 算法技术交接文档

生成日期：2026-07-19

本文档面向“接手本仓库的另一个模型或工程师”。目标是让读者不依赖当前对话，只读本文件就能理解这一版算法的任务定义、数据流、模型结构、训练评估流程、实验配置、现有结果和后续修改边界。

## 0. 先读结论

当前 `FlexiNet_ChassisFusion` 不是旧版标量 `speed` 回归项目，而是 nuScenes ego/body frame 下的二维纵侧向速度估计项目：

```text
target_velocity = [v_x, v_y]
base_velocity   = [base_vx, base_vy]
pred_velocity   = base_velocity + delta_velocity
```

- `v_x`：车辆前向速度，单位 `m/s`。
- `v_y`：车辆左向速度，单位 `m/s`。
- 默认 `base_velocity_source: zero_lateral`，即 `base_velocity = [base_vx, 0]`。
- 可选 `base_velocity_source: eovm_ekf`，把预计算 EOVM-EKF 结果作为二维 base velocity。
- 当前主方法配置是 `configs/velocity_xy_vkd_gated.yaml`。
- 当前主模型是 `VisionBranch + KinematicBranch + DynamicsBranch + ConfidenceGatedFusion`。
- 当前主融合模式是 `fusion_mode: residual_gated`，输出仍是残差式二维速度：`pred_velocity = base_velocity + delta_velocity`。

最重要的代码文件：

```text
dataset/nuscenes_chassis.py
model/fusion_model.py
model/flexinet_backbone.py
model/modules.py
train.py
test.py
utils/evaluation.py
analysis/inference.py
analysis/postprocess.py
configs/velocity_xy_vkd_gated.yaml
```

如果你发现 `README.md` 或历史文档还在描述单值 `speed` 任务，应以本文件、`AI_CODE_CONTEXT.md`、`速度XY改造使用说明.md` 和实际代码为准。

## 1. 当前版本的核心假设

本版算法的关键假设如下：

1. 监督标签来自 nuScenes `ego_pose`，不是来自 CAN bus 速度字段。
2. 输入可以使用相机图像序列和可部署底盘/CAN 信号。
3. 主实验使用 `fair_chassis`，不把 pose 派生速度/加速度作为可部署输入。
4. `pose_assisted` 只作为上限实验，因为它会加入和标签同源的 pose 派生量。
5. 默认 base velocity 只给出可靠纵向先验 `[base_vx, 0]`，侧向速度由模型学习残差。
6. 当前 dynamics branch 是 8 维动力学代理 MLP 分支，不等价于完整 EOVM-EKF。
7. EOVM-EKF 已有实现和预计算脚本，但作为主物理先验前必须先审计单位、符号、比例和车辆参数。

## 2. 仓库结构和职责

当前改造版目录：

```text
FlexiNet_ChassisFusion/
  configs/
    default.yaml
    velocity_xy_fair_gray.yaml
    velocity_xy_fair_flow.yaml
    velocity_xy_fair_rgb_flow_geom.yaml
    velocity_xy_pose_assisted.yaml
    velocity_xy_vision_only.yaml
    velocity_xy_vision_kinematics.yaml
    velocity_xy_kd_only.yaml
    velocity_xy_vkd_gated.yaml
    smoke_velocity_xy_vkd_gated.yaml

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

  analysis/
    inference.py
    postprocess.py
    evaluate_checkpoint.py
    compare_checkpoints.py

  eovm_ekf/
    eovm_ekf/
      eovm.py
      estimator.py
      parameters.py
      steering_filter.py
      types.py

  scripts/
    precompute_frames.py
    precompute_optical_flow.py
    audit_eovm_inputs.py
    precompute_eovm_base_velocity.py
    path_sanity_check.py
    profile_training_step.py
    analyze_vkd_experiments.py

  train.py
  test.py
  smoke_model.py
  tests/test_velocity_xy_pipeline.py
```

职责划分：

- `dataset/nuscenes_chassis.py`：构造图像序列、底盘时间序列、二维速度标签、base velocity、光流和几何先验。
- `model/flexinet_backbone.py` + `model/modules.py`：保留 FlexiNet 视觉时序特征提取器，输出 256 维视觉特征。
- `model/fusion_model.py`：实现 residual/direct/residual_gated 三种融合模式，以及 VKD 三分支和 confidence gate。
- `train.py`：训练入口，负责 dataset/model/loss/optimizer/eval/checkpoint。
- `test.py`：旧式单 checkpoint 评估入口。
- `analysis/evaluate_checkpoint.py`：推荐的训练后评估和分析入口，会生成 bad case、per-scene、per-bin、per-scenario 报告。
- `utils/evaluation.py`：二维指标、预测 CSV、诊断图、训练曲线和小结果复制。
- `eovm_ekf/`：外部 EOVM-EKF 物理速度估计器实现，可预计算 `[vx, vy]` 先验。

## 3. 数据流

### 3.1 样本构建

数据集类：

```python
NuScenesChassisFusionDataset
```

位置：

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
    "geometry": Tensor,        # [4] 或 empty Tensor

    # 兼容旧标量代码和派生指标
    "base_speed": Tensor,      # [1]
    "speed": Tensor,           # [1]

    "scene_name": str,
    "center_token": str,
    "center_timestamp": Tensor,
}
```

默认图像输入：

```text
images: [13, 1, 64, 64]
camera: CAM_FRONT
image_mode: gray
normalize: [0,255] -> [0,1] -> [-1,1]
```

RGB 配置时：

```text
images: [13, 3, 64, 64]
image_mode: rgb
```

注意：图像序列是按相机 `sample_data` 链构建的滑窗，中心帧为 `seq_len // 2`。

### 3.2 二维速度标签

标签生成函数：

```python
NuScenesChassisFusionDataset._compute_velocity_label(center_token)
```

实际逻辑：

```text
center_sd = sample_data(center_token)
prev_sd   = sample_data(center_sd.prev)
next_sd   = sample_data(center_sd.next)

p_prev = prev ego_pose translation
p_next = next ego_pose translation
dt     = (t_next - t_prev) / 1e6

v_global = (p_next - p_prev) / dt
v_ego    = rotate_global_velocity_to_center_ego_frame(v_global, center_pose.rotation)

target_velocity = [v_ego_x, v_ego_y]
```

坐标系转换函数：

```python
_rotate_global_velocity_to_ego(vx_global, vy_global, rotation)
```

含义：

- `v_x` 是中心帧车体系前向速度。
- `v_y` 是中心帧车体系左向速度。
- 边界帧没有 `prev` 或 `next` 时不作为样本，不用 0 填标签。
- 如果使用 frame cache 且 `labels.json` 存在，标签会直接从缓存读取，避免 DataLoader worker 重复加载 nuScenes 元数据。

### 3.3 底盘/CAN 特征

CAN 读取类：

```python
NuScenesCanBusAdapter
```

它按 `utime <= timestamp_us` 选择最近一条 CAN 消息，不使用未来 CAN 消息。

主实验底盘模式：

```yaml
data:
  chassis_mode: fair_chassis
```

`fair_chassis` 的 16 维顺序：

```text
0  vehicle_speed
1  odom_speed
2  yaw_rate
3  throttle
4  brake
5  steering
6  steering_feedback
7  longitudinal_accel
8  transversal_accel
9  fl_wheel_speed
10 fr_wheel_speed
11 rl_wheel_speed
12 rr_wheel_speed
13 wheel_speed_mean
14 wheel_speed_std
15 base_vx
```

主要单位和缩放：

```text
vehicle_speed       = vehicle_monitor.vehicle_speed / 3.6
odom_speed          = zoe_veh_info.odom_speed / 3.6
yaw_rate            = radians(vehicle_monitor.yaw_rate)
throttle            = vehicle_monitor.throttle / 1000
brake               = vehicle_monitor.brake / 126
steering            = vehicle_monitor.steering / 780
steering_feedback   = steeranglefeedback.value / 7.7
wheel_speeds        = raw wheel speed / 1000
base_vx priority    = vehicle_speed -> odom_speed -> wheel_speed_mean -> 0
```

`pose_assisted` 在 `fair_chassis` 后额外拼 6 维：

```text
pose_speed
pose_vel_x
pose_vel_y
pose_accel_x
pose_accel_y
pose_rotation_rate_z
```

重要警告：`pose_assisted` 可能泄漏和标签同源的 pose 信息，只能当上限实验或诊断实验，不应作为可部署结论。

### 3.4 底盘时间序列

配置：

```yaml
data:
  chassis_window_sec: 1.0
  chassis_hz: 20.0
  online_mode: false
```

实现函数：

```python
NuScenesCanBusAdapter.extract_feature_sequence(...)
```

默认窗口长度：

```text
sample_count = round(window_sec * hz) + 1
             = round(1.0 * 20.0) + 1
             = 21
```

所以默认 `chassis` shape 约为：

```text
[21, 16]  # fair_chassis
```

`online_mode` 含义：

- `false`：离线训练，窗口覆盖中心时间前后。
- `true`：在线/因果推理，只用中心时间之前的历史窗口。

### 3.5 Base Velocity

默认配置：

```yaml
data:
  base_velocity_source: zero_lateral
```

实际 base：

```text
base_velocity = [base_vx, 0.0]
```

其中 `base_vx` 来自 CAN 特征抽取时的优先级选择。

可选 EOVM：

```yaml
data:
  base_velocity_source: eovm_ekf
  eovm_base_root: /path/to/eovm_base_velocity
```

此时会用 `PrecomputedVelocityLookup` 从：

```text
<eovm_base_root>/<scene_name>.csv
```

读取 `timestamp_us <= center_timestamp` 的最近一条 `[vx, vy]`。

### 3.6 图像帧缓存

全量 nuScenes 不建议每个 epoch 直接重复解码原始 JPEG。推荐先运行：

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

缓存目录包含：

```text
manifest.json
labels.json
scene-xxxx.npy
```

`manifest.json` 会校验：

```text
img_height
img_width
image_mode
version
cam_channel
```

如果配置和缓存不一致，dataset 初始化会报错，避免错用缓存。

注意当前代码细节：

- `train.py` 会把 `frame_cache_root` 传给 dataset。
- `analysis/inference.py` 会把 `frame_cache_root` 传给 dataset。
- `test.py` 当前没有传 `frame_cache_root`，单独跑 `test.py` 时可能走原始图片读取路径。
- `scripts/path_sanity_check.py` 当前也没有把 `frame_cache_root` 传入 dataset，只检查路径存在。

因此推荐训练后评估优先用：

```bash
python analysis/evaluate_checkpoint.py ...
```

而不是直接依赖 `test.py`。

### 3.7 光流和几何先验

光流配置：

```yaml
data:
  flow_mode: precomputed
  flow_root: /path/to/precomputed_flow
```

dataset 会为 `tokens[:-1]` 读取每个 token 的 `.pt` 或 `.npy` 光流，支持：

```text
[2, H, W]
[H, W, 2]
```

batch 后 shape：

```text
[B, T-1, 2, H, W]
```

几何先验配置：

```yaml
data:
  geometry_mode: basic
```

当前 `basic` geometry 为中心底盘向量中的 4 维：

```text
[yaw_rate, steering, steering_feedback, base_vx]
```

如果 `geometry_mode: none`，返回 empty Tensor。

## 4. 模型结构

主模型类：

```python
FlexiNetChassisFusion
```

位置：

```text
model/fusion_model.py
```

### 4.1 FlexiNet 视觉 backbone

位置：

```text
model/flexinet_backbone.py
model/modules.py
```

输入：

```text
images: [B, T, C, H, W]
```

`FlexiNetBackbone.forward_features()` 内部先转置：

```text
[B, T, C, H, W] -> [B, C, T, H, W]
```

处理链路：

```text
ContextualMotionAnalysis
  -> AdaptiveFeatureTransformer
  -> SpatialFeatureExtraction
  -> MotionFeatureExtraction
  -> DynamicIntegrationGate
  -> AdaptiveAvgPool3dStatic(mode="1")
  -> visual_feat [B, 256]
```

各模块含义：

- `ContextualMotionAnalysis`：对相邻帧做绝对差分，提取时序运动变化。
- `AdaptiveFeatureTransformer`：用运动特征产生 attention，对 3D 卷积特征做自适应变换。
- `SpatialFeatureExtraction`：空间特征提取，包含 depthwise/pointwise/dilated 3D 卷积。
- `MotionFeatureExtraction`：时序卷积提取运动特征，并在空间维做平均。
- `DynamicIntegrationGate`：在空间分支和时序分支之间做通道级 gate 融合。
- `global_pool`：全局平均池化成 256 维向量。

预训练权重加载：

```python
model.load_flexinet_backbone(checkpoint_path, strict=False)
```

它只加载旧 FlexiNet checkpoint 中这些模块相关权重：

```text
cma_block
aft_block
sfe_module
mfe_module
dig
global_pool
```

旧标量速度回归头不会进入当前二维融合头。

### 4.2 KinematicBranch

核心类：

```python
ChassisSequenceEncoder
```

输入：

```text
chassis: [B, K, C]
```

默认配置：

```yaml
model:
  chassis_sequence_model: gru
  chassis_feature_dim: 8
```

GRU 模式：

```text
GRU(input_dim=C, hidden_dim=hidden_dim)
hidden[-1] -> Dropout -> Linear(hidden_dim, chassis_feature_dim) -> ReLU
```

输出：

```text
kinematic_feat: [B, chassis_feature_dim]
```

也支持 `mean` 或 `none`，这两种会对时间维求平均后走 MLP。

### 4.3 Optional FlowEncoder

启用条件：

```text
use_vision_branch == true
flow_input_channels > 0
flow_feature_dim > 0
```

输入：

```text
flow: [B, T-1, 2, H, W]
```

处理：

```text
reshape to [B*(T-1), 2, H, W]
2D Conv -> ReLU -> Conv -> ReLU -> AdaptiveAvgPool2d(1)
mean over time
Linear(32, flow_feature_dim) -> ReLU
```

输出拼接到视觉分支：

```text
vision_input_dim = 256 + flow_feature_dim + geometry_feature_dim
```

### 4.4 Optional GeometryEncoder

启用条件：

```text
use_vision_branch == true
geometry_input_dim > 0
geometry_feature_dim > 0
```

输入：

```text
geometry: [B, 4]
```

输出：

```text
geometry_feat: [B, geometry_feature_dim]
```

也拼接到视觉分支。

### 4.5 DynamicsBranch

核心类：

```python
DynamicsProxyEncoder
```

输入不是独立 dataset 字段，而是在模型内部从 `base_velocity` 和 `chassis` 中心时刻构造：

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

该顺序由测试固定：

```python
DYNAMICS_PROXY_FEATURE_NAMES == [
    "base_vx",
    "base_vy",
    "yaw_rate",
    "steering",
    "steering_feedback",
    "longitudinal_accel",
    "transversal_accel",
    "wheel_speed_std",
]
```

默认要求：

```python
dynamics_feature_dim == 8
```

输出：

```text
dynamics_feat: [B, branch_feature_dim]
```

重要：这个分支目前是“动力学代理特征分支”，不是完整 EKF 或轮胎模型。它让网络显式看到和动力学相关的 proxy，并通过 gate 学习何时信任它。

### 4.6 三种融合模式

#### residual

配置：

```yaml
model:
  fusion_mode: residual
```

逻辑：

```text
features = concat(active vision / kinematic / dynamics features)
delta_velocity = residual_speed_head(features)
pred_velocity = resolve_base_velocity(base_velocity, base_speed) + delta_velocity
```

这是旧 residual 思路扩展到二维速度。

#### direct

配置：

```yaml
model:
  fusion_mode: direct
```

逻辑：

```text
features = concat(active features)
pred_velocity = direct_speed_head(features)
```

它不加 `base_velocity`。当前 `vision_only` 使用 direct，是为了避免纯视觉消融被底盘先验污染。

#### residual_gated

配置：

```yaml
model:
  fusion_mode: residual_gated
  fusion_type: confidence_gated
```

逻辑：

```text
vision_feat    -> ProjectionBlock -> branch_dim
kinematic_feat -> ProjectionBlock -> branch_dim
dynamics_feat  -> DynamicsProxyEncoder -> branch_dim

stacked = [B, active_branch_count, branch_dim]
gate_logits = gate_mlp(stacked.flatten(1))
gate = softmax(gate_logits, dim=1)
fused = sum_i gate_i * branch_i

delta_velocity = FusionHead(fused)
pred_velocity = base_velocity + delta_velocity
```

特点：

- 只对 active branches 做 softmax。
- 关闭的分支不会以零向量混入 gate。
- `save_gate_weights: true` 时，评估阶段可保存 `gate_vision`、`gate_kinematic`、`gate_dynamics`。
- KD-only 的 active branches 是 `["kinematic", "dynamics"]`。
- VKD-gated 的 active branches 是 `["vision", "kinematic", "dynamics"]`。

### 4.7 Base Velocity 解析

模型内部函数：

```python
_resolve_base_velocity(delta, base_speed=None, base_velocity=None)
```

优先级：

1. 如果 `base_velocity` 存在，直接使用 `[B, 2]`。
2. 如果只有 `base_speed` 且 `output_dim == 2`，返回 `[base_speed, 0]`。
3. 如果只有 `base_speed` 且非二维输出，返回 `base_speed`。
4. 否则返回 `zeros_like(delta)`。

当前训练主路径会传 `base_velocity`。

## 5. 配置文件含义

### 5.1 当前主方法：velocity_xy_vkd_gated.yaml

路径：

```text
configs/velocity_xy_vkd_gated.yaml
```

关键配置：

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

model:
  fusion_mode: residual_gated
  fusion_type: confidence_gated
  output_dim: 2
  chassis_feature_dim: 8
  chassis_sequence_model: gru
  hidden_dim: 96
  dropout: 0.3
  freeze_backbone: true
  use_vision_branch: true
  use_kinematic_branch: true
  use_dynamics_branch: true
  branch_feature_dim: 64
  dynamics_feature_dim: 8
  save_gate_weights: true

train:
  batch_size: 32
  num_workers: 8
  prefetch_factor: 4
  epochs: 20
  learning_rate_head: 0.001
  learning_rate_backbone: 0.00003
  use_amp: true
  loss: velocity_smoothl1
  sigma_vx: 5.0
  sigma_vy: 0.5
  speed_loss_weight: 0.1
  grad_clip: 2.0
```

说明：

- 默认不启用光流和 geometry。
- 默认冻结 FlexiNet backbone，只训练融合头、底盘 GRU、projection、dynamics proxy 和 gate。
- 默认 dynamics branch 只使用 proxy 特征，不使用外部 EOVM。
- 默认 loss 是二维归一化 SmoothL1 加速度模长辅助项。

### 5.2 其他主要实验配置

```text
configs/default.yaml
```

普通 residual baseline，vision + kinematic，非 gated。

```text
configs/velocity_xy_fair_gray.yaml
```

灰度图像 + fair_chassis residual baseline，20 epoch。

```text
configs/velocity_xy_fair_flow.yaml
```

灰度图像 + precomputed optical flow + fair_chassis residual。

```text
configs/velocity_xy_fair_rgb_flow_geom.yaml
```

RGB 图像 + precomputed flow + basic geometry + fair_chassis residual。实际配置中图像仍是 `64x64`，batch size 为 24，backbone 仍冻结。

```text
configs/velocity_xy_pose_assisted.yaml
```

pose_assisted 上限实验，不作为可部署结论。

```text
configs/velocity_xy_vision_only.yaml
```

纯视觉消融：

```yaml
model:
  fusion_mode: direct
  use_vision_branch: true
  use_kinematic_branch: false
  use_dynamics_branch: false
```

```text
configs/velocity_xy_vision_kinematics.yaml
```

视觉 + 运动学 gated 消融：

```yaml
model:
  fusion_mode: residual_gated
  use_vision_branch: true
  use_kinematic_branch: true
  use_dynamics_branch: false
```

```text
configs/velocity_xy_kd_only.yaml
```

运动学 + 动力学代理 gated 消融：

```yaml
model:
  fusion_mode: residual_gated
  use_vision_branch: false
  use_kinematic_branch: true
  use_dynamics_branch: true
```

```text
configs/smoke_velocity_xy_vkd_gated.yaml
```

小规模流程测试：

```yaml
data:
  max_scenes: 3

train:
  batch_size: 4
  epochs: 1
  limit_train_batches: 10
  limit_eval_batches: 5
```

## 6. 训练流程

入口：

```bash
cd FlexiNet_ChassisFusion
python train.py --config configs/velocity_xy_vkd_gated.yaml
```

### 6.1 训练主线

`train.py` 主流程：

```text
load YAML config
set seed
build train dataset split="train"
build val dataset split="val"
build DataLoader
infer chassis_input_dim and input_channels
build FlexiNetChassisFusion
load FlexiNet pretrained backbone if vision branch enabled
freeze backbone if configured
build criterion
build AdamW optimizer with head/backbone param groups
build CosineAnnealingLR scheduler
train loop
  forward from batch
  loss
  backward with optional AMP
  grad clip
  optimizer step
eval every N epochs
save last checkpoint
save best checkpoint by selection_metric_mps
write metrics, predictions, plots
copy small results to results_git
```

注意：

- `train.py` 当前固定创建 `split="train"` 和 `split="val"`，不使用 YAML 中的 `data.split` 来决定训练/验证 split。
- backbone 冻结时参数仍在 optimizer param group 中，但 `requires_grad=False`，不会产生梯度。
- 如果 `use_vision_branch: false`，不会创建 backbone，也不会加载预训练视觉权重。

### 6.2 Batch Forward Contract

训练阶段统一通过：

```python
_model_forward_from_batch(model, batch, device)
```

它取：

```python
images = batch["images"]
chassis = batch["chassis"]
base_velocity = batch["base_velocity"]
velocity = batch["velocity"]
flow = batch.get("flow")
geometry = batch.get("geometry")
```

然后调用：

```python
pred = model(
    images,
    chassis,
    base_velocity=base_velocity,
    flow=flow_or_none,
    geometry=geometry_or_none,
)
```

返回：

```text
pred:          [B, 2]
velocity:      [B, 2]
base_velocity: [B, 2]
```

即使关闭视觉分支，forward 签名仍要求传 `images`，因为模型用 `images.shape[0]`、`images.device` 和 `images.dtype` 构造 dynamics proxy。

### 6.3 损失函数

默认损失：

```python
VelocitySmoothL1Loss
```

公式：

```text
sigma = [sigma_vx, sigma_vy]
L_vel = SmoothL1(pred / sigma, target / sigma)

如果 speed_loss_weight > 0:
  pred_speed = ||pred||_2
  target_speed = ||target||_2
  L = L_vel + speed_loss_weight * SmoothL1(pred_speed, target_speed)
```

默认：

```yaml
sigma_vx: 5.0
sigma_vy: 0.5
speed_loss_weight: 0.1
```

这样做是因为 `v_y` 数值通常远小于 `v_x`，如果不分轴归一化，侧向速度误差容易被纵向误差淹没。

可选知识损失：

```python
KnowledgeDrivenVelocityLoss
```

可选项：

```yaml
beta_loss_weight
prior_loss_weight
smoothness_loss_weight
consistency_loss_weight
beta_vx_eps
```

当前配置默认都为 0。细节：

- `beta_loss_weight`：在 `abs(target_vx) > beta_vx_eps` 时，对 sideslip angle `atan2(v_y, v_x)` 做 SmoothL1。
- `prior_loss_weight`：约束 `pred` 不要无理由偏离 `base_velocity`。
- `consistency_loss_weight`：用 `pred_vx * yaw_rate` 和 `transversal_accel` 做弱一致性。
- `smoothness_loss_weight > 0` 当前会直接报错，因为当前 head 输出单帧 `[B, 2]`，没有序列预测。

### 6.4 Checkpoint 和输出

训练输出目录由配置决定：

```yaml
train:
  log_dir: runs/FlexiNet_ChassisFusion_velocity_xy/vkd_gated
  results_dir: results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated
  save_dir: checkpoints/velocity_xy_vkd_gated
```

主要文件：

```text
checkpoints/velocity_xy_vkd_gated/
  last.pth
  best.pth

runs/FlexiNet_ChassisFusion_velocity_xy/vkd_gated/
  training_history.csv
  training_history.json
  train_val_loss_curve.png
  val_mae_rmse_curve.png
  last_eval/
    val_metrics.json
    val_predictions.csv
    val_speed_pred_vs_target.png
    val_endpoint_error_vs_speed.png
    val_endpoint_error_histogram.png
    val_delta_vx_histogram.png
    val_delta_vy_histogram.png
    val_speed_bin_endpoint_error.png
  best_eval/
    same files as last_eval
```

`best.pth` 选择指标：

```text
selection_metric_mps = mae_vx_mps + mae_vy_mps
```

## 7. 评估和分析

### 7.1 推荐评估入口

推荐：

```bash
python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --checkpoint checkpoints/velocity_xy_vkd_gated/best.pth \
  --output_dir outputs/velocity_xy_vkd_gated/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated/eval_best
```

它会：

1. 调用 `analysis/inference.py` 构建 dataset/model 并加载 checkpoint。
2. 调用 `train.run_eval()` 推理验证集。
3. 调用 `utils.evaluation.write_evaluation_artifacts()` 写标准指标、CSV 和图。
4. 调用 `analysis.postprocess.write_analysis_bundle()` 写分析报告。

输出：

```text
outputs/velocity_xy_vkd_gated/eval_best/
  val_metrics.json
  val_predictions.csv
  val_*.png
  analysis/
    val_bad_cases_top50.csv
    val_per_scene_metrics.csv
    val_per_speed_bin_metrics.csv
    val_per_scenario_metrics.csv
    val_report.md
```

不推荐只依赖 `test.py` 的原因：当前 `test.py` 没有把 `frame_cache_root` 传给 dataset，可能评估时不用预计算帧缓存。

### 7.2 run_eval 收集的信息

`train.run_eval()` 除了预测和标签，还会收集：

```text
scene_name
center_token
center_timestamp
yaw_rate
wheel_speed_std
longitudinal_accel
transversal_accel
gate_weights
gate_names
```

对于 `chassis` shape：

- `[B, K, C]`：取 `K // 2` 中心时刻。
- `[B, C]`：直接使用。

如果 geometry 非空，会优先从 geometry 的第一维取 yaw rate；否则从 chassis 中心向量取 yaw rate。

### 7.3 核心指标

位置：

```text
utils/evaluation.py
```

整体指标：

```text
mae_vx_mps
rmse_vx_mps
mae_vy_mps
rmse_vy_mps
vector_endpoint_error_mps
speed_magnitude_mae_mps
speed_magnitude_rmse_mps
r2_vx
r2_vy
selection_metric_mps
mae_vx_kmh
mae_vy_kmh
```

如果有 base velocity：

```text
base_mae_vx_mps
base_rmse_vx_mps
base_mae_vy_mps
base_rmse_vy_mps
base_speed_magnitude_mae_mps
gain_vs_base_vx_mps
gain_vs_base_vy_mps
gain_vs_base_vector_mps
```

`gain_vs_base_*` 的定义是：

```text
base_error - model_error
```

正数表示模型优于 base。

### 7.4 分场景指标

`compute_metrics()` 内部会构造这些分组：

```text
low_speed:          target speed < 2.0 m/s
mid_speed:          2.0 <= target speed < 10.0 m/s
high_speed:         target speed >= 10.0 m/s
high_lateral:       abs(target_vy) top 10%
low_speed_lateral:  speed < 2.0 and abs(target_vy) >= median
turning:            abs(yaw_rate) >= 0.1
straight:           abs(yaw_rate) < 0.1
wheel_inconsistent: wheel_speed_std top 10%
hard_accel:         abs(longitudinal_accel) top 10%
high_lat_accel:     abs(transversal_accel) top 10%
```

每组输出：

```text
{group}_sample_count
{group}_mae_vx_mps
{group}_mae_vy_mps
{group}_vector_endpoint_error_mps
```

这些指标用于判断模型是否只在平均场景好，还是在转弯、高侧向速度、轮速不一致、急加减速等困难工况也有价值。

### 7.5 预测 CSV 字段

`val_predictions.csv` 字段：

```text
scene_name
center_token
center_timestamp
target_vx_mps
target_vy_mps
pred_vx_mps
pred_vy_mps
base_vx_mps
base_vy_mps
error_vx_mps
error_vy_mps
endpoint_error_mps
base_endpoint_error_mps
delta_vx_mps
delta_vy_mps
yaw_rate_radps
longitudinal_accel_mps2
transversal_accel_mps2
wheel_speed_std_mps
gate_vision
gate_kinematic
gate_dynamics
```

如果 gate 或 proxy 不可用，对应列为空。

## 8. 当前实验结果

现有比较报告：

```text
analysis_reports/vkd_0704/REPORT_VKD_0704.md
analysis_reports/vkd_0704/comparison_summary.json
```

比较方法：

```text
vkd_gated          Vision + Kinematics + Dynamics gated
kd_only            Kinematics + Dynamics only
vision_only        Vision only
vision_kinematics  Vision + Kinematics
```

对齐样本：

```text
33148 validation samples
max target difference across aligned methods = 0
```

单 seed 结果，主指标 `vector_endpoint_error_mps`：

| method | vector EE | MAE vx | MAE vy | R2 vx | R2 vy | gain vs base EE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vkd_gated | 0.07438 | 0.06875 | 0.01782 | 0.99919 | 0.50826 | 0.11898 |
| kd_only | 0.07559 | 0.06995 | 0.01784 | 0.99916 | 0.50384 | 0.11777 |
| vision_kinematics | 0.07625 | 0.07049 | 0.01815 | 0.99913 | 0.47718 | 0.11710 |
| vision_only | 1.08914 | 1.08815 | 0.02150 | 0.81050 | 0.31053 | -0.89578 |

结论边界：

- `vkd_gated` 是这组单 seed 消融中最好的方法。
- 它相对 `kd_only` 的 vector endpoint error 低约 `0.00121 m/s`。
- 它相对 `vision_kinematics` 低约 `0.00187 m/s`。
- 它显著优于 `vision_only`，说明纯前视低分辨率视觉不足以单独完成纵向速度估计。
- 这不是统计显著性结论，不能在没有 repeated seeds 的情况下写成显著提升。

`vkd_gated` 当前 best eval 关键指标：

```text
mae_vx_mps                     0.068751
rmse_vx_mps                    0.100282
mae_vy_mps                     0.017819
rmse_vy_mps                    0.026045
vector_endpoint_error_mps       0.074382
speed_magnitude_mae_mps         0.068248
base_mae_vx_mps                 0.183282
base_mae_vy_mps                 0.038154
gain_vs_base_vx_mps             0.114531
gain_vs_base_vy_mps             0.020335
gain_vs_base_vector_mps         0.118976
```

困难场景结果：

```text
turning_vector_endpoint_error_mps          0.086518
high_lateral_vector_endpoint_error_mps     0.108752
wheel_inconsistent_vector_endpoint_error   0.085025
hard_accel_vector_endpoint_error_mps       0.086821
high_lat_accel_vector_endpoint_error_mps   0.086761
```

`vkd_gated` 在场景 gate 上的平均权重示例：

| scenario | gate vision | gate kinematic | gate dynamics |
| --- | ---: | ---: | ---: |
| turning | 0.6008 | 0.2321 | 0.1671 |
| high_lateral | 0.5816 | 0.2202 | 0.1982 |
| wheel_inconsistent | 0.5973 | 0.2342 | 0.1685 |
| hard_accel | 0.6124 | 0.1980 | 0.1896 |
| high_lat_accel | 0.6008 | 0.2293 | 0.1699 |

解释时要谨慎：gate 权重是模型内部融合权重，能提供解释线索，但不是严格的因果贡献证明。

## 9. EOVM-EKF 子模块

位置：

```text
eovm_ekf/
scripts/audit_eovm_inputs.py
scripts/precompute_eovm_base_velocity.py
```

EOVM-EKF 实现内容：

- RVST/EOVM 公式层。
- 13 状态 EKF prediction/update。
- Steering-angle Kalman filter，用于估计 `delta` 和 `delta_dot`。
- 输出速度估计 `vx, vy`。

EKF 状态顺序：

```text
[v_x, v_y, gamma, omega_1, omega_2, omega_3, omega_4, a_x, a_y, rho_1, rho_2, rho_3, rho_4]
```

SensorSample 包含：

```text
steering_angle
yaw_rate
wheel_speeds: [omega_1, omega_2, omega_3, omega_4]
ax
ay
```

预计算单 scene：

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

启用方式：

```yaml
data:
  base_velocity_source: eovm_ekf
  eovm_base_root: /mnt/sda2/zy/nuscenes/eovm_base_velocity
```

必须先审计：

```text
wheel speed 单位和比例
steering 字段符号和比例
yaw_rate 单位
longitudinal_accel / transversal_accel 方向
车辆几何参数
tire_radius
```

关键交接结论：

- 当前默认 VKD 不需要 EOVM CSV，也能跑全量 nuScenes。
- 当前 dynamics branch 使用 chassis proxy，不等于 EOVM。
- EOVM 是后续物理先验消融，不应在未审计前作为主结论。

## 10. 推荐运行流程

### 10.1 预计算灰度帧缓存

```bash
cd FlexiNet_ChassisFusion

python scripts/precompute_frames.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split trainval \
  --cam_channel CAM_FRONT \
  --height 64 --width 64 \
  --image_mode gray \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

### 10.2 路径和模型 sanity check

```bash
python scripts/path_sanity_check.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --skip-dataset \
  --skip-train-step
```

有数据后：

```bash
python scripts/path_sanity_check.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --skip-train-step
```

注意：此脚本当前没有把 `frame_cache_root` 传入 dataset，因此 dataset 探针可能不代表缓存路径实际使用情况。

### 10.3 Smoke 训练

```bash
python train.py --config configs/smoke_velocity_xy_vkd_gated.yaml
```

期望输出：

```text
train_loss=...
val_loss=...
val_mae_vx=...
val_mae_vy=...
Saved new best checkpoint ...
```

### 10.4 全量主方法训练

```bash
python train.py --config configs/velocity_xy_vkd_gated.yaml
```

### 10.5 消融实验

```bash
python train.py --config configs/velocity_xy_kd_only.yaml
python train.py --config configs/velocity_xy_vision_only.yaml
python train.py --config configs/velocity_xy_vision_kinematics.yaml
python train.py --config configs/velocity_xy_vkd_gated.yaml
```

推荐顺序：

```text
1. smoke_velocity_xy_vkd_gated
2. velocity_xy_kd_only
3. velocity_xy_vision_only
4. velocity_xy_vision_kinematics
5. velocity_xy_vkd_gated
6. audit_eovm_inputs
7. precompute_eovm_base_velocity
8. eovm variants
```

### 10.6 评估 best checkpoint

```bash
python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_vkd_gated.yaml \
  --checkpoint checkpoints/velocity_xy_vkd_gated/best.pth \
  --output_dir outputs/velocity_xy_vkd_gated/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/vkd_gated/eval_best
```

## 11. 验证命令

本地无完整 nuScenes 时，至少应跑：

```powershell
conda run -n speedXY_py310 python -m compileall dataset model utils train.py test.py smoke_model.py scripts analysis
conda run -n speedXY_py310 python -m pytest tests/test_velocity_xy_pipeline.py eovm_ekf/tests -q
conda run -n speedXY_py310 python smoke_model.py
conda run -n speedXY_py310 python smoke_model.py --with_flow
```

历史文档记录过：

```text
tests/test_velocity_xy_pipeline.py eovm_ekf/tests
14 passed
```

当前 `tests/test_velocity_xy_pipeline.py` 覆盖的关键契约包括：

- 二维 velocity label 使用中心差分并旋转到 ego frame。
- `fair_chassis` 不包含 pose 派生速度/加速度。
- 模型输出 shape 是 `[B, 2]`。
- `residual_gated` gate shape 是 `[B, active_branch_count]`，每行和为 1。
- KD-only 不创建视觉 backbone。
- `save_gate_weights=false` 时不保存 gate。
- dynamics proxy 特征顺序固定。
- 五个 VKD/消融配置都能构建二维模型。
- metrics 输出二维速度误差和分场景指标。
- knowledge loss 可 backward。
- predictions CSV 写 proxy 和 gate 列。
- frame cache normalization、label cache 和 manifest mismatch 行为。
- EOVM/precompute 脚本可以 `--help`。

## 12. 接手修改时不要破坏的接口

Dataset batch contract 必须保持：

```python
{
    "images": Tensor,
    "chassis": Tensor,
    "base_velocity": Tensor,
    "velocity": Tensor,
    "flow": Tensor,
    "geometry": Tensor,
    "base_speed": Tensor,
    "speed": Tensor,
    "scene_name": str,
    "center_token": str,
    "center_timestamp": Tensor,
}
```

模型输出默认必须保持：

```text
pred_velocity: [B, 2]
```

不要把 `output_dim` 改回 1，除非明确做旧标量 speed 消融。

不要把 `speed` / `base_speed` 当作当前主训练目标。它们只是兼容字段和派生指标。

不要把 `pose_assisted` 结果写成可部署主结论。

不要在未审计前把 EOVM-EKF 作为可靠主先验。

如果修改 chassis 特征顺序，必须同步检查：

```text
FAIR_CHASSIS_FEATURES
FAIR_CHASSIS_INDEX
DYNAMICS_PROXY_FEATURE_NAMES
FlexiNetChassisFusion._build_dynamics_proxy()
KnowledgeDrivenVelocityLoss._consistency_loss()
utils.evaluation.run_eval() 中的 proxy 抽取
tests/test_velocity_xy_pipeline.py
```

如果修改 evaluation 字段，必须同步检查：

```text
utils/evaluation.py
analysis/postprocess.py
analysis/compare_checkpoints.py
scripts/analyze_vkd_experiments.py
```

如果修改 dataset 参数，必须同步检查：

```text
train.py
test.py
analysis/inference.py
scripts/profile_training_step.py
scripts/path_sanity_check.py
smoke_model.py
tests/test_velocity_xy_pipeline.py
```

尤其注意 `frame_cache_root`：目前 `test.py` 和 `path_sanity_check.py` 没有传入它。

## 13. 给另一个模型的最小任务提示

如果要把本项目交给另一个模型，可以直接附上这段：

```text
请先阅读 `FlexiNet_ChassisFusion/ALGORITHM_HANDOFF.md`。

当前项目不是旧版标量 speed 估计，而是 nuScenes ego/body frame 下二维速度 `[v_x, v_y]` 估计。

核心合同：
- Dataset 返回 `images`, `chassis`, `base_velocity`, `velocity`, `flow`, `geometry`, `base_speed`, `speed` 和 metadata。
- Model 输出 `[B, 2]`。
- 主方法是 `configs/velocity_xy_vkd_gated.yaml`，使用 `fusion_mode: residual_gated`。
- 主预测形式是 `pred_velocity = base_velocity + delta_velocity`。
- 默认 `base_velocity = [base_vx, 0]`，EOVM-EKF 是可选且待审计的物理先验。
- `fair_chassis` 是主实验输入，`pose_assisted` 只作上限实验。
- 评估主指标是 `vector_endpoint_error_mps`, `mae_vx_mps`, `mae_vy_mps`, `gain_vs_base_*` 和分场景指标。

修改代码时不要把任务改回 1 维 speed，也不要破坏 dataset batch contract。
```

