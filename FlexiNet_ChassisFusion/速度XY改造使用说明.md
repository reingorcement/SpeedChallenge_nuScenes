# 纵侧向速度估计改造代码说明与使用方法

本文档记录当前 `feature/velocity-xy-fusion` 分支对 `FlexiNet_ChassisFusion` 的主要改动、设计目的和服务器端使用流程。

当前改动位置：

```text
D:\data\Master\01_Master_1\01_Research\09_SpeedChallenge_nuScenes_worktrees\velocity-xy-fusion\FlexiNet_ChassisFusion
```

运行 Python 命令时请使用 `speedXY_py310` 环境：

```powershell
conda run -n speedXY_py310 python ...
```

## 1. 改造目标

原版本任务是标量速度估计：

```text
pred_speed = base_speed + delta_v
```

本次改造把任务扩展为 ego/body frame 下的二维速度估计：

```text
target_velocity = [v_x, v_y]
base_velocity = [base_vx, base_vy]
pred_velocity = base_velocity + delta_velocity
```

其中：

- `v_x`：车辆前向速度，单位 m/s。
- `v_y`：车辆左向速度，单位 m/s。
- 默认第一版 `base_velocity = [base_vx, 0]`。
- 可选用预计算 `EOVM-EKF` 结果作为 `base_velocity = [base_vx, base_vy]`。

这样改的好处是：模型不再只学习速度大小，而是可以评估纵向、侧向、向量端点误差，以及在转弯/低速/横向高分位场景中的表现。

## 2. 主要代码改动

### 2.1 数据集与标签

文件：

```text
dataset/nuscenes_chassis.py
```

核心改动：

- 新增 `NuScenesChassisFusionDataset._compute_velocity_label()`。
- 用中心差分从 nuScenes `ego_pose` 生成二维速度标签。
- 将 global/world velocity 旋转到中心帧 ego/body frame。
- 样本构建时跳过没有 `prev` 或 `next` 的边界帧，避免用 `0` 作为错误监督。
- `__getitem__()` 新增返回：

```python
{
    "velocity": Tensor[2],
    "base_velocity": Tensor[2],
    "chassis": Tensor[K, C],
    "flow": Tensor[T-1, 2, H, W] 或 empty Tensor,
    "geometry": Tensor[4] 或 empty Tensor,
}
```

为了兼容旧代码，也保留：

```python
{
    "speed": Tensor[1],
    "base_speed": Tensor[1],
}
```

但训练和评估主路径已经改用 `velocity/base_velocity`。

### 2.2 fair_chassis 与 pose_assisted

文件：

```text
dataset/nuscenes_chassis.py
```

新增两种底盘特征模式：

```yaml
data:
  chassis_mode: fair_chassis
```

`fair_chassis` 只使用更接近真实车辆可部署的 CAN/底盘信号：

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
FL/FR/RL/RR wheel speed
wheel_speed_mean
wheel_speed_std
base_vx
```

`pose_assisted` 会额外加入 nuScenes pose 派生量：

```text
pose_speed
pose_vel_x
pose_vel_y
pose_accel_x
pose_accel_y
pose_rotation_rate_z
```

建议：

- 论文或主要实验使用 `fair_chassis`。
- `pose_assisted` 只作为上限实验，因为标签也来自 ego pose，pose 派生输入可能造成同源信息泄漏。

### 2.3 底盘时间序列建模

文件：

```text
dataset/nuscenes_chassis.py
model/fusion_model.py
```

新增配置：

```yaml
data:
  chassis_window_sec: 1.0
  chassis_hz: 20.0
  online_mode: false

model:
  chassis_sequence_model: gru
```

数据集会在中心图像时间附近构建 CAN 时间窗口：

- `online_mode: false`：离线训练，使用中心时间前后窗口。
- `online_mode: true`：在线/因果推理，只使用历史窗口。

模型侧新增 `ChassisSequenceEncoder`，默认使用 GRU 编码 `[K, C]` 的底盘时间序列。

这样做的好处是：`v_y` 对 `yaw_rate`、`steering`、`transversal_accel` 的历史变化更敏感，单时刻 CAN 向量不足以表示转向动态和传感器延迟。

### 2.4 多分支视觉运动融合

文件：

```text
model/fusion_model.py
smoke_model.py
```

新增可选分支：

- 原 FlexiNet 图像序列分支：`gray` 或 `rgb`。
- 预计算光流分支：`flow_mode: precomputed`。
- 几何先验分支：`geometry_mode: basic`，当前包含 `yaw_rate`、`steering`、`steering_feedback`、`base_vx`。

默认仍保留灰度序列 baseline，不强制启用光流。这样方便做消融：

1. 灰度序列 + fair_chassis。
2. 灰度序列 + 光流 + fair_chassis。
3. RGB 序列 + 光流 + geometry + fair_chassis。
4. pose_assisted 上限实验。

### 2.5 二维训练、损失与评估

文件：

```text
train.py
test.py
utils/evaluation.py
```

训练目标改为：

```python
loss = VelocitySmoothL1Loss(pred_velocity, target_velocity)
```

支持分轴归一化：

```yaml
train:
  loss: velocity_smoothl1
  sigma_vx: 5.0
  sigma_vy: 0.5
  speed_loss_weight: 0.1
```

这样做是为了避免 `v_y` 数值通常远小于 `v_x`，在未归一化损失中被纵向速度误差淹没。

评估指标新增：

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
low_speed / mid_speed / high_speed 分组误差
turning / straight 分组误差
high_lateral 分组误差
```

预测 CSV 输出也已改为二维字段：

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

## 3. 配置文件说明

默认配置已经改为二维 fair_chassis baseline：

```text
configs/default.yaml
```

新增消融配置：

```text
configs/velocity_xy_fair_gray.yaml
configs/velocity_xy_fair_flow.yaml
configs/velocity_xy_fair_rgb_flow_geom.yaml
configs/velocity_xy_pose_assisted.yaml
```

建议服务器实验顺序：

1. `velocity_xy_fair_gray.yaml`
2. `velocity_xy_fair_flow.yaml`
3. `velocity_xy_fair_rgb_flow_geom.yaml`
4. `velocity_xy_pose_assisted.yaml`

先跑 1 和 4，可以快速判断：

- `fair_chassis` 下二维任务是否成立。
- `pose_assisted` 上限能到什么程度。
- pose 同源输入是否显著提高结果。

再跑 2 和 3，判断光流、RGB 和几何先验是否对 `v_y` 和转弯场景有真实贡献。

## 4. 服务器端使用流程

当前 PC 端没有完整 nuScenes 数据集，完整训练建议在服务器上执行。

### 4.1 进入项目目录

```bash
cd /path/to/FlexiNet_ChassisFusion
```

根据服务器实际路径修改配置中的：

```yaml
data:
  dataroot: /mnt/sda2/zy/nuscenes
  can_bus_root: /mnt/sda2/zy/nuscenes/can_bus

model:
  pretrained_path: ../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

### 4.2 预计算图像帧缓存（强烈推荐，决定训练速度）

`v1.0-trainval` 在 `CAM_FRONT` 的 `sample_data` 链上做步长为 1 的滑窗，会产生约 16 万个样本，每个样本要读 `seq_len` 张原图。由于窗口重叠且 `shuffle=True`，每个 epoch 实际要解码上百万次 1600×900 的 JPEG，这是服务器训练慢的主要原因。

先把帧（以及速度标签）预计算成 64×64 的 uint8 小数组落盘，训练时直接读：

```bash
# 一次性覆盖 train + val（split=trainval/all 表示全部 scene）
conda run -n speedXY_py310 python scripts/precompute_frames.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split trainval \
  --cam_channel CAM_FRONT \
  --height 64 --width 64 \
  --image_mode gray \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

输出：
- `<output_root>/<scene>.npy`：每个 scene 一个 `[N, H, W]`（gray）或 `[N, H, W, 3]`（rgb）uint8 数组。
- `<output_root>/manifest.json`：记录 `version / cam_channel / img_size / image_mode` 与各 scene 的 token 顺序（用于 token→帧映射）。
- `<output_root>/labels.json`：每个有效中心帧的速度标签 `[vx, vy]`。

然后在配置里指向该目录：

```yaml
data:
  frame_cache_root: /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

注意：
- 缓存与配置的 `img_height/img_width/image_mode/version/cam_channel` 必须一致，否则训练启动时会**主动报错**（防止用错缓存）。RGB 实验（如 `velocity_xy_fair_rgb_flow_geom.yaml`，`image_mode: rgb`）需要单独用 `--image_mode rgb` 生成一份缓存。
- 缓存包含 `labels.json` 且底盘走 CAN bus 时，DataLoader worker 不再需要加载几个 GB 的 NuScenes 元数据，可显著降低内存并加快 worker 启动。
- 不设置 `frame_cache_root` 时完全走原 JPEG 路径，结果可复现。

### 4.2.1 跑 fair_chassis 灰度 baseline

```bash
conda run -n speedXY_py310 python train.py --config configs/velocity_xy_fair_gray.yaml
```

训练完成后结果会写到配置中的：

```text
runs/FlexiNet_ChassisFusion_velocity_xy/fair_gray
checkpoints/velocity_xy_fair_gray
results_git/FlexiNet_ChassisFusion_velocity_xy/fair_gray
```

### 4.2.2 训练加速相关参数

`train` 段新增两个可选项（默认保持原行为）：

```yaml
train:
  eval_every: 1     # 每 N 个 epoch 才跑全量 val（最后一个 epoch 必评）；val 同样很费时，设 2~5 可明显提速
  log_interval: 20  # 每 N 步刷新一次进度条 loss，避免每步 .item() 强制 GPU 同步
```

其它已内置的加速改动（无需配置）：

- 训练时张量拷贝改为 `non_blocking=True`，配合 `pin_memory` 异步搬运。
- 训练循环用 `loss.detach()` 累加，减少每步 GPU 同步。
- 底盘特征抽取去掉重复抽取、用纯 Python 计算轮速标准差、序列一次性建张量。

想量化确认瓶颈，可用分步计时脚本（分别报告 load / transfer / compute）：

```bash
# 不用缓存（原始 JPEG 路径）
conda run -n speedXY_py310 python scripts/profile_training_step.py --config configs/velocity_xy_fair_gray.yaml --batches 50
# 用帧缓存对比
conda run -n speedXY_py310 python scripts/profile_training_step.py --config configs/velocity_xy_fair_gray.yaml --batches 50 \
  --frame_cache_root /mnt/sda2/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray
```

### 4.2.3 小规模冒烟测试（先验证路径与流程跑通）

正式全量训练前，建议先用极小规模验证：缓存路径是否正确、`Dataset` 能否读缓存、模型前向/反向/评估/存档能否跑通。相关开关：

- `data.max_scenes`：只用前 N 个 scene（train/val 各取前 N）。
- `train.limit_train_batches` / `train.limit_eval_batches`：每个 epoch 只跑前 N 个 batch。

配套已提供 `configs/smoke_velocity_xy_fair_gray.yaml`（3 个 scene、batch=4、各 ~10 个 batch、独立输出目录），它指向独立的小缓存 `…/CAM_FRONT_64x64_gray_smoke`，**不依赖也不干扰**正在跑的全量预计算。

步骤（服务器，环境 `zy_speedest`）：

```bash
cd FlexiNet_ChassisFusion
# 1) 生成小缓存：train 和 val 各取前 3 个 scene，写入同一目录（脚本支持增量合并）
python scripts/precompute_frames.py \
  --dataroot /media/ps/AC78E69278E65B1A/zy/nuscenes \
  --version v1.0-trainval --split train --limit_scenes 3 --image_mode gray \
  --output_root /media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray_smoke
python scripts/precompute_frames.py \
  --dataroot /media/ps/AC78E69278E65B1A/zy/nuscenes \
  --version v1.0-trainval --split val --limit_scenes 3 --image_mode gray \
  --output_root /media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_frames/CAM_FRONT_64x64_gray_smoke

# 2) 冒烟训练（秒级~分钟级完成）
python train.py --config configs/smoke_velocity_xy_fair_gray.yaml
```

只要能看到训练打印 `train_loss` / `val_mae_vx` 并在 `checkpoints/smoke` 写出 `last.pth`，就说明整条链路（缓存→数据集→模型→评估→存档）通畅。确认无误后，等全量 `CAM_FRONT_64x64_gray` 预计算完成，再跑正式配置即可。

> `max_scenes` 用的是按 split 过滤后的前 N 个 scene，和 `precompute_frames.py --split <同一split> --limit_scenes N` 的取景顺序一致，所以缓存与训练用到的 scene 能对上。

### 4.3 预计算光流

光流不建议在 `Dataset.__getitem__()` 中实时计算，服务器上先预计算：

```bash
conda run -n speedXY_py310 python scripts/precompute_optical_flow.py \
  --dataroot /mnt/sda2/zy/nuscenes \
  --version v1.0-trainval \
  --split train \
  --cam_channel CAM_FRONT \
  --output_root /mnt/sda2/zy/nuscenes/precomputed_flow/CAM_FRONT \
  --width 64 \
  --height 64
```

注意：

- 当前脚本使用 OpenCV Farneback 光流，依赖 `opencv-python`。
- 后续如果要用 RAFT 或 GMFlow，建议仍保持相同输出格式：每个相机 token 一个 `.npy`，形状为 `[H, W, 2]` 或 `[2, H, W]`。

### 4.4 跑光流消融

确认 `configs/velocity_xy_fair_flow.yaml` 中：

```yaml
data:
  flow_mode: precomputed
  flow_root: /mnt/sda2/zy/nuscenes/precomputed_flow/CAM_FRONT
```

然后运行：

```bash
conda run -n speedXY_py310 python train.py --config configs/velocity_xy_fair_flow.yaml
```

### 4.5 跑 RGB + 光流 + 几何先验融合

```bash
conda run -n speedXY_py310 python train.py --config configs/velocity_xy_fair_rgb_flow_geom.yaml
```

这组配置会：

- 使用 RGB 输入。
- 解冻 backbone。
- 输入尺寸提高到 `128x128`。
- 启用光流分支和 geometry prior 分支。

显存占用会更高，必要时调小：

```yaml
train:
  batch_size: 4
```

### 4.6 跑 pose_assisted 上限实验

```bash
conda run -n speedXY_py310 python train.py --config configs/velocity_xy_pose_assisted.yaml
```

这组实验不能作为可部署主结论，只用于判断 pose 派生输入的上限效果。

## 5. EOVM-EKF 与动态载荷转移

相关文件：

```text
eovm_ekf/
scripts/audit_eovm_inputs.py
scripts/precompute_eovm_base_velocity.py
```

当前 `eovm_ekf/eovm_ekf/eovm.py` 已实现动态载荷转移权重：

```python
load_transfer_weights(ax, ay, geometry, tuning)
```

需要的数据：

- `longitudinal_accel`
- `transversal_accel`
- 四轮轮速
- `yaw_rate`
- `steering` 或 `steering_feedback`

当前 nuScenes CAN 代码能读取这些字段，但仍需要服务器端审计：

- 四轮轮速单位是否能转成 rad/s。
- steering 字段符号和比例是否正确。
- nuScenes ego vehicle 的 `l/w/h/tire_radius` 是否能标定。

### 5.1 审计 EOVM 输入字段

```bash
conda run -n speedXY_py310 python scripts/audit_eovm_inputs.py \
  --can_bus_root /mnt/sda2/zy/nuscenes/can_bus \
  --scene_name scene-0061
```

输出会显示：

- 各 CAN 消息记录数量。
- 四轮轮速、加速度、yaw_rate、steering 的数值范围。
- 用当前默认车辆参数计算出的动态载荷转移 `theta`。

### 5.2 预计算 EOVM base_velocity

如果审计确认单位和比例后，可以先按 scene 预计算 EOVM 先验：

```bash
conda run -n speedXY_py310 python scripts/precompute_eovm_base_velocity.py \
  --can_bus_root /mnt/sda2/zy/nuscenes/can_bus \
  --scene_name scene-0061 \
  --output_csv /mnt/sda2/zy/nuscenes/eovm_base_velocity/scene-0061.csv \
  --wheel_speed_scale 1.0 \
  --steering_scale 0.001282051282
```

然后在配置中启用：

```yaml
data:
  base_velocity_source: eovm_ekf
  eovm_base_root: /mnt/sda2/zy/nuscenes/eovm_base_velocity
```

说明：

- 第一版主实验仍建议使用 `base_velocity_source: zero_lateral`。
- EOVM 作为后续物理先验消融，不建议在未完成单位/参数标定前作为默认主模型。

## 6. 本地验证命令

这些命令已经在 PC 端 `speedXY_py310` 环境中跑过：

```powershell
conda run -n speedXY_py310 python -m pytest tests/test_velocity_xy_pipeline.py eovm_ekf/tests -q
```

结果：

```text
14 passed
```

编译检查：

```powershell
conda run -n speedXY_py310 python -m compileall dataset model utils train.py test.py smoke_model.py scripts
```

模型冒烟测试：

```powershell
conda run -n speedXY_py310 python smoke_model.py
conda run -n speedXY_py310 python smoke_model.py --with_flow
```

两次输出 shape 都应为：

```text
Output shape: (2, 2)
```

## 7. 注意事项

1. 当前完整训练没有在 PC 端跑，因为 PC 端没有完整 nuScenes 数据集。
2. `flow_root`、`dataroot`、`can_bus_root`、`pretrained_path` 需要在服务器上按实际路径修改。
3. `fair_chassis` 是主实验模式，`pose_assisted` 只做上限实验。
4. 光流第一版使用 OpenCV Farneback，后续可替换为 RAFT/GMFlow，但建议保持预计算文件接口不变。
5. EOVM-EKF 已有动态载荷转移公式，但接入主实验前必须审计轮速单位、转向比例和车辆几何参数。
6. 评估主指标建议优先看：

```text
mae_vx_mps
mae_vy_mps
vector_endpoint_error_mps
gain_vs_base_vx_mps
gain_vs_base_vy_mps
turning_vector_endpoint_error_mps
high_lateral_mae_vy_mps
```

## 8. 推荐实验顺序

```text
Step 1: fair_gray
  验证二维标签、fair_chassis、base_velocity=[base_vx,0] 是否成立。

Step 2: pose_assisted
  得到 pose 同源上限，判断 fair 模式还有多大差距。

Step 3: fair_flow
  判断光流是否改善 v_y、转弯和 high_lateral 样本。

Step 4: fair_rgb_flow_geom
  判断 RGB、更高分辨率、光流、几何先验的组合收益。

Step 5: EOVM audit and base_velocity
  单位和参数审计后，再比较 EOVM base_velocity 与 zero_lateral base_velocity。
```

