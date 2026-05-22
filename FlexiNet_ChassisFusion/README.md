# FlexiNet 底盘运动学融合项目

本项目用于在 nuScenes 数据集上，对官方 FlexiNet 单目自车速度估计模型进行微调，并加入 nuScenes CAN bus expansion 中的底盘运动学信息。

默认设计保留原始 FlexiNet 的视觉主干网络，将原本只输出速度的回归头替换为多模态融合回归头，并采用残差式预测：

```text
predicted_speed = chassis_base_speed + learned_speed_correction
```

这个设计主要面向小数据量场景下的域适配。对于 `nuScenes-mini`，建议先只训练新加的底盘编码器和融合头；只有当验证误差仍然有稳定下降空间时，再考虑解冻视觉主干网络。

## 项目结构

```text
FlexiNet_ChassisFusion/
  configs/
    default.yaml
  dataset/
    __init__.py
    nuscenes_chassis.py
  model/
    __init__.py
    modules.py
    flexinet_backbone.py
    fusion_model.py
  utils/
    __init__.py
    metrics.py
    seed.py
  train.py
  test.py
  smoke_model.py
  README.md
```

## 文件功能说明

### `configs/default.yaml`

项目的中心配置文件，用于管理数据路径、模型设置和训练超参数。

重要字段：

```yaml
data:
  dataroot: .../data/nuscenes_mini
  version: v1.0-mini
  seq_len: 13
  cam_channel: CAM_FRONT
  can_bus_root: .../data/nuscenes_mini/can_bus

model:
  pretrained_path: .../FlexiNet/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
  fusion_mode: residual
  freeze_backbone: true

train:
  batch_size: 4
  epochs: 40
  learning_rate_head: 0.001
  learning_rate_backbone: 0.00003
```

如果要修改路径、序列长度、批大小、融合模式，或者是否冻结视觉主干，优先修改这个文件。

### `dataset/nuscenes_chassis.py`

定义训练和测试使用的 nuScenes 数据集。

主要类：

- `NuScenesCanBusAdapter`：读取官方 CAN bus JSON 文件，并将底盘消息按图像时间戳对齐。
- `NuScenesChassisFusionDataset`：返回图像序列、底盘特征向量、底盘基础速度和目标速度。

单个样本的返回格式：

```python
{
    "images": Tensor,      # [T, 1, H, W]，默认 [13, 1, 64, 64]
    "chassis": Tensor,    # [16]，底盘/CAN 特征向量
    "base_speed": Tensor, # [1]，底盘速度先验，单位 m/s
    "speed": Tensor,      # [1]，目标速度，单位 m/s
}
```

目标速度由中心相机帧和下一帧之间的 ego pose 位移计算得到。CAN 特征只从满足 `utime <= image_timestamp` 的消息中选取，因此不会使用未来 CAN 信息。

期望的 CAN bus 文件布局：

```text
data/nuscenes_mini/can_bus/
  scene-0061_pose.json
  scene-0061_vehicle_monitor.json
  scene-0061_zoe_veh_info.json
  scene-0061_steeranglefeedback.json
  ...
```

当前底盘特征向量的顺序如下：

```text
0  vehicle_speed
1  odom_speed
2  pose_speed
3  yaw_rate
4  throttle
5  brake
6  steering
7  steering_feedback
8  longitudinal_accel
9  transversal_accel
10 wheel_speed_mean
11 wheel_speed_std
12 accel_x
13 accel_y
14 rotation_rate_z
15 base_speed
```

所有速度相关数值都应理解为 `m/s`。部分 CAN 字段已经做了归一化或缩放，以避免特征量级差异过大。

### `model/modules.py`

包含从原始 FlexiNet 项目复制过来的视觉网络基础模块：

- `ContextualMotionAnalysis`
- `AdaptiveFeatureTransformer`
- `SpatialFeatureExtraction`
- `MotionFeatureExtraction`
- `DynamicIntegrationGate`
- `AdaptiveAvgPool3dStatic`

这个文件应尽量保持接近官方 FlexiNet 实现。除非你明确要修改视觉网络结构，否则不建议改动这里。

### `model/flexinet_backbone.py`

将原始 FlexiNet 的视觉模块封装成一个特征提取器。

输入：

```text
[B, T, C, H, W]
```

默认输入形状：

```text
[B, 13, 1, 64, 64]
```

输出：

```text
[B, 256]
```

这个 256 维向量就是融合模型使用的视觉特征。原始 FlexiNet 的 `fc` 速度回归头在这里被有意移除。

### `model/fusion_model.py`

定义多模态融合模型。

主类：

```python
FlexiNetChassisFusion
```

内部模块：

- `backbone`：FlexiNet 视觉特征提取器。
- `chassis_encoder`：将 16 维底盘特征向量映射为紧凑特征的 MLP。
- `residual_speed_head`：预测速度修正量 `delta_v`。
- `direct_speed_head`：当 `fusion_mode: direct` 时，直接预测速度。

支持的模式：

```text
residual: predicted_speed = base_speed + delta_v
direct:   predicted_speed = fusion_head(visual_feature, chassis_feature)
```

重要方法：

```python
load_flexinet_backbone(checkpoint_path)
freeze_backbone()
unfreeze_backbone()
```

`load_flexinet_backbone` 只从原始 checkpoint 中加载官方 FlexiNet 的视觉模块权重，并忽略旧的 `fc` 回归头。

### `train.py`

主训练入口。

主要职责：

- 从 YAML 配置文件读取参数。
- 构建训练集和验证集。
- 加载官方 FlexiNet 预训练权重。
- 按配置冻结视觉主干网络。
- 训练底盘编码器和融合回归头。
- 保存 `checkpoints/last.pth` 和 `checkpoints/best.pth`。

运行：

```bash
conda activate vision_est
cd D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/FlexiNet_ChassisFusion
python train.py --config configs/default.yaml
```

### `test.py`

评估入口。

主要职责：

- 加载已保存的 checkpoint。
- 在验证集上进行推理。
- 输出 loss、MAE 和 RMSE。

运行：

```bash
conda activate vision_est
cd D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/FlexiNet_ChassisFusion
python test.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

### `smoke_model.py`

模型级 smoke test 脚本。它不需要 nuScenes 数据，只检查模型能否成功构建、加载预训练权重并完成一次前向传播。

运行：

```bash
conda activate vision_est
cd D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/FlexiNet_ChassisFusion
python smoke_model.py --pretrained ../FlexiNet/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

期望输出：

```text
Output shape: (2, 1)
Output sample: [...]
```

### `utils/metrics.py`

简单的评估指标工具：

- `mae`
- `rmse`

由 `test.py` 使用。

### `utils/seed.py`

复现实验用的随机种子工具。它会设置：

- Python `random`
- NumPy
- PyTorch CPU/CUDA
- cuDNN deterministic mode

## 环境要求

使用你已经配置好的环境：

```bash
conda activate vision_est
```

环境中至少需要包含：

```text
torch
torchvision
numpy
Pillow
tqdm
PyYAML
nuscenes-devkit
```

如果无法导入 `nuscenes`，需要在 `vision_est` 环境中安装 `nuscenes-devkit`。

## 数据假设

默认路径假设当前仓库结构如下：

```text
09_SpeedChallenge_nuScenes/
  data/
    nuscenes_mini/
      maps/
      samples/
      sweeps/
      v1.0-mini/
      can_bus/
  FlexiNet/
    pretrained_models/
      checkpoint_epoch_100_nuimage_L1_best.pth
  FlexiNet_ChassisFusion/
```

如果服务器上的路径不同，请修改：

```text
configs/default.yaml
```

重点修改这些字段：

```yaml
data:
  dataroot: /path/to/nuscenes_mini
  can_bus_root: /path/to/nuscenes_mini/can_bus

model:
  pretrained_path: /path/to/checkpoint_epoch_100_nuimage_L1_best.pth
```

## 推荐操作流程

### 第 1 步：先在不加载数据的情况下检查模型

```bash
python smoke_model.py --pretrained ../FlexiNet/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

这一步可以在加载 nuScenes 之前，提前发现 `torch` 缺失、checkpoint 路径错误、模型维度错误等问题。

### 第 2 步：检查数据路径

确认以下目录存在：

```text
data/nuscenes_mini/v1.0-mini
data/nuscenes_mini/samples
data/nuscenes_mini/sweeps
data/nuscenes_mini/can_bus
```

确认 CAN 文件类似下面这种命名：

```text
scene-0061_pose.json
scene-0061_vehicle_monitor.json
scene-0061_zoe_veh_info.json
scene-0061_steeranglefeedback.json
```

### 第 3 步：训练冻结视觉主干的融合模型

使用默认配置：

```bash
python train.py --config configs/default.yaml
```

默认行为：

```text
freeze_backbone: true
fusion_mode: residual
seq_len: 13
batch_size: 4
```

这是在 `nuScenes-mini` 上最稳妥的起步设置。

### 第 4 步：评估模型

```bash
python test.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

重点关注：

```text
MAE  越低越好
RMSE 越低越好
```

### 第 5 步：最后再考虑解冻视觉主干

如果验证集 MAE 长时间停在较高水平，并且训练过程稳定，再修改：

```yaml
model:
  freeze_backbone: false

train:
  learning_rate_backbone: 0.00001
```

视觉主干的学习率要明显小于融合头的学习率。

## 操作与优化指南

### 显存较小时

从下面配置开始：

```yaml
train:
  batch_size: 1
  num_workers: 0
  use_amp: true
```

如果可以正常运行，再增加到：

```yaml
batch_size: 2
```

然后再尝试：

```yaml
batch_size: 4
```

不要优先降低 `seq_len`。FlexiNet 原本就是围绕 13 帧时序上下文设计的，缩短序列长度会改变任务设定。

### 想提高训练速度时

使用：

```yaml
train:
  use_amp: true
  num_workers: 4
```

在 Windows 上，如果 dataloader 启动不稳定，可以降低为：

```yaml
num_workers: 0
```

在 Linux 服务器上，可以尝试：

```yaml
num_workers: 4
```

或者：

```yaml
num_workers: 8
```

具体取决于 CPU 和磁盘读取速度。

### 想提高 `nuScenes-mini` 上的训练稳定性时

使用保守设置：

```yaml
model:
  freeze_backbone: true
  fusion_mode: residual

train:
  learning_rate_head: 0.001
  learning_rate_backbone: 0.00003
  weight_decay: 0.0001
  loss: smoothl1
```

如果验证 loss 抖动很大：

```yaml
train:
  learning_rate_head: 0.0003
  batch_size: 2
```

如果训练 loss 很快下降，但验证 loss 变差，说明模型过拟合。此时应继续冻结视觉主干；只有在确认数据质量没有问题之后，再考虑调整融合头大小或 dropout。

### 什么时候用 `residual`，什么时候用 `direct`

默认使用 residual：

```yaml
model:
  fusion_mode: residual
```

当 CAN bus 能提供有用的基础速度时，residual 模式最合适。模型只需要学习速度修正量。

做消融实验时可以使用 direct：

```yaml
model:
  fusion_mode: direct
```

direct 模式用于测试模型是否能直接从视觉特征和底盘特征的融合结果中估计速度。

推荐的实验表：

```text
A. 原始 FlexiNet 预训练模型，不微调
B. 只微调 FlexiNet 视觉回归头
C. 只使用底盘运动学信息的 baseline
D. FlexiNet + direct late fusion
E. FlexiNet + residual fusion
```

当前项目已经实现了 D 和 E。A、B、C 是写最终实验结论前建议补充的 baseline。

### 学习率规则

冻结视觉主干时：

```yaml
learning_rate_head: 0.001
```

解冻视觉主干时：

```yaml
learning_rate_head: 0.0003
learning_rate_backbone: 0.00001
```

不要在 `nuScenes-mini` 上用 `1e-3` 的学习率训练整个模型。数据量太小，这样很容易破坏预训练视觉表示。

### 序列长度

推荐：

```yaml
seq_len: 13
```

原因：

- 官方 FlexiNet 是围绕 13 帧时序输入构建的。
- 预训练 checkpoint 也是在这个时序设定下训练的。
- 更短的序列会削弱运动线索，也会让与官方模型的比较不够干净。

只有在 `batch_size: 1` 且开启 AMP 之后仍然显存不足时，才考虑减少 `seq_len`。

### 图像尺寸

默认：

```yaml
img_height: 64
img_width: 64
```

这与原始项目风格一致，并且显存占用较低。

如果 GPU 显存足够，并且想测试更高的视觉细节，可以尝试：

```yaml
img_height: 96
img_width: 96
```

这会带来更高显存占用和更慢的训练速度。

### CAN 特征质量检查

在信任训练结果之前，先检查：

- `base_speed` 不是大部分样本都为 0。
- CAN 时间戳能和图像时间戳对齐。
- `vehicle_speed`、`odom_speed` 或 `pose_speed` 的数值范围合理。
- 验证集 scene 有对应的 CAN 文件。

危险信号：

```text
base_speed is zero for most samples
```

可能原因：

- `can_bus_root` 路径错误
- scene JSON 文件缺失
- CAN 消息文件命名和预期布局不同
- 时间戳单位不是微秒

### Pose Fallback

代码中提供了一个用于检查训练流程的 fallback 模式：

```bash
python train.py --config configs/default.yaml --allow_pose_fallback
```

这个模式使用历史 ego pose 构造一个临时的“类底盘特征向量”。它只用于调试，不应作为真实 CAN bus 底盘融合结果进行汇报。

## 服务器使用方式

在服务器上，需要复制或挂载：

```text
data/nuscenes_mini
data/nuscenes_mini/can_bus
FlexiNet/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
FlexiNet_ChassisFusion
```

然后把 `configs/default.yaml` 中的路径改成服务器路径。

推荐的服务器运行命令：

```bash
conda activate vision_est
cd /path/to/09_SpeedChallenge_nuScenes/FlexiNet_ChassisFusion
python train.py --config configs/default.yaml
```

长时间训练时：

```bash
nohup python train.py --config configs/default.yaml > train.log 2>&1 &
```

然后监控日志：

```bash
tail -f train.log
```

## 常见问题

### `ModuleNotFoundError: No module named 'torch'`

说明当前没有运行在正确环境中。

```bash
conda activate vision_est
```

### `ModuleNotFoundError: No module named 'nuscenes'`

需要安装或激活包含 `nuscenes-devkit` 的环境。

### `CAN bus root not found`

检查：

```yaml
data:
  can_bus_root: ...
```

该目录下应包含类似下面的文件：

```text
scene-0061_pose.json
scene-0061_vehicle_monitor.json
```

### Checkpoint 路径错误

检查：

```yaml
model:
  pretrained_path: ...
```

默认期望文件为：

```text
FlexiNet/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

### CUDA 显存不足

按顺序尝试：

```yaml
batch_size: 1
use_amp: true
num_workers: 0
```

只有这样仍然不够时，再考虑降低：

```yaml
seq_len: 9
```

### 验证结果很差

在改模型结构之前，先检查：

1. 官方预训练 checkpoint 是否真的加载成功？
2. `seq_len` 是否仍然是 13？
3. `base_speed` 是否非零，并且单位是否为 `m/s`？
4. 训练/验证 scene 划分是否正确？
5. 第一阶段训练时视觉主干是否冻结？
6. 验证 loss 是否优于 chassis-only baseline？

## 开发建议

保持原始 `FlexiNet/` 目录不变。当前项目被设计为独立项目，目的是让实验结果可以和下载下来的原始 FlexiNet 项目进行干净比较。

推荐的修改策略：

- 在 `configs/default.yaml` 中修改路径和超参数。
- 在 `dataset/nuscenes_chassis.py` 中修改 CAN 特征提取逻辑。
- 在 `model/fusion_model.py` 中修改融合结构。
- 除非你明确要设计新的视觉主干，否则不要修改 `model/modules.py`。

