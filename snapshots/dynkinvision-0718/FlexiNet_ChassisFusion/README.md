# FlexiNet_ChassisFusion

`FlexiNet_ChassisFusion` 是一个面向 nuScenes 自车速度估计任务的多模态融合项目。它保留 FlexiNet 的视觉时序骨干网络，并加入 nuScenes CAN bus / 底盘运动学特征，通过视觉序列与底盘特征联合估计车辆速度。

默认使用残差式融合：

```text
predicted_speed = base_speed + learned_delta_v
```

其中 `base_speed` 来自 CAN bus 中的车速、轮速或 pose 速度先验，模型学习的是相对底盘速度先验的修正量 `delta_v`。这种设计适合在底盘信号本身已经较强时，用视觉时序信息补充和校正速度估计。

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
    evaluation.py
  train.py
  test.py
  smoke_model.py
  train.py.bak
  README.md
  checkpoints/
  runs/
  outputs/
  results_git/
```

## 代码文件说明

### `configs/default.yaml`

项目的中心配置文件，用于管理数据路径、模型参数、训练超参数和输出目录。

当前默认配置中的关键字段包括：

```yaml
data:
  dataroot: /mnt/sda2/zy/nuscenes
  version: v1.0-trainval
  seq_len: 13
  img_height: 64
  img_width: 64
  cam_channel: CAM_FRONT
  can_bus_root: /mnt/sda2/zy/nuscenes/can_bus
  allow_pose_fallback: false
  image_cache_size: 4096

model:
  pretrained_path: ../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
  fusion_mode: residual
  chassis_feature_dim: 16
  hidden_dim: 128
  dropout: 0.3
  freeze_backbone: true

train:
  batch_size: 16
  num_workers: 1
  epochs: 10
  shuffle_train: false
  learning_rate_head: 0.001
  learning_rate_backbone: 0.00003
  weight_decay: 0.0001
  use_amp: true
  seed: 42
  log_dir: runs/FlexiNet_ChassisFusion
  results_dir: results_git/FlexiNet_ChassisFusion
  save_dir: checkpoints
  loss: smoothl1
  grad_clip: 2.0
```

如果需要修改 nuScenes 路径、CAN bus 路径、预训练权重路径、batch size、学习率、是否冻结视觉骨干或输出目录，优先修改这个文件。

### `dataset/nuscenes_chassis.py`

定义训练和测试使用的 nuScenes 数据集。

主要类：

- `NuScenesCanBusAdapter`：读取 CAN bus JSON 文件，并按图像时间戳对齐底盘消息。
- `NuScenesChassisFusionDataset`：构建图像序列、底盘特征向量、底盘速度先验和目标速度。

单个样本返回：

```python
{
    "images": Tensor,            # [T, 1, H, W]，默认 [13, 1, 64, 64]
    "chassis": Tensor,           # [16]，底盘/CAN 特征向量
    "base_speed": Tensor,        # [1]，底盘速度先验，单位 m/s
    "speed": Tensor,             # [1]，目标速度，单位 m/s
    "scene_name": str,
    "center_token": str,
    "center_timestamp": Tensor,
}
```

目标速度由中心相机帧和下一帧之间的 ego pose 位移计算得到。CAN 特征只从满足 `utime <= image_timestamp` 的消息中选取，因此不会使用未来 CAN 信息。

期望的 CAN bus 文件布局类似：

```text
can_bus/
  scene-0061_pose.json
  scene-0061_vehicle_monitor.json
  scene-0061_zoe_veh_info.json
  scene-0061_steeranglefeedback.json
  ...
```

当前 16 维底盘特征顺序如下：

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

`base_speed` 的优先级为：

```text
vehicle_speed -> odom_speed -> pose_speed -> 0
```

### `model/modules.py`

包含 FlexiNet 视觉骨干使用的基础模块：

- `ContextualMotionAnalysis`
- `AdaptiveFeatureTransformer`
- `SpatialFeatureExtraction`
- `MotionFeatureExtraction`
- `DynamicIntegrationGate`
- `AdaptiveAvgPool3dStatic`

这些模块主要负责时序运动差分、视觉特征变换、空间特征提取、运动特征提取和动态融合。除非要重新设计视觉骨干，否则通常不需要改动这里。

### `model/flexinet_backbone.py`

将 FlexiNet 视觉模块封装成一个特征提取器。

输入形状：

```text
[B, T, C, H, W]
```

默认输入：

```text
[B, 13, 1, 64, 64]
```

输出：

```text
[B, 256]
```

这个 256 维向量就是融合模型使用的视觉特征。原始 FlexiNet 的速度回归头没有保留在这个封装中。

### `model/fusion_model.py`

定义多模态融合模型 `FlexiNetChassisFusion`。

内部模块：

- `backbone`：FlexiNet 视觉特征提取器。
- `chassis_encoder`：将 16 维底盘特征映射为紧凑特征的 MLP。
- `residual_speed_head`：在 `residual` 模式下预测速度修正量 `delta_v`。
- `direct_speed_head`：在 `direct` 模式下直接预测速度。

支持的模式：

```text
residual: predicted_speed = base_speed + delta_v
direct:   predicted_speed = fusion_head(visual_feature, chassis_feature)
```

重要方法：

```python
load_flexinet_backbone(checkpoint_path, strict=False)
freeze_backbone()
unfreeze_backbone()
```

`load_flexinet_backbone()` 只从原始 checkpoint 中加载 FlexiNet 视觉骨干相关权重，并忽略旧的速度回归头。

### `train.py`

训练入口。

主要职责：

- 从 YAML 配置文件读取参数。
- 构建训练集和验证集。
- 加载原始 FlexiNet 预训练权重。
- 按配置冻结或解冻视觉骨干。
- 训练底盘编码器和融合回归头。
- 保存 `checkpoints/last.pth` 和 `checkpoints/best.pth`。
- 写出训练历史、验证指标、预测文件和诊断图。

运行：

```bash
cd FlexiNet_ChassisFusion
python train.py --config configs/default.yaml
```

### `test.py`

评估入口。

主要职责：

- 加载已保存的 checkpoint。
- 在验证集上进行推理。
- 输出 loss、MAE、RMSE、R2 和相对 `base_speed` 的增益。
- 写出完整预测 CSV、指标 JSON 和评估图。

运行：

```bash
cd FlexiNet_ChassisFusion
python test.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

默认输出目录：

```text
outputs/FlexiNet_ChassisFusion/eval/
results_git/FlexiNet_ChassisFusion/test_eval/
```

### `smoke_model.py`

模型级 smoke test 脚本。它不需要 nuScenes 数据，只检查模型能否成功构建、可选加载预训练权重，并完成一次随机输入前向传播。

运行：

```bash
cd FlexiNet_ChassisFusion
python smoke_model.py --pretrained ../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

期望输出：

```text
Output shape: (2, 1)
Output sample: [...]
```

### `utils/evaluation.py`

评估和结果写出工具。

主要功能：

- `EvaluationResult`：保存预测值、目标值、base speed 和样本元信息。
- `compute_metrics()`：计算 MAE、RMSE、R2、base speed baseline 和模型相对 baseline 的增益。
- `save_predictions_csv()`：保存逐样本预测结果。
- `plot_training_history()`：绘制训练过程曲线。
- `plot_prediction_diagnostics()`：绘制预测诊断图。
- `write_evaluation_artifacts()`：统一写出指标、预测 CSV 和诊断图。
- `copy_small_results_for_git()`：将较小的 JSON、CSV 和 PNG 复制到 `results_git/`。

### `utils/metrics.py`

简单的评估指标工具，包含：

- `mae`
- `rmse`

### `utils/seed.py`

复现实验用的随机种子工具。它会设置：

- Python `random`
- NumPy
- PyTorch CPU/CUDA
- cuDNN deterministic mode

## 实验输出目录说明

项目中有三个主要结果目录：`runs/`、`outputs/` 和 `results_git/`。它们的定位不同。

### `runs/`

`runs/` 是训练阶段的完整日志和验证结果目录，由 `train.py` 根据 `train.log_dir` 写出。

当前目录结构：

```text
runs/
  FlexiNet_ChassisFusion/
    training_history.csv
    training_history.json
    train_val_loss_curve.png
    val_mae_rmse_curve.png
    last_eval/
      val_metrics.json
      val_predictions.csv
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
    best_eval/
      val_metrics.json
      val_predictions.csv
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
```

各文件含义：

- `training_history.csv`：逐 epoch 的训练/验证指标表格，适合用 Excel、Python 或论文绘图脚本继续分析。
- `training_history.json`：与 CSV 内容一致，但保留 JSON 格式，适合程序读取。
- `train_val_loss_curve.png`：训练 loss 与验证 loss 随 epoch 的变化曲线，用于判断收敛情况和过拟合风险。
- `val_mae_rmse_curve.png`：验证集 MAE、RMSE 以及 base speed baseline 的误差曲线，用于观察模型是否逐步超过底盘速度先验。
- `last_eval/`：最后一个 epoch 的验证结果。
- `best_eval/`：验证 loss 最优 checkpoint 对应的验证结果。

`runs/` 中会保留完整的 `val_predictions.csv`，因此它适合做后续误差分析、重画图、筛选 bad case 或按 scene 分组统计。

### `outputs/`

`outputs/` 是单独运行 `test.py` 后生成的完整评估输出目录。

当前目录结构：

```text
outputs/
  FlexiNet_ChassisFusion/
    eval/
      val_metrics.json
      val_predictions.csv
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
```

这些文件来自命令：

```bash
python test.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

各文件含义：

- `val_metrics.json`：测试入口在验证集上计算出的汇总指标，包括 loss、MAE、RMSE、R2、base speed baseline 误差和模型相对 baseline 的增益。
- `val_predictions.csv`：逐样本预测结果，包含 scene 名称、中心帧 token、时间戳、目标速度、模型预测速度、base speed、误差、绝对误差和残差修正量。
- `val_pred_vs_target.png`：预测速度与真实速度的散点对比图，同时绘制 `base_speed` 和理想对角线。点越靠近对角线，说明速度估计越准确。
- `val_residual_vs_speed.png`：预测残差 `pred_speed - target_speed` 随真实速度的分布。用于观察模型是否在低速、中速或高速区间存在系统性偏差。
- `val_abs_error_histogram.png`：绝对误差分布直方图。用于观察多数样本的误差集中范围，以及是否存在长尾大误差。
- `val_delta_v_histogram.png`：模型学习到的残差修正量 `delta_v = pred_speed - base_speed` 的分布。用于判断模型主要是在小幅校正底盘先验，还是经常做大幅修正。
- `val_speed_bin_mae.png`：按真实速度区间分桶后的 MAE 对比图，比较 `base_speed` baseline 与模型预测在不同速度段的误差。

`outputs/` 保留完整测试结果，是复核 `test.py` 输出和重新绘制图表时最直接的数据来源。

### `results_git/`

`results_git/` 是适合纳入版本管理或论文整理的小型结果目录。它由 `copy_small_results_for_git()` 从 `runs/` 或 `outputs/` 复制一部分较小文件而来，默认不复制大型逐样本预测 CSV。

当前目录结构：

```text
results_git/
  FlexiNet_ChassisFusion/
    training_history.csv
    training_history.json
    train_val_loss_curve.png
    val_mae_rmse_curve.png
    last_eval/
      val_metrics.json
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
    best_eval/
      val_metrics.json
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
    test_eval/
      val_metrics.json
      val_pred_vs_target.png
      val_residual_vs_speed.png
      val_abs_error_histogram.png
      val_delta_v_histogram.png
      val_speed_bin_mae.png
      val_true_vs_estimated_comparison.svg
      val_true_vs_estimated_zoom_first1000.svg
```

其中：

- 根目录下的 `training_history.*` 和训练曲线来自训练过程。
- `last_eval/` 来自训练最后一个 epoch 的验证结果摘要。
- `best_eval/` 来自训练期间验证 loss 最优的 checkpoint 结果摘要。
- `test_eval/` 来自单独运行 `test.py` 后的结果摘要。
- `val_true_vs_estimated_comparison.svg` 是基于 `outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv` 额外绘制的真值-估计值综合对比图。
- `val_true_vs_estimated_zoom_first1000.svg` 是前 1000 个验证样本的局部放大图，同时展示真值、模型估计值和 `base_speed`。

`results_git/` 适合用于快速查看实验结论、写报告、同步到 Git 或放入论文附录。若需要逐样本预测细节，应回到 `runs/` 或 `outputs/` 中读取 `val_predictions.csv`。

## 图表文件说明

### 训练曲线

`train_val_loss_curve.png`

- 横轴：epoch。
- 纵轴：loss。
- 曲线：训练 loss 与验证 loss。
- 用途：判断训练是否收敛，验证 loss 是否反弹，以及是否存在过拟合迹象。

`val_mae_rmse_curve.png`

- 横轴：epoch。
- 纵轴：速度误差，单位 `m/s`。
- 曲线：验证 MAE、验证 RMSE、base speed baseline MAE。
- 用途：判断模型是否随着训练逐步优于直接使用底盘速度先验。

### 预测质量图

`val_pred_vs_target.png`

- 横轴：真实速度 `target_speed_mps`。
- 纵轴：预测速度。
- 内容：模型预测点、`base_speed` baseline 点和理想对角线。
- 用途：查看整体拟合程度。点越贴近对角线，预测越准确；若点整体偏离对角线，说明存在系统性高估或低估。

`val_residual_vs_speed.png`

- 横轴：真实速度 `target_speed_mps`。
- 纵轴：预测残差 `pred_speed_mps - target_speed_mps`。
- 用途：检查误差是否随速度变化。例如高速段残差持续为正，说明模型在高速段偏高估；持续为负则说明偏低估。

`val_abs_error_histogram.png`

- 横轴：绝对误差 `abs(pred_speed_mps - target_speed_mps)`。
- 纵轴：样本数量。
- 用途：查看误差集中区间和长尾样本比例。左侧越集中，说明多数样本预测越稳定。

`val_delta_v_histogram.png`

- 横轴：残差修正量 `delta_v = pred_speed_mps - base_speed_mps`。
- 纵轴：样本数量。
- 用途：观察模型对底盘速度先验的修正幅度。如果大多数值接近 0，说明模型主要做小幅校正；如果分布很宽，需要检查 CAN 先验质量或模型是否过度修正。

`val_speed_bin_mae.png`

- 横轴：真实速度分桶。
- 纵轴：MAE，单位 `m/s`。
- 内容：不同速度区间下 `base_speed` baseline 与模型预测的 MAE 对比。
- 用途：判断模型在哪些速度段优于或弱于底盘速度先验。

### 真值-估计值对比图

`val_true_vs_estimated_comparison.svg`

- 数据来源：`outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv`。
- 内容：全验证集下采样后的真值/估计值趋势、真值-估计值散点图、预测误差趋势。
- 用途：同时查看时间顺序上的速度拟合情况、整体相关性和误差波动。

`val_true_vs_estimated_zoom_first1000.svg`

- 数据来源：`outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv`。
- 内容：前 1000 个验证样本的真值速度、模型估计速度和 `base_speed`。
- 用途：局部检查模型估计是否跟随真实速度变化，以及相比底盘速度先验是否更贴近真值。

## 当前测试结果摘要

`results_git/FlexiNet_ChassisFusion/test_eval/val_metrics.json` 中记录了当前 `test.py` 评估结果：

```text
MAE  = 0.0895 m/s = 0.3223 km/h
RMSE = 0.1246 m/s = 0.4485 km/h
R2   = 0.99874

Base MAE  = 0.1995 m/s = 0.7180 km/h
Base RMSE = 0.2813 m/s = 1.0126 km/h

MAE gain vs base  = 0.1099 m/s
RMSE gain vs base = 0.1567 m/s
```

这说明当前融合模型在验证集上优于直接使用 `base_speed` 作为速度估计。

## 环境要求

建议使用已经配置好的训练环境。环境中至少需要：

```text
torch
torchvision
numpy
Pillow
tqdm
PyYAML
nuscenes-devkit
matplotlib
```

`matplotlib` 用于保存训练曲线和评估诊断图。

## 数据假设

默认路径假设服务器上存在：

```text
/mnt/sda2/zy/nuscenes/
  maps/
  samples/
  sweeps/
  v1.0-trainval/
  can_bus/
```

如果服务器路径不同，需要修改：

```text
configs/default.yaml
```

重点修改：

```yaml
data:
  dataroot: /path/to/nuscenes
  can_bus_root: /path/to/nuscenes/can_bus

model:
  pretrained_path: /path/to/checkpoint_epoch_100_nuimage_L1_best.pth
```

## 推荐操作流程

### 1. 检查模型前向传播

```bash
python smoke_model.py --pretrained ../FlexiNet-main/pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth
```

### 2. 训练模型

```bash
python train.py --config configs/default.yaml
```

### 3. 评估最佳 checkpoint

```bash
python test.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

### 4. 查看结果

优先查看：

```text
results_git/FlexiNet_ChassisFusion/test_eval/val_metrics.json
results_git/FlexiNet_ChassisFusion/test_eval/val_true_vs_estimated_comparison.svg
results_git/FlexiNet_ChassisFusion/test_eval/val_true_vs_estimated_zoom_first1000.svg
results_git/FlexiNet_ChassisFusion/test_eval/val_pred_vs_target.png
results_git/FlexiNet_ChassisFusion/test_eval/val_speed_bin_mae.png
```

如需逐样本分析，查看：

```text
outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv
```

## 训练策略建议

### 小数据量或 nuScenes-mini

建议从冻结视觉骨干开始：

```yaml
model:
  freeze_backbone: true
  fusion_mode: residual
```

这样训练主要集中在 `chassis_encoder` 和融合回归头上，降低过拟合和破坏预训练视觉特征的风险。

### 解冻视觉骨干

在确认冻结骨干训练稳定后，再考虑：

```yaml
model:
  freeze_backbone: false

train:
  learning_rate_head: 0.0003
  learning_rate_backbone: 0.00001
```

视觉骨干学习率应明显低于融合头学习率。

### 显存不足

优先调整：

```yaml
train:
  batch_size: 1
  use_amp: true
  num_workers: 0
```

如果仍然不足，再考虑降低 `img_height` / `img_width` 或 `seq_len`。

## residual 与 direct 模式

默认推荐：

```yaml
model:
  fusion_mode: residual
```

当 CAN bus 能提供可靠速度先验时，`residual` 更适合，因为模型只需要学习修正量。

如果要做消融实验，可以使用：

```yaml
model:
  fusion_mode: direct
```

`direct` 模式会忽略 `base_speed + delta_v` 结构，直接从拼接后的视觉和底盘特征预测速度。

建议至少比较：

```text
1. base_speed baseline
2. FlexiNet + chassis direct fusion
3. FlexiNet + chassis residual fusion
```

## 开发建议

- 修改路径、学习率、batch size、AMP、输出目录等，优先改 `configs/default.yaml`。
- 修改 CAN bus 特征提取逻辑，主要改 `dataset/nuscenes_chassis.py`。
- 修改融合方式，主要改 `model/fusion_model.py`。
- 修改视觉骨干结构，才需要动 `model/modules.py` 和 `model/flexinet_backbone.py`。
- `train.py.bak` 是旧版训练脚本备份，当前运行入口以 `train.py` 为准。

