# FlexiNet_ChassisFusion 学习计划

## 学习目标

完整掌握 `FlexiNet_ChassisFusion` 项目的代码结构、数据流、模型结构、训练逻辑、测试逻辑和实验结果解释方式，达到可以独立修改、复现实验、解释结果和排查问题的程度。

## 总体阅读顺序

不要从模型细节开始。推荐顺序是：

```text
README.md
configs/default.yaml
train.py
dataset/nuscenes_chassis.py
utils/evaluation.py
model/fusion_model.py
model/flexinet_backbone.py
model/modules.py
test.py
smoke_model.py
outputs/ runs/ results_git/
```

原因是：先理解项目目标和运行主线，再理解数据如何进入模型，然后看模型怎么融合，最后看结果如何产生和解释。

## 阶段 1：建立项目全局地图

### 先看文件

```text
README.md
configs/default.yaml
```

### 要弄清楚的问题

- 这个项目要解决什么任务？
- 输入是什么？输出是什么？
- 默认使用哪个 nuScenes 版本？
- 图像序列长度是多少？
- 底盘特征维度是多少？
- 模型默认是 `residual` 还是 `direct`？
- 视觉骨干默认冻结还是解冻？
- 训练结果、测试结果分别写到哪里？

### 读完后应该能说清楚

这个项目用 nuScenes 图像序列和 CAN bus 底盘特征估计 ego vehicle 速度。默认输入为 `[B, 13, 1, 64, 64]` 的灰度图像序列和 16 维底盘特征，模型在 `residual` 模式下输出：

```text
predicted_speed = base_speed + delta_v
```

## 阶段 2：读训练主线

### 先看文件

```text
train.py
```

### 推荐阅读顺序

1. `main()`
2. `load_config_from_yaml()`
3. `build_criterion()`
4. `run_eval()`
5. 训练循环里的 batch 处理逻辑
6. checkpoint 保存逻辑
7. 结果写出逻辑

### 要弄清楚的问题

- 配置文件如何进入代码？
- train / val 数据集在哪里创建？
- DataLoader 的 batch 中有哪些键？
- 模型在哪里创建？
- 预训练 FlexiNet 权重在哪里加载？
- `freeze_backbone` 在哪里生效？
- loss 是怎么计算的？
- optimizer 有几个参数组？
- AMP 和梯度裁剪在哪里做？
- best checkpoint 依据什么指标保存？
- `runs/` 和 `results_git/` 是什么时候写出的？

### 读完后应该能画出的主线

```text
default.yaml
  -> train.py
  -> NuScenesChassisFusionDataset
  -> DataLoader batch
  -> FlexiNetChassisFusion
  -> loss(pred, speed)
  -> optimizer.step()
  -> run_eval()
  -> checkpoints + runs + results_git
```

## 阶段 3：读数据集和标签构造

### 先看文件

```text
dataset/nuscenes_chassis.py
```

### 推荐阅读顺序

1. `NuScenesChassisFusionDataset.__init__()`
2. `_scene_names_for_split()`
3. `_load_frame_tensor()`
4. `_compute_speed_label()`
5. `NuScenesCanBusAdapter.extract_feature_vector()`
6. `NuScenesChassisFusionDataset.__getitem__()`
7. `_compute_pose_fallback()`

### 要弄清楚的问题

- scene 是如何按 split 选择的？
- 一个样本为什么是连续 `seq_len` 帧？
- 中心帧是如何选的？
- 图像如何 resize、灰度化、归一化？
- 目标速度 `speed` 是怎么从 ego pose 算出来的？
- CAN bus 消息如何按时间戳对齐？
- 16 维底盘特征每一维是什么？
- `base_speed` 的优先级是什么？
- `allow_pose_fallback` 是什么，为什么只能调试用？

### 读完后应该能说清楚

`__getitem__()` 返回的是模型真正吃进去的样本。重点不是 nuScenes API 本身，而是这个项目如何把 camera sample_data、ego pose 和 CAN bus JSON 对齐成一个训练样本。

## 阶段 4：读评估和结果生成

### 先看文件

```text
utils/evaluation.py
```

### 推荐阅读顺序

1. `EvaluationResult`
2. `compute_metrics()`
3. `save_predictions_csv()`
4. `write_evaluation_artifacts()`
5. `plot_training_history()`
6. `plot_prediction_diagnostics()`
7. `copy_small_results_for_git()`

### 要弄清楚的问题

- MAE、RMSE、R2 是怎么计算的？
- `base_mae_mps` 和 `base_rmse_mps` 是什么？
- `mae_gain_vs_base_mps` 如何解释？
- `val_predictions.csv` 每一列是什么？
- 每张 PNG 图是从哪些列画出来的？
- 为什么 `results_git/` 默认没有 `val_predictions.csv`？

### 读完后应该能解释的图

```text
train_val_loss_curve.png
val_mae_rmse_curve.png
val_pred_vs_target.png
val_residual_vs_speed.png
val_abs_error_histogram.png
val_delta_v_histogram.png
val_speed_bin_mae.png
val_true_vs_estimated_comparison.svg
val_true_vs_estimated_zoom_first1000.svg
```

## 阶段 5：读融合模型

### 先看文件

```text
model/fusion_model.py
```

### 推荐阅读顺序

1. `ChassisEncoder`
2. `FusionHead`
3. `FlexiNetChassisFusion.__init__()`
4. `FlexiNetChassisFusion.forward()`
5. `load_flexinet_backbone()`
6. `freeze_backbone()` / `unfreeze_backbone()`

### 要弄清楚的问题

- 16 维底盘特征如何变成底盘 embedding？
- 视觉特征是多少维？
- 底盘特征和视觉特征在哪里拼接？
- `residual` 和 `direct` 模式的区别是什么？
- `base_speed` 在 residual 模式中起什么作用？
- 为什么加载预训练权重时要忽略旧回归头？

### 读完后应该能写出的 forward 逻辑

```text
images -> FlexiNetBackbone -> visual_feat [B, 256]
chassis_vec -> ChassisEncoder -> chassis_feat [B, chassis_feature_dim]
[visual_feat, chassis_feat] -> FusionHead

residual:
  pred = base_speed + delta_v

direct:
  pred = direct_speed_head(fused)
```

## 阶段 6：读视觉骨干

### 先看文件

```text
model/flexinet_backbone.py
model/modules.py
```

### 推荐阅读顺序

1. `FlexiNetBackbone.forward_features()`
2. `ContextualMotionAnalysis`
3. `AdaptiveFeatureTransformer`
4. `SpatialFeatureExtraction`
5. `MotionFeatureExtraction`
6. `DynamicIntegrationGate`
7. `AdaptiveAvgPool3dStatic`

### 要弄清楚的问题

- 输入为什么要从 `[B, T, C, H, W]` 变成 `[B, C, T, H, W]`？
- CMA 如何使用相邻帧差分？
- AFT 如何用运动信息调制特征？
- SFE 和 MFE 分别偏向空间还是运动？
- DIG 如何融合空间和运动特征？
- 最后为什么输出 `[B, 256]`？

### 学习建议

这一阶段不要一开始就陷入每个卷积层的参数。先把张量形状流走通，再回头理解每个模块的意义。

## 阶段 7：读测试入口和 smoke test

### 先看文件

```text
test.py
smoke_model.py
```

### 要弄清楚的问题

- `test.py` 和 `train.py` 中的验证逻辑有什么复用关系？
- checkpoint 是怎么加载的？
- `test.py` 的输出目录和 `train.py` 的输出目录有什么区别？
- `smoke_model.py` 为什么不需要 nuScenes 数据？
- smoke test 能发现哪些问题，不能发现哪些问题？

## 阶段 8：读实验结果

### 先看目录

```text
runs/FlexiNet_ChassisFusion/
outputs/FlexiNet_ChassisFusion/eval/
results_git/FlexiNet_ChassisFusion/
```

### 推荐阅读顺序

1. `results_git/FlexiNet_ChassisFusion/test_eval/val_metrics.json`
2. `outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv`
3. `results_git/FlexiNet_ChassisFusion/test_eval/val_true_vs_estimated_comparison.svg`
4. `results_git/FlexiNet_ChassisFusion/test_eval/val_pred_vs_target.png`
5. `results_git/FlexiNet_ChassisFusion/test_eval/val_speed_bin_mae.png`
6. `runs/FlexiNet_ChassisFusion/training_history.csv`
7. `runs/FlexiNet_ChassisFusion/train_val_loss_curve.png`
8. `runs/FlexiNet_ChassisFusion/val_mae_rmse_curve.png`

### 要弄清楚的问题

- 当前模型 MAE / RMSE 是多少？
- 相比 `base_speed` baseline 提升多少？
- 误差主要集中在哪些速度段？
- 是否有系统性高估或低估？
- 训练过程中是否出现过拟合？
- best checkpoint 和 last checkpoint 差异大不大？

## 最终验收标准

学完后，你应该能独立回答：

- 这个项目的输入、输出和任务目标是什么？
- 一个训练样本是怎么从 nuScenes 和 CAN bus 中构造出来的？
- 目标速度标签是怎么来的？
- 16 维底盘特征分别是什么？
- `base_speed` 和 `delta_v` 分别是什么？
- `residual` 和 `direct` 融合模式有什么区别？
- FlexiNet 视觉骨干输出的特征维度是多少？
- 训练时哪些参数被冻结，哪些参数被训练？
- checkpoint、训练曲线、评估图分别保存在哪里？
- `runs/`、`outputs/`、`results_git/` 各自适合做什么？
- 当前实验结果是否优于 base speed baseline？
- 如果模型效果不好，应该优先检查数据、特征、训练还是模型？

## 建议学习节奏

### 第一天

- 读 `README.md`
- 读 `configs/default.yaml`
- 读 `train.py`
- 画出训练主流程图

### 第二天

- 读 `dataset/nuscenes_chassis.py`
- 手动跟踪一个样本从 scene 到 `__getitem__()` 的构造过程
- 写清楚 16 维底盘特征和速度标签来源

### 第三天

- 读 `utils/evaluation.py`
- 打开 `val_predictions.csv`
- 对照图表理解每张评估图

### 第四天

- 读 `model/fusion_model.py`
- 读 `model/flexinet_backbone.py`
- 只追张量形状，不纠结全部卷积细节

### 第五天

- 读 `model/modules.py`
- 理解 FlexiNet 视觉骨干每个模块的作用
- 回头复述完整 forward 流程

### 第六天

- 读 `test.py` 和 `smoke_model.py`
- 对照 `outputs/`、`runs/`、`results_git/` 理解训练结果和测试结果
- 总结当前实验结论

