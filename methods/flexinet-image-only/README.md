# FlexiNet：纯图像速度回归

## 原版 `FlexiNet-main`

方法 ID：`flexinet-image-original`。13 帧灰度图像序列，默认 64×64；`FlexiNet.forward(x)` 只接收图像，输出单一车速大小。没有显式光流输入，也不融合底盘信号。

`chassis_aligned` 表示与融合实验对齐场景：数据加载器按 CAN 文件是否存在筛选场景，模型不读取 CAN 数值作为输入。ego pose 用于生成标签。原 README 的“optical flow”等历史描述不替代当前代码的实际输入契约。

在 `FlexiNet-main/` 中配置数据路径后运行：

```bash
python train.py --mode nuscenes --config configs/train_chassis_aligned.yaml
python test.py --config configs/test_chassis_aligned.yaml
```

原预训练权重、nuScenes 微调的 best/last 权重、原 LICENSE 均保留。未在本次整理中重训或复测完整数据集精度。

## 派生版 `FlexiNet_ImageXY`

方法 ID：`flexinet-image-xy`。共享旧融合工程的实现，仅保留 `configs/velocity_xy_vision_only.yaml` 这一运行配置：direct 输出、灰度图像、运动学和动力学分支关闭、光流和几何特征关闭，输出 vx/vy。

**预测不依赖底盘字段，但历史数据加载和训练入口仍读取 CAN 数据；尚不是完全解除 CAN 文件依赖的独立部署包。** 不要使用默认配置启动。

```bash
cd FlexiNet_ImageXY
python train.py --config configs/velocity_xy_vision_only.yaml
python test.py --config configs/velocity_xy_vision_only.yaml --checkpoint checkpoints/velocity_xy_vision_only/best.pth
```

原灰度 vision-only 权重随本目录保留。原配置里的服务器数据路径需要在自己的副本中调整。共享模块中其他模型类的存在不代表此配置启用了它们。

V3 的 RGB direct 也是图像回归派生方法，代码和权重统一保存在 `method/purevision-flow` 的 `PureVision_V3/`，避免复制同一模型形成两套维护源。方法目录见仓库 `docs/METHODS.md`。

两个子项目分别进入目录、使用独立 Python 进程运行，不合并它们的 `model`/`dataset` 导入路径。
