# 光流纯视觉速度估计

本分支维护 `methods/purevision-flow/`。入口与运行边界见 [方法说明](methods/purevision-flow/README.md)。

- [本分支来源清单](docs/SOURCES.json)
- [方法分类](docs/METHODS.md)
- [验证记录](docs/VALIDATION.md)
- [整合主分支](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/main)
- [历史归档](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/archive/legacy)

运行 `python tools/verify_manifest.py` 检查交付文件。各子项目分别进入目录运行，不合并全局 `model`/`dataset` 导入路径。完整精度评价与正式训练状态见方法说明。

本快照包含：PureVision_V3、补偿速度估计、PureVision_V4。新融合头需要正式训练，现有 V3 权重不能当成 V4 成品权重。
