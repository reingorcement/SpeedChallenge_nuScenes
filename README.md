# SpeedChallenge_nuScenes

按方法整理的视觉速度估计仓库。先看 [方法对照](docs/METHODS.md)、[分支与恢复](docs/BRANCHES.md)、[来源清单](docs/SOURCES.json) 和 [本次验证](docs/VALIDATION.md)。

| 方法路线 | 目录 | 分支 |
|---|---|---|
| 普通 CNN：图像输入，无光流/底盘融合 | [cnn-image-only](methods/cnn-image-only/README.md) | baseline/cnn-image-only |
| FlexiNet：图像回归，无光流/底盘预测融合 | [flexinet-image-only](methods/flexinet-image-only/README.md) | baseline/flexinet-image-only |
| 光流纯视觉：V3、补偿版、V4 | [purevision-flow](methods/purevision-flow/README.md) | method/purevision-flow |
| 历史底盘融合、KD/VKD、V2 和完整实验记录 | [archive/legacy](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/archive/legacy) | archive/legacy |

main 汇总三个方法目录。各子项目分别进入目录运行，保留独立模型、配置、权重和输出；算法文件未因本次分类整理而修改。代码来源见 docs/SOURCES.json。

V3 与 V4 的训练/推理入口均已入库。V4 的 F_RC/F_R 可运行，新融合头仍需正式训练和独立评价。旧底盘融合方法不会因为名称含 visual 而被归类成严格纯视觉。

原始图像数据集、生成光流缓存和本机压缩包不放入方法目录。原远程实验结果通过归档分支及标签保全。各子项目已有权重保留；无需为本次整理安装 Git LFS。

```bash
python tools/verify_manifest.py
```

此命令只检查当前分支交付文件是否完整，不运行训练或访问数据集。运行方法前阅读对应目录 README；不要把不同旧项目的 `model`、`dataset` 加到同一个进程的导入路径。
