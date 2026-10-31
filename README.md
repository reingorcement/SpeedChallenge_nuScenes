# SpeedChallenge_nuScenes

**当前进度（2026-10-31）：** R02 BASE/FLEX 三seed完整对照已完成，FLEX确定为后续开发主线（不宣称已证明统计优越）。本轮R03_Frozen、R03_FT、R02_FLEX_CONT的实现、真实梯度与恢复验证已通过，见[实现与运行协议](methods/purevision-flow/PureVision_V5/docs/next/R03_FLEX_IMPLEMENTATION_20261031.md)。正式启动收据和后续进度以[结果分支](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results)为准。全部1000场景CAM_FRONT光流已完成，复用已有缓存。

**V5 GitHub 审核入口：** [AI 审核导航](methods/purevision-flow/PureVision_V5/AI_REVIEW.md) · [完整交接](methods/purevision-flow/PureVision_V5/docs/HANDOFF_20261030.md) · [六组正式结果](methods/purevision-flow/PureVision_V5/docs/RESULT_20261030_S2.md) · [实验凭证](methods/purevision-flow/PureVision_V5/docs/review_evidence/20261030_s2/README.md) · [用户原计划](methods/purevision-flow/PureVision_V5/docs/V5_EXECUTION_PLAN_USER.md)。2026-10-30 首轮 R00—R05 均已完成；结果仍限于单种子、冻结 RGB 的 PLANAR_PROXY 开发集比较。完整数据、缓存和本轮新 checkpoint 位于服务器，不随文档上传。

按方法整理的视觉速度估计仓库。先看 [方法对照](docs/METHODS.md)、[分支与恢复](docs/BRANCHES.md)、[来源清单](docs/SOURCES.json) 和 [本次验证](docs/VALIDATION.md)。

| 方法路线 | 目录 | 分支 |
|---|---|---|
| 普通 CNN：图像输入，无光流/底盘融合 | [cnn-image-only](methods/cnn-image-only/README.md) | baseline/cnn-image-only |
| FlexiNet：图像回归，无光流/底盘预测融合 | [flexinet-image-only](methods/flexinet-image-only/README.md) | baseline/flexinet-image-only |
| 光流纯视觉：V3、补偿版、V4 | [purevision-flow](methods/purevision-flow/README.md) | method/purevision-flow |
| V5：独立光流编码与特征融合 | [PureVision_V5](methods/purevision-flow/PureVision_V5/README.md) | method/purevision-v5 |
| 历史底盘融合、KD/VKD、V2 和完整实验记录 | [archive/legacy](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/archive/legacy) | archive/legacy |

main 汇总三个方法目录。各子项目分别进入目录运行，保留独立模型、配置、权重和输出；算法文件未因本次分类整理而修改。代码来源见 docs/SOURCES.json。

V3 与 V4 的训练/推理入口均已入库。V4 的 F_RC/F_R 可运行，新融合头仍需正式训练和独立评价。旧底盘融合方法不会因为名称含 visual 而被归类成严格纯视觉。

V5 的 S2 实现在 [PureVision_V5](methods/purevision-flow/PureVision_V5/README.md)：独立光流空间/时间编码、单二维输出和 FB 提示/硬筛选。使用明确的后轴中点 `PLANAR_PROXY` 标签；训练状态与实测结果按该目录记录，不沿用旧方法精度。

原始图像数据集、生成光流缓存和本机压缩包不放入方法目录。原远程实验结果通过归档分支及标签保全。各子项目已有权重保留；无需为本次整理安装 Git LFS。

```bash
python tools/verify_manifest.py
```

此命令检查交付文件哈希（V5 分支包含新增代码），不运行训练或访问数据集。V5 的实际流程检查见[验证记录](methods/purevision-flow/PureVision_V5/docs/VALIDATION.md)。运行方法前阅读对应目录 README；不要把不同旧项目的 `model`、`dataset` 加到同一个进程的导入路径。
