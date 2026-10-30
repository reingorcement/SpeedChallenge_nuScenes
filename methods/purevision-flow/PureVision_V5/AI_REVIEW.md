# 给 GitHub 审核 AI 的入口

本分支已包含 V5 的全部模型、数据处理、光流、训练、评价、队列、巡视源码，
六组配置、测试，以及本轮说明与交接。请先读下面的交接和结果，再核对源码。
不要根据旧 README 的历史快照推断当前实验仍在排队。

## 阅读顺序

1. [完整中文交接](docs/HANDOFF_20261030.md)：实际进度、设计、设置、故障、边界及未解决问题。
2. [六组正式结果](docs/RESULT_20261030_S2.md)：完成状态、指标、解释限制与 checkpoint 校验和。
3. [GitHub 可读实验凭证](docs/review_evidence/20261030_s2/README.md)：实际配置、训练终态、独立评价收据、逐轮摘要、巡视和关闭证据。
4. [用户原始 V5 计划](docs/V5_EXECUTION_PLAN_USER.md)：原文件的逐字节副本，不代表全部后续路线已实施；本轮实际范围和调整见交接及下表。
5. [模型设计](docs/MODEL_DESIGN.md)、[数据审计](docs/DATA_AUDIT.md)、[光流审计](docs/FLOW_AUDIT.md)、[计划调整](docs/ADJUSTMENTS.md)。
6. [运行操作](docs/OPERATIONS.md)、[验证记录](docs/VALIDATION.md)、[队列故障](docs/INCIDENT_20261030_QUEUE.md)、[AMP 故障](docs/INCIDENT_20261030_AMP.md)。

原计划副本来源为用户原工作区的 `V5_最终执行计划.md`，SHA-256：
`f4a40f98e9df11812d3947b62384dac3cb1d594e3e76f925b15d65080a40a7a3`。
原工作区中的文件仍保留，本分支没有替换或提交原工作区的未提交工作。

## 版本和当前状态

- 成功训练/评价固定 commit：`3979268699ce6db46d536cf22dc16f02b969a465`。
- [固定训练源码](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/3979268699ce6db46d536cf22dc16f02b969a465/methods/purevision-flow/PureVision_V5)，标签 `v5-s2-ready-v0.3`。
- 六组结果记录 commit：`c1098dbcf504fc78340c1b8824aea698991c3699`，标签 `v5-s2-results-v0.1`。
- 本次交接发布只补文档和小型凭证，不改变该固定训练源码、配置或已完成结果。
- 截至 2026-10-30 下午，R00--R05 都完成 20 轮和独立 dev 评价；队列已退出，任务 cron 已自动移除，无下一轮训练。
- S3、G0/G1、S4、多种子、RGB 微调及 KITTI 速度训练尚未完成。

“独立评价”是单独加载 checkpoint 重算 **同一 dev**，并非独立测试集。
这些是冻结 V3 RGB 表示、单 seed 的 **PLANAR_PROXY** 结果，不是完整真实体轴任务结论。

## 源码导航

| 审核问题 | 文件 |
|---|---|
| 标签参考点、非均匀导数、坐标、时间 | [labels.py](purevision_v5/labels.py)、[data.py](purevision_v5/data.py)、[prepare_data.py](prepare_data.py) |
| 真实 forward、时间卷积、统一二维头、FB 对照 | [model.py](purevision_v5/model.py)、[六组配置](configs/) |
| 冻结 RGB、旧 CMA、区域顺序和特征缓存 | [rgb.py](purevision_v5/rgb.py)、[features.py](purevision_v5/features.py)、[保留的 RGB 源码](purevision_v5/_rgb_reference/) |
| RAFT、输入分辨率、向量缩放和 FB 阈值 | [flow.py](purevision_v5/flow.py)、[prepare_flow.py](prepare_flow.py) |
| 配对、输入白名单、缓存和划分隔离 | [datasets.py](purevision_v5/datasets.py)、[packed_flow.py](purevision_v5/packed_flow.py) |
| loss、AMP、真实更新数、checkpoint 和恢复 | [training.py](purevision_v5/training.py)、[train.py](train.py) |
| 指标、选模、侧向规则与独立评价 | [evaluation.py](purevision_v5/evaluation.py)、[evaluate.py](evaluate.py)、[collect_results.py](collect_results.py) |
| 固定代码、队列、GPU 锁、进程不确定状态 | [common.py](purevision_v5/common.py)、[pipeline.py](pipeline.py) |
| 启动验证与真实定时巡视 | [startup_watch.py](startup_watch.py)、[monitor.py](monitor.py)、[scripts](scripts/) |
| 自动化测试 | [tests](tests/) |

## 请重点审核，而不是先重写工程

1. 后轴中点、当前 yaw 系、m/s 和 PLANAR_PROXY 定义是否一致；未来 pose 是否严格只用于监督。
2. V3 RGB 权重历史来源及 scene/log 划分的证据和限制；R00 含 CMA 且没有原始外观旁路的含义。
3. 数值光流符号、幅值、单位、dt、下采样、合法性及 FB 的计算是否一致。
4. R01/R03 是否同时改变容量和时空表示；R04/R05 是否除硬屏蔽外保持相同提示和数值流来源。
5. AMP 跳步和恢复逻辑是否正确，是否把 processed batches 错当实际参数更新。
6. 仅 31 个非零侧向窗口、dev 复用、单 seed、末轮最优如何限制结论；flow-only 最好的结果应如何解释。
7. 已知维护项：评价规则硬编码与 JSON 脱节、RGB 缓存未逐 JPEG 绑定 SHA、粗尺度掩码近似、外部巡视环境文件权限过宽。
8. 下一轮应先验证预算/种子/RGB 微调还是新结构；不要把 S3/几何模块当成未经结果支持的默认补丁。

请把意见区分为：阻断当前结果可信度、限制结论范围、影响下一轮的维护问题、尚未证实的假设。
当前 `lateral_rule` 配置与硬编码相同，但仅改 JSON 不会改变实际评价规则；本次文档发布不声称已修复它。

## GitHub 能审核的范围

GitHub 包含源码、配置、测试、原计划、详细交接、结果摘要及选择性导出的工件凭证。
完整 nuScenes/KITTI 数据、光流数组、原始预测 CSV、大日志和本轮新 checkpoint 仍在服务器，
位置与校验信息在交接及凭证里。原有版本已跟踪的预训练权重仍保留。

凭证是服务器工件的副本或注明来源的摘要，不是重新训练，不等于外部审核者已经独立复验原数据和模型。
仅访问 GitHub 的 AI 可以审代码、配置、公式、证据内部一致性和结论边界，不能声称已经重跑六组实验。
本发布不包含 `network.env`、认证配置或凭据。不要依据外部绝对路径假定当前审核环境能访问训练机。

## 可运行的仓库检查

仓库根目录：

```bash
python tools/verify_manifest.py
```

已准备相同依赖环境时，在 V5 目录执行 CPU 测试：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m pytest -q -p no:cacheprovider tests
```

本轮记录的版本为 Python 3.11.17、torch 2.10.0+cu128、torchvision 0.25.0+cu128。
完整正式命令在凭证的 `pipeline_plan.json`；不要直接重跑已完成的队列或覆盖既有 run ID。
