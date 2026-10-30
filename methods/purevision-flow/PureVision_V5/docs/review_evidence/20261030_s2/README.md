# 本轮可供 GitHub 审核的实验凭证

这是 2026-10-30 六组已完成实验的选择性工件导出，不是新实验。
训练和独立 dev 评价使用 commit `3979268699ce6db46d536cf22dc16f02b969a465`，
run ID 为 `20261030_s42_r2_R00` 至 `20261030_s42_r2_R05`。
完整解释见[交接](../../HANDOFF_20261030.md)及[结果说明](../../RESULT_20261030_S2.md)。

## 来源和真实性边界

服务器工件根目录：
`/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5`。

[source_index.json](source_index.json)逐项记录 55 份来源工件的绝对路径、来源 SHA-256、字节数、导出路径、导出 SHA-256 和转换方式：

- `byte_copy` 是原始文件逐字节副本。
- `derived` 是注明删减方式的摘要。指标摘要和逐轮摘要只移除 `per_scene`；缓存摘要只移除大规模映射并加入条目数。
- 每组 `final_artifact_hashes.json` 是本次从最终 best/last checkpoint 和预测 CSV 重新读取计算的 SHA-256 清单；没有包含这些文件的内容。这 6 份现场计算清单不属于 `source_index.json` 的复制/摘要记录，其文件哈希由仓库根目录 `FILE_MANIFEST.json` 覆盖。
- 缓存数组 SHA 来自生成时索引，本次导出没有重新读取数十 GB 数组计算哈希。
- 源数据和工件未随文档修改。本目录也不证明 GitHub 审核者已经重新运行训练或独立验证原始数据。

全部指标仍是单 seed、冻结 RGB、PLANAR_PROXY、同一 dev 的选模与独立重算；并非独立测试集结果。

## 总览与实际命令

| 文件 | 内容 |
|---|---|
| [pipeline_plan.json](pipeline_plan.json) | 原始固定版本、依赖、GPU 分配、全部实际启动命令 |
| [pipeline_status.json](pipeline_status.json) | 队列完成状态 |
| [results.json](results.json) | 原始结果汇总，包括每场景指标、可比较性、侧向接受检查、来源身份和预测 SHA |
| [dataset_audit.json](dataset_audit.json) | 原始轻量数据/标签核验、时间分布、合成及敏感性结果 |
| [source_index.json](source_index.json) | 55 份复制/摘要工件的来源及导出校验和、转换方式 |

## 每组实验

| 实验 | 实际配置与契约 | 训练终态 | 独立评价收据 | 指标摘要 | 逐轮摘要 | 启动核验 | 最终工件 SHA |
|---|---|---|---|---|---|---|---|
| R00 | [run](R00/run.json) | [status](R00/status.json) | [evaluation](R00/evaluation.json) | [metrics](R00/metrics_summary.json) | [epochs](R00/epoch_summary.json) | [startup](R00/startup_proof_summary.json) | [hashes](R00/final_artifact_hashes.json) |
| R01 | [run](R01/run.json) | [status](R01/status.json) | [evaluation](R01/evaluation.json) | [metrics](R01/metrics_summary.json) | [epochs](R01/epoch_summary.json) | [startup](R01/startup_proof_summary.json) | [hashes](R01/final_artifact_hashes.json) |
| R02 | [run](R02/run.json) | [status](R02/status.json) | [evaluation](R02/evaluation.json) | [metrics](R02/metrics_summary.json) | [epochs](R02/epoch_summary.json) | [startup](R02/startup_proof_summary.json) | [hashes](R02/final_artifact_hashes.json) |
| R03 | [run](R03/run.json) | [status](R03/status.json) | [evaluation](R03/evaluation.json) | [metrics](R03/metrics_summary.json) | [epochs](R03/epoch_summary.json) | [startup](R03/startup_proof_summary.json) | [hashes](R03/final_artifact_hashes.json) |
| R04 | [run](R04/run.json) | [status](R04/status.json) | [evaluation](R04/evaluation.json) | [metrics](R04/metrics_summary.json) | [epochs](R04/epoch_summary.json) | [startup](R04/startup_proof_summary.json) | [hashes](R04/final_artifact_hashes.json) |
| R05 | [run](R05/run.json) | [status](R05/status.json) | [evaluation](R05/evaluation.json) | [metrics](R05/metrics_summary.json) | [epochs](R05/epoch_summary.json) | [startup](R05/startup_proof_summary.json) | [hashes](R05/final_artifact_hashes.json) |

注意：启动核验中的 checkpoint SHA 对应当时已保存的版本，之后正常训练会更新 best/last，
因此不应要求它等于最终 SHA。最终 best SHA 必须与独立评价收据及 `results.json` 一致，本次导出已核对。
逐轮 `epoch` 从 0 开始，汇总 `best_epoch` 从 1 开始；20 轮总计 41,560 个处理批次，AMP 跳步数另计。

## 缓存与比较条件

- [冻结 RGB 缓存摘要](cache/rgb_features_summary.json)
- [前向光流摘要](cache/forward_packed_summary.json)
- [双向/FB 缓存摘要](cache/bidirectional_packed_summary.json)

两种 flow/valid 数组 SHA 相同，双向缓存记录了 158,513 对的逐项一致性验证。
FB mask 的 SHA 不同是预期行为。索引中的 `verified_remote_commit: null` 是该打包入口未单独请求远端检查的记录；
不能仅由此推断代码未推送，应结合固定队列计划和版本管理记录判断。

## 巡视与故障证据

- [原调度安装记录](monitoring/schedule.json)、[六组启动检查完成](monitoring/watcher_20261030_s42_r2.json)、[结束后自动移除调度](monitoring/auto_disabled.json)。
- 实际定时 Astra 元数据：[12:00](monitoring/20261030T040001Z_3237516.meta.json)、[13:00](monitoring/20261030T050001Z_3254030.meta.json)、[14:00](monitoring/20261030T060001Z_3267366.meta.json)，均为北京时间；文件名时间为 UTC。
- 队列故障：[现场证据](incidents/20261030T040315Z_dependency_blocking_evidence.json)、[隔离复现](incidents/reproduction_9b859c2.json)、[数值代码一致性](incidents/numerical_identity_9b859c2_recovery.json)、[恢复启动](incidents/recovery_launch.json)。
- AMP：[隔离复现](incidents/amp_overflow_reproduction_gpu1.json)、[修复验证](incidents/amp_overflow_fix_gpu1.json)。

`schedule.json` 是历史安装/验证记录，不代表现在还有任务；结束状态以 `auto_disabled.json` 和交接记录为准。
巡视元数据记录模型、触发类型、退出状态和实际 shell 读取证据，但不等同于再次独立审计所有训练行为。
没有上传 `network.env`、凭据、crontab 整份备份、CLI 完整日志、原始训练日志、预测 CSV、数据、缓存数组或本轮新 checkpoint。
元数据中的绝对路径和日志引用指向服务器，GitHub 本身不会提供这些外部文件。
