# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261030_pilot_s42_R02_512 | R02_512 | completed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — | 0.9466885675627843 |
| 20261030_pilot_s42_flow1024_0 | flow1024_0 | failed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | —/— | — |
| 20261030_pilot_s42_flow1024_1 | flow1024_1 | failed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | —/— | — |
| 20261030_pilot_s42_pack1024 | pack1024 | blocked | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | —/— | — |
| 20261030_pilot_s42_R02_1024 | R02_1024 | blocked | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — | — |
| 20261030_pilot_s42_retry1_flow1024_0 | flow1024_0 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | 4642/— | — |
| 20261030_pilot_s42_retry1_flow1024_1 | flow1024_1 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | 4628/— | — |
| 20261030_pilot_s42_retry1_pack1024 | pack1024 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | —/— | — |
| 20261030_pilot_s42_retry1_R02_1024 | R02_1024 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | — | 0.8791637729320062 |
| 20261030_pilot_s43s44_R02_512_s43 | R02_512_s43 | completed | `252fb5f44c290c0d894faf8373329ba27105604e` | — | 1.0100648945304322 |
| 20261030_pilot_s43s44_R02_1024_s43 | R02_1024_s43 | completed | `252fb5f44c290c0d894faf8373329ba27105604e` | — | 0.8563112247326278 |
| 20261030_pilot_s43s44_R02_512_s44 | R02_512_s44 | completed | `252fb5f44c290c0d894faf8373329ba27105604e` | — | 0.9168585051180084 |
| 20261030_pilot_s43s44_R02_1024_s44 | R02_1024_s44 | completed | `252fb5f44c290c0d894faf8373329ba27105604e` | — | 0.8205532018835945 |
| 20261030_full_nuscenes_front1024_flow1024_0 | flow1024_0 | running | `87d7468f911da5579bbbc39f79ec6bc85fa16dd6` | 18104/77623 | — |
| 20261030_full_nuscenes_front1024_flow1024_1 | flow1024_1 | running | `87d7468f911da5579bbbc39f79ec6bc85fa16dd6` | 18400/77277 | — |
| 20261030_full_nuscenes_front1024_flow1024_2 | flow1024_2 | running | `87d7468f911da5579bbbc39f79ec6bc85fa16dd6` | 18504/77339 | — |
| 20261030_full_nuscenes_front1024_pack_train_dev | pack_train_dev | queued | `87d7468f911da5579bbbc39f79ec6bc85fa16dd6` | —/— | — |
| 20261030_full_nuscenes_front1024_verify_full_cache | verify_full_cache | queued | `87d7468f911da5579bbbc39f79ec6bc85fa16dd6` | —/— | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
