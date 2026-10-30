# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。训练源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | training commit | dev EPE |
|---|---|---|---|---|
| 20261030_pilot_s42_R02_512 | R02_512 | completed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | 0.9466885675627843 |
| 20261030_pilot_s42_flow1024_0 | flow1024_0 | failed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — |
| 20261030_pilot_s42_flow1024_1 | flow1024_1 | failed | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — |
| 20261030_pilot_s42_pack1024 | pack1024 | blocked | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — |
| 20261030_pilot_s42_R02_1024 | R02_1024 | blocked | `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` | — |
| 20261030_pilot_s42_retry1_flow1024_0 | flow1024_0 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | — |
| 20261030_pilot_s42_retry1_flow1024_1 | flow1024_1 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | — |
| 20261030_pilot_s42_retry1_pack1024 | pack1024 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | — |
| 20261030_pilot_s42_retry1_R02_1024 | R02_1024 | completed | `be327b7f91f5f7e5ba46b70dede116a21c551652` | 0.8791637729320062 |
| 20261030_pilot_s43s44_R02_512_s43 | R02_512_s43 | running | `252fb5f44c290c0d894faf8373329ba27105604e` | — |
| 20261030_pilot_s43s44_R02_1024_s43 | R02_1024_s43 | running | `252fb5f44c290c0d894faf8373329ba27105604e` | — |
| 20261030_pilot_s43s44_R02_512_s44 | R02_512_s44 | queued | `252fb5f44c290c0d894faf8373329ba27105604e` | — |
| 20261030_pilot_s43s44_R02_1024_s44 | R02_1024_s44 | queued | `252fb5f44c290c0d894faf8373329ba27105604e` | — |

每个 run 目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
