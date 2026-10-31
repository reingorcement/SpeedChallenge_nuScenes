# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261031_r03_flex_s42_R03_FT | R03_FT | running | `2922f8c4527b6c46eb664d02a4a041cd2c45f588` | — | — |
| 20261031_r03_flex_s42_R03_Frozen | R03_Frozen | completed | `2922f8c4527b6c46eb664d02a4a041cd2c45f588` | — | 0.20364393914279363 |
| 20261031_r03_flex_s42_R02_FLEX_CONT | R02_FLEX_CONT | completed | `2922f8c4527b6c46eb664d02a4a041cd2c45f588` | — | 0.1911382971946551 |
| 20261031_r03_flex_s42_compare | compare | queued | `2922f8c4527b6c46eb664d02a4a041cd2c45f588` | —/— | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
