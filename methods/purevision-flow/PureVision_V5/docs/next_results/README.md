# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261104_0025_s23_geometry_GEOMETRY_FULL | GEOMETRY_FULL | running | `c09c602f1296994336c8d079435f835a7ccaa462` | —/— | — |
| 20261104_0025_s23_geometry_PILOT_A | PILOT_A | running | `c09c602f1296994336c8d079435f835a7ccaa462` | — | — |
| 20261104_0025_s23_geometry_PILOT_B | PILOT_B | running | `c09c602f1296994336c8d079435f835a7ccaa462` | — | — |
| 20261104_0025_s23_geometry_FULL_A | FULL_A | queued | `c09c602f1296994336c8d079435f835a7ccaa462` | — | — |
| 20261104_0025_s23_geometry_FULL_B | FULL_B | queued | `c09c602f1296994336c8d079435f835a7ccaa462` | — | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
