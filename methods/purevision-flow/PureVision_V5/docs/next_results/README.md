# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261101_2129_fusion_abc_s42_cache | cache | queued | `8eac695a778ee7bedc40ffe23fc42214c854811b` | —/— | — |
| 20261101_2129_fusion_abc_s42_compare | compare | queued | `8eac695a778ee7bedc40ffe23fc42214c854811b` | —/— | — |
| 20261101_2129_fusion_abc_s42_A | A | queued | `8eac695a778ee7bedc40ffe23fc42214c854811b` | — | — |
| 20261101_2129_fusion_abc_s42_B | B | queued | `8eac695a778ee7bedc40ffe23fc42214c854811b` | — | — |
| 20261101_2129_fusion_abc_s42_C | C | queued | `8eac695a778ee7bedc40ffe23fc42214c854811b` | — | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
