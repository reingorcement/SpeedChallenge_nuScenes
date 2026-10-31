# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261031_r02_base_flex_s42_R02_BASE | R02_BASE | completed | `a847938681d60ae86e0e293e24ffa7d2e74e20a9` | — | 0.21514150947435656 |
| 20261031_r02_base_flex_s42_R02_FLEX | R02_FLEX | completed | `a847938681d60ae86e0e293e24ffa7d2e74e20a9` | — | 0.19162981273705193 |
| 20261031_r02_base_flex_s42_compare | compare | completed | `a847938681d60ae86e0e293e24ffa7d2e74e20a9` | —/— | — |
| 20261031_r02_base_flex_s43_s44_R02_BASE_s43 | R02_BASE_s43 | failed | `ace7519f8277026982cc843c4a95932ee64eeabc` | — | — |
| 20261031_r02_base_flex_s43_s44_R02_FLEX_s43 | R02_FLEX_s43 | completed | `ace7519f8277026982cc843c4a95932ee64eeabc` | — | 0.18307130493691354 |
| 20261031_r02_base_flex_s43_s44_R02_BASE_s44 | R02_BASE_s44 | failed | `ace7519f8277026982cc843c4a95932ee64eeabc` | — | — |
| 20261031_r02_base_flex_s43_s44_R02_FLEX_s44 | R02_FLEX_s44 | completed | `ace7519f8277026982cc843c4a95932ee64eeabc` | — | 0.18297927335656355 |
| 20261031_r02_base_flex_s43_s44_compare | compare | blocked | `ace7519f8277026982cc843c4a95932ee64eeabc` | —/— | — |
| 20261031_r02_base_retry_s43_s44_R02_BASE_s43 | R02_BASE_s43 | completed | `7857976ace26707923bae5a0e8e7f9d4bb1067ee` | — | 0.20453527452944956 |
| 20261031_r02_base_retry_s43_s44_R02_BASE_s44 | R02_BASE_s44 | completed | `7857976ace26707923bae5a0e8e7f9d4bb1067ee` | — | 0.2057318953064576 |
| 20261031_r02_base_retry_s43_s44_compare | compare | completed | `7857976ace26707923bae5a0e8e7f9d4bb1067ee` | —/— | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
