# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261102_0052_fusion_grid_s42_cache46 | cache46 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | 9150/9150 | — |
| 20261102_0052_fusion_grid_s42_publish46 | publish46 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | —/— | — |
| 20261102_0052_fusion_grid_s42_dual_summary | dual_summary | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | —/— | — |
| 20261102_0052_fusion_grid_s42_compare | compare | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | —/— | — |
| 20261102_0052_fusion_grid_s42_J23 | J23 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.4730789881462918 |
| 20261102_0052_fusion_grid_s42_S23 | S23 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.3939757528186787 |
| 20261102_0052_fusion_grid_s42_S46 | S46 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.4391302283926206 |
| 20261102_0052_fusion_grid_s42_J46 | J46 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.47218452599489064 |
| 20261102_0052_fusion_grid_s42_F23 | F23 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.3990972977483189 |
| 20261102_0052_fusion_grid_s42_R23 | R23 | completed | `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d` | — | 0.8716735553871353 |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
