# V5 下一阶段实验状态

这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。

| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |
|---|---|---|---|---|---|
| 20261102_1545_fusion_confirm_tail_S23_TAIL_FT | S23_TAIL_FT | running | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_S23_FROZEN_CONT | S23_FROZEN_CONT | running | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_S23_s43 | S23_s43 | running | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_J23_s43 | J23_s43 | queued | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_F23_s43 | F23_s43 | queued | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_S23_s44 | S23_s44 | queued | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_J23_s44 | J23_s44 | queued | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |
| 20261102_1545_fusion_confirm_tail_F23_s44 | F23_s44 | queued | `f1fb5402e559847eb4ea51554a69daf50b6bc7dd` | — | — |

训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。
`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。
没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。
