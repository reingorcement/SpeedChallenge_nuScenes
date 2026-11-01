# 候选三seed复核

状态：**partial**；已核验 10/12 份结果（本轮新增 6/8）。

原 seed42 固定提交 `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d`；seed43/44 固定提交 `c8c2370271312c654200a169d10f55bd0a76cc54`。

以下均值±样本标准差按seed计算（ddof=1），正式指标包含全部8个dev场景。

| 组别 | 已完成seed | Best dev EPE | Last dev EPE | Best train EPE | dev−train EPE |
|---|---|---:|---:|---:|---:|
| J23 | 42,43,44 | 0.475710 ± 0.002296 | 0.488020 ± 0.008852 | 0.203667 ± 0.044471 | 0.272043 ± 0.044400 |
| S23 | 42,43,44 | 0.416605 ± 0.022631 | 0.430192 ± 0.013566 | 0.164035 ± 0.021995 | 0.252570 ± 0.044532 |
| F23 | 42,43,44 | 0.405153 ± 0.005514 | 0.414658 ± 0.010463 | 0.159092 ± 0.014726 | 0.246061 ± 0.018133 |
| S46 | 42 | 0.439130 (n=1) | 0.461279 (n=1) | 0.176976 (n=1) | 0.262154 (n=1) |

| 组别 | Seed | Best轮 | Best dev EPE | Last dev EPE（轮20） | Best train EPE |
|---|---:|---:|---:|---:|---:|
| J23 | 42 | 13 | 0.473079 | 0.486954 | 0.207096 |
| S23 | 42 | 12 | 0.393976 | 0.418796 | 0.184186 |
| F23 | 42 | 9 | 0.399097 | 0.404873 | 0.171751 |
| S46 | 42 | 12 | 0.439130 | 0.461279 | 0.176976 |
| J23 | 43 | 12 | 0.477308 | 0.497356 | 0.246324 |
| S23 | 43 | 18 | 0.439239 | 0.445198 | 0.140571 |
| F23 | 43 | 12 | 0.409883 | 0.425689 | 0.162595 |
| S46 | 43 | 待完成 | — | — | — |
| J23 | 44 | 19 | 0.476744 | 0.479749 | 0.157581 |
| S23 | 44 | 12 | 0.416601 | 0.426582 | 0.167348 |
| F23 | 44 | 19 | 0.406479 | 0.413412 | 0.142930 |
| S46 | 44 | 待完成 | — | — | — |

| 配对（候选−基线） | ΔEPE均值±SD | 改善/退化/持平seed | 稳定性 |
|---|---:|---:|---|
| S23_minus_J23 | -0.059105 ± 0.020536 | 3/0/0 | improved_all_three |
| S23_minus_F23 | 0.011452 ± 0.017277 | 1/2/0 | mixed_across_seeds |
| S46_minus_S23 | 0.045154 (n=1) | 0/1/0 | pending_all_three_seeds |

| 组别 | vx MAE | vx RMSE | vx有符号偏差 | vy MAE | vy RMSE |
|---|---:|---:|---:|---:|---:|
| J23 | 0.472807 ± 0.002549 | 0.816839 ± 0.026666 | -0.092338 ± 0.100480 | 0.022564 ± 0.000302 | 0.032987 ± 0.000155 |
| S23 | 0.412952 ± 0.022985 | 0.683037 ± 0.045392 | -0.137155 ± 0.032337 | 0.025378 ± 0.003054 | 0.035292 ± 0.002843 |
| F23 | 0.401288 ± 0.005661 | 0.678432 ± 0.042025 | -0.142469 ± 0.037441 | 0.025755 ± 0.002818 | 0.035909 ± 0.002437 |
| S46 | 0.434988 (n=1) | 0.798597 (n=1) | -0.115390 (n=1) | 0.027647 (n=1) | 0.036750 (n=1) |

scene-1013/0402敏感性仅诊断，不改变正式全场景结果：

| 配对 | 全场景ΔEPE | 去1013诊断 | 去0402诊断 | 去两者诊断 |
|---|---:|---:|---:|---:|
| S23_minus_J23 | -0.059105 ± 0.020536 | -0.046181 ± 0.004068 | -0.096245 ± 0.027728 | -0.087228 ± 0.022398 |
| S23_minus_F23 | 0.011452 ± 0.017277 | 0.026872 ± 0.032036 | -0.008226 ± 0.022591 | 0.006584 ± 0.020939 |
| S46_minus_S23 | 0.045154 (n=1) | -0.006583 (n=1) | 0.083669 (n=1) | 0.029455 (n=1) |

侧向 abs(vy)≥0.2 m/s：独立窗口 **13**、场景 **1**，状态 **insufficient_data**。重复seed不增加独立样本数，不能将该子集数量乘3。

**12份结果未齐全；当前仅报告已完成记录，不给候选胜负或下一阶段执行结论。**

逐seed最佳轮次、best/last、两轴误差、曲线、全部场景配对及速度分段见CSV；来源与恢复一致性校验见JSON。
已完成记录未检出来源、完整预算、最佳选择、恢复绑定、预测指标、同seed初始化或实际批次异常。

解释边界：

- Three seeds reuse the same 32/8 scenes and 7324/1826 windows; seed variation does not create new independent data.
- Mean and sample SD use ddof=1 across the three run metrics, not pooled predictions or per-window pseudo-replication.
- All-scene window-weighted dev EPE remains the formal selection metric. Excluding scene-1013/0402 is diagnostic only.
- Online train loss is not best-checkpoint train EPE. Train/dev gap uses the selected checkpoint.
- Cross-grid effects include integer position encoding; S46 effects cannot be attributed to one factor alone.
- Historical epoch order remains seed + zero-based epoch; different seeds may share shifted epoch permutations.
- F23 is a matched 2x3 diagnostic. Repeated seeds do not establish lateral adequacy or authorize a change of research direction.
- Source encoders were trained on original full train and selected on original full dev; this pilot is not held-out evidence.
