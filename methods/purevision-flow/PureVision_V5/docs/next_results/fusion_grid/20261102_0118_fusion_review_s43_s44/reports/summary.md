# 候选三seed复核

状态：**partial**；已核验 7/12 份结果（本轮新增 3/8）。

原 seed42 固定提交 `6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d`；seed43/44 固定提交 `c8c2370271312c654200a169d10f55bd0a76cc54`。

以下均值±样本标准差按seed计算（ddof=1），正式指标包含全部8个dev场景。

| 组别 | 已完成seed | Best dev EPE | Last dev EPE | Best train EPE | dev−train EPE |
|---|---|---:|---:|---:|---:|
| J23 | 42,43 | 0.475194 ± 0.002991 | 0.492155 ± 0.007356 | 0.226710 ± 0.027739 | 0.248484 ± 0.024748 |
| S23 | 42,43 | 0.416607 ± 0.032006 | 0.431997 ± 0.018669 | 0.162379 ± 0.030840 | 0.254229 ± 0.062846 |
| F23 | 42,43 | 0.404490 ± 0.007626 | 0.415281 ± 0.014719 | 0.167173 ± 0.006474 | 0.237317 ± 0.014101 |
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
| J23 | 44 | 待完成 | — | — | — |
| S23 | 44 | 待完成 | — | — | — |
| F23 | 44 | 待完成 | — | — | — |
| S46 | 44 | 待完成 | — | — | — |

| 配对（候选−基线） | ΔEPE均值±SD | 改善/退化/持平seed | 稳定性 |
|---|---:|---:|---|
| S23_minus_J23 | -0.058587 ± 0.029015 | 2/0/0 | pending_all_three_seeds |
| S23_minus_F23 | 0.012117 ± 0.024379 | 1/1/0 | pending_all_three_seeds |
| S46_minus_S23 | 0.045154 (n=1) | 0/1/0 | pending_all_three_seeds |

| 组别 | vx MAE | vx RMSE | vx有符号偏差 | vy MAE | vy RMSE |
|---|---:|---:|---:|---:|---:|
| J23 | 0.472269 ± 0.003356 | 0.801446 ± 0.000763 | -0.042357 ± 0.072139 | 0.022737 ± 0.000028 | 0.032970 ± 0.000216 |
| S23 | 0.412921 ± 0.032506 | 0.669508 ± 0.054980 | -0.141802 ± 0.044292 | 0.026001 ± 0.004040 | 0.035624 ± 0.003938 |
| F23 | 0.400317 ± 0.007644 | 0.677070 ± 0.059338 | -0.127600 ± 0.038436 | 0.027361 ± 0.000634 | 0.037215 ± 0.001282 |
| S46 | 0.434988 (n=1) | 0.798597 (n=1) | -0.115390 (n=1) | 0.027647 (n=1) | 0.036750 (n=1) |

scene-1013/0402敏感性仅诊断，不改变正式全场景结果：

| 配对 | 全场景ΔEPE | 去1013诊断 | 去0402诊断 | 去两者诊断 |
|---|---:|---:|---:|---:|
| S23_minus_J23 | -0.058587 ± 0.029015 | -0.045302 ± 0.005334 | -0.107638 ± 0.027549 | -0.100158 ± 0.000407 |
| S23_minus_F23 | 0.012117 ± 0.024379 | 0.034118 ± 0.041684 | -0.021010 ± 0.006329 | -0.000711 ± 0.023613 |
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
