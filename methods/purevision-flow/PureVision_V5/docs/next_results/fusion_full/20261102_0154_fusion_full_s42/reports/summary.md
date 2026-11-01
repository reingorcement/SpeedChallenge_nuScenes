# 完整数据 J23/S23/F23 seed42 结果

状态：**partial**；实验 `20261102_0154_fusion_full_s42`；代码 `bc9a3d8addbdb5413589e0b2121d87ee85e5b026`。

结论仅适用于本次完整数据seed42；完整train评估未完成时先交付dev核心结果。

| 组别 | Best轮 | Best dev EPE | Last dev EPE | Best train EPE | vx MAE/RMSE | vy MAE/RMSE | vx/vy偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| J23 | 20 | 0.184029 | 0.184029 | 0.100880 | 0.181020/0.320827 | 0.016503/0.025987 | -0.017897/0.000528 |
| F23 | 19 | 0.187446 | 0.187624 | 0.107092 | 0.184164/0.350759 | 0.016931/0.026382 | -0.018915/0.001089 |

A：相同完整23480窗口的匹配比较

| 比较（候选−基线） | ΔEPE | 相对变化 | 改善/退化/持平场景 |
|---|---:|---:|---:|
| F23_minus_J23 | 0.003417 | 1.86% | 60/43/0 |

B：按身份选取原pilot1826窗口，完整best保持不变

| 比较（候选−基线） | ΔEPE | 相对变化 | 改善/退化/持平场景 |
|---|---:|---:|---:|
| J23 | -0.203049 | -42.92% | 8/0/0 |
| F23 | -0.129322 | -32.40% | 8/0/0 |

C：固定历史参照，跨训练预算的实际精度比较

| 比较（候选−基线） | ΔEPE | 相对变化 | 改善/退化/持平场景 |
|---|---:|---:|---:|
| J23_minus_FLEX_original | -0.007601 | -3.97% | 48/55/0 |
| J23_minus_FLEX_continued | -0.007109 | -3.72% | 53/50/0 |
| J23_minus_RGB_appearance | -0.021039 | -10.26% | 78/25/0 |
| F23_minus_FLEX_original | -0.004184 | -2.18% | 63/40/0 |
| F23_minus_FLEX_continued | -0.003692 | -1.93% | 56/47/0 |
| F23_minus_RGB_appearance | -0.017622 | -8.59% | 79/24/0 |

固定历史checkpoint（阶段预算单独列出，不等于全部从零训练）：

| 历史参照 | 阶段预算 | 选中轮 | 完整dev EPE |
|---|---:|---:|---:|
| FLEX_original | 40 | 37 | 0.191630 |
| FLEX_continued | 40 | 29 | 0.191138 |
| RGB_appearance | 18 | 1 | 0.205068 |

重点场景和高速段（正式指标始终保留全部场景）：

| 组别 | scene-0402 EPE | scene-1013 EPE | 10–15m/s EPE | 10–15m/s vx偏差 |
|---|---:|---:|---:|---:|
| J23 | 0.190614 | 0.820085 | 0.362489 | -0.197044 |
| F23 | 0.195803 | 0.880667 | 0.328084 | -0.133027 |

| 分区 | abs(vy)≥0.2独立窗口 | 场景 | 侧向充分性 |
|---|---:|---:|---|
| train_core | 136 | 30 | sufficient_data |
| dev | 31 | 5 | insufficient_data |

逐轮两轴损失、全部103场景、速度分段、配对差异、0402/1013敏感性及来源见CSV/JSON。
曲线只支持提出下一步建议，不自动延长预算；cosine末端学习率为0，不能原样续跑视为有效扩训。
本轮未授权额外seed、模型、网格、训练轮数或编码器微调。

解释边界：

- Only this full-data seed42 is compared; pilot three-seed evidence cannot establish stable full-data superiority.
- A is matched new-reader training; C is historical accuracy context with different architectures, encoder training, precision and stage budgets.
- B restricts the already full-dev-selected best predictions by (scene,token) to the original 1826 pilot windows; no pilot-specific checkpoint selection.
- B jointly changes training coverage, update count and dev selection coverage, so it is not isolated data-volume causality.
- Official metrics include all scenes. scene-0402/1013 exclusion is diagnostic only and never changes selection or sampling.
- Online training loss is not best-checkpoint train EPE. Missing train EPE stays pending until its single full inference completes.
- PLANAR_PROXY and dev-selected frozen source; no held-out/test conclusion. Lateral adequacy follows the original independent-window rule.
