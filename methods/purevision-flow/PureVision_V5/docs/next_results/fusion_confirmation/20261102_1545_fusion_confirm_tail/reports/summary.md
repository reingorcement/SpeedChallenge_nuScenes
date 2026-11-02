# 完整数据三种子复核与 S23 新阶段20轮

状态：**partial**；实验 `20261102_1545_fusion_confirm_tail`；新代码 `f1fb5402e559847eb4ea51554a69daf50b6bc7dd`。

A 仅统计原完整数据协议 seed42/43/44；B 是从 seed42 父 best19 开始的新20轮阶段，二者分别报告。

## A 原协议三种子复核

| 组别 | seed | best轮 | best dev EPE | last dev EPE | best train EPE | dev−train |
|---|---:|---:|---:|---:|---:|---:|
| J23 | 42 | 20 | 0.184029 | 0.184029 | 0.100880 | 0.083149 |
| S23 | 42 | 19 | 0.178472 | 0.178734 | 0.099001 | 0.079471 |
| F23 | 42 | 19 | 0.187446 | 0.187624 | 0.107092 | 0.080354 |
| J23 | 43 | 19 | 0.182913 | 0.183771 | 0.101901 | 0.081012 |
| S23 | 43 | 19 | 0.179544 | 0.180326 | 0.096788 | 0.082756 |
| F23 | 43 | 19 | 0.185462 | 0.186273 | 0.106679 | 0.078782 |
| J23 | 44 | 17 | 0.187179 | 0.187292 | 0.104551 | 0.082628 |
| S23 | 44 | pending | pending | pending | pending | pending |
| F23 | 44 | pending | pending | pending | pending | pending |

| 组别 | 已有seed数 | best dev EPE均值 ± 样本SD |
|---|---:|---:|
| J23 | 3 | 0.184707 ± 0.002212 |
| S23 | 2 | 0.179008 ± 0.000758 |
| F23 | 2 | 0.186454 ± 0.001403 |

样本SD按 ddof=1；尚不足三seed时仅描述进度，不作稳定性结论。

| 配对（前者−后者） | ΔEPE均值 ± SD | 改善seed/已有seed | 方向 |
|---|---:|---:|---|
| S23_minus_J23 | -0.004463 ± 0.001547 | 2/2 | pending_all_three_seeds |
| S23_minus_F23 | -0.007446 ± 0.002162 | 2/2 | pending_all_three_seeds |
| F23_minus_J23 | 0.002983 ± 0.000614 | 0/2 | pending_all_three_seeds |

## B 两组 S23 新20轮阶段

父模型 `20261102_0154_fusion_full_s42_S23` best19，epoch0 dev EPE=0.178472；SHA `39d517001824414da1a49b6660a4cf5fdb9545c484cc87da0203032b068a5171`。
编码器固定起点 SHA `ab25fd6c4a7cd14a99a189f56e6b47913ed0f946a60cff2d0c9102479bc30ff8`。两阶段均新建 AdamW 与 cosine20；每组41560次更新。

| 组别 | 阶段best轮 | 阶段best dev | 阶段last dev | best train | 是否替换父模型 | epoch0含入后的选中轮 |
|---|---:|---:|---:|---:|---|---:|
| S23_FROZEN_CONT | 3 | 0.179102 | 0.179261 | 0.099592 | parent_epoch0 | 0 |
| S23_TAIL_FT | pending | pending | pending | pending | pending | pending |

| 对照（前者−后者） | Δdev EPE | 相对% | Δbest train EPE | 改善/退化场景 |
|---|---:|---:|---:|---:|
| S23_FROZEN_CONT_minus_parent_epoch0 | 0.000630 | 0.352919 | 0.000591 | 40/63 |

S23_FROZEN_CONT_minus_parent_epoch0：10–15m/s 共 1708 个窗口，ΔEPE=0.005411，Δvx偏差=-0.017252。偏差变化需与绝对偏差共同查看，正变化本身不等于精度提升。


## 两轴、侧向与完整证据

| 运行 | vx MAE/RMSE | vy MAE/RMSE | vx/vy偏差 | 非零vy MAE/RMSE | 近零误报率 |
|---|---:|---:|---:|---:|---:|
| S23_s42 | 0.175216/0.317155 | 0.016839/0.026312 | -0.021498/0.001938 | 0.232243/0.262340 | 0.000000 |
| J23_s42 | 0.181020/0.320827 | 0.016503/0.025987 | -0.017897/0.000528 | 0.232692/0.265516 | 0.000000 |
| F23_s42 | 0.184164/0.350759 | 0.016931/0.026382 | -0.018915/0.001089 | 0.228438/0.260998 | 0.000000 |
| S23_s43 | 0.176356/0.329192 | 0.016639/0.026133 | -0.014707/0.000512 | 0.232538/0.261244 | 0.000000 |
| J23_s43 | 0.179891/0.318238 | 0.016454/0.026003 | -0.005329/0.000878 | 0.234359/0.265828 | 0.000000 |
| F23_s43 | 0.182354/0.346112 | 0.016591/0.025963 | -0.010742/0.000102 | 0.228282/0.262193 | 0.000000 |
| J23_s44 | 0.184044/0.331866 | 0.016850/0.026558 | -0.018668/0.003003 | 0.235531/0.267421 | 0.000000 |
| S23_FROZEN_CONT_s42 | 0.175835/0.315150 | 0.017043/0.026244 | -0.017669/0.000981 | 0.230431/0.261085 | 0.000000 |

侧向正/负方向偏差、恒零vy参照及各子集独立样本数见 lateral_diagnostics.csv。dev非零侧向仍为31窗口/5场景，**insufficient_data**。
曲线数据见 learning_curves.csv；online train loss不是best-checkpoint完整train EPE。逐场景、速度段绝对值与配对差异均已导出；原始预测与checkpoint只引用路径及SHA，不重复推理。

## 限制与后续

- A contains only original full 20-epoch protocol seeds42/43/44. Parent-initialized stage20 runs are never pooled into A.
- Sample SD uses ddof=1 across training seeds. Three seeds are descriptive; repeated dev windows are not new independent labels.
- Order policy is seed+zero_based_epoch: same numbered epochs differ across seeds, but offset epochs can reuse permutations.
- B is seed42 only; no cross-seed stability claim and no attribution to an individual encoder branch from joint tail unfreezing.
- Stage best is selected only among epochs1..20. Parent epoch0 is separately eligible for replacement, with parent retained on ties.
- All 23480 dev windows remain the official metric. Focus-scene exclusions are diagnostic only; no holdout/test selection.
- Online train loss is not best-checkpoint train EPE; each best full train evaluation is performed once by its trainer.
- Dev nonzero lateral has 31 windows/5 scenes: insufficient_data remains, regardless of EPE or repeated seeds. PLANAR_PROXY is not true sideslip.
- Near-zero diagnostic is abs(true vy)<0.05 m/s, alarm abs(pred vy)>=0.2 m/s; no label or selection rule changes.
- High-speed bins above15m/s retain their small independent sample counts; signed bias closer to zero and accuracy are reported separately.
- No additional run, seed, budget, architecture or freeze change is automatically authorized by this report.

下一步由实际三seed配对方向、速度段取舍和B的验证收益共同审阅；本报告不自动追加训练。B仅seed42，任何稳定性复核须另行授权。
