# FlexiNet 与 ChassisFusion 速度估计结果汇报

生成日期：2026-05-29

项目位置：`09_SpeedChallenge_nuScenes`

本报告已用最新训练好的纯视觉 FlexiNet 测试结果覆盖旧版视觉结果。对比对象仍为同一批对齐验证样本上的三种方法：

- `Pure Vision FlexiNet`：最新纯视觉模型，输入前视图像序列。
- `Base Speed Baseline`：ChassisFusion 数据中的底盘速度先验 `base_speed`。
- `FlexiNet + ChassisFusion`：视觉特征与底盘运动学/动力学特征的 residual fusion 模型。

## 1. 核心结论

新版纯视觉 FlexiNet 的结果已经明显优于旧版视觉估计。最新纯视觉模型在对齐验证集上的 MAE 为 `1.2856 m/s`，约 `4.6283 km/h`，R2 为 `0.7137`。这说明纯视觉分支已经学到了有效速度尺度，不再是旧报告中接近失效的状态。

但是，从当前结果看，纯视觉模型仍然明显弱于底盘速度先验和融合模型。`base_speed` baseline 的 MAE 为 `0.1995 m/s`，而 ChassisFusion 的 MAE 进一步降低到 `0.0895 m/s`。因此，当前最可靠的结论是：**最新纯视觉模型有显著改进，但视觉+底盘 residual fusion 仍然是最优方案。**

三组方法在同一批验证样本上完成对齐，共 `33148` 个样本，`center_token` 对齐，最大标签差异为 `0 m/s`。

| 方法 | MAE (m/s) | MAE (km/h) | RMSE (m/s) | RMSE (km/h) | R2 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pure Vision FlexiNet | 1.2856 | 4.6283 | 1.8809 | 6.7713 | 0.7137 | 新版纯视觉已有明显速度估计能力 |
| Base Speed Baseline | 0.1995 | 0.7180 | 0.2813 | 1.0126 | 0.9936 | 底盘速度先验非常强 |
| FlexiNet + ChassisFusion | 0.0895 | 0.3223 | 0.1246 | 0.4485 | 0.9987 | 融合模型在强 baseline 上继续提升 |

相对于 `base_speed` baseline，ChassisFusion 的 MAE 下降 `0.1099 m/s`，相对下降 `55.1%`；RMSE 下降 `0.1567 m/s`，相对下降 `55.7%`。

相对于最新纯视觉模型，ChassisFusion 的 MAE 下降 `1.1961 m/s`，相对下降 `93.0%`。

## 2. 数据与对齐协议

本次重新生成分析报告使用以下输入文件：

- 最新纯视觉预测：`FlexiNet-main/outputs/FlexiNet-main/chassis_aligned_eval/val_predictions.csv`
- 最新纯视觉指标：`FlexiNet-main/outputs/FlexiNet-main/chassis_aligned_eval/val_metrics.json`
- ChassisFusion 预测：`FlexiNet_ChassisFusion/outputs/FlexiNet_ChassisFusion/eval/val_predictions.csv`
- ChassisFusion 指标：`FlexiNet_ChassisFusion/outputs/FlexiNet_ChassisFusion/eval/val_metrics.json`

对齐方式为 `center_token` 精确匹配，并使用 ChassisFusion CSV 的样本顺序作为最终分析顺序。纯视觉 CSV 行数为 `33148`，ChassisFusion CSV 行数为 `33148`，最终对齐样本数为 `33148`。

## 3. 指标解释

速度单位默认为 `m/s`，报告中同时给出 `km/h` 换算。核心指标包括：

- MAE：平均绝对误差，是主要精度指标。
- RMSE：均方根误差，对大误差更敏感。
- R2：决定系数，衡量预测能否解释真实速度变化。
- 误差分位数：用于观察典型误差和长尾误差。

## 4. 定量结果

| 方法 | P50 abs error (m/s) | P90 abs error (m/s) | P95 abs error (m/s) | P99 abs error (m/s) | Max abs error (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure Vision FlexiNet | 0.8367 | 3.0892 | 4.0402 | 6.5937 | 10.5709 |
| Base Speed Baseline | 0.1512 | 0.4606 | 0.5875 | 0.8832 | 1.6873 |
| FlexiNet + ChassisFusion | 0.0671 | 0.2065 | 0.2574 | 0.3694 | 0.8995 |

从分位数看，新版纯视觉的中位绝对误差为 `0.8367 m/s`，已经显著优于旧版视觉估计；但高分位误差仍然较大，P95 绝对误差为 `4.0402 m/s`。这说明纯视觉分支在多数样本上已能估计速度趋势，但在部分场景中仍不够稳定。

ChassisFusion 的 P95 绝对误差为 `0.2574 m/s`，明显低于 `base_speed` 的 `0.5875 m/s`，说明 residual correction 不只改善平均误差，也降低了较高分位误差。

## 5. Residual Correction 行为

ChassisFusion 的输出形式为：

```text
predicted_speed = base_speed + learned_delta_v
```

当前验证集上，修正量 `delta_v = pred_speed - base_speed` 的统计如下：

| 统计项 | 数值 (m/s) |
| --- | ---: |
| mean | 0.0947 |
| std | 0.2226 |
| min | -1.2207 |
| max | 1.1644 |
| P05 | -0.2926 |
| P50 | 0.0924 |
| P95 | 0.4705 |

该分布说明模型大多数情况下是在强底盘先验附近做有限幅度修正，而不是脱离 `base_speed` 重新预测完整速度。这符合 residual fusion 的设计目标。

## 6. 曲线对比图

重新生成的文件如下：

![First 1000 speed comparison](first1000_true_vs_vision_base_fusion.svg)

第一张图显示前 1000 个对齐验证样本上的四条曲线：

- True speed
- Latest pure vision FlexiNet prediction
- Base speed baseline
- FlexiNet + ChassisFusion prediction

新版纯视觉曲线相比旧版结果已经更接近真实速度，但仍能看到局部偏差和波动。`base_speed` 与 ChassisFusion 更贴近真实速度，其中 ChassisFusion 在局部细节上进一步修正 `base_speed`。

![Base and fusion detail](first1000_true_vs_base_fusion_detail.svg)

第二张图去掉纯视觉曲线，便于观察 `base_speed` 与 ChassisFusion 的细节差异。由于纯视觉误差尺度仍大于底盘和融合方法，细节图更适合展示 residual correction 的效果。

用于画图的对齐数据保存于：

- `analysis_reports/flexinet_speed_comparison/aligned_first1000_curve_data.csv`

统计摘要保存于：

- `analysis_reports/flexinet_speed_comparison/comparison_summary.json`

## 7. 结果解释

### 7.1 最新纯视觉模型的意义

最新纯视觉 FlexiNet 的 MAE 为 `1.2856 m/s`，R2 为 `0.7137`。这说明纯视觉模型已经能从图像时序中恢复一部分自车速度信息。与旧版视觉估计相比，本次结果应被解读为纯视觉路线取得了实际进展。

不过，纯视觉估计仍受到图像分辨率、场景纹理、动态目标、低速运动线索弱、遮挡和光照变化等因素影响。当前结果还不足以说明纯视觉可以替代底盘信号。

### 7.2 为什么底盘 baseline 仍然很强

`base_speed` 直接来自车辆底盘/运动学相关信号，与 ego pose 推导的速度标签高度相关，因此 baseline 本身已经很强。它的 MAE 为 `0.1995 m/s`，远低于纯视觉模型。

这也是报告中必须保留 `base_speed` 对比的原因：如果只比较纯视觉和 ChassisFusion，会高估融合模型相对于底盘先验的增益。

### 7.3 ChassisFusion 的价值

ChassisFusion 在强底盘 baseline 上仍将 MAE 从 `0.1995 m/s` 降到 `0.0895 m/s`。这说明模型学习到的 residual correction 有效，不是简单复制底盘速度。

当前证据支持的结论是：视觉和底盘特征的 residual fusion 能够在强底盘先验基础上进一步提升速度估计精度。

## 8. 局限与后续建议

1. 当前结果只来自一次训练和一次测试，若用于论文或正式结题，应补充多随机种子实验。
2. 纯视觉模型已有明显改进，但还需要进一步分析哪些场景导致高分位误差。
3. ChassisFusion 优于 `base_speed`，但还需要 chassis-only、without-base-speed、direct-fusion-vs-residual 等消融实验来拆分视觉贡献。
4. 当前视觉窗口围绕中心帧构造，若目标是在线部署，需要改成只使用历史帧并重新评估。
5. 建议继续按速度段、加减速状态、转弯状态和场景类型分析误差来源。

## 9. 汇报时建议强调

1. 本次报告已替换为最新纯视觉模型结果，旧版纯视觉结果不再作为当前结论依据。
2. 新版纯视觉 MAE 为 `1.2856 m/s`，说明视觉分支已经有效学习速度信息。
3. `base_speed` baseline 仍然很强，MAE 为 `0.1995 m/s`，因此融合模型必须和它比较才有说服力。
4. ChassisFusion 在 `base_speed` 基础上进一步将 MAE 降至 `0.0895 m/s`，说明 residual fusion 学到了有效修正。

## 10. 文件索引

本次覆盖生成的文件：

- `analysis_reports/flexinet_speed_comparison/REPORT_FOR_ADVISOR_SPEED_COMPARISON.md`
- `analysis_reports/flexinet_speed_comparison/REPORT_FOR_ADVISOR_SPEED_COMPARISON.pdf`
- `analysis_reports/flexinet_speed_comparison/first1000_true_vs_vision_base_fusion.svg`
- `analysis_reports/flexinet_speed_comparison/first1000_true_vs_base_fusion_detail.svg`
- `analysis_reports/flexinet_speed_comparison/aligned_first1000_curve_data.csv`
- `analysis_reports/flexinet_speed_comparison/comparison_summary.json`
