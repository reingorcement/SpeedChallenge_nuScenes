# 分辨率预试验已完成，等待用户决定

2026-10-30，Asia/Shanghai。32 train / 8 dev场景，seed42，两组各40完整轮。

| RAFT宽×高 | dev EPE m/s | MAE vx | MAE vy | 更新 / AMP skips |
|---|---:|---:|---:|---:|
|512×288|0.946689|0.944315|0.025546|4600 / 0|
|1024×576|0.879164|0.877233|0.024956|4600 / 0|

高分辨率EPE下降7.13%，6/8场景改善，2/8退化。总体侧向规则通过，但13个非零侧向窗口仅来自1场景且有局部退化，完整侧向验收仍是证据不足。PLANAR_PROXY、单seed、同dev重算，不是独立测试。

低run `20261030_pilot_s42_R02_512`，training commit `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9`。
高run `20261030_pilot_s42_retry1_R02_1024`，training commit `be327b7f91f5f7e5ba46b70dede116a21c551652`。
初始化、40轮顺序和核心数值源码均核验一致。当前提交只补完成文档，不是新的训练版本。

[完整比较与实际证据](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results/comparisons/resolution_20261030_pilot_s42_retry1)。
[成本、局部退化与待决策选项](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results/comparisons/resolution_decision_20261030)。
[巡视、发布及控制命令证据](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results/evidence/operations_active_verification_20261030.json)。

推荐先使用现成缓存补seed43/44的四个短训练，再决定全量分辨率。也可由用户选择临时采用1024×576，或成本优先保留512×288。**尚未批准/执行这些后续选项；没有开始全量高分辨率缓存、Flex或R03正式训练。**

初次高缓存网络前置失败保留，恢复时没有重训低组；CSV发布CRLF误报已修复。新代码最终186测试通过。工件、日志、完整恢复checkpoint均在私有ART下，GitHub保存报告/预测及权重哈希，权重原件没有上传。
