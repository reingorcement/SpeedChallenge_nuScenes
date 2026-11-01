# PureVision V5: S2 image/flow feature fusion

**2026-11-01 冻结编码器 A/B/C pilot：** 已实现公共 FP32 特征接入、三组新融合读出和独立训练/评价入口，仅通过12个真实窗口的短预检；**未启动正式20轮训练、未生成完整特征缓存、未改变巡视调度**。见[实施计划与精确入口](docs/next/FUSION_ABC_PILOT_PLAN.md)及[预检收据](docs/next/evidence/fusion_abc/)。来源固定为外观＋差分 R03 的已训练 best，保留学习后的外观旁路；本轮仅比较冻结特征的读出能力。

**2026-11-01 RGB信息路径对照：** 旧R03_FT用户主动暂停，主对照锁定前18完整epoch内的best。新增一组原始外观＋CMA差分旁路，停止18轮、余弦周期40轮；见[设计与自审](docs/next/R03_RGB_PATH_DESIGN_20261101.md)。旧FT不恢复，Frozen/CONT的40轮结果不混入18轮主比较。


**Latest phase:** R02 BASE/FLEX 三seed完整40轮对照已完成，FLEX为已确定开发主线。当前R03 Frozen/FT与同起点CONT已实现并通过必要真实验证；见[实现、自审和运行协议](docs/next/R03_FLEX_IMPLEMENTATION_20261031.md)。正式训练/评价/巡视状态以[结果分支](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results)为准。非零侧向仍31窗口/5场景、insufficient_data；下面六组是历史首轮。

**GitHub review:** [审核导航](AI_REVIEW.md) · [完整中文交接](docs/HANDOFF_20261030.md) · [六组完成结果](docs/RESULT_20261030_S2.md) · [可读实验凭证](docs/review_evidence/20261030_s2/README.md) · [用户原计划](docs/V5_EXECUTION_PLAN_USER.md)。All six first-round runs completed; the queue and scheduled patrols have ended. Training code is pinned at `3979268699ce6db46d536cf22dc16f02b969a465`; later handoff commits do not modify that checkout.

Five causal CAM_FRONT images and their four adjacent flow fields predict one
`[vx, vy]` vector in m/s. The current target is **PLANAR_PROXY**, expressed in the
current yaw frame at the **nuScenes rear axle midpoint**. It is not a verified
three-dimensional body/centre-of-mass velocity target.

The first comparison asks whether explicit flow helps the same **frozen V3 RGB
representation**. It does not establish superiority over a fully trained image
model. The RGB representation itself includes CMA absolute image differences;
the numerical signed flow branch never passes through CMA.

| Config | Computation |
|---|---|
| R00 | Frozen RGB descriptor + four measured historical dt values -> 2D head |
| R01 | Add fixed spatial-grid pooled flow/validity + linear projection |
| R02 | Independent spatial/temporal flow encoder + dt, without RGB |
| R03 | Independent spatial CNN + four-pair temporal convolution + RGB feature concatenation |
| R04 | R03 with binary FB reliability hint; retain every legal forward flow |
| R05 | Same hint/threshold as R04; additionally zero FB-rejected flow |

See [model design](docs/MODEL_DESIGN.md), [data evidence](docs/DATA_AUDIT.md) and
[flow evidence](docs/FLOW_AUDIT.md). Configurations share the downstream trainer;
there are no independent speed predictors to fuse. R03--R05 have identical
parameter shapes. RAFT is frozen. No segmentation, geometry loss, depth or
dynamic ground-normal module is included in this first round.

## Initial training and acceptance policy

Seed 42; AdamW lr 1e-3, weight decay 1e-4; batch 64; at most 20 epochs; cosine
learning rate to 1e-4; patience 5 on dev EPE (minimum decrease 1e-4 m/s).
SmoothL1 operates on errors divided by fixed `[5,1]` m/s scales, beta 1. All
reported predictions and metrics remain in m/s. RGB/RAFT weights and BN are
frozen; downstream models train from fresh same-policy random initialization.

Before observing candidate results, the lateral acceptance rule is fixed:
reject an improvement claim if dev MAE(vy) exceeds R00 by more than
`max(0.01 m/s, 5% of R00 MAE(vy))`, either overall or on `abs(target_vy)>=0.2 m/s`.
The latter needs at least 100 windows and 5 scenes; otherwise report insufficient
evidence, not a pass. EPE gain alone is insufficient. These are exploratory
engineering tolerances, not measured statistical significance or label accuracy.

The original scene split remains 582 train / 103 dev / 75 historical diagnostic
holdout / 75 final holdout. It is not log-disjoint. Both holdouts remain sealed in
this implementation. Historical label/model results do not populate the V5 table.

## Environment and storage

Use `/home/zhangyu/.conda/envs/speedest/bin/python`, Python 3.11, PyTorch
2.10.0+cu128 and torchvision 0.25.0+cu128. The shell's default Python is a
different environment. Raw data, flow caches, descriptor caches, logs and
checkpoints remain outside Git, under
`/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/`.

Formal training requires clean detached code whose exact commit is present on
the remote V5 branch or immutable `v5-code-<12hex>` tag. Run directories must be
new. Resume also uses a new run ID and rejects changed code/data/label/loss
contracts. Checkpoints store optimizer, scheduler, AMP, RNG and batch progress.

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B -m pytest -q -p no:cacheprovider tests
```

See the [first run record](docs/RUN_20261030.md), [operations](docs/OPERATIONS.md)
and [verification evidence](docs/VALIDATION.md) for concrete versions, commands
and artifact locations. `collect_results.py` exports status and independently
verified metrics from a fixed pipeline plan. A passed unit test is not a trained
model result.
