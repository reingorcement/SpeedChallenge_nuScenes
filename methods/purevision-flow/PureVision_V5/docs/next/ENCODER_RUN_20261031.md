# R02_BASE / R02_FLEX 正式启动记录

本轮现已完成，见[最终结果](ENCODER_RESULTS_20261031.md)。以下保留启动当时记录。

2026-10-31 13:50（北京时间）两组已启动，GPU1/2分别为BASE/FLEX，两张都是RTX5090。这里记录启动，**不是40轮完成报告**。完整数据、共同协议和算法见[设计自审](ENCODER_COMPARISON_20261031.md)。

- training commit：`a847938681d60ae86e0e293e24ffa7d2e74e20a9`；代码、配置及tag `v5-code-a847938681d6`在启动前已推送并以实际`ls-remote`核实。
- detached运行目录：`/home/zhangyu/.local/share/speedest-v5-next/code/a847938681d6/methods/purevision-flow/PureVision_V5`。后续文档提交不改变它。
- plan identity：`0f23c724cb8225d60ccfd5b0b1647363479f2bf8c39ab0fb31d912f1092248a1`。
- run IDs：`20261031_r02_base_flex_s42_R02_BASE`、`20261031_r02_base_flex_s42_R02_FLEX`。
- 第一次完整dev：BASE EPE `0.5267899634`，FLEX `0.3665267229` m/s；不得把首轮数值当作最终选模比较。
- 初始共同头SHA两组均为`7de467aac88b930505733a8d7817bb1e1095fd075a804a27081d16121f7388a9`；完整模型/编码器初始化哈希各自记录在run.json。

完整计划/启动命令：`/home/zhangyu/.local/share/speedest-v5-next/pipeline/20261031_r02_base_flex_s42/plan.json`。日志`logs/<run_id>.log`，权重`runs/<run_id>/{last,best}.pt`，独立评价及逐样本预测`evaluations/<run_id>/`。上述相对位置均在同一个artifact根下。

正式协调入口（已执行，**不要重复启动同一前缀**）：

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B \
 /home/zhangyu/.local/share/speedest-v5-next/code/a847938681d6/methods/purevision-flow/PureVision_V5/encoder_comparison.py \
 --root /home/zhangyu/.local/share/speedest-v5-next \
 --manifest /mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1/dataset.json \
 --run-prefix 20261031_r02_base_flex_s42
```

协调、每组训练、独立生命周期/发布分别挂tmux；可`tmux attach -t v5next_20261031_r02_base_flex_s42_R02_BASE`（FLEX同理）。训练命令、完整评价命令都在plan对应job的`commands`中，含具体数据/光流/config/GPU/output路径。停止/新ID恢复方法见设计文档，不在原run目录覆盖恢复。

两组均已通过步骤1/2/3、首次dev、last/best完整状态CPU重载检查；证据[启动核验及Astra探针](evidence/encoder_startup_20261031.json)。任务在已持续运行10天以上的服务器上执行，与当前对话进程分离，仍依赖checkpoint应对机器故障。

## 本轮真实巡视和发布

13:52:38安装本轮5分钟server、30分钟Astra cron。13:53:01 cron触发Astra探针，13:53:57成功，读取11个工件，run IDs绑定本轮两组及排队中的compare。13:55:01新的server定时快照已产生；没有拿上一轮缓存巡视的成功记录代替。[服务器回执](evidence/encoder_server_cron_20261031.json)。

结果发布器独立使用`results/purevision-v5-next` worktree。首次提交推送后最终远端读回遇到SSH超时，保留pending；随后13:54实际cron探针成功推送并核实`bffd2b1a7dda32db37e32f20dbc7f37fc6e836f0`。临时分钟探针已恢复为每小时第17分钟发布，另由生命周期程序在里程碑发布、失败后重试；[发布探针回执](evidence/encoder_publisher_cron_probe_20261031.json)。网络失败未中断训练。安装的整点/半点Astra和小时发布周期，须以之后的真实触发记录核实；这里已证明本轮服务器访问和cron执行能力，不冒称所有未来触发已执行。

现阶段出现少量动态loss scale导致的AMP跳步，三种计数分别保留；不是每处理一批都发生optimizer update。暂无需修改模型或重启的异常，正常波动不触发干预。

## GitHub 实时工件

- [结果状态与导航](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results)
- [BASE运行收据、指标与预测](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results/experiments/20261031_r02_base_flex_s42_R02_BASE)
- [FLEX运行收据、指标与预测](https://github.com/reingorcement/SpeedChallenge_nuScenes/tree/results/purevision-v5-next/methods/purevision-flow/PureVision_V5/docs/next_results/experiments/20261031_r02_base_flex_s42_R02_FLEX)

每组40轮后独立evaluate best checkpoint；第三个compare作业核对全过程并输出EPE、分量误差、逐场景、侧向子集、常量基线、成本及选择建议。公开逐窗口CSV与权重哈希，完整权重留服务器。本轮plan对应终态报告经远端核实后才移除专属巡视及发布；中间发布延迟不等于训练停止。侧向31窗口/5场景保持insufficient_data，单seed差异不叫显著优势。
