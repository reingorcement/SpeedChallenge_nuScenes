# 全量 CAM_FRONT 光流：正式运行记录

2026-10-30 23:09:23（Asia/Shanghai）快照：**正式生成运行中，1712 / 232239 对，约0.74%**。三个分片均为running、只启动一次；无已报告错误。后续train/dev打包与独立核验均queued，尚未完成。本阶段是光流预计算，没有启动新的R02/Flex/R03训练，也没有新增测速指标。

## 用户选择和固定版本

- 用户确认全部场景、仅CAM_FRONT；采用预试验后的1024×576前向RAFT，保留原生流及128×72派生视图。
- 正式源码commit：`87d7468f911da5579bbbc39f79ec6bc85fa16dd6`。
- 远端分支：`method/purevision-v5-next`；已推送tag：`v5-code-87d7468f911d`。
- 正式启动前已独立ls-remote核实分支与tag均精确等于该commit。268项回归测试通过。
- 固定运行目录：`/home/zhangyu/.local/share/speedest-v5-next/code/87d7468f911d/methods/purevision-flow/PureVision_V5`，detached且干净；后续文档提交不修改它。
- prefix：`20261030_full_nuscenes_front1024`；plan identity：`cbc303bbcd40f1189db741cab7cce6f3938d6a3be3e3326ce182555ce95cf620`。
- 协调器23:02:47启动；23:03:22启动三个worker；正式全量入口分别复核已推送版本后开始计算。

## 当前任务与成本

| 作业 | GPU | 目标帧对 | 快照状态 |
|---|---|---:|---|
| flow1024_0 | 0 | 77623 | running |
| flow1024_1 | 1 | 77277 | running |
| flow1024_2 | 2 | 77339 | running |
| pack_train_dev | CPU | 原manifest所需158513对 | queued |
| verify_full_cache | CPU | 全232239对覆盖及最终抽查 | queued |

三卡各一个batch8任务，峰值CUDA分配约9.45GB/卡。前1712对已写约10.21GB，累计合计5.03对/s；按当时最慢分片估计生成还需约13.0小时，约10月31日中午，另计打包和收尾核验。这是吞吐外推，不是完成承诺。数据盘同步读写明显限制速度；不能套用NVMe pilot约两小时的估计。

覆盖1000场景，850 trainval + 150 official test；完整预计约1.385TB原生与派生文件，另约8.77GB train/dev packed。原监督835场景不改，未分配15场景与全部holdout/official test只做视觉缓存。两版本分目录保存。原9270对pilot允许严格来源复用，既有512缓存和旧分支保持不变。

## 已实际验证

启动前真实HDD短批验证三卡生成、零重算复用、跨盘原字节adoption和native-only恢复。正式启动后分别重载三片首个包，核对native/compact SHA、源图像/时间/场景、单位和精确派生数值。检查时已累计312对；之后计数推进到1712对，证明真实持续运行。完整收尾还没有发生，不把上述首批检查当作全量完成。

服务器证据根 `ART=/home/zhangyu/.local/share/speedest-v5-next`：

- `evidence/full_flow_publication_20261030.json`：正式代码/tag远端证明。
- `evidence/full_flow_initial_verification_20261030.json`：首批包独立核验，SHA `d14c69b3fbfbc6ac69d4e9658b4d9ecd54e8293b5109de25f5694ac2da108151`。
- `evidence/full_flow_first1000_progress_20261030.json`：实际1712对时的完整进度与估时快照。
- `plans/20261030_full_nuscenes_front1024_launch.json`：完整启动argv。
- `pipeline/20261030_full_nuscenes_front1024/plan.json`：所有worker/pack/verify命令及输入身份；同目录`status.json`、`jobs/*.json`、`reports/*.json`为实时状态。
- `logs/20261030_full_nuscenes_front1024_*.log`：完整日志。

首次运行状态已发布到 `results/purevision-v5-next`，commit `890c2a173162ac0d0191170c8fc7028e463e04a6`（23:04:48远端核实）。后续发布会继续更新结果分支，历史提交保留。结果入口是该分支 `methods/purevision-flow/PureVision_V5/docs/next_results/README.md`；状态快照必须连同时间阅读。

## 巡视与后台持续运行

手动Astra与一次**真实cron探针**均成功：手动记录23:04:55，cron探针23:09:59；各读取12个目标，包含本次三分片的进度报告、worker状态和日志，缺失读取为0。巡视使用固定87d746版本及gpt-6-astra。23:10:23已安装每5分钟服务器检查、每30分钟Astra；发布器每小时17分执行。探针已自动移除，当前恰好三条本任务cron，旧配置有备份。

**调度链路已由真实cron执行验证，正式5分钟服务器周期已在23:15:02自然触发成功；首次定时发布和半小时Astra在本记录时仍分别待23:17、23:30。** 安装状态与触发状态分开记录；本文件不提前声称这些未来周期已经成功。后续是否成功应读取`ART/monitoring/schedule.json`、`last_astra.json`和`publisher_schedule.json`。

证据为 [本轮运维验证](evidence/operations_active_verification_20261030_full_nuscenes_front1024.json)，SHA `cb4dcaa9fe9ac1ff076e91dbc80d5c24f1c3b19481f83d06f57c7799f7fa7a01`。实际cron读取证据位于`ART/monitoring/astra/20261030T150901Z_3355229.meta.json`。巡视只读诊断并留记录，不自动改算法或重启。确认代码bug后仍须开发目录修复、测试、推送，以新版本/run恢复，不能改正在运行的目录。

`v5next_20261030_full_nuscenes_front1024_completion_publish`会话等待本plan终态并触发结果发布，只有本轮五个job均终态且新报告远端SHA核实后才退出；网络故障会留下记录并重试。终态后服务器检查停止本任务巡视，发布器在成功发布终态后停止。该机制依赖持续运行的实际服务器，机器断电/重启后的恢复仍需检查已有缓存与状态。

辅助完成发布器在收尾审查中修复两个明确bug：超时未捕获可能导致退出，以及失败收据覆盖初次成功后无法自行重试。v2锁定本commit/plan的初次远端证明，失败每5分钟重试；3项隔离测试通过。只替换辅助completion会话，三个光流worker PID及固定源码均未改变。实际运行代码的逐字节[审核快照](evidence/completion_20261030_full_nuscenes_front1024.py)、[测试](evidence/test_completion_20261030_full_nuscenes_front1024.py)与[修复凭证](evidence/completion_watcher_fix_20261030_full_nuscenes_front1024.json)均已保存。

## 位置与操作

数据缓存：

```text
/mnt/sdb2/Workspace/zhangyu/dataset/nuscenes/derived/optical_flow/raft_large_C_T_SKHT_V2_1024x576_fwd_v1_20261030/
```

运行中的tmux：`v5next_20261030_full_nuscenes_front1024_coordinator`及同前缀`_flow1024_0`、`_flow1024_1`、`_flow1024_2`。例如：

```bash
tmux attach -t v5next_20261030_full_nuscenes_front1024_flow1024_0
tail -n 30 /home/zhangyu/.local/share/speedest-v5-next/logs/20261030_full_nuscenes_front1024_flow1024_0.log
/home/zhangyu/.conda/envs/speedest/bin/python -B /home/zhangyu/.local/share/speedest-v5-next/code/87d7468f911d/methods/purevision-flow/PureVision_V5/monitor.py --root /home/zhangyu/.local/share/speedest-v5-next --json
```

连接后Ctrl-b再d仅断开观察，作业继续；要停止某一明确worker，连接后Ctrl-C并核对状态。协调器意外退出、worker仍在运行时可从原固定源码用`full_flow_pipeline.py --resume-plan <原plan.json绝对路径>`恢复管理；不会自动重启终态worker。不要重复执行已存在prefix的新启动命令。失败后保留失败记录，定位原因，再按兼容性使用新run及来源清单恢复。

完整范围、数值合同、验证边界和恢复设计见 [实施说明](FULL_FLOW_20261030.md)。
