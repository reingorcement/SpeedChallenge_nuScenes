# 三 seed 分辨率预试验：结果、限制和待决策

2026-10-30（Asia/Shanghai）。用户批准的 seed43/44 四组已全部完成；连同原 seed42，共六个有效小规模 R02 结果。

| 训练 seed | 512×288 EPE | 1024×576 EPE | 高分辨率改善 |
|---:|---:|---:|---:|
| 42 | 0.946689 | 0.879164 | 7.13% |
| 43 | 1.010065 | 0.856311 | 15.22% |
| 44 | 0.916859 | 0.820553 | 10.50% |

均值±样本标准差（m/s，n=3、ddof=1）：512 为 **0.957871±0.047599**，1024 为 **0.852009±0.029541**；相对均值改善 **11.05%**。配对差值 high−low 为 −0.105861±0.043902 m/s。7/8 场景按三 seed 均值改善，scene-0566 仍退化；6/8 场景在三个 seed 均改善。

## 侧向风险没有消失

| seed | 13 窗口低/高 EPE | 低 MAE vy | 高 MAE vy | 高−低 MAE vy |
|---:|---:|---:|---:|---:|
| 42 | 0.662744 / 1.461329 | 0.180087 | 0.183006 | +0.002919 |
| 43 | 1.160439 / 1.134048 | 0.160523 | 0.181031 | +0.020507 |
| 44 | 0.298941 / 1.506400 | 0.178421 | 0.185869 | +0.007447 |

三个 seed 的总体 vy MAE 均改善，总体侧向阈值均通过；但非零侧向的13个窗口仅来自1个场景，三个 seed 的此子集 vy MAE 均变差。seed43 的子集增量约0.0205 m/s，超过0.01 m/s的数值容差；因未达到预先规定的100窗口/5场景门槛，正式子集验收状态仍是 insufficient_data，不能偷改为 pass，也不能把13窗口乘三算成39。子集EPE在seed42/44显著退化，scene-0566平均EPE增加约0.0806 m/s。

1024组平均vx偏差−0.237742 m/s，512为−0.218171 m/s，绝对偏差增加。1024 seed44最优点为第40轮，统一40轮只是初筛预算，不能声称已充分收敛。

## 推荐和决定权

**推荐将1024×576作为下一阶段待验证的前端，派生视图仍保持128×72。** 理由是三个seed总体收益方向一致、平均改善约11%，且7/8场景均值改善；足以支持投入全量开发数据验证，不足以宣称全面胜出或通过侧向验收。下一阶段必须继续报告完整开发集的非零vy覆盖、逐场景结果和既定侧向规则。

另一可行选项是保留512×288，以复用完整缓存和降低成本。两者均由用户选择；本报告没有代替用户锁定分辨率，没有启动全量RAFT、Flex或R03。当前证据无需为了获得更好数字继续无止境增加相同小样本seed。

## 比较条件及真实成本

- 固定原32 train_core / 8 dev场景、7324/1826窗口；selection seed仍42，training seeds为42/43/44。
- 六组均40完整epoch、4600实际更新、0 AMP skips。RAFT冻结；独立光流空间编码器、四步时间模块和二维速度头共同训练。
- 低/高同seed的初始化、40轮样本顺序一致；10份数值核心源码身份相同。跨seed沿用seed+epoch顺序政策，邻近seed排列有移位复用；这是敏感性检查，不是显著性检验。
- 仍为nuScenes后轴参考点的PLANAR_PROXY、同dev选模并重算，不是独立测试、真实体轴任务完成或跨数据集泛化证明。
- 本轮新增RAFT=0对，新增四组训练分别约91.62、85.01、78.77、77.50秒（低43/高43/低44/高44）；另有网络核验、数据读取和评价时间。
- 历史1024实测9270对约55.27GB native+compact；线性外推全158513对约880.23GiB，另packed约8.16GiB，双GPU约129分钟，仅为估计。建议原生/派生放数据盘、packed放NVMe，复用已有帧对，不在系统盘堆满约0.95TB新缓存。
- KITTI仍缺连续Raw/OXTS/真实时间等必需项；本轮没有KITTI测速训练。

## 版本、日志和审核入口

- 四个新run training_commit：`252fb5f44c290c0d894faf8373329ba27105604e`；192项测试通过。模型和训练数值未改。
- 新run前缀：`20261030_pilot_s43s44_`，后缀 `R02_512_s43` / `R02_1024_s43` / `R02_512_s44` / `R02_1024_s44`。
- 原seed42分别沿用 `20261030_pilot_s42_R02_512` / `20261030_pilot_s42_retry1_R02_1024`，实际commit分别为 `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9` / `be327b7f91f5f7e5ba46b70dede116a21c551652`。
- 工件根：`/home/zhangyu/.local/share/speedest-v5-next`。`runs/<run_id>/`含run/config/epoch记录及best.pt、last.pt完整状态；`evaluations/<run_id>/`含重算指标及逐样本预测；`logs/<run_id>.log`为日志。
- 固定源码：`code/252fb5f44c29/`；计划及实际逐job命令：`pipeline/20261030_pilot_s43s44/plan.json`；启动receipt：`plans/20261030_pilot_s43s44_launch.json`。
- 三seed总报告SHA256：`26d351a3eb0b3d9edfe188a544fc7c040e64b2d6f1e51bee4b96ffbeddba0ec9`；相邻目录 `resolution_multiseed_20261030_pilot_s43s44/summary.json` 可核对原始统计、权重/预测哈希。
- GitHub仅保存配置、指标、逐样本预测和工件哈希，完整权重/大日志/缓存保存在服务器。报告提交与训练提交分开记录。
- 本轮无训练失败或算法bug。存在GitHub链路偶发延迟，真实远端校验成功后才开始正式训练；没有放宽版本检查。
- 本轮真实cron、手动Astra、逐run首次更新/验证/完整checkpoint证据及最终清理状态见 `evidence/operations_*_20261030_pilot_s43s44.json`。安装状态与实际触发分开记录，终态后停止对应巡视。
- 正确源码tag为 `v5-code-252fb5f44c29`。误输入的同提交别名 `v5-code-252fb5f01270` 保留历史，不用于定位训练。

## 后续接手

先读取GitHub结果分支同名comparison和实际服务器状态，确认当前已完成且没有训练作业，再读取用户的分辨率选择。不要重复启动本轮coordinator，也不要对已完成40轮作业做无意义resume。选择1024后才准备缺失帧对的全量缓存，保留现有pilot和低缓存；固定缓存与协议后才开展R02/Flex结构对照。未选择前可做独立设计，但不启动依赖该选择的正式任务。

实际训练/评价完整argv在计划中；例如一个已执行的低seed43训练为：

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B /home/zhangyu/.local/share/speedest-v5-next/code/252fb5f44c29/methods/purevision-flow/PureVision_V5/train.py --config /home/zhangyu/.local/share/speedest-v5-next/code/252fb5f44c29/methods/purevision-flow/PureVision_V5/configs/next/R02_raft512_s43.json --manifest /mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1/dataset.json --selection /home/zhangyu/.local/share/speedest-v5-next/plans/resolution_pilot_s42_32t8v.json --flow /home/zhangyu/.cache/speedest-v5/flows_20261030/forward_packed --device cuda:1 --workers 4 --output /home/zhangyu/.local/share/speedest-v5-next/runs/20261030_pilot_s43s44_R02_512_s43
```

上述命令是已执行记录，不应直接再次执行覆盖输出。GPU1低43→44、GPU2高43→44；每个worker持有GPU锁，全程tmux。查看日志可用 `tail -n 30 <ART>/logs/<run_id>.log`，连接运行中的worker用 `tmux attach -t v5next_<run_id>`；完成后会话退出是预期行为，当前状态看终态凭证。停止只针对明确的本轮会话，以Ctrl-C发中断并核对已有checkpoint，禁止全局kill。恢复需新run ID、相同固定版本/配置/输入并以原 `last.pt` 加 `--resume`，由完整状态兼容检查决定；本轮均已完成，不需要恢复。
