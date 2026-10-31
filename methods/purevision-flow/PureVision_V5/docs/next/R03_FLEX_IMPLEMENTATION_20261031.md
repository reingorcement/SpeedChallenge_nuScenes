# FLEX + RGB：实现、自审与运行协议

2026-10-31。用户已授权 Frozen / FT / CONT 三组；FLEX 已被确定为开发主线。本阶段精度优先，成本只记录，不重新比较 BASE/FLEX，不重算 RAFT。原[准备方案](R03_FLEX_PREPARATION_20261031.md)保留为设计历史；本记录描述已实现内容。

## 本次实现与设计依据

旧 factory 仅允许 R02 使用 FLEX，RGB 数据接口只提供冻结 descriptor，且 RGB forward 强制 no_grad。现已打通 FLEX R03 与真实 RGB 微调：`RGBDescriptor`持有 V3 backbone 和 region_projection；Frozen 固定参数及全部 BN 状态，FT 走五帧真实像素，所有 RGB 模块进入 optimizer。两者共享同一个模型工厂及 checkpoint/evaluation 路径。Frozen 训练复用经过真实等价检查的 FP32 descriptor，同时 checkpoint 仍保存完整 RGB 权重和 BN，支持在线图像推理。

融合固定 RGB384 + FLEX64 + 四真实dt4 → Linear452→192 / ReLU / Linear192→2，输出 `[vx,vy]` m/s。CONT 使用原68维输入头。没有额外门控、FB、硬筛选或几何损失。光流 FLEX 的逐对空间 stem、SFE、MFE（三层时间核3、感受野7、对四步历史零填充）和有符号u/v路径全部沿用；RGB 中的 CMA 只属于原 RGB 权重架构，不替代光流。

三组固定 seed42 R02_FLEX best（来源 epoch37），不挑选三个seed中的最优者：

```
/home/zhangyu/.local/share/speedest-v5-next/runs/20261031_r02_base_flex_s42_R02_FLEX/best.pt
SHA256 4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823
```

显式复制 flow_encoder 完整state、原头flow/dt列、bias、末层，新RGB列精确置零。初始化收据记录来源run/commit/文件SHA、各模块state SHA及映射核验。Frozen/FT同RGB初始化SHA `1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82`，整个融合头一致。三组新建AdamW、scheduler、AMP与随机状态，不跨结构resume。

## 必要验证与自审结论

证据：[64个真实窗口检查](evidence/r03_real_preflight_20261031.json)、[实际训练器中断恢复检查](evidence/r03_trainer_recovery_20261031.json)。这些是有限 smoke 检查，不是正式研究指标。

- 真实 train/dev 为132978/23480窗口、582/103场景，场景与图像token无交叉。检查64个不同训练场景的五帧、四流、四dt和标签配对；dt为0.049993–0.200000秒，采样流范围−32.71875至15.75（128×72视图像素位移），未做逐帧标准化或绝对值替代。
- 三组初始FP32预测最大绝对差0；预先锁定容差atol=1e-5、rtol=1e-5。Frozen的新RGB输出与原FP32特征缓存最大差0；FT同初始RGB状态的计算差约1.07e-6。
- 三组各3次真实AMP optimizer更新，0 skips；FT首次RGB上游梯度为零符合零列初始化，后续CMA/AFT/SFE/MFE/DIG/projection均有有限非零梯度且参数更新。Frozen权重和BN完整state hash不变。
- 真实batch64实测：FT峰值45,066,672,128 bytes（约42 GiB），热身后每步约0.72秒；Frozen缓存路径约0.013秒，CONT约0.013秒。它们只是同GPU0驻留输入的小样本测量，不能代替实际全程吞吐。
- 因显存足够，三组统一微batch64，无梯度累积（最后一个batch50）；FT的RGB/flow BN按实际微batch更新，Frozen的RGB BN固定。评价batch8、FP32。若未来改微batch，须作为新配置与run记录。
- 真实训练器分别在step3中断、step5恢复，完整状态和CPU恢复检查通过；独立构造模型后严格加载包含RGB的state，用16个真实dev窗口检查评价。单元测试另验证断点续训与连续训练状态一致。
- 全套测试288项通过；之后巡视/发布部分的针对性测试54项通过（含恢复测试复核）。任何正式运行的配置、commit以启动收据为准。

实现修改：`rgb.py/model.py/datasets.py/training.py`；新增`initialization.py/optimization.py/rgb_frames.py`；复用`train.py/evaluate.py`而不复制训练器。新增三份`configs/next/*.json`、真实验证脚本、融合对照队列与完成收集器。保存包含RGB/BN、optimizer两组状态、scheduler、AMP、RNG、epoch/batch游标、累计loss和更新/skip计数。

自审通过：推理输入只包含历史视觉及四个dt；标签和未来pose保持在离线数据侧。原划分和PLANAR_PROXY后轴中点/当前yaw坐标语义不变。主要剩余不确定性是正式泛化收益，不以代码通过预测算法一定有效。

## 数据复用和一项读取修正

仍使用已有1000场景CAM_FRONT完成缓存；训练只读原train_core/dev的1024×576 RAFT生成、128×72 packed视图。native约1.385TB不重复扫描或生成。冻结RGB缓存为 `.../01_speedest_vision_artifacts/v5/features/v3_direct_fp32_v2`，身份 `c955ae53d33ca5cc12d29ccfc451a46fee6a4291e05622f4c5f22b7c27a36128`，不使用旧低精度特征缓存。

FT的真实五帧图像先按原PIL bilinear规则缩放为uint8像素视图，保留帧token、文件名、原文件SHA、分区和序列SHA；不缓存可训练特征、不含标签。目录为`/home/zhangyu/.local/share/speedest-v5-next/rgb_frames/v3_uint8_train_dev_v1`，完成身份与小样本等价证据由启动收据补充。

发现packed reader的token合法性校验调用`cache_path('.')`，每个pair又查询当前目录的两个schema路径，在FUSE磁盘上产生数十万次无关查询。新做法仅抽出并复用同一token合法性函数，配对时间/文件SHA/shape校验不变；原有限检查中止后重新验证通过。影响加载开销，不改变缓存、数值或实验设计。

## 固定协议和资源

三组seed42，各新增40完整epoch，不早停，dev EPE选best；相同样本顺序`randperm(seed+epoch)`、原loss、有效batch64、AMP、clip2、AdamW wd1e-4。flow/head lr1e-3；FT RGB lr1e-4；每组各自余弦衰减至初值0.1，避免单一eta_min让RGB学习率不衰减。每组计划处理83,120 batches，实际optimizer updates与AMP skips单独记录；若有skip，不补额外epoch。

FT在GPU0（98GB RTX PRO 6000）运行，Frozen在GPU2（32GB RTX5090）运行；CONT在GPU2接续Frozen的训练和独立评价。GPU1现有桌面/ToDesk进程保留。队列有GPU锁及进程身份检查，不往同GPU叠加作业。

可训练/总参数：CONT 76,850/76,850，Frozen 150,578/2,243,762，FT 2,243,762/2,243,762。Frozen缓存的计算优势按实际情况报告，不据此设置精度选型门槛。

## 版本、启动、恢复与巡视

正式代码/config提交推送且核对远端SHA后，从该commit建立detached worktree。运行目录独立，运行后不修改工作树或配置。预计run前缀`20261031_r03_flex_s42`；实际commit、启动命令、环境、输入身份、日志和权重路径在`run.json`及结果分支`plans/`、`experiments/`中记录。

```bash
# 在固定worktree的PureVision_V5目录中；各路径展开值见发布的plan.json
python fusion_comparison.py --root "$ART" --manifest "$MANIFEST" \
  --run-prefix 20261031_r03_flex_s42 --features "$FEATURES" \
  --rgb-frames "$RGB_FRAMES" --verification "$VERIFICATION"
# 队列中的FT实际调用形式；Frozen把--rgb-frames换为--features，CONT两者都不传
python train.py --config configs/next/R03_FT.json --manifest "$MANIFEST" \
  --flow "$PACK" --rgb-frames "$RGB_FRAMES" --device cuda:0 --workers 4 --output "$RUN"
python evaluate.py --checkpoint "$RUN/best.pt" --manifest "$MANIFEST" \
  --flow "$PACK" --rgb-frames "$RGB_FRAMES" --device cuda:0 --workers 4 --output "$EVALUATION"
tmux attach -t v5next_20261031_r03_flex_s42_R03_FT
tail -n 30 "$ART/logs/20261031_r03_flex_s42_R03_FT.log"
tmux send-keys -t v5next_20261031_r03_flex_s42_R03_FT C-c
# 仅同commit、配置和任务语义可恢复；指定新run目录并另登记新run ID
python train.py --config configs/next/R03_FT.json --manifest "$MANIFEST" \
  --flow "$PACK" --rgb-frames "$RGB_FRAMES" --device cuda:0 --workers 4 \
  --resume "$RUN/last.pt" --output "$NEW_RUN"
```

checkpoint每250步或120秒保存，并在完整epoch保存last及dev最佳best；信号中断在更新边界保存，恢复不继承未完成累积。服务器为持续运行主机，tmux负责脱离终端，checkpoint负责恢复，不把tmux等同于机器持久性保证。

调度：服务器5分钟、Astra只读30分钟、独立发布每小时（整点后17分）及生命周期里程碑。新run使用本轮新调度回执，启动后仅轻量确认，不在前台等待首个dev。首次训练步、首次dev及完整checkpoint检查由CPU生命周期流程完成。Astra正常只读当前阶段摘要，异常才看日志，不修改运行代码、不根据发布延迟重启训练。异常与关键进展进入GitHub记录；服务器CLI不能主动在当前聊天中插入新消息。全部训练、评价、比较及实际远端发布成功后才清理本轮调度，失败不能冒充完成。

## 完成后的比较边界

Frozen vs CONT回答冻结RGB增益；FT vs Frozen回答RGB微调增益；FT vs CONT回答整体微调融合收益。发布全部dev EPE、vx/vy MAE/RMSE/bias、逐场景、逐窗口预测、零vy及只在train拟合的中位数常量基线、更新数/AMP skips/成本。非零侧向仍31个独立窗口/5场景，维持100窗口/5场景门槛及`insufficient_data`。RGB初始化曾用同dev选模；原场景划分并非log-disjoint；不把开发集PLANAR_PROXY表现称为独立真实体轴任务成绩。单seed差异不称显著或稳定优越。
