# 冻结编码器 A/B/C 融合 pilot：实现与短预检

2026-11-01。本阶段仅实现、固定配置和小样本验证。**没有启动20轮正式训练，没有生成完整9150窗口特征缓存，没有安装或改变任何队列/巡视调度。** 旧 `20261031_r03_flex_s42_R03_FT` 保持用户主动中断。

## 已核对来源与边界

只读 `git ls-remote` 确认开发 HEAD 为 `4cf83c5cb19df8ada2e600f0c66a0905decbff83`，结果 HEAD 为 `b50f4b4555b420778af5e8a43baafdc9f486a9ce`。服务器未发现运行中的 V5 训练/队列，旧阶段已完成收尾；用户 crontab 无 V5 条目。GPU0、2空闲；GPU1有桌面显存占用，不操作该进程。历史代码没有本次 A/B/C 实现。

固定 encoder checkpoint：

`/home/zhangyu/.local/share/speedest-v5-next/runs/20261101_rgb_path_s42_R03_APPEAR_FT_s42_18ep/best.pt`

SHA256：`ab25fd6c4a7cd14a99a189f56e6b47913ed0f946a60cff2d0c9102479bc30ff8`。

该模型训练 commit 为 `4cf83c5…`，best 来自第1轮。先按 checkpoint 中的 config 构造完整 R03，再 `load_state_dict(strict=True)` 加载全部144个状态项。没有调用 V3 `load_pretrained`。已训练 appearance stem 及 `appearance_alpha=0.007355934474617243` 被保留；完整源状态摘要为 `0ea639695bca8079965340e0df9b1364f579c85bc83d4dca7b7ef7607a4e7996`。

复用选择文件：`/home/zhangyu/.local/share/speedest-v5-next/plans/resolution_pilot_s42_32t8v.json`。

- selection identity：`83abcfc4565b80a06547aef4ad6bb3c0cd8b4d6a3156c5164653d47f91302b5f`。
- 原 manifest SHA256：`a74f7963f7c9e0b529fefef75f5be1c191a7a2fccbddd3678b82f825ee0de32a`。
- 现有选择算法重建、序列校验和 VelocityDataset 实例化共同确认：原 train_core 的32场景、7324窗口；原 dev 的8场景、1826窗口。没有静默换样本。
- 两区场景及窗口使用的全部 image tokens 不交叉；标签/光流/图像接口沿用现有 contract。

**研究范围仅为当前冻结表示下的融合/读出能力。** 编码器已经见过完整训练集，来源 checkpoint 已由既有完整 dev 选出；32/8仅指本轮新增模块的训练/选模范围。原划分不是 log-disjoint，任务仍是后轴中点 PLANAR_PROXY，不是重新建立的独立泛化测试。

## 设计、自审与实际接入点

保留既有 RGB 的 CMA+外观旁路、AFT、SFE、MFE、DIG，以及 FLEX flow stem、SFE、MFE。新增冻结 wrapper、公共缓存和三个从头初始化的读出模型。已有区域投影、flow projection 和速度头不参与新预测，也不复制其权重。输出直接是物理单位 `[vx,vy]`，没有旧速度相加或速度残差。

FrozenFusionExtractor 冻结整个已加载源模型的参数，包括 alpha、BN affine。`train()` 永远执行 eval，forward 为 no_grad；父模块调用 train 也不能改变其状态。所有 buffers（BN running_mean/var、num_batches_tracked等）通过前后完整状态哈希检查。正式新模块训练仅加载缓存，不构造编码器，优化器中不可能包含编码器参数。

光流输入保留 signed u/v、valid、hint=valid。非有限或无效位置按原 R03 规则置零，无 FB、新筛选或重算 RAFT。实际路径是：

`stem -> x[N,32,4,9,16] -> {sfe(x), temporal(x)}`

两分支并行，**不是** `temporal(sfe(x))`。SFE 输出 `[N,32,4,9,16]`，MFE 输出 `[N,32,4,1,1]`。RGB 全部五帧进入现有 backbone；DIG feature_map 为 `[N,256,5,72,128]`，带有窗口时间上下文，不能描述为单帧独立特征。

缓存 shape/dtype：

| 字段 | 形状 | 操作 |
|---|---|---|
| F_sp | `[N,4,6,32]` | SFE逐时间位置仅空间2×3池化 |
| F_tm | `[N,4,32]` | 既有MFE空间汇聚，保留四步 |
| R | `[N,4,6,256]` | RGB逐时间位置仅空间2×3池化，保留五步后索引0..3 |
| dt | `[N,4]` | 实测历史间隔，原始秒数 |

全部 FP32。区域顺序为上到下、左到右；token顺序 t,row,col。旧 RGB 描述路径翻转的是 pooled height 维；新接入绕过此翻转和旧 projection。F01/F12/F23/F34 分别关联 RGB 输出0/1/2/3，不采用5→4的时间自适应池化。

实际 adaptive_avg_pool2d 的零基闭开边界：

| 模态 | 行 | 列 |
|---|---|---|
| flow 9×16 | `[0,5), [4,9)` | `[0,6), [5,11), [10,16)` |
| RGB 72×128 | `[0,36), [36,72)` | `[0,43), [42,86), [85,128)` |

非整除时bin有重叠。这是同一逻辑网格的粗粒度对应，不是像素对齐或相同感受野。

公共缓存每区分开保存 `*_features.pt` 与 `*_annotations.pt`；标签、scene/token/time及帧对列表只在annotations，模型拒绝额外字段。index记录 checkpoint/代码/源码文件哈希、固定清单、实际样本清单、预处理/RAFT视图版本、接入点、dtype/shape、区域/时间顺序、池化边界、文件哈希及冻结核验。加载同时校验固定来源与选择身份，不能只凭内部自洽哈希接受另一份缓存。三组读取同一cache identity。

## 三组模型和初始化

| 模型 | 实际结构 | 可训练参数 |
|---|---|---:|
| A | F_sp时间均值后C,row,col展开192；F_tm按C,T展开128；concat320→64 ReLU。R时间均值后共享256→64 ReLU、区域/通道展开384。64+384+dt4→192 GELU→2 | 124354 |
| B | 区域F_sp32+广播F_tm32+RGB256→64→LN→固定位置编码；两层独立64→256 GELU→64残差+LN；24token均值，+dt4→192 GELU→2 | 100738 |
| C | 与B相同输入；两层独立标准TransformerEncoderLayer，64维/4head/FFN128/GELU/dropout0/post-norm；相同均值和速度头 | 101250 |

实例化参数量逐一匹配。所有Linear带bias，所有LN affine=True、eps=1e-5。固定sin/cos位置编码为时间32、行16、列16维，base10000、幅度1；没有可训练位置参数。F_tm广播只增加共享上下文，不新增空间信息。B各token不交换信息；C所有历史token可互相注意，不加因果mask/CLS/额外最终LN/其他注意力池化。

使用隔离 seed42 的 PyTorch 默认初始化政策。B/C 的 input_projection、input_norm、完整head从同一初始化显式复制，随后独立存储和训练；公共状态哈希进入运行收据。C两层独立实例化，attention初始权重不同，避免TransformerEncoder默认克隆。同一seed不替代显式公共复制。A全部新读出参数重新初始化，不能引用历史FT指标作为A成绩。

## 已固定但未启动的训练协议

三份配置：`configs/next/FUSION_ABC_{A,B,C}.json`。

- seed42，20个完整epoch，batch64，drop_last=False，无梯度累积；每轮115batch（末batch28），每组总2300batch。
- 原 `epoch_order(7324,42,zero_based_epoch)`；同样本基序，每轮顺序哈希落盘。
- FP32，AMP/TF32关闭；相同 speedest Python/PyTorch环境。
- AdamW lr1e-3、weight_decay1e-4、betas(.9,.999)、eps1e-8，grad clip2。
- CosineAnnealingLR(T_max20,eta_min0)，完整epoch结束后step，记录当轮实际及下一轮LR。
- SmoothL1(pred/[5,1],target/[5,1],beta1,mean)。每轮同一1826 dev样本，按样本数汇总整体EPE。严格更低才覆盖best，完全相同保留更早轮。
- best/last包含新增模型、优化器、scheduler、RNG、epoch/batch游标、processed/updates/AMP0及数据/代码身份。支持同版本独立目录恢复，SIGTERM/INT安全边界保存。显示epoch从1开始。
- 不早停、不搜索单组超参、不增训某组。完整运行后独立重载best导出train/dev指标和逐样本预测。

## GPU及未来启动入口

| 物理GPU | UUID | 计划 | 进程内部 |
|---|---|---|---|
| 0 PRO6000 | GPU-cedbf5f4-647b-640c-1244-babecaef88e7 | 先公共缓存；完成后C | cuda:0 |
| 1 RTX5090 | GPU-5dbd27d5-0ae2-0ddc-ec5b-2077e99643e2 | B，保留桌面进程 | cuda:0 |
| 2 RTX5090 | GPU-b2986926-3ad1-16bc-f86c-28aae0c46598 | A | cuda:0 |

启动脚本 `scripts/launch_fusion_abc.sh` 默认只打印命令；只有 `--execute` 才执行。未来执行需要clean、已推送的detached worktree。先cache阶段，cache完备后才允许train阶段；训练各组独立tmux、run目录、日志。暂停/中断不会被当作完成自动触发评价。脚本**不安装队列或巡视**；本阶段没有执行以下正式动作。

在将来固定worktree的V5目录内，精确入口为：

```bash
bash scripts/launch_fusion_abc.sh --stage cache \
  --cache /home/zhangyu/.local/share/speedest-v5-next/fusion_abc_cache/s42_32t8v \
  --run-root /home/zhangyu/.local/share/speedest-v5-next/fusion_abc_runs \
  --run-id fusion_abc_s42_pilot

bash scripts/launch_fusion_abc.sh --stage train \
  --cache /home/zhangyu/.local/share/speedest-v5-next/fusion_abc_cache/s42_32t8v \
  --run-root /home/zhangyu/.local/share/speedest-v5-next/fusion_abc_runs \
  --run-id fusion_abc_s42_pilot
```

以上是安全dry-run入口。后续获得正式启动指令、固定新训练commit后再添加 `--execute`；本阶段不添加。缓存入口为 `fusion_abc_extract.py`；训练入口 `fusion_abc_train.py`；独立评价入口 `fusion_abc_evaluate.py`；有限预检入口 `verify_fusion_abc.py`。

## 预检、收据与自审结论

预检只用8个train、4个dev真实窗口，覆盖各区多个场景，按清单等距选择并保存确切index/token。一次性A/B/C各3次更新；另外以同一小缓存检查真实训练CLI和独立评价CLI。预检模型、优化器和指标全部标记preflight，不继承到正式训练，也不报告为研究精度。

收据位于 `docs/next/evidence/fusion_abc/`；服务器保留完整小样本工件和日志：

- 来源审计：`/home/zhangyu/.local/share/speedest-v5-next/evidence/fusion_abc_source_audit_20261101.json`
- 最终真实接入预检：`/home/zhangyu/.local/share/speedest-v5-next/verification/fusion_abc_final_20261101/receipt.json`
- 训练/独立评价入口预检：`/home/zhangyu/.local/share/speedest-v5-next/verification/fusion_abc_20261101/`

检查包括：图像/flow/dt/标签身份、实际接入shape、逐时间位置空间池化、在线/落盘一致、全encoder state不变、B/C显式相同公共初始化、有限梯度/真实更新、保存/重载、B无跨token而C有跨token响应、dev尾batch正确权重、20轮学习率政策。

自审：三组改变的是冻结特征的读出结构，缓存、来源和训练协议共同锁定；没有引入光流门控/空间权重或新损失。必要的接口风险已针对性覆盖，正式运行尚未开始，也没有模型精度比较结果。主要研究局限仍是先前全数据编码器训练/dev选模和单seed小范围读出训练。
