# 基于FLEX的R03-Frozen / R03-FT准备方案

状态：基于实际forward、数据与优化器路径的设计，尚未实现或启动这三组融合/继续训练实验。当前正式范围是R02的成对seed43/44复现。FLEX已由用户指定为主线；精度优先，无BASE等参数限制，不以低成本主动压缩结构。

## 已核实的复用边界

`purevision_v5/model.py`当前FLEX仅允许R02；R03仍默认旧BASE编码器。当前R03 head输入是RGB384+flow64+dt4=452；R02是flow64+dt4=68，两者第一层权重形状不同，不能直接strict加载整个R02 state。

`rgb.py:FrozenRGB`选择性加载V3的backbone与region_projection，要求causal T5、RGB72×128、direct_pretrain/regions。它同时强制requires_grad(False)、train(False)及forward的no_grad装饰器。`datasets.py`对所有RGB实验读取固定384维descriptor缓存；`training.py`只有VelocityModel进入AdamW。**只改rgb_policy配置或requires_grad开关无法实现RGB微调**，冻结特征缓存不可能向RGB backbone反传。

真实RGB forward：五帧[B,5,3,72,128]，PIL bilinear RGB及x/127.5-1 → backbone.forward_feature_map（内部CMA/AFT/SFE/MFE/DIG）→ adaptive pool (1,2,3)，H翻转保持near-to-far次序 → 六个256D区域经Linear256→64/ReLU → flatten384。RGB内已有CMA是原权重结构；它不替代独立光流分支的有符号u/v路径。光流保持当前FLEX完整结构及四真实dt。

V3初始化权重SHA `1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82`，历史证据见[数据审计](../DATA_AUDIT.md)：旧train_core训练/dev选模，同原场景划分，checkpoint未自带实际split文件hash。这是RGB初始化，不复用旧velocity head排名；保留同dev曾参与预训练选模的局限。实际新阶段启动前再次核实文件与SHA和缓存契约，不以目录名推断齐全。

## 推荐的共同起点与融合

建议将已完成seed42 R02_FLEX的best.pt固定为共同起点（epoch37，SHA `4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823`），不从seed42/43/44里挑最优seed。具体启动须以阶段配置锁定；这样补seed结果不会悄悄改变后续初始化选择。

准备三个新run：R02_FLEX_CONT、R03_Frozen、R03_FT。三者都加载同一R02 flow encoder，继续训练其所有参数。Frozen/FT均从同一V3 RGB state开始，使用同形状452→192→2速度头，头的完整初始state逐字节相同。

基本融合保留RGB全部384维，与FLEX64和dt4直接拼接，不为等参数或节省成本压缩RGB。为复用R02预测功能，明确把源head第一层的flow64/dt4列复制到新452输入头对应列，bias及末层也复制；新RGB384列初始化为零。CONT保持原68输入head。三者step0预测应在明确fp32数值容差内一致，既有flow/head子块hash相同，Frozen/FT整个头hash相同。新RGB列零初始化意味着**首次反向RGB上游梯度为零是预期**；更新RGB列后必须在后续小样本步骤证明RGB各模块具有有限非零梯度且权重实际变化，不能用一次head更新冒充微调。若此初始化实测长期阻断RGB学习，另记录修正而非静默变更。

三组均重新建立optimizer/scheduler/AMP/RNG训练状态；这是同起点的新任务预算，不继承旧40轮结束的学习率或动量。保留来源checkpoint及加载映射收据，不使用--resume跨模型语义恢复。

## 真实FT路径及首版预算建议

需要将RGB descriptor提取抽为可训练公共模块，Frozen是该模块的冻结包装；FT forward不能含no_grad/detach。VelocityModel/factory持有RGB子模块；数据集按显式rgb_policy提供原五帧images或经验证的冻结descriptor，不同时混用。FT optimizer显式列入backbone及region_projection，记录各参数组名称、lr、tensor数与hash。模型保存/恢复/评价统一factory，checkpoint含RGB state及BN buffers，不能在评价时重新套回固定V3权重。

推荐FT首版更新整个RGB backbone/projection（含BN affine），BN使用训练态统计并在dev冻结；Frozen的权重和BN统计均固定eval。二者差别包括正常微调带来的BN统计更新，必须如实记录；不把FT的BN变化当作意外bug。后续如需固定BN统计是单独配置/对照，不能训练中临时改。

建议额外共同预算为40个完整epoch（每组均新40轮，即共同来源40轮+额外40轮），batch64有效样本、相同seed42和逐epoch样本顺序、原损失/clip2/AMP、AdamW。flow/head lr=1e-3、RGB FT lr=1e-4，各组均余弦衰减至初值0.1倍，wd1e-4；严格dev EPE选模、不早停。Frozen/FT须同微batch及累积策略；优先实测batch64显存和RGB在线解码吞吐，再固定实际微batch，不能为了速度只给Frozen更多epoch。若FT需累积，处理batch、optimizer updates、AMP skips另统计；BN行为依赖微batch，不能称与batch64 BN完全等价。这个新阶段预算/初始化是推荐设计，尚未据此开始训练。

CONT必须与Frozen/FT同时保留，享有同起点、同额外40轮及flow/head优化协议；比较融合收益以CONT为直接基线，不以停止在40轮的R02作唯一基线。另保留原R02的40轮结果展示额外训练的影响。未来若改为从随机flow初始化，必须三组统一，不能混作同起点对照。

## 最小实施与验收清单

修改rgb.py公共提取器、model.py融合/初始化映射、datasets.py在线图像、training.py参数组、evaluation/startup factory；增加三份配置与实际RGB微调测试，复用现有训练脚本。

先验证原Frozen输出与新冻结包装/缓存一致、帧token和流/标签窗口对齐、推理字段仅历史图像/流/dt、R02到融合head映射和step0预测一致；再检查FT中CMA/AFT/SFE/MFE/DIG/projection逐模块梯度/实际更新、Frozen不更新及BN模式；验证包括RGB状态的保存/加载/恢复和独立评价。实际图像读盘、缓存完整性、batch64显存和吞吐尚待该实施阶段测量，不虚报为通过。

侧向31窗口/5场景继续insufficient_data；重复seed、额外epoch和RGB输入均不增加独立侧向证据。heldout仍不参与选模。后续共同起点/预算若需实质调整，集中给出原因及选项后再执行。
