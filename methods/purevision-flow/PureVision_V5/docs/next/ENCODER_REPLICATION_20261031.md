# FLEX主线决策与BASE/FLEX成对补seed43/44

2026-10-31，用户明确选择当前R02_FLEX作为后续主线，以预测精度为主要目标。参数、吞吐、显存和墙钟继续公开，但不再作为采用门槛，后续结构不要求与BASE等参数，也不主动为降低成本压缩。保留BASE及全部历史结果。这个决定是开发路线选择，不把seed42的10.93% EPE改善写成已经证明稳定优越；原报告中的建议保留为当时记录，本决策更新后续执行方向。

## 简短实施及自审

- 复用已完成seed42两组及全量缓存证据。本次不改模型、训练、数据和数值评价代码；新增四份配置，仅将seed/head_seed改成43或44。继续完整582/103场景、132978/23480窗口、40轮、batch64、AdamW/余弦、同损失与dev EPE选模、不早停。
- `encoder_multiseed.py`复用既有tmux worker、锁和训练/独立评价命令；`collect_encoder_multiseed.py`重新核对六个run的实际收据和预测，汇总三个成对seed。引用固定seed42 plan identity及完成报告SHA，禁止混入pilot/20轮结果。数值身份还包含FLEX实际导入的`_rgb_reference/modules.py`。
- `encoder_lifecycle.py`支持首批两个实际作业的启动门槛，后续两个排队作业各自核验；不会把排队称为通过验证。`collect_resolution.inspect_run`为新stage明确使用full_train_dev范围。记录三种计数：processed batches、optimizer updates、AMP skips。
- 风险：GPU1目前有ToDesk桌面服务，保持原状。seed43在GPU0（RTX PRO6000 Max-Q），seed44在GPU2（RTX5090）；同seed的BASE/FLEX先后运行在同卡，两个seed并行，最多两个训练进程。不同seed硬件不同，不用跨卡墙钟差归因编码器。每个run记录实际设备与成本。
- 验证：两组原算法已有完整前向、反向、恢复及正式40轮证据。本次新增seed的有符号/时间输入梯度、共享头hash、真实packed前向/更新/中断恢复/独立小样本评价，另检查排队门槛和协议锁定。只重读约8.77GB packed，不重扫约1.385TB原生缓存。

数值路径未变化，公平性检查通过后再推送代码/config/证据，核实远端SHA并建立detached worktree。未通过必要检查时不启动正式训练；不因无关RGB准备阻塞这四组。

## 固定协议与证据

seed42源plan：`/home/zhangyu/.local/share/speedest-v5-next/pipeline/20261031_r02_base_flex_s42/plan.json`，identity `0f23c724cb8225d60ccfd5b0b1647363479f2bf8c39ab0fb31d912f1092248a1`。原training commit `a847938681d60ae86e0e293e24ffa7d2e74e20a9`；原完成summary SHA `5aa3ffb431408b24e67a42e6655b793ee6550a74e3423812156d8d788474370d`。本轮使用新操作代码commit，训练数值文件逐字节比对相同。

每对head_seed显式等于该训练seed，在隔离CPU RNG中重新初始化同形状速度头；实际head state SHA必须相同。其余模块采用相同默认初始化政策；每轮样本顺序独立由seed+epoch生成并记录SHA。同seed BASE/FLEX逐epoch一致，不要求不同seed相同。

数据、标签、流单位及边界继承[原设计](ENCODER_COMPARISON_20261031.md)：PLANAR_PROXY后轴中点，五帧/四对历史流和四个真实dt；RAFT冻结1024×576、训练流128×72，signed u/v为该训练分辨率的像素位移。无RGB、FB或几何输入。holdout不参与训练、选择或本轮汇总。

侧向规则不变：abs(vy)>=0.2m/s，至少100窗口/5场景；实际仍31个独立窗口/5场景，insufficient_data。三个训练seed不把31变成93或增加场景数。汇总报告每个seed的EPE/分量MAE、RMSE、偏差、逐场景/侧向/常量基线和成对差值；均值与样本标准差仅描述初始化敏感性，不宣称统计显著优越。

## 运行与发布约定

本轮前缀：`20261031_r02_base_flex_s43_s44`；四个run后缀为`R02_BASE_s43`、`R02_FLEX_s43`、`R02_BASE_s44`、`R02_FLEX_s44`。artifact根`/home/zhangyu/.local/share/speedest-v5-next`。plan保存命令、固定commit、config与缓存身份；logs/<run_id>.log、runs/<run_id>/{last,best}.pt、evaluations/<run_id>/。40轮结束各自独立评价best，再执行合并核验。

每个实验tmux名`v5next_<run_id>`；重连`tmux attach -t <session>`，查看`tail -f <log>`。需要停止时向相应训练child PID发送SIGTERM，先核实PID身份；等待完整中断checkpoint落盘，不kill GPU上无关进程。恢复需新run ID、同固定commit/config/data与完整last.pt，使用train.py --resume <last.pt>；恢复操作不能冒充全新重复实验，必须记录来源。正常训练不随损失波动重启。

首批BASE43/44的前3步、首次dev、完整checkpoint重载正常后：5分钟服务器检查、30分钟Astra只读巡视。FLEX43/44启动后另生成各自proof。使用本轮新plan安装并验证实际cron触发，旧巡视已结束的成功记录不能替代。独立report worktree每小时及里程碑发布；网络失败保留pending并重试，正常训练继续。四组训练/评价/比较完成且本轮终态远端发布确认后移除本轮巡视。启动记录将补充实际training commit、启动证据与调度回执。

R03仅准备方案，见[R03-Frozen/FT及同起点flow-only对照](R03_FLEX_PREPARATION_20261031.md)，不混入本轮补seed预算。

## 启动前验证结果

测试套件278项通过（13.86秒）。真实packed核验seed43、44全部通过，分别见[evidence43](evidence/encoder_preflight_s43_20261031.json)、[evidence44](evidence/encoder_preflight_s44_20261031.json)。两seed各两模型都执行真实数据梯度/更新检查，训练中断3步后恢复到5步，独立评价128个dev窗口；这些仅为smoke验证，不报告为正式精度。数据量及分区边界与原完整协议相同。

seed43共同头SHA：`57f1f2dd990120c44e046d901257017cf25fcb0984b940075f2c3f1a2aea89be`；seed44：`48d8166ffa9388bfb0d52cccb7adbe9d5c99fc8b84008a1c1330e41e1a74c639`。同seed两模型一致，不同seed不同。新计划已通过原seed42报告、数值源文件、四份配置及缓存身份校验；正式plan会在固定已推送commit下重新生成，不能把当前开发工作区的dry plan当正式收据。

## 首次启动故障与保留有效训练（14:44后）

首轮正式代码`ace7519f8277026982cc843c4a95932ee64eeabc`已推送且远端branch/tag双重核实。首次前缀`20261031_r02_base_flex_s43_s44`的两个BASE在训练前再次核实GitHub版本时失败：tmux worker新会话继承tmux服务器旧代理，未继承已经验证可用的任务网络配置；TLS失败后SSH代理连接超时。两组均没有run.json、训练step或checkpoint，保留failed记录，不计入训练预算，不覆盖原run ID。

随后两个FLEX的远端实际校验成功，已经正常训练并产生有效dev/checkpoint，**保留其运行代码、配置、优化器和预算，不重启**。修复只让新tmux worker通过已有secure_exec.py加载任务私有network.json，来源与正常发布器相同；继续实际远端校验，不使用缓存SHA冒充联网成功，不改系统/global Git设置，不公开代理凭据。模型/训练数值路径完全不变。

新前缀`20261031_r02_base_retry_s43_s44`只补启动两个BASE（新ID），分别在同卡FLEX结束后运行；新比较器显式采用旧plan中的两个有效FLEX，记录各自真实training commit。新plan绑定旧plan字节SHA及身份，验证保留FLEX的真实run状态、数值代码与配置，禁止将失败BASE当作已训练结果。排队实际变为每seed FLEX→BASE，是网络故障后的资源安排，不改变数据、40轮、初始化和选模规则。旧比较job因BASE失败为blocked，由新比较job在全部有效run完成后取代。失败和被取代的操作记录一并发布。

原协调/生命周期进程将交接给新协调器；只结束操作进程，不向两个FLEX训练/worker发送停止信号。新生命周期先核验两个保留FLEX的首轮证据并启用本轮新调度，BASE接续时单独核验。结果发布同时覆盖seed42、首次计划和恢复计划。调度安装及实际触发需另见新回执，首次publisher探针脚本出现read_json参数误用，已在finally恢复小时调度；该未成功探针不能声称验证通过。

修复后61项针对排队、初始化、巡视、发布的测试通过；恢复plan的实际服务器输入校验也通过。恢复不会更换FLEX的初始随机状态或继承失败BASE状态。
