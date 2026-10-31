# R02_BASE / R02_FLEX 实施、协议与自审

本轮仅比较完整原 train_core/dev 上的 flow-only 编码器。开发起点及已核实远端 HEAD：`3015247e0f0da52460e51c5bbf35389db8201f02`。旧成果与分支保留；不重算全量光流，不读取 holdout/official test 标签，不加入 RGB/FB/几何模块。两组均新训 seed42、40个完整 epoch，旧 pilot/20轮结果不充当直接基线。

## 算法与数据路径

BASE 沿用原 R02：逐对 4→16→32→64 空间卷积，时间核 `(3,1,1)`，区域池化到 `(T,H,W)=(1,2,3)`，384→64 投影。FLEX 逐对 4→16→32→32 stem 后，并行使用 SFE(32,32) 和 MFE(32,16,32)，得到192维空间表示和128维有序时间表示，拼接后320→64投影。两组追加四个 dt，统一68→192→2速度头。

实际输入 `[B,4,4,72,128]`，四个通道是有符号u/v、valid、valid。最后一个valid沿用原R02接口，不是FB提示，没有新增筛选；现有端点合法性处理不变。光流保持128×72视图中的像素位移，四个真实dt以秒输入并除以固定1/12作尺度变换，不除入光流。无未来pose或未来时间作为模型输入。

FLEX 的 Conv3d 数据为 `[B,C,T,H,W]`。SFE只有空间卷积，不混合T；它的区域池化输出T=1、H=2、W=3。MFE的三层时间核均为3、stride1，时间感受野7；四步历史以两端零填充维持T=4。其池化只平均H/W，输出`[B,32,4,1,1]`，按C,T固定顺序展平。所有有效时间上下文都在这四对历史内；没有读取预测时刻之后的帧。感受野大于窗口、短窗口填充贡献是保留的候选局限，不能解释为七步真实观测。

固定作者来源：[FlexiNet modules.py, e25127f6912733213d46881d1139c9632e2f0114](https://github.com/Geekgineer/FlexiNet/blob/e25127f6912733213d46881d1139c9632e2f0114/model/modules.py)。已沿作者SFE/MFE实际forward核对；复用仓库已有等价整理版`_rgb_reference/modules.py`（SHA256 `78b1d491d8e3435b0492510799ad1f9774910108e16bbc2c232658b371cfcbb8`），不复制历史工程，也不调用其CMA/AFT/gate。整理版不是作者文件的逐字节副本。保留20261030草案结构；参数74354/76850，FLEX多3.36%。比较归因是整个编码器，包括BN、宽度及池化差异，不单独归因SFE或MFE。

## 共同协议及有限验证

原清单 SHA256 `a74f7963f7c9e0b529fefef75f5be1c191a7a2fccbddd3678b82f825ee0de32a`；train_core 582场景/132978窗口，dev103场景/23480窗口。训练与dev场景、图像token交集为空。PLANAR_PROXY、nuScenes后轴中点、原划分不按log隔离等限制不变。

复用已完成的1024×576 RAFT、128×72 packed视图。原生成commit `87d7468f911da5579bbbc39f79ec6bc85fa16dd6`；完成summary SHA256 `96946ac49805c1d4d57668bd0ff48db7cffa24371e2e277458dadcba8516b25c`；packed索引SHA256 `cce797035ce44b07e5f64ce0e560bfeb68a5f17bb1cb46ac8c16398a56782414`。位置见[全量缓存完成报告](FULL_FLOW_COMPLETE_20261031.md)。本轮读取8.77GB packed数组的既有校验入口和代表性真实batch，没有再扫1.385TB原生内容。

两组共同：batch64、无累积、AdamW lr0.001/wd0.0001、SmoothL1尺度[5,1]m/s、beta1、grad clip2、AMP、余弦40轮至0.0001；不早停；严格dev EPE选best。样本顺序由独立`randperm(seed+epoch)`产生并逐epoch记录SHA，与模型消耗随机数无关。编码器使用PyTorch默认Conv/Linear初始化，BN weight1/bias0。共同头在隔离CPU RNG中显式head_seed42重新初始化，两组头state SHA必须一致；不能仅凭全局seed推断。

真实检查入口`verify_encoders.py`：全量清单统计及分区交集、代表性batch64、每个空间/时间/投影/速度头（FLEX含SFE）梯度有限且真实参数更新；真实训练器中断3批、恢复至5批；恢复后独立评价128个dev样本。检查结果仅为smoke，不是模型精度成果。dt分位数[min,p1,median,p99,max]秒约[0.043824,0.049994,0.100000,0.102829,0.300008]。

同一GPU1 RTX5090、batch64、10次warmup+100次驻留batch的AMP forward/backward/AdamW：BASE约21298窗口/s、FLEX约10531窗口/s；峰值allocated约176.0/179.4MB。这是短时计算基准，排除数据读取、dev及checkpoint，未包含正式梯度裁剪；不能宣称等成本，正式墙钟与峰值另记录。两组初始头SHA均为`7de467aac88b930505733a8d7817bb1e1095fd075a804a27081d16121f7388a9`。详细证据在`evidence/encoder_preflight_20261031.json`。

自审：输入单位、时间轴、标签、分区和梯度路径一致；原R02默认行为兼容；训练/恢复/评价/启动核验均由同一模型工厂构造FLEX；共同初始化显式核对。新增代码局限是候选整体归因、短时成本测量及单seed；不能预设FLEX更好。

## 固定运行与巡视

正式入口`encoder_comparison.py`，两组GPU1/2（同型号5090）并行，完成后第三个CPU作业`collect_encoders.py`独立核对40轮、83120处理批次、更新/跳步恒等关系、逐epoch样本顺序、best选模、预测标签覆盖、checkpoint哈希。每次运行的plan记录实际代码SHA、全部命令和路径；只在代码/config推送且远端核实后，用detached worktree启动。

根目录`/home/zhangyu/.local/share/speedest-v5-next`；前缀`20261031_r02_base_flex_s42`；训练会话`v5next_<prefix>_R02_BASE`及`..._R02_FLEX`；日志`logs/<prefix>_<name>.log`；完整恢复状态`runs/<prefix>_<name>/{last,best}.pt`；独立评价`evaluations/<prefix>_<name>/`；比较`reports/encoder_comparison_<prefix>/summary.{json,md}`。运行收据保存启动命令及环境。

`encoder_lifecycle.py`不控制训练，只核验首轮、触发里程碑发布及管理本轮cron。两组步骤1/2/3、第一次dev、CPU重载完整checkpoint均通过后安装5分钟server/30分钟Astra及实际cron探针；小时发布由独立结果worktree执行，网络失败保留pending并重试。巡视状态含processed_batches、optimizer_updates、amp_skips、最新/最佳dev、日志/checkpoint更新时间、GPU/磁盘和发布回执。安装不等于实际触发，启用后必须记录本轮plan/run真实读取回执。

停止：向plan中记录的训练child PID发送SIGTERM，训练器保存last后退出；不用kill -9。恢复必须在相同固定代码/配置/数据下，以新run ID运行原train命令并加`--resume <原run>/last.pt`。原队列不自动重启失败作业；若用户决定恢复，需要注册新的运行与巡视记录。重连`tmux attach -t <会话名>`，查看`tail -f <日志>`。运行在持久服务器，tmux不依赖当前对话进程；机器故障仍依赖完整checkpoint恢复。

全部训练、评价和比较终态在GitHub实际发布后，才停止本轮专属巡视/发布，保留历史。结果分支与训练commit分开记录；大权重及缓存不进入Git。

## 判读规则

完整dev非零侧向为31窗口/5场景，仍不足100窗口/5场景，保持`insufficient_data`。整体vy MAE退化容忍`max(0.01m/s,5% BASE MAE)`不变。本轮复核零vy及仅train_core拟合的中位数vy基线，并新增报告train中位数二维常量；不使用dev拟合常量。

结果同时报告EPE、vx/vy MAE/RMSE/bias、逐场景、非零子集、参数/吞吐/峰值显存/完整墙钟。若FLEX改善EPE且整体侧向不过度退化，只能作为补seed43/44的候选；不足的侧向证据与单seed限制不能消除。本轮不自动开始RGB Frozen/FT。
