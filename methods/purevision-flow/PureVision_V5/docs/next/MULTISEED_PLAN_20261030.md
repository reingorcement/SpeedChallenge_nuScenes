# 已批准：补充训练seed43、44

用户于2026-10-30（Asia/Shanghai）选择“先补两个seed，再选分辨率”。当前授权是四个小规模R02训练与评价，未选择全量分辨率。

## 实施与自审

- 同一原manifest和seed42选定的32train/8dev场景，7324/1826窗口；selection seed与training seed是两个独立概念。
- 已有512/1024缓存只读复用，模型视图统一128×72，不重新生成RAFT，不改变标签/时间/参考点/有效掩码规则。
- 四份配置仅改变训练seed=43/44和既定expected_flow；各40完整epoch、batch64、原loss/AdamW/cosine、禁早停、dev EPE选模。
- 两RTX5090分担低/高组，逐卡先43再44，避免GPU叠加；每个新run重新初始化，不继承seed42优化器或权重。
- 新队列复用pipeline/tmux，新增三seed收集器复用已有CSV/标签/checkpoint/epoch验真。模型、训练器、数据和数值评价路径保持逐字一致。
- 每个seed的两分辨率必须具有相同初始参数及40轮顺序；跨seed可以不同。seed42保留原run/原training commit，只引用其已完成结果。
- 每次worker命令前核验源plan、selection、两个packed index、配置和数值代码身份；正式训练仍须推送成功及固定detached工作目录。
- 新报告只在三对均完整可比时给均值、sample SD(ddof=1)和配对差值；失败/不完整不会填补假指标。
- 顺序沿用原 `seed + epoch` 规则，邻近seed的逐轮排列存在移位复用；本轮是初始化/顺序敏感性检查，不把三个seed称为统计显著性证明。
- 三seed不增加独立场景或标签数。非零侧向仍仅13窗口/1场景，不把重复训练的预测当成39个独立样本。

验证安排：针对新增队列/配置/验真接口的测试，真实CUDA检查seed初始化和更新；已有恢复/加载验证继续适用，因为数值源码未改。正式启动后实际核查步数、首次验证、checkpoint及各run的Astra读取。

沿用每5分钟宿主检查、每30分钟Astra、独立结果发布。真实定时触发证据与安装状态分开记录，完成后清理本轮条目。新结果进入 `results/purevision-v5-next` 的 `docs/next_results/`；全量分辨率依然由用户在结果后决定。
