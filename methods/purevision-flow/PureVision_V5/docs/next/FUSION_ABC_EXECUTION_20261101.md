# A/B/C pilot 正式执行记录

用户已授权公共冻结缓存生成、A/B/C 各20轮、独立评价和最终比较。本记录是提交前的锁定安排；实际进度以结果分支的运行回执为准，不将缓存生成称为训练已启动。

## 固定身份与核验

实验前缀 `20261101_2129_fusion_abc_s42`；任务后缀 `cache`、`A`、`B`、`C`、`compare`。已提交的机器可读定义为 `FUSION_ABC_EXECUTION_20261101.json`，原32/8场景清单的原字节副本为 `FUSION_ABC_SCENES_s42_32t8v.json`。来源checkpoint、全部SHA、配置SHA、GPU UUID和预算均在定义中。

启动前只读核实远端开发 `13fc4525dad7a2c0518666cf03c60ba98ca2ba33`、结果 `b50f4b4555b420778af5e8a43baafdc9f486a9ce`。无V5训练或巡视残留；GPU无计算进程，GPU1桌面保留，旧FT为interrupted。实际启动前再核对资源。严格复用已训练外观参数、原RAFT/RGB缓存、7324/1826窗口；无需重编码RAFt或改模型。

本次数值实现和三份训练配置未变。新增范围为缓存进度/物理GPU/显存/耗时记录，固定DAG、生命周期、独立结果汇总和发布适配。既有27项相关测试及新汇总/运维回归共73项通过。补做1 train+1 dev真实缓存及1次一次性GPU更新，完整状态重载通过；见 `evidence/fusion_abc/execution_preflight.json`。观测代码首次调用在CUDA上下文初始化前重置显存计数而报错，已修复并通过重检，未影响任何正式run。原完整有限预检继续有效。

## 执行与调度

提交并推送后，以精确远端SHA创建 `$ART/code/20261101_2129_fusion_abc_s42_<sha7>` detached worktree；实际训练SHA写入plan和每份run。后续结果提交不作为训练版本。ART为 `/home/zhangyu/.local/share/speedest-v5-next`。

固定目录的V5工程内，入口：

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B fusion_abc_pipeline.py \
  --run-spec docs/next/FUSION_ABC_EXECUTION_20261101.json --materialize
/home/zhangyu/.conda/envs/speedest/bin/python -B fusion_abc_pipeline.py \
  --plan /home/zhangyu/.local/share/speedest-v5-next/pipeline/20261101_2129_fusion_abc_s42/plan.json
```

协调器和各任务挂独立tmux。GPU0以FP32、eval/no_grad生成一份缓存，完整检查来源、覆盖、形状、有限值和样本对应后，C→GPU0、B→GPU1、A→GPU2并行。所有命令UUID绑定，进程内均cuda:0；目标卡有计算进程则排队，不迁移抢占。每组115批/轮、20轮、2300更新，训练仅读取缓存及新增模型。每组随后独立重载best评价完整pilot train/dev；全部完成后才汇总。

缓存 `$ART/fusion_abc_cache/20261101_2129_fusion_abc_s42`；run `$ART/runs/20261101_2129_fusion_abc_s42_{A,B,C}`；日志 `$ART/logs/20261101_2129_fusion_abc_s42_*.log`；比较 `$ART/reports/fusion_abc_20261101_2129_fusion_abc_s42`。

`fusion_abc_lifecycle.py --plan PLAN --publisher-command PRIVATE_COMMAND --install` 安装本轮5分钟服务器、30分钟Astra、每小时17分独立发布，并保存精确cron行、下次触发及本轮scope。实际cron探针验证服务器读取、Astra只读执行与远端发布，收据在 `$ART/monitoring/fusion_abc_20261101_2129_fusion_abc_s42/`。同入口不加install启动后台生命周期，检查首次验证/完整checkpoint和里程碑；网络失败只保留并重试发布。只在最终比较完成且远端核验成功后删除本轮精确cron条目。若提前全部完成，记录实际触发是否观察到，不人为延长。

调度使用既有Codex CLI非交互只读执行能力（[官方文档](https://learn.chatgpt.com/docs/non-interactive-mode)）；必须以本轮实际回执证实启用，历史成功不替代。凭据、网络控制、完整权重、特征缓存不入Git。

## 汇总与停止边界

结果分支公开运行标识/固定训练commit、manifest与缓存哈希、参数/更新数、最佳/最后dev、独立train/dev及vx/vy指标、曲线、配对逐样本/场景结果、三组差异、GPU与耗时/显存。未全部20轮完成不形成最终比较。

仅单seed、少量场景的新读出筛选；编码器已在完整训练集训练且由既有dev选模。A/B同时改变融合与读出，B/C也不是仅注意力的严格消融；异构GPU时间不作等硬件效率比较。非零侧向不足100窗口/5场景仍为insufficient_data。不新增seed、全量训练或改编码器；旧FT保持中断。
