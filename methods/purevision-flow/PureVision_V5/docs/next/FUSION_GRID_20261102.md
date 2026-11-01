# 六组冻结特征融合与网格 pilot

执行定版用户方案 `FUSION_GRID_USER_PLAN_20261102.txt`。固定六组、seed42、20 epoch、7324/1826窗口、batch64、FP32。训练直接输出物理单位[vx,vy]。

新代码局限于 `fusion_grid*`、`prepare_fusion_grid.py`、`train_fusion_grid.py`、六份配置以及结果发布适配；历史模型和实验产物保持原样。J23由历史build_triplet的C得到，其他模型显式复制共同参数。新4×6缓存直接由冻结原始feature map池化；复用2×3经过来源、全量样本身份及少量真实数值核验。

每个完整epoch保存不可变checkpoint；`recovery.json`原子绑定当前和历史best及SHA。`best.pt`/`last.pt`只是导出硬链接，恢复不依赖它们。恢复使用相同固定实现、配置、缓存并写新目录，完整epoch以外的更新作废并单列。跨设备/软件不保证位级轨迹一致。

队列：PRO6000提取4×6，两张5090先运行J23/S23；4×6清单发布后优先S46/J46，单路填空。每卡最多一个正式进程。四双路及六组完成分别由队列触发汇总/发布，不等30分钟巡视。沿用5分钟服务器检查、30分钟Astra、每小时GitHub发布。

正式入口：固定已推送detached worktree中 `fusion_grid_pipeline.py --materialize --run-spec docs/next/FUSION_GRID_EXECUTION_20261102.json`。先发布2×3就绪清单，再安装巡视并启动同一入口 `--plan <plan.json>`。所有输出保存在外部产物根目录，运行代码固定后不再修改。

分离token与更细网格分别有匹配对照，但网格包含配套位置编码变化。单种子仅作候选筛选，单路只与2×3直接比较。最终候选复核建议和可审核提示词由报告给出，不自动补seed或扩大数据。
