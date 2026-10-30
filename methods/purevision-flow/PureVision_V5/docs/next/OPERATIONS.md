# 下一阶段运行、巡视与结果发布

本页是操作说明，不是后台任务已启用的证明。人工时间使用 Asia/Shanghai，机器凭证保留 UTC。
当前仅授权 512×288 与 1024×576 的小规模 R02 比较；分辨率选择及后续全量计算由用户确认。

## 目录与权限

- 新工件/控制根 `ART=/home/zhangyu/.local/share/speedest-v5-next`；任务账号 `zhangyu`，目录700。
- `ART/monitoring` 为700，`network.json` 为600，只保存 HTTP(S)/ALL/NO_PROXY 的大小写白名单变量；Python按JSON数据读取，**不再 source shell env**。不公开代理值或凭据。
- 固定训练与运维源码：`ART/code/<training_commit前12位>/methods/purevision-flow/PureVision_V5`，来自已推送、核对远端SHA的干净detached worktree。
- 独立结果worktree：`ART/report-worktree`，分支 `results/purevision-v5-next`。开发分支是 `method/purevision-v5-next`。
- 旧 `/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/monitoring` 及相关父目录的 `root:root 0777` 遗留**未原位修复**，旧结果保留。本轮保护的是新home目录中的控制文件，不声称整个旧仓库父路径已经安全。

大数据、光流、日志和checkpoint在worktree外。复用旧缓存只读；新版本不得覆盖旧缓存身份。

## 当前真实验证状态

2026-10-30 **21:17:28**，开发版脚本的真实Astra手动调用成功；模型为 `gpt-6-astra`，CLI退出0，完成了指定宿主快照的实际读取。

凭证：`ART/monitoring/astra/20261030T131656Z_3310040.meta.json`，同前缀 `.log/.txt/.request.json/.snapshot.json` 保留原始事件和报告。
该调用是**无正式作业的开发smoke**，`run_ids=[]`，不能证明某个run通过启动检查，也不能证明cron已经启用。
38项针对性测试、shell语法检查和diff检查通过，记录在 `ART/evidence/operations_validation.json`；这些测试不替代固定版本中的真实运行核验。
随后检查发现startup watcher CLI仍有旧“必须六组”的限制，已改为接受非空、身份唯一且模型受支持的作业清单，并加入真实子进程CLI的两组pilot入口测试；不启动真实训练或Astra来模拟失败状态。

本文写入时，正式调度尚未安装。后续是否启用，应实际查看crontab、固定代码SHA、`schedule.json`以及新的 `trigger=cron_probe/scheduled` 读取成功凭证；不要沿用本段手动smoke或旧轮次历史状态。

## 三项独立机制

| 机制 | 频率与职责 | 成功/停止依据 |
|---|---|---|
| 宿主轻量检查 | 每5分钟；PID及start ticks、子进程、步数、日志/状态、GPU、盘、checkpoint时间 | alive/dead/unknown分别记录；unknown保留巡视；不控制训练 |
| Astra只读诊断 | 每30分钟；固定CLI、300秒超时、独立flock | JSONL中精确读取helper的成功 `command_execution`，并覆盖指定run/路径/read receipts；结束语本身不足以通过 |
| 结果发布 | 每小时17分；独立锁、600秒超时、独立结果worktree | 只提交白名单报告；远端SHA确认后算published；网络失败pending补传，不影响训练 |

外层宿主快照负责真实PID/GPU命名空间；Astra sandbox看不到设备或PID时，不能据此判训练死亡。
监控仅接纳明确终态，或同worker状态下两次确认worker/child/tmux消失且作业锁释放的协调器证据；解析失败、权限不足、父死子活均不能冒充终止。

首次启动watcher检查步骤1/2/3、首次完整验证以及last/best的CPU恢复状态，再发起指定run的Astra读取。checkpoint会被后续原子替换，所以首次检查SHA与最终SHA分别记录。
巡视不改算法、超参、配置，不自动kill/restart/resume。明确bug由开发目录修复、测试、提交推送；是否重跑按影响判断，研究决断交用户。

## 固定版本安装与查看

以下是待填入真实commit/plan后执行的命令，不表示已经运行：

```bash
ART=/home/zhangyu/.local/share/speedest-v5-next
PY=/home/zhangyu/.conda/envs/speedest/bin/python
FIXED_V5="$ART/code/<commit12>/methods/purevision-flow/PureVision_V5"
PLAN="$ART/pipeline/<实际prefix>/plan.json"

# 单次宿主读取；然后单次Astra。读取meta的success、run_ids、read_evidence。
"$PY" -B "$FIXED_V5/monitor.py" --root "$ART" --write --json
bash "$FIXED_V5/scripts/astra_patrol.sh" "$ART" manual

# 真实cron探针：每分钟临时条目，触发后移除；必须等待并核对实际凭证。
"$PY" -B "$FIXED_V5/monitor.py" --root "$ART" --install-probe "$FIXED_V5"

# 仅在固定版本手动及真实cron均验证后安装正式5分钟/30分钟条目。
"$PY" -B "$FIXED_V5/monitor.py" --root "$ART" --install-cron "$FIXED_V5"

crontab -l
cat "$ART/monitoring/STATUS.md"
cat "$ART/monitoring/last_astra.json"
tmux list-sessions
tmux attach -t <plan.json中的实际session>
tail -f "$ART/logs/<实际run_id>.log"
```

`plan.json`、`pipeline/<prefix>/jobs/*.json`、`monitoring/jobs.json`记录真实tmux名、命令、PID及start ticks。
各run的 `run.json/status.json/metrics.jsonl` 记录实际配置、数据/缓存、training commit、处理批次、optimizer updates及AMP skips。
启动后需确认步数推进、首次验证和完整恢复checkpoint，不以“tmux存在”替代。

确需人工停止时，核对实际child PID/start ticks后发送TERM，让训练器先写last恢复状态；不直接杀整个tmux。
恢复必须新run ID、兼容模型/数据/损失合同、固定代码与独立日志，使用训练入口 `--resume <原last.pt> --output <新run目录>`，同时记录parent run；语义改变不盲目续训。

## 发布器设置与核验

发布器只写 `docs/next_results/`，使用 `experiments/<run_id>/` 避开仓库忽略 `runs/` 的规则；另允许导出 `ART/evidence/*.json` 和 `ART/reports/resolution*/summary.{json,md}`。
JSONL仅取完整行；JSON可能重新排版，`source_sha256`与`export_sha256`分别记录。小型逐样本预测可以发布，权重仅发布路径/哈希；大日志、原始数据、光流和网络env均不进入Git。

先用实际参数手动发布：

```bash
"$PY" -B "$FIXED_V5/publish_results.py" \
  --root "$ART" --worktree "$ART/report-worktree" --plan "$PLAN" \
  --push-url git@github.com:reingorcement/SpeedChallenge_nuScenes.git \
  --stop-when-terminal
```

需要多计划时重复 `--plan`。`ART/monitoring/publisher_command.json` 是权限600的私有控制文件，格式为 `{"command":["<绝对PY>","-B","<FIXED_V5>/publish_results.py","--root","<ART>","--worktree","<ART>/report-worktree","--plan","<实际PLAN>","--push-url","git@github.com:reingorcement/SpeedChallenge_nuScenes.git","--stop-when-terminal"]}`。
使用实际绝对路径，不把尖括号占位符写入正式文件。

```bash
"$PY" -B "$FIXED_V5/monitor.py" --root "$ART" \
  --install-publisher "$FIXED_V5" \
  --publisher-command-file "$ART/monitoring/publisher_command.json"
cat "$ART/monitoring/publication.json"
cat "$ART/monitoring/publisher_schedule.json"
```

发布前拒绝worktree未知修改；不会全仓 `git add -A`、自动合并或force push。scoped SSH代理只用于本次Git网络命令，不改全局Git/SSH配置。
首次启动、首次验证、完成和失败应前台及时调用发布，不必等待整点。无人值守期间每小时同步，有变化才提交。
`publication.json`记录training commits、report commit、远端核验与pending原因；真正定时执行才将publisher schedule标记为verified_trigger。

所有作业真正终止后，宿主检查可以移除本任务 `_SERVER/_ASTRA/_PROBE` 条目，**保留独立 `_PUBLISHER`**；最终结果成功推送后publisher才停止自己的条目，不影响其他项目。
人工拆除命令分别为 `monitor.py --root "$ART" --remove-cron` 和 `--remove-publisher`；原crontab有备份。
tmux、cron和本地持久磁盘均不保证服务器永不故障，完整checkpoint仍是恢复基础。
