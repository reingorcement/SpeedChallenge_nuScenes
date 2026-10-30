# V5 训练与巡视操作

训练、缓存和巡视工件根目录为 `/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5`。正式训练代码必须来自已推送 commit 的 detached worktree，运行期间不编辑该目录。开发目录和训练目录分开；本页的 `FIXED_V5` 指实际锁定工作目录中的 `methods/purevision-flow/PureVision_V5`，实际路径记录在每次 launch/run 和 `monitoring/schedule.json`。

## 训练会话与恢复

每个作业对应唯一 run ID、tmux session 和独立日志。`runs/<run_id>/launch.json` 记录 PID/session/device，建议同时保存 `/proc/<pid>/stat` 的 start_ticks，供巡视排除 PID 重用。`run.json` 记录 commit、配置、数据/特征/标签版本；`status.json` 保存步数及终态；`metrics.jsonl` 保存训练和验证指标。

```bash
tmux list-sessions
tmux attach -t <launch.json中的session>
tail -f /mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/runs/<run_id>/stdout.log
```

人工确认需要停止时，向 **launch.json 中经核实的训练 PID** 发送 `kill -TERM <pid>`。训练器捕获 TERM/INT 后先保存可恢复的 `last.pt`，然后将状态写为 `interrupted`。等待 checkpoint 更新时间和退出状态确认；不直接 kill 整个 tmux 会话。巡视本身永不发送停止信号。

恢复由人工启动为新 run ID，沿用兼容任务/配置/标签/模型语义，记录原 run ID：

```bash
cd "$FIXED_V5"
/home/zhangyu/.conda/envs/speedest/bin/python -B train.py \
  --config <原实际配置> --manifest <原dataset.json> \
  --features <原feature缓存目录> --flow <原flow缓存目录> \
  --resume <原run目录>/last.pt --output <新run目录> --device <原GPU>
```

不需要光流的 R00 省略 `--flow`；参数必须与任务契约匹配。恢复命令仍须放进新 tmux 会话，日志落盘。`last.pt` 保存模型、优化器、调度器、AMP scaler、随机状态、epoch、下一个 batch 与累计指标；旧标签/损失/模型语义发生变化时重新训练，不能盲目恢复。

tmux 只能使进程脱离当前交互。本机 cron 和持久磁盘已经可用，但这不构成永不断电/永不回收保证；checkpoint 是恢复基础。不能把运行中称为完成，不能把完成称为取得改善。

## 巡视实现和状态

- `monitor.py` 在 **训练宿主机进程/设备命名空间** 检查 PID、训练进度、最新指标、日志异常、GPU/显存、磁盘及 checkpoint 更新时间，写入 `monitoring/snapshots/`、`latest.json` 和 `STATUS.md`。正常波动或日志异常文本只记录为诊断信号，不改训练。
- `scripts/server_check.sh` 使用 flock 防止重叠，60 秒超时，每5分钟运行一次。训练状态明确区分 `queued/running/completed/failed/interrupted`。运行标记仍在而 PID 消失时标为 interrupted，不伪造成功或错误原因。
- 缓存的 `complete`/`done` 统一显示为 completed，展示 `pairs_processed/counts` 或 `completed/total` 进度；依赖失败后的 `blocked` 是终态，全部作业 completed/failed/interrupted/cancelled/blocked 后停止对应定时。
- `scripts/astra_patrol.sh` 使用独立 flock、300 秒超时，通过本机既有 Codex CLI 的 **gpt-6-astra** 执行只读巡视。CLI 参数包含 `--ephemeral --sandbox read-only`、禁止交互审批。Astra 必须实际读取宿主机工件，并以 `ASTRA_PATROL_OK` 结束；脚本检查退出码、最终回答及真实 shell 调用痕迹。报告、CLI 日志和结果元数据进入 `monitoring/astra/`。
- **sandbox 边界**：只读 CLI 能读取同机工件，但可能看不到宿主机 PID/GPU 设备。Astra 依据外层 `monitor.py` 的宿主机快照判断 PID/GPU，直接读取状态和日志交叉核验，不能将 sandbox 的 GPU 不可见误报为训练机器故障。
- 不存在正式作业时，scheduled Astra 不调用模型；只留下服务器快照。所有登记作业终态后自动移除本任务两条 cron。尚未启动但已计划的实验须在 registry 中登记为 queued，防止阶段间提前结束巡视。

外部缓存/排队作业可登记 `monitoring/jobs.json`，不需要修改训练目录：

```json
{"jobs":[{"job_id":"cache_rgb_v1","kind":"cache","status":"running","pid":12345,"pid_start_ticks":123456,"session":"v5-cache-rgb","device":"cuda:1","log_path":"/absolute/cache.log","status_path":"/absolute/optional-status.json"},{"job_id":"R03_pending","kind":"training","status":"queued"}]}
```

实际 run 目录自动发现；若 registry 与 run 同一作业，用相同 `job_id`。缓存任务完成后由其启动包装器/人工更新终态。没有退出状态证据时，巡视不会猜测缓存任务成功。

## 定时安装、验证和移除

官方 [Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app) 不意味着 tmux 会自动唤醒模型；本项目实际使用服务器 cron 调用 [non-interactive Codex CLI](https://learn.chatgpt.com/docs/non-interactive-mode)。已核查本机 CLI 帮助、cron daemon 和直接访问工件的能力。

```bash
# 单次轻量检查；不依赖 Codex 连接。
/home/zhangyu/.conda/envs/speedest/bin/python -B "$FIXED_V5/monitor.py" --write

# 单次 Astra；先验证成功，再启用定时。
bash "$FIXED_V5/scripts/astra_patrol.sh" /mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5 manual

# 真正的每分钟临时 cron 探针，触发一次后自动删除探针行。
/home/zhangyu/.conda/envs/speedest/bin/python -B "$FIXED_V5/monitor.py" --install-probe "$FIXED_V5"

# 正式每5分钟服务器检查、每小时0分 Astra。仅接受干净 detached worktree。
/home/zhangyu/.conda/envs/speedest/bin/python -B "$FIXED_V5/monitor.py" --install-cron "$FIXED_V5"

# 只移除本任务的 cron 行，保留其他任务。
/home/zhangyu/.conda/envs/speedest/bin/python -B "$FIXED_V5/monitor.py" --remove-cron
```

每次改动 crontab 前，原始内容保存在 `monitoring/crontab_backups/`；只管理带该 artifact root 唯一标记的 V5 行，不删除其他条目。`schedule.json` 的初态为 `installed_pending_observed_trigger`；须以 `astra/*.meta.json` 中 `trigger=cron_probe, success=true` 的实测记录证明真实调度执行。仅安装 crontab 或手动运行成功不等于定时巡视已经验证。

本机交互环境通过已有 `127.0.0.1:7900` HTTP(S) 代理访问模型，而 cron 不继承交互环境。第一次真实 cron 探针因此连接超时，已保留失败日志。巡视脚本会读取外部 `monitoring/network.env` 中显式保存的代理变量；不把认证信息写入代码、日志或该文件。代理服务不可用时报告巡视失败，不能宣称模型已完成巡视。

可手动核查：`crontab -l`、`monitoring/STATUS.md`、`latest.json`、`last_astra.json`、`astra/*.meta.json`、`cron.log`。连接失败、超时或没有真实 shell 读取均记录失败，不声称自动回访有效；服务器轻量检查仍可独立运行。异常先定位，再由开发流程修复、验证、提交并决定受影响实验是否重跑。

### 已实际验证的调度证据

2026-10-30 上海时间 11:22:47，加入代理环境后的完整脚本在精简环境下手动成功。随后由真实 cron 于 **11:24:01** 触发同一外部只读快照脚本，**11:24:46** 完成：`model=gpt-6-astra, exit_code=0, success=true, actual_shell_read_observed=true`。文件为 `monitoring/astra/20261030T032401Z_3226289.meta.json`、同名 `.txt/.log`；临时探针行已自动移除。此前 11:20 的缺代理失败记录保留，没有隐藏失败。

此探针验证了服务器定时调度、模型连接、本机工件可达及报告落盘，测试时还没有正式作业。**正式每小时配置的实际安装状态以 `monitoring/schedule.json` 和 crontab 为准**，不能把探针成功写成正式训练已被巡视。探针代码位于外部 `monitoring/probe_code_v3_20261030`；正式配置必须指向随后已提交的固定 detached worktree。
