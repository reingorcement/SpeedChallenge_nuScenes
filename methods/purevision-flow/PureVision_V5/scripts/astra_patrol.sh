#!/usr/bin/env bash
set -euo pipefail
V5_ARTIFACT_ROOT="${1:-/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5}"
V5_TRIGGER="${2:-manual}"
V5_STARTUP_PROOF="${3:-}"
V5_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_PYTHON=/home/zhangyu/.conda/envs/speedest/bin/python
V5_CODEX=/home/zhangyu/.vscode-server/extensions/openai.chatgpt-26.1007.21434/bin/linux-x86_64/codex
if [[ "$V5_TRIGGER" == startup ]]; then
  V5_STARTUP_RUN="$($V5_PYTHON - "$V5_ARTIFACT_ROOT" "$V5_STARTUP_PROOF" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
proof = Path(sys.argv[2]).resolve()
if not proof.is_file() or not proof.is_relative_to(root / 'monitoring/startup') or proof.name != 'startup_proof.json':
    raise ValueError('Startup proof must exist beneath this artifact root monitoring/startup')
value = json.loads(proof.read_text())
run_id = value['run_id']
if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id) or proof.parent.name != run_id:
    raise ValueError('Startup proof run identity/path mismatch')
print(run_id)
PY
  )"
fi
if [[ -f "$V5_ARTIFACT_ROOT/monitoring/network.env" ]]; then
  source "$V5_ARTIFACT_ROOT/monitoring/network.env"
fi
mkdir -p "$V5_ARTIFACT_ROOT/monitoring/astra"
exec 9>"$V5_ARTIFACT_ROOT/monitoring/astra.lock"
/usr/bin/flock -n 9 || exit 0
V5_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
V5_PREFIX="$V5_ARTIFACT_ROOT/monitoring/astra/$V5_STAMP"
/usr/bin/timeout 60 "$V5_PYTHON" -B "$V5_CODE_ROOT/monitor.py" --root "$V5_ARTIFACT_ROOT" --write --json > "$V5_PREFIX.snapshot.json"
if [[ "$V5_TRIGGER" == scheduled ]]; then
  V5_ACTIVE="$($V5_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["active_count"])' "$V5_PREFIX.snapshot.json")"
  if [[ "$V5_ACTIVE" == 0 ]]; then
    "$V5_PYTHON" -B "$V5_CODE_ROOT/monitor.py" --root "$V5_ARTIFACT_ROOT" --auto-disable > /dev/null
    exit 0
  fi
fi
cat > "$V5_PREFIX.prompt.txt" <<'PROMPT'
你是本机 PureVision V5 的只读 Astra 巡视员。用户只授权巡视，不授权修改算法/代码/配置/标签/训练参数，不授权启动、停止、重启或续训作业，不授权发送消息或访问外网。
重要执行边界：外层 monitor.py 在训练宿主机命名空间采集真实 PID/GPU/磁盘信息。你的只读 sandbox 可能隔离 /proc 和 GPU 设备；PID 存活与 GPU 状态以 monitoring/latest.json 的宿主机快照为准，不运行 nvidia-smi/ps/kill 来将 sandbox 不可见误判成宿主机故障。你能直接读取同机训练工件，用状态和日志内容交叉检查快照。
先实际使用 shell 读取当前工作目录 monitoring/latest.json，再读取相关 runs/*/status.json 或缓存作业日志末尾（最多80行）。如果当前没有正式作业，读取 data/nuscenes_proxy_v1/dataset.json 的 counts 验证本机可达；明确“无正式作业”。禁止读取凭据、认证配置、用户其他项目或真实数据图像。
核查进程与步数、训练和验证指标、异常日志、GPU/显存、磁盘、checkpoint 更新时间。区分 queued/running/completed/failed/interrupted，不把代码跑通或训练启动称为有效结果。正常损失波动不触发改算法；异常只写定位建议。写简短中文巡视报告，列出实际读取路径、异常、下一次人工核查动作。只在真正完成本地读取后以独立一行 ASTRA_PATROL_OK 结束。只把报告写到最终回答，文件保存由外层脚本负责。
PROMPT
if [[ "$V5_TRIGGER" == startup ]]; then
  cat >> "$V5_PREFIX.prompt.txt" <<PROMPT

本次是指定实验的首次启动巡视，目标 run_id 为 $V5_STARTUP_RUN。必须实际读取以下文件，不能仅根据其他缓存作业或全局快照宣布完成：
- $V5_STARTUP_PROOF
- $V5_ARTIFACT_ROOT/runs/$V5_STARTUP_RUN/run.json
- $V5_ARTIFACT_ROOT/runs/$V5_STARTUP_RUN/status.json
- $V5_ARTIFACT_ROOT/runs/$V5_STARTUP_RUN/metrics.jsonl 的首条完整记录
- $V5_ARTIFACT_ROOT/logs/$V5_STARTUP_RUN.log 的最初训练步记录和末尾（各最多80行）
核对该 run 的实际步骤1、2、3、首次验证、startup_proof 中 last.pt/best.pt 的 CPU 加载证据及 commit/contract 一致性。不要自行加载大 checkpoint；CPU 检查已由外层完成。最终报告明确写出此 run_id、实际读取路径及通过/缺失项；缺少目标实验文件时不能把其他作业状态当成该实验验证成功。
PROMPT
fi
set +e
/usr/bin/timeout 300 "$V5_CODEX" exec --ephemeral --skip-git-repo-check --sandbox read-only --model gpt-6-astra \
  -c 'approval_policy="never"' -c 'model_reasoning_effort="low"' --color never \
  -C "$V5_ARTIFACT_ROOT" --output-last-message "$V5_PREFIX.txt" - < "$V5_PREFIX.prompt.txt" > "$V5_PREFIX.log" 2>&1
V5_EXIT_CODE=$?
set -e
"$V5_PYTHON" -B "$V5_CODE_ROOT/monitor.py" --root "$V5_ARTIFACT_ROOT" --patrol-result "$V5_PREFIX" --patrol-exit-code "$V5_EXIT_CODE" --trigger "$V5_TRIGGER"
exit "$V5_EXIT_CODE"
