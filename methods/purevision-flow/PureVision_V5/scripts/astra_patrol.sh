#!/usr/bin/env bash
set -euo pipefail
umask 077
V5_ARTIFACT_ROOT="${1:-/home/zhangyu/.local/share/speedest-v5-next}"
V5_TRIGGER="${2:-manual}"
V5_STARTUP_PROOF="${3:-}"
V5_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_PYTHON=/home/zhangyu/.conda/envs/speedest/bin/python
V5_CODEX=/home/zhangyu/.vscode-server/extensions/openai.chatgpt-26.1007.21434/bin/linux-x86_64/codex
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
V5_PREPARE=("$V5_PYTHON" -B "$V5_CODE_ROOT/scripts/prepare_patrol.py" "$V5_ARTIFACT_ROOT" "$V5_PREFIX")
if [[ "$V5_TRIGGER" == startup ]]; then
  V5_PREPARE+=(--startup "$V5_STARTUP_PROOF")
fi
"${V5_PREPARE[@]}"
set +e
/usr/bin/timeout 300 "$V5_PYTHON" -B "$V5_CODE_ROOT/scripts/secure_exec.py" --network "$V5_ARTIFACT_ROOT/monitoring/network.json" --   "$V5_CODEX" exec --json --ephemeral --skip-git-repo-check --sandbox read-only --model gpt-6-astra   -c 'approval_policy="never"' -c 'model_reasoning_effort="low"' --color never   -C "$V5_ARTIFACT_ROOT" --output-last-message "$V5_PREFIX.txt" - < "$V5_PREFIX.prompt.txt" > "$V5_PREFIX.log" 2>&1
V5_EXIT_CODE=$?
set -e
"$V5_PYTHON" -B "$V5_CODE_ROOT/monitor.py" --root "$V5_ARTIFACT_ROOT" --patrol-result "$V5_PREFIX" --patrol-exit-code "$V5_EXIT_CODE" --trigger "$V5_TRIGGER"
exit "$V5_EXIT_CODE"
