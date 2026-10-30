#!/usr/bin/env bash
set -euo pipefail
V5_ARTIFACT_ROOT="${1:-/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5}"
V5_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_PYTHON=/home/zhangyu/.conda/envs/speedest/bin/python
mkdir -p "$V5_ARTIFACT_ROOT/monitoring"
exec 9>"$V5_ARTIFACT_ROOT/monitoring/server_check.lock"
/usr/bin/flock -n 9 || exit 0
/usr/bin/timeout 60 "$V5_PYTHON" -B "$V5_CODE_ROOT/monitor.py" --root "$V5_ARTIFACT_ROOT" --write --json --auto-disable
