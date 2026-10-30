#!/usr/bin/env bash
set -euo pipefail
umask 077
V5_ARTIFACT_ROOT="$1"
V5_COMMAND_FILE="$2"
V5_TRIGGER="${3:-manual}"
V5_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_PYTHON=/home/zhangyu/.conda/envs/speedest/bin/python
# publish_results.py owns its own flock; this wrapper never edits job/algorithm state.
/usr/bin/timeout 600 "$V5_PYTHON" -B "$V5_CODE_ROOT/scripts/run_publisher.py" "$V5_ARTIFACT_ROOT" "$V5_COMMAND_FILE" --trigger "$V5_TRIGGER"
