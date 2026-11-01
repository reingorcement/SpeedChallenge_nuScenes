#!/usr/bin/env bash
set -euo pipefail
abc_code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/zhangyu/.conda/envs/speedest/bin/python -B "$abc_code_root/fusion_abc_launch.py" "$@"
