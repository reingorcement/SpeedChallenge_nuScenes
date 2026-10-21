#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p logs

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" 2>&1 | tee "logs/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

run_step train_fair_gray \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_fair_gray.yaml

run_step train_pose_assisted \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_pose_assisted.yaml

echo "[$(date '+%F %T')] All baseline runs finished."
