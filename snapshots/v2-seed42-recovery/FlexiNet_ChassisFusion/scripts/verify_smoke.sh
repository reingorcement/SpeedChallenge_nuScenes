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

run_step sanity_fair_gray \
  "${PYTHON_BIN}" -u scripts/path_sanity_check.py \
    --config configs/velocity_xy_fair_gray.yaml

run_step sanity_pose_assisted \
  "${PYTHON_BIN}" -u scripts/path_sanity_check.py \
    --config configs/velocity_xy_pose_assisted.yaml

run_step flow_backend_smoke \
  "${PYTHON_BIN}" -u scripts/precompute_optical_flow.py \
    --dataroot /media/ps/AC78E69278E65B1A/zy/nuscenes \
    --version v1.0-trainval \
    --split trainval \
    --cam_channel CAM_FRONT \
    --output_root /media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_flow/CAM_FRONT \
    --width 64 \
    --height 64 \
    --device auto \
    --limit 100

echo "[$(date '+%F %T')] Smoke verification finished."
