#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
FLOW_ROOT="${FLOW_ROOT:-${DATA_ROOT}/precomputed_flow/CAM_FRONT}"
FLOW_DEVICE="${FLOW_DEVICE:-auto}"
FLOW_WIDTH="${FLOW_WIDTH:-64}"
FLOW_HEIGHT="${FLOW_HEIGHT:-64}"
FLOW_SPLIT="${FLOW_SPLIT:-trainval}"
FLOW_LIMIT="${FLOW_LIMIT:-0}"

mkdir -p logs

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" 2>&1 | tee "logs/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

run_step precompute_flow_smoke \
  "${PYTHON_BIN}" -u scripts/precompute_optical_flow.py \
    --dataroot "${DATA_ROOT}" \
    --version v1.0-trainval \
    --split "${FLOW_SPLIT}" \
    --cam_channel CAM_FRONT \
    --output_root "${FLOW_ROOT}" \
    --width "${FLOW_WIDTH}" \
    --height "${FLOW_HEIGHT}" \
    --device "${FLOW_DEVICE}" \
    --limit 100

if [[ "${FLOW_LIMIT}" == "0" ]]; then
  run_step precompute_flow_full \
    "${PYTHON_BIN}" -u scripts/precompute_optical_flow.py \
      --dataroot "${DATA_ROOT}" \
      --version v1.0-trainval \
      --split "${FLOW_SPLIT}" \
      --cam_channel CAM_FRONT \
      --output_root "${FLOW_ROOT}" \
      --width "${FLOW_WIDTH}" \
      --height "${FLOW_HEIGHT}" \
      --device "${FLOW_DEVICE}"
else
  run_step "precompute_flow_limit_${FLOW_LIMIT}" \
    "${PYTHON_BIN}" -u scripts/precompute_optical_flow.py \
      --dataroot "${DATA_ROOT}" \
      --version v1.0-trainval \
      --split "${FLOW_SPLIT}" \
      --cam_channel CAM_FRONT \
      --output_root "${FLOW_ROOT}" \
      --width "${FLOW_WIDTH}" \
      --height "${FLOW_HEIGHT}" \
      --device "${FLOW_DEVICE}" \
      --limit "${FLOW_LIMIT}"
fi

run_step train_fair_flow \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_fair_flow.yaml

run_step train_fair_rgb_flow_geom \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_fair_rgb_flow_geom.yaml

echo "[$(date '+%F %T')] Flow precompute and training runs finished."
