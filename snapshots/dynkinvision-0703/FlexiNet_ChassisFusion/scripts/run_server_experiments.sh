#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs/server_experiments}"
STATUS_FILE="${STATUS_FILE:-${LOG_ROOT}/status.tsv}"
RUN_SANITY="${RUN_SANITY:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"

mkdir -p "${LOG_ROOT}"

timestamp() {
  date '+%F %T'
}

write_status_header() {
  if [[ ! -f "${STATUS_FILE}" ]]; then
    printf "timestamp\texperiment\tstatus\tconfig\texit_code\telapsed_sec\tlog_file\n" > "${STATUS_FILE}"
  fi
}

append_status() {
  local experiment="$1"
  local status="$2"
  local config="$3"
  local exit_code="$4"
  local elapsed_sec="$5"
  local log_file="$6"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(timestamp)" "${experiment}" "${status}" "${config}" "${exit_code}" "${elapsed_sec}" "${log_file}" \
    >> "${STATUS_FILE}"
}

run_logged() {
  local experiment="$1"
  local config="$2"
  shift 2

  local log_file="${LOG_ROOT}/${experiment}.log"
  local start_epoch
  local end_epoch
  local elapsed_sec
  local exit_code

  start_epoch="$(date +%s)"
  append_status "${experiment}" "STARTED" "${config}" "-" "-" "${log_file}"
  echo "[$(timestamp)] START ${experiment}"
  echo "[$(timestamp)] Log: ${log_file}"

  set +e
  "$@" 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  end_epoch="$(date +%s)"
  elapsed_sec="$((end_epoch - start_epoch))"

  if [[ "${exit_code}" -eq 0 ]]; then
    append_status "${experiment}" "PASSED" "${config}" "${exit_code}" "${elapsed_sec}" "${log_file}"
    echo "[$(timestamp)] DONE  ${experiment} (${elapsed_sec}s)"
  else
    append_status "${experiment}" "FAILED" "${config}" "${exit_code}" "${elapsed_sec}" "${log_file}"
    echo "[$(timestamp)] FAIL  ${experiment} (${elapsed_sec}s, exit=${exit_code})"
    echo "[$(timestamp)] See log: ${log_file}"
    exit "${exit_code}"
  fi
}

write_status_header

echo "[$(timestamp)] Project: ${PROJECT_DIR}"
echo "[$(timestamp)] Logs: ${LOG_ROOT}"
echo "[$(timestamp)] Status: ${STATUS_FILE}"
echo "[$(timestamp)] Python: ${PYTHON_BIN}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  run_logged sanity_vkd_gated_model configs/velocity_xy_vkd_gated.yaml \
    "${PYTHON_BIN}" -u scripts/path_sanity_check.py \
      --config configs/velocity_xy_vkd_gated.yaml \
      --skip-dataset \
      --skip-train-step

  run_logged sanity_vkd_gated_dataset configs/velocity_xy_vkd_gated.yaml \
    "${PYTHON_BIN}" -u scripts/path_sanity_check.py \
      --config configs/velocity_xy_vkd_gated.yaml \
      --skip-train-step
fi

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_logged smoke_vkd_gated configs/smoke_velocity_xy_vkd_gated.yaml \
    "${PYTHON_BIN}" -u train.py \
      --config configs/smoke_velocity_xy_vkd_gated.yaml
fi

run_logged velocity_xy_vkd_gated configs/velocity_xy_vkd_gated.yaml \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_vkd_gated.yaml

run_logged velocity_xy_kd_only configs/velocity_xy_kd_only.yaml \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_kd_only.yaml

run_logged velocity_xy_vision_only configs/velocity_xy_vision_only.yaml \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_vision_only.yaml

run_logged velocity_xy_vision_kinematics configs/velocity_xy_vision_kinematics.yaml \
  "${PYTHON_BIN}" -u train.py \
    --config configs/velocity_xy_vision_kinematics.yaml

echo "[$(timestamp)] All requested server experiments finished."
