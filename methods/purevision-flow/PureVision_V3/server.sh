#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/server.env"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "${command}" in
  check)
    "${PYTHON_BIN}" scripts/check_package.py --cuda
    "${PYTHON_BIN}" -m pytest tests/test_v3_visual_only.py tests/test_v3_frozen_ablation.py -q
    ;;
  prepare)
    "${PYTHON_BIN}" scripts/precompute_frames.py \
      --dataroot "${NUSCENES_ROOT}" --version v1.0-trainval --split trainval \
      --height 72 --width 128 --image_mode rgb --output_root "${V3_FRAME_ROOT}"
    "${PYTHON_BIN}" scripts/precompute_v2_optical_flow.py \
      --dataroot "${NUSCENES_ROOT}" --output_root "${V3_FLOW_ROOT}" \
      --split_manifest logs/v2_split_manifest.json \
      --height 288 --width 512 --model raft_large
    ;;
  test|test-smoke)
    run_tag="${V3_RUN_TAG:-v3_${command}_$(date +%Y%m%d_%H%M%S)}"
    args=(
      --checkpoint "${V3_CHECKPOINT}"
      --dataroot "${NUSCENES_ROOT}" --flow_root "${V3_FLOW_ROOT}"
      --frame_cache_root "${V3_FRAME_ROOT}"
      --split_manifest logs/v2_split_manifest.json
      --output_dir "outputs/${run_tag}"
      --batch_size "${V3_TEST_BATCH_SIZE}" --num_workers "${V3_TEST_NUM_WORKERS}"
    )
    if [[ "${command}" == "test-smoke" ]]; then args+=(--max_batches 2); fi
    "${PYTHON_BIN}" scripts/run_v3_ablation.py "${args[@]}" "$@"
    ;;
  train)
    bash scripts/run_v3_visual_only_pipeline.sh "$@"
    ;;
  *)
    echo "Usage: bash server.sh {check|prepare|test-smoke|test|train}"
    echo "Set paths in server.env first. Test uses the bundled frozen fusion checkpoint."
    if [[ "${command}" != "help" ]]; then exit 2; fi
    ;;
esac
