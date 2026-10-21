#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATAROOT="${NUSCENES_ROOT:-/data/nuscenes}"
FLOW_ROOT="${V3_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
FRAME_ROOT="${V3_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
CONFIG="${V3_CONFIG:-configs/velocity_xy_v3_visual_only.yaml}"
RUN_TAG="${V3_RUN_TAG:-v3_visual_only_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="outputs/${RUN_TAG}"
CHECKPOINT_ROOT="checkpoints/${RUN_TAG}"
ANALYSIS_ROOT="${RUN_ROOT}/stage1_test_analysis"
DIRECT_BATCH_SIZE="${V3_DIRECT_BATCH_SIZE:-16}"
FUSION_BATCH_SIZE="${V3_FUSION_BATCH_SIZE:-4}"
NUM_WORKERS="${V3_NUM_WORKERS:-8}"
PREFETCH_FACTOR="${V3_PREFETCH_FACTOR:-2}"
read -r -a SEEDS <<< "${V3_SEEDS:-42 3407 2026}"

mkdir -p logs "${RUN_ROOT}" "${CHECKPOINT_ROOT}"
exec > >(tee -a "logs/${RUN_TAG}.log") 2>&1
trap 'status=$?; echo "V3 pipeline failed at line ${LINENO} with status ${status}."; exit "${status}"' ERR

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Required path does not exist: $1" >&2
    return 1
  fi
}

checkpoint_complete() {
  local checkpoint="$1"
  "${PYTHON_BIN}" scripts/v3_pipeline_checks.py checkpoint \
    --checkpoint "${checkpoint}" --config "${CONFIG}"
}

train_stage() {
  local stage="$1"
  local seed="$2"
  local save_dir="$3"
  local output_dir="$4"
  local initial_checkpoint="${5:-}"
  local last_checkpoint="${save_dir}/last.pth"
  local best_checkpoint="${save_dir}/best.pth"
  local batch_size="${FUSION_BATCH_SIZE}"
  if [[ "${stage}" == "direct_pretrain" ]]; then
    batch_size="${DIRECT_BATCH_SIZE}"
  fi
  local args=(
    --config "${CONFIG}"
    --stage "${stage}"
    --seed "${seed}"
    --save_dir "${save_dir}"
    --output_dir "${output_dir}"
    --dataroot "${DATAROOT}"
    --flow_root "${FLOW_ROOT}"
    --frame_cache_root "${FRAME_ROOT}"
    --batch_size "${batch_size}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
  )
  if [[ -n "${initial_checkpoint}" ]]; then
    args+=(--init_checkpoint "${initial_checkpoint}")
  fi
  if [[ -f "${last_checkpoint}" ]] && checkpoint_complete "${last_checkpoint}"; then
    require_path "${best_checkpoint}"
    echo "Skipping completed ${stage} seed ${seed}: ${best_checkpoint}"
    return
  fi
  if [[ -f "${last_checkpoint}" ]]; then
    args+=(--resume "${last_checkpoint}")
    echo "Resuming ${stage} seed ${seed} from ${last_checkpoint}"
  fi
  "${PYTHON_BIN}" train_v3.py "${args[@]}"
  require_path "${best_checkpoint}"
}

echo "Run tag: ${RUN_TAG}"
echo "Dataset: ${DATAROOT}"
echo "Flow cache: ${FLOW_ROOT}"
echo "Frame cache: ${FRAME_ROOT}"
echo "Runtime: direct_batch=${DIRECT_BATCH_SIZE}, fusion_batch=${FUSION_BATCH_SIZE}, workers=${NUM_WORKERS}, prefetch=${PREFETCH_FACTOR}"
require_path "${DATAROOT}"
require_path "${FLOW_ROOT}"
require_path "${FRAME_ROOT}"
require_path "${CONFIG}"
require_path "logs/v2_split_manifest.json"

"${PYTHON_BIN}" -c 'import matplotlib, torch; assert torch.cuda.is_available(), "CUDA is required"; print(f"torch={torch.__version__}, cuda={torch.cuda.get_device_name(0)}, matplotlib={matplotlib.__version__}")'
"${PYTHON_BIN}" -m pytest tests/test_v3_visual_only.py -q

for seed in "${SEEDS[@]}"; do
  direct_checkpoint_dir="${CHECKPOINT_ROOT}/direct_seed_${seed}"
  direct_output_dir="${RUN_ROOT}/dev/direct_seed_${seed}"
  fusion_checkpoint_dir="${CHECKPOINT_ROOT}/fusion_seed_${seed}"
  fusion_output_dir="${RUN_ROOT}/dev/fusion_seed_${seed}"
  train_stage direct_pretrain "${seed}" "${direct_checkpoint_dir}" "${direct_output_dir}"
  train_stage fusion "${seed}" "${fusion_checkpoint_dir}" "${fusion_output_dir}" \
    "${direct_checkpoint_dir}/best.pth"
done

"${PYTHON_BIN}" scripts/evaluate_v3_holdout.py \
  --config "${CONFIG}" --stage geometry_audit --seed 42 \
  --output_dir "${RUN_ROOT}/stage1_test/geometry" \
  --dataroot "${DATAROOT}" --flow_root "${FLOW_ROOT}" \
  --frame_cache_root "${FRAME_ROOT}"

DIRECT_PREDICTIONS=()
FUSION_PREDICTIONS=()
for seed in "${SEEDS[@]}"; do
  direct_dir="${RUN_ROOT}/stage1_test/direct_seed_${seed}"
  fusion_dir="${RUN_ROOT}/stage1_test/fusion_seed_${seed}"
  "${PYTHON_BIN}" scripts/evaluate_v3_holdout.py \
    --config "${CONFIG}" --stage direct_pretrain --seed "${seed}" \
    --checkpoint "${CHECKPOINT_ROOT}/direct_seed_${seed}/best.pth" \
    --output_dir "${direct_dir}" --dataroot "${DATAROOT}" \
    --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}"
  "${PYTHON_BIN}" scripts/evaluate_v3_holdout.py \
    --config "${CONFIG}" --stage fusion --seed "${seed}" \
    --checkpoint "${CHECKPOINT_ROOT}/fusion_seed_${seed}/best.pth" \
    --output_dir "${fusion_dir}" --dataroot "${DATAROOT}" \
    --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}"
  DIRECT_PREDICTIONS+=("${direct_dir}/predictions.csv")
  FUSION_PREDICTIONS+=("${fusion_dir}/predictions.csv")
done

"${PYTHON_BIN}" scripts/analyze_v3_results.py \
  --seeds "${SEEDS[@]}" \
  --direct "${DIRECT_PREDICTIONS[@]}" \
  --fusion "${FUSION_PREDICTIONS[@]}" \
  --samples 10000 --output_dir "${ANALYSIS_ROOT}"

echo "V3 pipeline completed successfully."
echo "Checkpoints: ${CHECKPOINT_ROOT}"
echo "Holdout predictions: ${RUN_ROOT}/stage1_test"
echo "Analysis report: ${ANALYSIS_ROOT}/analysis-report.md"
