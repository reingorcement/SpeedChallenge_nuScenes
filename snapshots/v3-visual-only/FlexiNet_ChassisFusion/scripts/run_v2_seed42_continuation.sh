#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${V2_RUN_TAG:-accelerated_20260719_143555}"
DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
FRAME_ROOT="${V2_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
GEOMETRY_ROOT="${V2_GEOMETRY_ROOT:-${DATAROOT}/precomputed_geometry_v2/CAM_FRONT}"
RUN_ROOT="outputs/${RUN_TAG}"
CHECKPOINT_ROOT="checkpoints/${RUN_TAG}"
RUNTIME_PATH="logs/v2_resolved_runtime_${RUN_TAG}.tsv"
SPLIT_MANIFEST="logs/v2_split_manifest.json"
TRAINING_STATS="logs/v2_training_stats.json"
VISUAL_CHECKPOINT="${CHECKPOINT_ROOT}/visual_seed_42/best.pth"
KINEMATIC_CHECKPOINT="${CHECKPOINT_ROOT}/kinematic_seed_42/best.pth"
VISUAL_METRICS="${RUN_ROOT}/visual_seed_42/best_eval/metrics.json"
KINEMATIC_METADATA="${RUN_ROOT}/kinematic_seed_42/run_metadata.json"
JOINT_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/joint_seed_42"
JOINT_OUTPUT_DIR="${RUN_ROOT}/joint_seed_42"
JOINT_BEST="${JOINT_CHECKPOINT_DIR}/best.pth"
JOINT_LAST="${JOINT_CHECKPOINT_DIR}/last.pth"
JOINT_COMPLETE="${JOINT_OUTPUT_DIR}/completed.json"
PROVENANCE="${RUN_ROOT}/seed42_continuation/provenance.json"
LOG_PATH="logs/v2_seed42_continuation_${RUN_TAG}.log"

mkdir -p logs "${RUN_ROOT}/seed42_continuation" "${JOINT_CHECKPOINT_DIR}" "${JOINT_OUTPUT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ ! -f "${RUNTIME_PATH}" ]]; then
  echo "Runtime profile not found: ${RUNTIME_PATH}" >&2
  exit 1
fi

runtime_value() {
  local key="$1"
  awk -F '\t' -v key="${key}" '$1 == key { print $2 }' "${RUNTIME_PATH}"
}

MICRO_BATCH_SIZE="$(runtime_value micro_batch_size)"
EFFECTIVE_BATCH_SIZE="$(runtime_value effective_batch_size)"
NUM_WORKERS="$(runtime_value num_workers)"
PREFETCH_FACTOR="$(runtime_value prefetch_factor)"
for value in MICRO_BATCH_SIZE EFFECTIVE_BATCH_SIZE NUM_WORKERS PREFETCH_FACTOR; do
  if [[ ! "${!value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid ${value}=${!value} in ${RUNTIME_PATH}" >&2
    exit 1
  fi
done

validation_args=(
  --config configs/velocity_xy_v2_visual_kinematic.yaml
  --visual_checkpoint "${VISUAL_CHECKPOINT}"
  --kinematic_checkpoint "${KINEMATIC_CHECKPOINT}"
  --visual_metrics "${VISUAL_METRICS}"
  --kinematic_run_metadata "${KINEMATIC_METADATA}"
  --split_manifest "${SPLIT_MANIFEST}"
  --training_stats "${TRAINING_STATS}"
  --frame_cache_root "${FRAME_ROOT}"
  --flow_root "${FLOW_ROOT}"
  --geometry_cache_root "${GEOMETRY_ROOT}"
  --dataroot "${DATAROOT}"
  --can_bus_root "${CAN_ROOT}"
  --output "${PROVENANCE}"
)

if [[ -f "${JOINT_COMPLETE}" ]]; then
  python scripts/validate_v2_seed42_artifacts.py "${validation_args[@]}" \
    --completion_marker "${JOINT_COMPLETE}" --joint_checkpoint "${JOINT_BEST}"
  echo "Validated completed joint seed 42; skipping training."
else
  if [[ -f "${JOINT_BEST}" && ! -f "${JOINT_LAST}" ]]; then
    echo "Joint best checkpoint exists without last.pth or a completion marker; refusing overwrite." >&2
    exit 1
  fi
  if [[ -f "${JOINT_LAST}" ]]; then
    python scripts/validate_v2_seed42_artifacts.py "${validation_args[@]}" \
      --resume_checkpoint "${JOINT_LAST}"
  else
    python scripts/validate_v2_seed42_artifacts.py "${validation_args[@]}"
  fi

  SMOKE_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/joint_seed_42_smoke"
  SMOKE_OUTPUT_DIR="${RUN_ROOT}/joint_seed_42_smoke"
  if [[ ! -f "${JOINT_LAST}" ]]; then
    python train_v2.py \
      --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml \
      --smoke --init_checkpoint "${KINEMATIC_CHECKPOINT}" \
      --visual_baseline_metrics "${VISUAL_METRICS}" \
      --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
      --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
      --geometry_cache_root "${GEOMETRY_ROOT}" --geometry_cache_mode required \
      --split_manifest "${SPLIT_MANIFEST}" \
      --batch_size "${MICRO_BATCH_SIZE}" \
      --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" \
      --save_dir "${SMOKE_CHECKPOINT_DIR}" --output_dir "${SMOKE_OUTPUT_DIR}"
  fi

  train_args=(
    --config configs/velocity_xy_v2_visual_kinematic.yaml
    --init_checkpoint "${KINEMATIC_CHECKPOINT}"
    --visual_baseline_metrics "${VISUAL_METRICS}"
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}"
    --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}"
    --geometry_cache_root "${GEOMETRY_ROOT}" --geometry_cache_mode required
    --split_manifest "${SPLIT_MANIFEST}"
    --batch_size "${MICRO_BATCH_SIZE}"
    --effective_batch_size "${EFFECTIVE_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}"
    --save_dir "${JOINT_CHECKPOINT_DIR}" --output_dir "${JOINT_OUTPUT_DIR}"
  )
  if [[ -f "${JOINT_LAST}" ]]; then
    python train_v2.py "${train_args[@]}" --resume "${JOINT_LAST}"
  else
    python train_v2.py "${train_args[@]}"
  fi
  python scripts/validate_v2_seed42_artifacts.py "${validation_args[@]}" \
    --completion_marker "${JOINT_COMPLETE}" --joint_checkpoint "${JOINT_BEST}"
fi

GEOMETRY_TEST_DIR="${RUN_ROOT}/stage1_test/geometry"
VISUAL_TEST_DIR="${RUN_ROOT}/stage1_test/visual_seed_42"
JOINT_TEST_DIR="${RUN_ROOT}/stage1_test/joint_seed_42"

python scripts/evaluate_v2_holdout.py \
  --config configs/velocity_xy_v2_geometry.yaml --mode geometry_only --seed 42 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" --variants baseline \
  --output_dir "${GEOMETRY_TEST_DIR}"

python scripts/evaluate_v2_holdout.py \
  --config configs/velocity_xy_v2_visual.yaml --checkpoint "${VISUAL_CHECKPOINT}" \
  --mode visual --seed 42 --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output_dir "${VISUAL_TEST_DIR}"

python scripts/evaluate_v2_holdout.py \
  --config configs/velocity_xy_v2_visual_kinematic.yaml --checkpoint "${JOINT_BEST}" \
  --mode visual_kinematic --seed 42 --dataroot "${DATAROOT}" \
  --can_bus_root "${CAN_ROOT}" --flow_root "${FLOW_ROOT}" \
  --frame_cache_root "${FRAME_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
  --output_dir "${JOINT_TEST_DIR}"

python scripts/analyze_v2_stage1.py \
  --preliminary --seeds 42 \
  --geometry_predictions "${GEOMETRY_TEST_DIR}/predictions.csv" \
  --visual_predictions "${VISUAL_TEST_DIR}/predictions.csv" \
  --joint_predictions "${JOINT_TEST_DIR}/predictions.csv" \
  --training_stats "${TRAINING_STATS}" \
  --output_dir "${RUN_ROOT}/stage1_test_report_seed42_preliminary"

echo "Seed-42 continuation complete."
echo "Report: ${RUN_ROOT}/stage1_test_report_seed42_preliminary/stage1_report.md"
