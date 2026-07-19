#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${NUSCENES_COMPARISON_PROJECT_ROOT:-${HOST_PROJECT_ROOT}}"
export NUSCENES_COMPARISON_PROJECT_ROOT="${PROJECT_ROOT}"
PROFILE="${EXPERIMENT_PROFILE:-legacy}"
python "${SCRIPT_DIR}/compatibility_check.py" --project-root "${PROJECT_ROOT}" --profile "${PROFILE}"
cd "${PROJECT_ROOT}"

DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
FRAME_ROOT="${V2_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
GEOMETRY_ROOT="${V2_GEOMETRY_ROOT:-${DATAROOT}/precomputed_geometry_v2/CAM_FRONT}"
SPLIT_MANIFEST="${V2_SPLIT_MANIFEST:-logs/v2_split_manifest.json}"
KINEMATIC_STATS="${V2_KINEMATIC_STATS:-logs/v2_training_stats.json}"
LEGACY_SPLIT_MANIFEST="${COMPARISON_SPLIT_MANIFEST:-logs/nuscenes_comparison_split_manifest.json}"
RUN_ID="${COMPARISON_RUN_ID:-comparison_$(date +%Y%m%d_%H%M%S)}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_DOMAIN_SHIFT="${RUN_DOMAIN_SHIFT:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEGACY_BATCH_SIZE="${LEGACY_BATCH_SIZE:-32}"
V2_BATCH_SIZE="${V2_BATCH_SIZE:-8}"
SEEDS=(42 3407 2026)
EXPERIMENT="${SCRIPT_DIR}"
OUTPUT_ROOT="outputs/nuscenes_version_comparison/${RUN_ID}"
CHECKPOINT_ROOT="checkpoints/nuscenes_version_comparison/${RUN_ID}"

if [[ "${PROFILE}" == "legacy" ]]; then
  mkdir -p "${OUTPUT_ROOT}" "${CHECKPOINT_ROOT}" logs
  printf '%s\n' "${RUN_ID}" > logs/nuscenes_version_comparison_latest_run.txt
  if [[ ! -f "${LEGACY_SPLIT_MANIFEST}" ]]; then
    python "${EXPERIMENT}/create_split_manifest.py" \
      --dataroot "${DATAROOT}" --can-bus-root "${CAN_ROOT}" \
      --output "${LEGACY_SPLIT_MANIFEST}" --seed 42
  fi
  python "${EXPERIMENT}/server_preflight.py" \
    --project-root "${PROJECT_ROOT}" --profile legacy \
    --dataroot "${DATAROOT}" --can-bus-root "${CAN_ROOT}" \
    --split-manifest "${LEGACY_SPLIT_MANIFEST}" --require-cuda

  legacy_common_args=(
    --run-id "${RUN_ID}"
    --dataroot "${DATAROOT}"
    --can-bus-root "${CAN_ROOT}"
    --split-manifest "${LEGACY_SPLIT_MANIFEST}"
    --num-workers "${NUM_WORKERS}"
  )
  legacy_frame_root="${LEGACY_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
  if [[ -f "${legacy_frame_root}/manifest.json" ]]; then
    legacy_common_args+=(--frame-cache-root "${legacy_frame_root}")
  else
    echo "[info] Shared legacy frame cache not found; using raw nuScenes images."
  fi

  train_legacy() {
    local algorithm="$1"
    local seed="$2"
    local checkpoint="${CHECKPOINT_ROOT}/${algorithm}/seed_${seed}/best.pth"
    if [[ -f "${checkpoint}" ]]; then
      echo "[skip] ${algorithm} seed=${seed}"
      return
    fi
    python "${EXPERIMENT}/train_comparison.py" \
      --algorithm "${algorithm}" --seed "${seed}" --domain all \
      --batch-size "${LEGACY_BATCH_SIZE}" "${legacy_common_args[@]}"
  }

  for seed in "${SEEDS[@]}"; do
    train_legacy legacy_kd "${seed}"
    train_legacy legacy_vkd "${seed}"
  done

  evaluate_legacy() {
    local algorithm="$1"
    local seed="$2"
    local partition="$3"
    shift 3
    local output="${OUTPUT_ROOT}/evaluation/${partition}/all/${algorithm}/seed_${seed}/predictions.csv"
    if [[ -f "${output}" ]]; then
      echo "[skip] evaluation ${algorithm} seed=${seed} partition=${partition}"
      return
    fi
    local checkpoint_args=()
    if [[ "${algorithm}" != "base_velocity" ]]; then
      checkpoint_args=(--checkpoint "${CHECKPOINT_ROOT}/${algorithm}/seed_${seed}/best.pth")
    fi
    python "${EXPERIMENT}/evaluate_comparison.py" \
      --algorithm "${algorithm}" --seed "${seed}" --partition "${partition}" --domain all \
      "${checkpoint_args[@]}" "${legacy_common_args[@]}" "$@"
  }

  evaluate_legacy base_velocity 42 train_core --variants clean
  THRESHOLDS="${OUTPUT_ROOT}/protocol_thresholds.json"
  python "${EXPERIMENT}/analyze_comparison.py" fit-thresholds \
    --predictions "${OUTPUT_ROOT}/evaluation/train_core/all/base_velocity/seed_42/predictions.csv" \
    --output "${THRESHOLDS}"
  evaluate_legacy base_velocity 42 stage1_test
  for seed in "${SEEDS[@]}"; do
    evaluate_legacy legacy_kd "${seed}" stage1_test
    evaluate_legacy legacy_vkd "${seed}" stage1_test
  done
  mapfile -t legacy_predictions < <(find "${OUTPUT_ROOT}/evaluation/stage1_test/all" -name predictions.csv -print | sort)
  python "${EXPERIMENT}/analyze_comparison.py" report \
    --predictions "${legacy_predictions[@]}" \
    --manifest "${LEGACY_SPLIT_MANIFEST}" --thresholds "${THRESHOLDS}" \
    --required-algorithms legacy_kd legacy_vkd \
    --output-dir "${OUTPUT_ROOT}/stage1_report"
  echo "Legacy comparison complete: ${OUTPUT_ROOT}/stage1_report/comparison_report.md"
  exit 0
fi

common_args=(
  --run-id "${RUN_ID}"
  --dataroot "${DATAROOT}"
  --can-bus-root "${CAN_ROOT}"
  --flow-root "${FLOW_ROOT}"
  --frame-cache-root "${FRAME_ROOT}"
  --geometry-cache-root "${GEOMETRY_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --kinematic-stats "${KINEMATIC_STATS}"
  --num-workers "${NUM_WORKERS}"
)

mkdir -p "${OUTPUT_ROOT}" "${CHECKPOINT_ROOT}" logs
printf '%s\n' "${RUN_ID}" > logs/nuscenes_version_comparison_latest_run.txt

if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python scripts/create_v2_split_manifest.py \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --output "${SPLIT_MANIFEST}" --seed 42
fi
python -c "from dataset.v2_splits import load_split_manifest; load_split_manifest('${SPLIT_MANIFEST}')"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  python scripts/audit_v2_data.py \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --partition train_core \
    --output logs/v2_data_audit.json
  python scripts/precompute_v2_optical_flow.py \
    --dataroot "${DATAROOT}" --output_root "${FLOW_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --partitions train_core dev stage1_test --width 512 --model raft_large
  python scripts/precompute_frames.py \
    --dataroot "${DATAROOT}" --version v1.0-trainval --split trainval \
    --cam_channel CAM_FRONT --output_root "${FRAME_ROOT}" \
    --height 72 --width 128 --image_mode rgb
  python scripts/prepare_v2_training_stats.py \
    --config configs/velocity_xy_v2_visual.yaml \
    --audit logs/v2_data_audit.json --output "${KINEMATIC_STATS}" \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}"
  python scripts/precompute_v2_geometry.py \
    --config configs/velocity_xy_v2_visual.yaml \
    --output_root "${GEOMETRY_ROOT}" --partitions train_core dev \
    --batch_size "${V2_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}"
fi

train_one() {
  local algorithm="$1"
  local seed="$2"
  local domain="${3:-all}"
  shift 3 || true
  local checkpoint="${CHECKPOINT_ROOT}/${algorithm}/seed_${seed}/best.pth"
  if [[ -f "${checkpoint}" ]]; then
    echo "[skip] ${algorithm} seed=${seed} domain=${domain}"
    return
  fi
  local batch_size="${LEGACY_BATCH_SIZE}"
  if [[ "${algorithm}" == v2_* ]]; then
    batch_size="${V2_BATCH_SIZE}"
  fi
  python "${EXPERIMENT}/train_comparison.py" \
    --algorithm "${algorithm}" --seed "${seed}" --domain "${domain}" \
    --batch-size "${batch_size}" "${common_args[@]}" "$@"
}

for seed in "${SEEDS[@]}"; do
  train_one legacy_kd "${seed}" all
  train_one legacy_vkd "${seed}" all
  train_one v2_visual "${seed}" all
  visual_checkpoint="${CHECKPOINT_ROOT}/v2_visual/seed_${seed}/best.pth"
  visual_metrics="${OUTPUT_ROOT}/train/v2_visual/seed_${seed}/best_eval/metrics.json"
  train_one v2_kinematic_pretrain "${seed}" all \
    --init-checkpoint "${visual_checkpoint}" --visual-baseline-metrics "${visual_metrics}"
  kinematic_checkpoint="${CHECKPOINT_ROOT}/v2_kinematic_pretrain/seed_${seed}/best.pth"
  train_one v2_visual_kinematic "${seed}" all \
    --init-checkpoint "${kinematic_checkpoint}" --visual-baseline-metrics "${visual_metrics}"
done

evaluate_one() {
  local algorithm="$1"
  local seed="$2"
  local partition="$3"
  local domain="$4"
  shift 4
  local output="${OUTPUT_ROOT}/evaluation/${partition}/${domain}/${algorithm}/seed_${seed}/predictions.csv"
  if [[ -f "${output}" ]]; then
    echo "[skip] evaluation ${algorithm} seed=${seed} partition=${partition} domain=${domain}"
    return
  fi
  local checkpoint_args=()
  if [[ "${algorithm}" != "base_velocity" && "${algorithm}" != "v2_geometry" ]]; then
    checkpoint_args=(--checkpoint "${CHECKPOINT_ROOT}/${algorithm}/seed_${seed}/best.pth")
  fi
  python "${EXPERIMENT}/evaluate_comparison.py" \
    --algorithm "${algorithm}" --seed "${seed}" \
    --partition "${partition}" --domain "${domain}" \
    "${checkpoint_args[@]}" "${common_args[@]}" "$@"
}

evaluate_one base_velocity 42 train_core all --variants clean
THRESHOLDS="${OUTPUT_ROOT}/protocol_thresholds.json"
python "${EXPERIMENT}/analyze_comparison.py" fit-thresholds \
  --predictions "${OUTPUT_ROOT}/evaluation/train_core/all/base_velocity/seed_42/predictions.csv" \
  --output "${THRESHOLDS}"

evaluate_one base_velocity 42 stage1_test all
evaluate_one v2_geometry 42 stage1_test all --include-calibration
for seed in "${SEEDS[@]}"; do
  evaluate_one legacy_kd "${seed}" stage1_test all
  evaluate_one legacy_vkd "${seed}" stage1_test all
  evaluate_one v2_visual "${seed}" stage1_test all
  evaluate_one v2_visual_kinematic "${seed}" stage1_test all --include-calibration
done

mapfile -t stage1_predictions < <(find "${OUTPUT_ROOT}/evaluation/stage1_test/all" -name predictions.csv -print | sort)
python "${EXPERIMENT}/analyze_comparison.py" report \
  --predictions "${stage1_predictions[@]}" \
  --manifest "${SPLIT_MANIFEST}" --thresholds "${THRESHOLDS}" \
  --output-dir "${OUTPUT_ROOT}/stage1_report"

if [[ "${RUN_DOMAIN_SHIFT}" == "1" ]]; then
  DOMAIN_RUN_ID="${RUN_ID}_boston_to_singapore"
  domain_checkpoint_root="checkpoints/nuscenes_version_comparison/${DOMAIN_RUN_ID}"
  domain_output_root="outputs/nuscenes_version_comparison/${DOMAIN_RUN_ID}"
  domain_manifest="${domain_output_root}/boston_training_manifest.json"
  domain_stats="${domain_output_root}/boston_training_stats.json"
  mkdir -p "${domain_output_root}"
  if [[ ! -f "${domain_manifest}" ]]; then
    python -c "from experiments.nuscenes_version_comparison.adapters.dataset_adapter import write_training_domain_manifest; write_training_domain_manifest('${SPLIT_MANIFEST}', '${domain_manifest}', version='v1.0-trainval', domain='boston')"
  fi
  if [[ ! -f "${domain_stats}" ]]; then
    python scripts/prepare_v2_training_stats.py \
      --config configs/velocity_xy_v2_visual.yaml \
      --audit logs/v2_data_audit.json --output "${domain_stats}" \
      --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
      --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
      --split_manifest "${domain_manifest}"
  fi
  domain_common_args=(
    --run-id "${DOMAIN_RUN_ID}"
    --dataroot "${DATAROOT}"
    --can-bus-root "${CAN_ROOT}"
    --flow-root "${FLOW_ROOT}"
    --frame-cache-root "${FRAME_ROOT}"
    --geometry-cache-root "${GEOMETRY_ROOT}"
    --split-manifest "${SPLIT_MANIFEST}"
    --kinematic-stats "${domain_stats}"
    --num-workers "${NUM_WORKERS}"
  )
  domain_train() {
    local algorithm="$1"
    local seed="$2"
    shift 2
    if [[ -f "${domain_checkpoint_root}/${algorithm}/seed_${seed}/best.pth" ]]; then
      echo "[skip] domain training ${algorithm} seed=${seed}"
      return
    fi
    local batch_size="${LEGACY_BATCH_SIZE}"
    if [[ "${algorithm}" == v2_* ]]; then
      batch_size="${V2_BATCH_SIZE}"
    fi
    python "${EXPERIMENT}/train_comparison.py" \
      --algorithm "${algorithm}" --seed "${seed}" --domain boston \
      --batch-size "${batch_size}" "${domain_common_args[@]}" "$@"
  }
  domain_evaluate() {
    local algorithm="$1"
    local seed="$2"
    local output="${domain_output_root}/evaluation/stage1_test/singapore/${algorithm}/seed_${seed}/predictions.csv"
    if [[ -f "${output}" ]]; then
      echo "[skip] domain evaluation ${algorithm} seed=${seed}"
      return
    fi
    python "${EXPERIMENT}/evaluate_comparison.py" \
      --algorithm "${algorithm}" --seed "${seed}" --partition stage1_test --domain singapore \
      --variants clean "${domain_common_args[@]}" \
      --checkpoint "${domain_checkpoint_root}/${algorithm}/seed_${seed}/best.pth"
  }
  for seed in "${SEEDS[@]}"; do
    domain_train legacy_kd "${seed}"
    domain_train legacy_vkd "${seed}"
    domain_train v2_visual "${seed}"
    domain_train v2_kinematic_pretrain "${seed}" \
      --init-checkpoint "${domain_checkpoint_root}/v2_visual/seed_${seed}/best.pth" \
      --visual-baseline-metrics "${domain_output_root}/train/v2_visual/seed_${seed}/best_eval/metrics.json"
    domain_train v2_visual_kinematic "${seed}" \
      --init-checkpoint "${domain_checkpoint_root}/v2_kinematic_pretrain/seed_${seed}/best.pth" \
      --visual-baseline-metrics "${domain_output_root}/train/v2_visual/seed_${seed}/best_eval/metrics.json"
    for algorithm in legacy_kd legacy_vkd v2_visual_kinematic; do
      domain_evaluate "${algorithm}" "${seed}"
    done
  done
  mapfile -t domain_predictions < <(find "${domain_output_root}/evaluation/stage1_test/singapore" -name predictions.csv -print | sort)
  python "${EXPERIMENT}/analyze_comparison.py" report \
    --predictions "${domain_predictions[@]}" \
    --manifest "${SPLIT_MANIFEST}" --thresholds "${THRESHOLDS}" \
    --required-algorithms legacy_kd legacy_vkd v2_visual_kinematic \
    --output-dir "${domain_output_root}/domain_shift_report"
fi

echo "Comparison complete: ${OUTPUT_ROOT}/stage1_report/comparison_report.md"
