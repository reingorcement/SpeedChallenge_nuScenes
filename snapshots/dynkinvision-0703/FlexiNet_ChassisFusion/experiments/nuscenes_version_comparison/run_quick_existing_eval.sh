#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE="${EXPERIMENT_PROFILE:-legacy}"
DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
SPLIT_MANIFEST="${COMPARISON_SPLIT_MANIFEST:-logs/nuscenes_comparison_split_manifest.json}"
RUN_ID="${COMPARISON_RUN_ID:-quick_existing_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${QUICK_MAX_SAMPLES:-512}"
BATCH_SIZE="${QUICK_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VARIANTS="${QUICK_VARIANTS:-clean can_missing yaw_bias_pos wheel_scale_low camera_missing}"
LEGACY_KD_CHECKPOINT="${LEGACY_KD_CHECKPOINT:-checkpoints/velocity_xy_kd_only/best.pth}"
LEGACY_VKD_CHECKPOINT="${LEGACY_VKD_CHECKPOINT:-checkpoints/velocity_xy_vkd_gated/best.pth}"

python "${SCRIPT_DIR}/compatibility_check.py" --project-root "${PROJECT_ROOT}" --profile "${PROFILE}"
if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python "${SCRIPT_DIR}/create_split_manifest.py" \
    --dataroot "${DATAROOT}" --can-bus-root "${CAN_ROOT}" \
    --output "${SPLIT_MANIFEST}" --seed 42
fi
python "${SCRIPT_DIR}/server_preflight.py" \
  --project-root "${PROJECT_ROOT}" --profile "${PROFILE}" \
  --dataroot "${DATAROOT}" --can-bus-root "${CAN_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" --require-cuda \
  --check-existing-checkpoints \
  --legacy-kd-checkpoint "${LEGACY_KD_CHECKPOINT}" \
  --legacy-vkd-checkpoint "${LEGACY_VKD_CHECKPOINT}"

common_args=(
  --profile "${PROFILE}"
  --run-id "${RUN_ID}"
  --dataroot "${DATAROOT}"
  --can-bus-root "${CAN_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --legacy-kd-checkpoint "${LEGACY_KD_CHECKPOINT}"
  --legacy-vkd-checkpoint "${LEGACY_VKD_CHECKPOINT}"
  --max-samples "${MAX_SAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

legacy_frame_root="${LEGACY_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_64x64_gray}"
if [[ -f "${legacy_frame_root}/manifest.json" ]]; then
  common_args+=(--legacy-frame-cache-root "${legacy_frame_root}")
else
  echo "[info] Legacy frame cache not found; using raw nuScenes images."
fi

if [[ "${PROFILE}" == "full" ]]; then
  v2_frame_root="${V2_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
  v2_flow_root="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
  common_args+=(
    --v2-frame-cache-root "${v2_frame_root}"
    --flow-root "${v2_flow_root}"
    --kinematic-stats "${V2_KINEMATIC_STATS:-logs/v2_training_stats.json}"
    --v2-visual-checkpoint "${V2_VISUAL_CHECKPOINT:-checkpoints/velocity_xy_v2_visual/best.pth}"
    --v2-visual-kinematic-checkpoint "${V2_VISUAL_KINEMATIC_CHECKPOINT:-checkpoints/velocity_xy_v2_visual_kinematic/best.pth}"
    --disable-v2-geometry-cache
  )
fi

# VARIANTS is intentionally word-split so callers can provide a space-separated list.
python "${SCRIPT_DIR}/quick_evaluate_existing.py" \
  "${common_args[@]}" --variants ${VARIANTS}

echo "Quick existing-checkpoint report: outputs/nuscenes_version_comparison/${RUN_ID}/quick_existing_eval/quick_report.md"
