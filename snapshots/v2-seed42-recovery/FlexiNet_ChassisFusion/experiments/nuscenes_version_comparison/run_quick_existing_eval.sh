#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${NUSCENES_COMPARISON_PROJECT_ROOT:-${HOST_PROJECT_ROOT}}"
export NUSCENES_COMPARISON_PROJECT_ROOT="${PROJECT_ROOT}"

python "${SCRIPT_DIR}/compatibility_check.py" --project-root "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
RUN_ID="${COMPARISON_RUN_ID:-quick_existing_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${QUICK_MAX_SAMPLES:-512}"
BATCH_SIZE="${QUICK_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VARIANTS="${QUICK_VARIANTS:-clean can_missing yaw_bias_pos wheel_scale_low camera_missing}"
LEGACY_FRAME_ROOT="${LEGACY_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_64x64_gray}"
V2_FRAME_ROOT="${V2_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
V2_FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
V2_GEOMETRY_ROOT="${V2_GEOMETRY_ROOT:-${DATAROOT}/precomputed_geometry_v2/CAM_FRONT}"
V2_SPLIT_MANIFEST="${V2_SPLIT_MANIFEST:-logs/v2_split_manifest.json}"
V2_KINEMATIC_STATS="${V2_KINEMATIC_STATS:-logs/v2_training_stats.json}"
LEGACY_KD_CHECKPOINT="${LEGACY_KD_CHECKPOINT:-checkpoints/velocity_xy_kd_only/best.pth}"
LEGACY_VKD_CHECKPOINT="${LEGACY_VKD_CHECKPOINT:-checkpoints/velocity_xy_vkd_gated/best.pth}"
V2_VISUAL_CHECKPOINT="${V2_VISUAL_CHECKPOINT:-checkpoints/velocity_xy_v2_visual/best.pth}"
V2_VISUAL_KINEMATIC_CHECKPOINT="${V2_VISUAL_KINEMATIC_CHECKPOINT:-checkpoints/velocity_xy_v2_visual_kinematic/best.pth}"

python "${SCRIPT_DIR}/quick_evaluate_existing.py" \
  --run-id "${RUN_ID}" \
  --dataroot "${DATAROOT}" \
  --can-bus-root "${CAN_ROOT}" \
  --split-manifest "${V2_SPLIT_MANIFEST}" \
  --kinematic-stats "${V2_KINEMATIC_STATS}" \
  --legacy-frame-cache-root "${LEGACY_FRAME_ROOT}" \
  --v2-frame-cache-root "${V2_FRAME_ROOT}" \
  --flow-root "${V2_FLOW_ROOT}" \
  --geometry-cache-root "${V2_GEOMETRY_ROOT}" \
  --legacy-kd-checkpoint "${LEGACY_KD_CHECKPOINT}" \
  --legacy-vkd-checkpoint "${LEGACY_VKD_CHECKPOINT}" \
  --v2-visual-checkpoint "${V2_VISUAL_CHECKPOINT}" \
  --v2-visual-kinematic-checkpoint "${V2_VISUAL_KINEMATIC_CHECKPOINT}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --variants ${VARIANTS} \
  --disable-v2-geometry-cache

echo "Quick existing-checkpoint report: outputs/nuscenes_version_comparison/${RUN_ID}/quick_existing_eval/quick_report.md"
