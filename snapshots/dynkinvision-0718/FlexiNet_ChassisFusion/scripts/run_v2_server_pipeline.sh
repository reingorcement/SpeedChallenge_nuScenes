#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
FRAME_ROOT="${V2_FRAME_ROOT:-${DATAROOT}/precomputed_frames/CAM_FRONT_72x128_rgb}"
GEOMETRY_ROOT="${V2_GEOMETRY_ROOT:-${DATAROOT}/precomputed_geometry_v2/CAM_FRONT}"
RUN_TAG="${V2_RUN_TAG:-accelerated_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="outputs/${RUN_TAG}"
CHECKPOINT_ROOT="checkpoints/${RUN_TAG}"
EFFECTIVE_BATCH_SIZE=8
SPLIT_MANIFEST="logs/v2_split_manifest.json"
PROFILE_PATH="logs/v2_throughput_profile_${RUN_TAG}.json"
SEEDS=(42 3407 2026)

common_args() {
  printf '%s\n' \
    --dataroot "${DATAROOT}" \
    --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --frame_cache_root "${FRAME_ROOT}"
}

run_online_stage() {
  local config="$1"
  shift
  mapfile -t args < <(common_args)
  python train_v2.py --config "${config}" "${args[@]}" \
    --geometry_cache_mode off \
    --batch_size "${MICRO_BATCH_SIZE:-8}" \
    --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --prefetch_factor "${PREFETCH_FACTOR:-2}" "$@"
}

run_cached_stage() {
  local config="$1"
  shift
  mapfile -t args < <(common_args)
  python train_v2.py --config "${config}" "${args[@]}" \
    --geometry_cache_root "${GEOMETRY_ROOT}" \
    --geometry_cache_mode required \
    --batch_size "${MICRO_BATCH_SIZE}" \
    --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" "$@"
}

select_micro_batch() {
  local log_path="logs/v2_max_memory_smoke_${RUN_TAG}.log"
  mapfile -t args < <(common_args)
  if python train_v2.py \
    --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml "${args[@]}" \
    --geometry_cache_root "${GEOMETRY_ROOT}" --geometry_cache_mode required \
    --num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" \
    --save_dir "${CHECKPOINT_ROOT}/smoke" --output_dir "${RUN_ROOT}/smoke" \
    --smoke --batch_size 8 --effective_batch_size 8 2>&1 | tee "${log_path}" >&2; then
    echo 8
    return
  fi
  if ! grep -Eqi "CUDA out of memory|OutOfMemoryError" "${log_path}"; then
    echo "Smoke failed for a non-OOM reason; refusing batch fallback." >&2
    return 1
  fi
  echo "CUDA OOM at batch 8; retrying batch 4 with accumulation 2." >&2
  python train_v2.py \
    --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml "${args[@]}" \
    --geometry_cache_root "${GEOMETRY_ROOT}" --geometry_cache_mode required \
    --num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" \
    --save_dir "${CHECKPOINT_ROOT}/smoke" --output_dir "${RUN_ROOT}/smoke" \
    --smoke --batch_size 4 --effective_batch_size 8 2>&1 | tee "${log_path}" >&2
  echo 4
}

mkdir -p logs "${RUN_ROOT}" "${CHECKPOINT_ROOT}" "${FRAME_ROOT}" "${GEOMETRY_ROOT}"
printf '%s\n' "${RUN_TAG}" > logs/v2_latest_run.txt
if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python scripts/create_v2_split_manifest.py \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --output "${SPLIT_MANIFEST}" --seed 42
else
  python -c "from dataset.v2_splits import load_split_manifest; load_split_manifest('${SPLIT_MANIFEST}')"
fi

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
  --audit logs/v2_data_audit.json --dataroot "${DATAROOT}" \
  --can_bus_root "${CAN_ROOT}" --flow_root "${FLOW_ROOT}" \
  --frame_cache_root "${FRAME_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
  --output logs/v2_training_stats.json

MICRO_BATCH_SIZE=8
NUM_WORKERS=8
PREFETCH_FACTOR=2
run_online_stage configs/velocity_xy_v2_geometry.yaml \
  --save_dir "${CHECKPOINT_ROOT}/geometry" --output_dir "${RUN_ROOT}/geometry"

python scripts/precompute_v2_geometry.py \
  --config configs/velocity_xy_v2_visual.yaml --output_root "${GEOMETRY_ROOT}" \
  --partitions train_core dev stage1_test --batch_size 8 --num_workers 8 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}"

python scripts/verify_v2_geometry_cache.py \
  --config configs/velocity_xy_v2_visual.yaml --geometry_cache_root "${GEOMETRY_ROOT}" \
  --partition dev --samples 256 --batch_size 8 --seed 42 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output "logs/v2_geometry_cache_verification_${RUN_TAG}.json"

python scripts/profile_v2_training.py \
  --config configs/velocity_xy_v2_visual.yaml \
  --geometry_cache_root "${GEOMETRY_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --workers 4 8 12 16 --prefetch_factor 4 \
  --warmup_batches 20 --measured_batches 200 --min_speedup 1.2 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
  --output "${PROFILE_PATH}"

NUM_WORKERS="$(python -c "import json; print(json.load(open('${PROFILE_PATH}'))['selected_num_workers'])")"
PREFETCH_FACTOR="$(python -c "import json; print(json.load(open('${PROFILE_PATH}'))['selected_prefetch_factor'])")"
MICRO_BATCH_SIZE="$(select_micro_batch)"
printf 'run_tag\t%s\nmicro_batch_size\t%s\neffective_batch_size\t%s\nnum_workers\t%s\nprefetch_factor\t%s\n' \
  "${RUN_TAG}" "${MICRO_BATCH_SIZE}" "${EFFECTIVE_BATCH_SIZE}" \
  "${NUM_WORKERS}" "${PREFETCH_FACTOR}" > "logs/v2_resolved_runtime_${RUN_TAG}.tsv"

run_cached_stage configs/velocity_xy_v2_visual.yaml \
  --max_epochs 1 --max_train_batches 5000 --max_val_batches 200 \
  --save_dir "${CHECKPOINT_ROOT}/visual_diagnostic" \
  --output_dir "${RUN_ROOT}/visual_diagnostic"

VISUAL_CKPT="${CHECKPOINT_ROOT}/visual_seed_42"
VISUAL_OUT="${RUN_ROOT}/visual_seed_42"
KIN_CKPT="${CHECKPOINT_ROOT}/kinematic_seed_42"
KIN_OUT="${RUN_ROOT}/kinematic_seed_42"
JOINT_CKPT="${CHECKPOINT_ROOT}/joint_seed_42"
JOINT_OUT="${RUN_ROOT}/joint_seed_42"
run_cached_stage configs/velocity_xy_v2_visual.yaml \
  --save_dir "${VISUAL_CKPT}" --output_dir "${VISUAL_OUT}"
run_cached_stage configs/velocity_xy_v2_kinematic.yaml \
  --init_checkpoint "${VISUAL_CKPT}/best.pth" \
  --visual_baseline_metrics "${VISUAL_OUT}/best_eval/metrics.json" \
  --save_dir "${KIN_CKPT}" --output_dir "${KIN_OUT}"
run_cached_stage configs/velocity_xy_v2_visual_kinematic.yaml \
  --init_checkpoint "${KIN_CKPT}/best.pth" \
  --visual_baseline_metrics "${VISUAL_OUT}/best_eval/metrics.json" \
  --save_dir "${JOINT_CKPT}" --output_dir "${JOINT_OUT}"

for seed in 3407 2026; do
  visual_ckpt="${CHECKPOINT_ROOT}/visual_seed_${seed}"
  visual_out="${RUN_ROOT}/visual_seed_${seed}"
  kin_ckpt="${CHECKPOINT_ROOT}/kinematic_seed_${seed}"
  kin_out="${RUN_ROOT}/kinematic_seed_${seed}"
  joint_ckpt="${CHECKPOINT_ROOT}/joint_seed_${seed}"
  joint_out="${RUN_ROOT}/joint_seed_${seed}"
  run_cached_stage configs/velocity_xy_v2_visual.yaml \
    --seed "${seed}" --save_dir "${visual_ckpt}" --output_dir "${visual_out}"
  run_cached_stage configs/velocity_xy_v2_kinematic.yaml \
    --seed "${seed}" --init_checkpoint "${visual_ckpt}/best.pth" \
    --visual_baseline_metrics "${visual_out}/best_eval/metrics.json" \
    --save_dir "${kin_ckpt}" --output_dir "${kin_out}"
  run_cached_stage configs/velocity_xy_v2_visual_kinematic.yaml \
    --seed "${seed}" --init_checkpoint "${kin_ckpt}/best.pth" \
    --visual_baseline_metrics "${visual_out}/best_eval/metrics.json" \
    --save_dir "${joint_ckpt}" --output_dir "${joint_out}"
done

python scripts/evaluate_v2_holdout.py \
  --config configs/velocity_xy_v2_geometry.yaml --mode geometry_only --seed 42 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --frame_cache_root "${FRAME_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" --variants baseline \
  --output_dir "${RUN_ROOT}/stage1_test/geometry"

VISUAL_PREDICTIONS=()
JOINT_PREDICTIONS=()
for seed in "${SEEDS[@]}"; do
  visual_ckpt="${CHECKPOINT_ROOT}/visual_seed_${seed}/best.pth"
  joint_ckpt="${CHECKPOINT_ROOT}/joint_seed_${seed}/best.pth"
  visual_dir="${RUN_ROOT}/stage1_test/visual_seed_${seed}"
  joint_dir="${RUN_ROOT}/stage1_test/joint_seed_${seed}"
  python scripts/evaluate_v2_holdout.py \
    --config configs/velocity_xy_v2_visual.yaml --checkpoint "${visual_ckpt}" \
    --mode visual --seed "${seed}" --dataroot "${DATAROOT}" \
    --can_bus_root "${CAN_ROOT}" --flow_root "${FLOW_ROOT}" \
    --frame_cache_root "${FRAME_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --variants baseline no_visual_residual no_wheel_scaling no_yaw_compensation \
      missing_can stale_can shuffled_images shuffled_flow shuffled_can \
    --output_dir "${visual_dir}"
  python scripts/evaluate_v2_holdout.py \
    --config configs/velocity_xy_v2_visual_kinematic.yaml --checkpoint "${joint_ckpt}" \
    --mode visual_kinematic --seed "${seed}" --dataroot "${DATAROOT}" \
    --can_bus_root "${CAN_ROOT}" --flow_root "${FLOW_ROOT}" \
    --frame_cache_root "${FRAME_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --output_dir "${joint_dir}"
  VISUAL_PREDICTIONS+=("${visual_dir}/predictions.csv")
  JOINT_PREDICTIONS+=("${joint_dir}/predictions.csv")
done

set +e
python scripts/analyze_v2_stage1.py \
  --seeds "${SEEDS[@]}" \
  --geometry_predictions "${RUN_ROOT}/stage1_test/geometry/predictions.csv" \
  --visual_predictions "${VISUAL_PREDICTIONS[@]}" \
  --joint_predictions "${JOINT_PREDICTIONS[@]}" \
  --training_stats logs/v2_training_stats.json \
  --output_dir "${RUN_ROOT}/stage1_test_report" --enforce
ANALYSIS_STATUS=$?
set -e
exit "${ANALYSIS_STATUS}"
