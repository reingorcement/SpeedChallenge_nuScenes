#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
EFFECTIVE_BATCH_SIZE=8
SPLIT_MANIFEST="logs/v2_split_manifest.json"
SEEDS=(42 3407 2026)

select_micro_batch() {
  local log_path="logs/v2_max_memory_smoke.log"
  mkdir -p logs
  if python train_v2.py \
    --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --smoke --batch_size 8 --effective_batch_size 8 2>&1 | tee "${log_path}" >&2; then
    echo 8
    return
  fi
  if ! grep -Eqi "CUDA out of memory|OutOfMemoryError" "${log_path}"; then
    echo "Smoke failed for a non-OOM reason; refusing to hide the error with a smaller batch." >&2
    return 1
  fi
  echo "CUDA OOM at batch 8; retrying batch 4 with accumulation 2." >&2
  python train_v2.py \
    --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --smoke --batch_size 4 --effective_batch_size 8 2>&1 | tee "${log_path}" >&2
  echo 4
}

run_stage() {
  local config="$1"
  shift
  python train_v2.py --config "${config}" \
    --dataroot "${DATAROOT}" \
    --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --batch_size "${MICRO_BATCH_SIZE}" \
    --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" "$@"
}

mkdir -p logs outputs/stage1_test
if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python scripts/create_v2_split_manifest.py \
    --dataroot "${DATAROOT}" \
    --can_bus_root "${CAN_ROOT}" \
    --output "${SPLIT_MANIFEST}" \
    --seed 42
else
  python -c "from dataset.v2_splits import load_split_manifest; load_split_manifest('${SPLIT_MANIFEST}')"
fi

python scripts/audit_v2_data.py \
  --dataroot "${DATAROOT}" \
  --can_bus_root "${CAN_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --partition train_core \
  --output logs/v2_data_audit.json

python scripts/precompute_v2_optical_flow.py \
  --dataroot "${DATAROOT}" \
  --output_root "${FLOW_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --partitions train_core dev stage1_test \
  --width 512 \
  --model raft_large

python scripts/prepare_v2_training_stats.py \
  --config configs/velocity_xy_v2_visual.yaml \
  --audit logs/v2_data_audit.json \
  --dataroot "${DATAROOT}" \
  --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output logs/v2_training_stats.json

MICRO_BATCH_SIZE="$(select_micro_batch)"
printf 'micro_batch_size\t%s\neffective_batch_size\t%s\n' \
  "${MICRO_BATCH_SIZE}" "${EFFECTIVE_BATCH_SIZE}" > logs/v2_resolved_batch.tsv

run_stage configs/velocity_xy_v2_geometry.yaml
run_stage configs/velocity_xy_v2_visual.yaml
run_stage configs/velocity_xy_v2_kinematic.yaml
run_stage configs/velocity_xy_v2_visual_kinematic.yaml

for seed in 3407 2026; do
  visual_ckpt="checkpoints/velocity_xy_v2_visual_seed_${seed}"
  visual_out="outputs/velocity_xy_v2_visual_seed_${seed}"
  kin_ckpt="checkpoints/velocity_xy_v2_kinematic_seed_${seed}"
  kin_out="outputs/velocity_xy_v2_kinematic_seed_${seed}"
  joint_ckpt="checkpoints/velocity_xy_v2_visual_kinematic_seed_${seed}"
  joint_out="outputs/velocity_xy_v2_visual_kinematic_seed_${seed}"

  run_stage configs/velocity_xy_v2_visual.yaml \
    --seed "${seed}" --save_dir "${visual_ckpt}" --output_dir "${visual_out}"
  run_stage configs/velocity_xy_v2_kinematic.yaml \
    --seed "${seed}" --init_checkpoint "${visual_ckpt}/best.pth" \
    --visual_baseline_metrics "${visual_out}/best_eval/metrics.json" \
    --save_dir "${kin_ckpt}" --output_dir "${kin_out}"
  run_stage configs/velocity_xy_v2_visual_kinematic.yaml \
    --seed "${seed}" --init_checkpoint "${kin_ckpt}/best.pth" \
    --visual_baseline_metrics "${visual_out}/best_eval/metrics.json" \
    --save_dir "${joint_ckpt}" --output_dir "${joint_out}"
done

python scripts/evaluate_v2_holdout.py \
  --config configs/velocity_xy_v2_geometry.yaml \
  --mode geometry_only \
  --seed 42 \
  --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
  --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
  --variants baseline \
  --output_dir outputs/stage1_test/geometry

VISUAL_PREDICTIONS=()
JOINT_PREDICTIONS=()
for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "42" ]]; then
    visual_ckpt="checkpoints/velocity_xy_v2_visual/best.pth"
    joint_ckpt="checkpoints/velocity_xy_v2_visual_kinematic/best.pth"
  else
    visual_ckpt="checkpoints/velocity_xy_v2_visual_seed_${seed}/best.pth"
    joint_ckpt="checkpoints/velocity_xy_v2_visual_kinematic_seed_${seed}/best.pth"
  fi
  visual_dir="outputs/stage1_test/visual_seed_${seed}"
  joint_dir="outputs/stage1_test/joint_seed_${seed}"
  python scripts/evaluate_v2_holdout.py \
    --config configs/velocity_xy_v2_visual.yaml \
    --checkpoint "${visual_ckpt}" \
    --mode visual \
    --seed "${seed}" \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --variants baseline no_visual_residual no_wheel_scaling no_yaw_compensation \
      missing_can stale_can shuffled_images shuffled_flow shuffled_can \
    --output_dir "${visual_dir}"
  python scripts/evaluate_v2_holdout.py \
    --config configs/velocity_xy_v2_visual_kinematic.yaml \
    --checkpoint "${joint_ckpt}" \
    --mode visual_kinematic \
    --seed "${seed}" \
    --dataroot "${DATAROOT}" --can_bus_root "${CAN_ROOT}" \
    --flow_root "${FLOW_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --output_dir "${joint_dir}"
  VISUAL_PREDICTIONS+=("${visual_dir}/predictions.csv")
  JOINT_PREDICTIONS+=("${joint_dir}/predictions.csv")
done

set +e
python scripts/analyze_v2_stage1.py \
  --seeds "${SEEDS[@]}" \
  --geometry_predictions outputs/stage1_test/geometry/predictions.csv \
  --visual_predictions "${VISUAL_PREDICTIONS[@]}" \
  --joint_predictions "${JOINT_PREDICTIONS[@]}" \
  --training_stats logs/v2_training_stats.json \
  --output_dir outputs/stage1_test_report \
  --enforce
ANALYSIS_STATUS=$?
set -e
exit "${ANALYSIS_STATUS}"
