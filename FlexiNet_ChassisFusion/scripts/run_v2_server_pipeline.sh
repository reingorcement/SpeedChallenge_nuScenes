#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${NUSCENES_ROOT:-/media/ps/AC78E69278E65B1A/zy/nuscenes}"
CAN_ROOT="${NUSCENES_CAN_ROOT:-${DATAROOT}/can_bus}"
FLOW_ROOT="${V2_FLOW_ROOT:-${DATAROOT}/precomputed_flow_v2/CAM_FRONT}"
EFFECTIVE_BATCH_SIZE=8

select_micro_batch() {
  local log_path="logs/v2_max_memory_smoke.log"
  mkdir -p logs
  if python train_v2.py \
    --config configs/smoke_velocity_xy_v2_visual_kinematic.yaml \
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
    --smoke --batch_size 4 --effective_batch_size 8 2>&1 | tee "${log_path}" >&2
  echo 4
}

run_stage() {
  local config="$1"
  shift
  python train_v2.py --config "${config}" \
    --batch_size "${MICRO_BATCH_SIZE}" \
    --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" "$@"
}

python scripts/audit_v2_data.py \
  --dataroot "${DATAROOT}" \
  --can_bus_root "${CAN_ROOT}" \
  --output logs/v2_data_audit.json

python scripts/precompute_v2_optical_flow.py \
  --dataroot "${DATAROOT}" \
  --output_root "${FLOW_ROOT}" \
  --width 512 \
  --model raft_large

python scripts/prepare_v2_training_stats.py \
  --config configs/velocity_xy_v2_visual.yaml \
  --audit logs/v2_data_audit.json \
  --output logs/v2_training_stats.json

MICRO_BATCH_SIZE="$(select_micro_batch)"
printf 'micro_batch_size\t%s\neffective_batch_size\t%s\n' \
  "${MICRO_BATCH_SIZE}" "${EFFECTIVE_BATCH_SIZE}" > logs/v2_resolved_batch.tsv

run_stage configs/velocity_xy_v2_geometry.yaml
run_stage configs/velocity_xy_v2_visual.yaml
run_stage configs/velocity_xy_v2_kinematic.yaml
run_stage configs/velocity_xy_v2_visual_kinematic.yaml

python scripts/evaluate_v2_counterfactual.py \
  --config configs/velocity_xy_v2_visual_kinematic.yaml \
  --checkpoint checkpoints/velocity_xy_v2_visual_kinematic/best.pth \
  --output outputs/velocity_xy_v2_visual_kinematic/counterfactual.json

python scripts/compare_v2_stages.py \
  --visual outputs/velocity_xy_v2_visual/best_eval/metrics.json \
  --joint outputs/velocity_xy_v2_visual_kinematic/best_eval/metrics.json \
  --output outputs/velocity_xy_v2_stage_comparison.json

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

python scripts/compare_v2_multiseed.py \
  --seeds 42 3407 2026 \
  --visual \
    outputs/velocity_xy_v2_visual/best_eval/metrics.json \
    outputs/velocity_xy_v2_visual_seed_3407/best_eval/metrics.json \
    outputs/velocity_xy_v2_visual_seed_2026/best_eval/metrics.json \
  --joint \
    outputs/velocity_xy_v2_visual_kinematic/best_eval/metrics.json \
    outputs/velocity_xy_v2_visual_kinematic_seed_3407/best_eval/metrics.json \
    outputs/velocity_xy_v2_visual_kinematic_seed_2026/best_eval/metrics.json \
  --output outputs/velocity_xy_v2_multiseed_comparison.json
