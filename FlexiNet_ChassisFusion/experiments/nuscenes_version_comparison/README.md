# nuScenes Version Comparison

This package compares the retained legacy KD/VKD architecture with the current
V2 visual and visual-kinematic architecture without modifying either source
implementation.

The current V2 is **visual geometry plus a weak kinematic correction**. It does
not contain a dynamics-student branch. Results from this package must not be
described as validation of a complete visual/kinematic/dynamics V2 system.

## Layout

- `configs/`: the shared causal protocol and per-algorithm overlays.
- `adapters/`: split-aware datasets and unified model inference contracts.
- `perturbations.py`: semantic sensor and V2-only calibration perturbations.
- `train_comparison.py`: legacy training and delegation to the V2 staged trainer.
- `evaluate_comparison.py`: aligned clean/stress inference and standard CSV output.
- `analyze_comparison.py`: train-derived thresholds and scene-level paired statistics.
- `run_server_pipeline.sh`: resumable three-seed server workflow.
- `RUNTIME_ESTIMATE.md`: log-based one-GPU and multi-GPU runtime ranges.

All experiment artifacts are written below:

```text
outputs/nuscenes_version_comparison/<run_id>/
checkpoints/nuscenes_version_comparison/<run_id>/
```

## Server Run

From `FlexiNet_ChassisFusion`:

```bash
COMPARISON_RUN_ID=stage1_$(date +%Y%m%d_%H%M%S) \
NUSCENES_ROOT=/path/to/nuscenes \
bash experiments/nuscenes_version_comparison/run_server_pipeline.sh
```

### Running From The Legacy Checkout

Commit `7de2aad` does not contain V2, explicit manifest scene selection, or the
causal current-index model interface. Copying this folder into that checkout is
supported only when a current source tree is supplied:

```bash
cd /path/to/legacy/FlexiNet_ChassisFusion
NUSCENES_COMPARISON_PROJECT_ROOT=/path/to/current/FlexiNet_ChassisFusion \
NUSCENES_ROOT=/path/to/nuscenes \
bash experiments/nuscenes_version_comparison/run_server_pipeline.sh
```

The experiment scripts are loaded from the legacy checkout, while model,
dataset, config, and trainer dependencies are loaded from the current source
tree. Outputs and checkpoints are written under the current source tree.

Run the compatibility check before a long server job:

```bash
python experiments/nuscenes_version_comparison/compatibility_check.py \
  --project-root /path/to/current/FlexiNet_ChassisFusion
```

The checker fails immediately and lists missing files if the supplied tree is a
pure legacy checkout.

Useful controls:

```bash
PREPARE_DATA=0          # reuse verified frame/flow/geometry caches
RUN_DOMAIN_SHIFT=0      # skip Boston-to-Singapore retraining
NUM_WORKERS=8
LEGACY_BATCH_SIZE=32
V2_BATCH_SIZE=8
```

The pipeline keeps `final_system_test` sealed. Dataset adapters reject any
partition other than `train_core`, `dev`, and `stage1_test`.

## Quick Existing-Checkpoint Run

For a 1-2 hour screening test without retraining, run:

```bash
COMPARISON_RUN_ID=quick_existing_$(date +%Y%m%d_%H%M%S) \
NUSCENES_ROOT=/path/to/nuscenes \
QUICK_MAX_SAMPLES=512 \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

This evaluates existing KD/VKD/V2 checkpoints, aligns predictions on the common
`scene_name + center_token` subset, and writes
`outputs/nuscenes_version_comparison/<run_id>/quick_existing_eval/quick_report.md`.
Default variants are `clean`, `can_missing`, `yaw_bias_pos`,
`wheel_scale_low`, and `camera_missing`.

The script always evaluates `base_velocity`, `legacy_kd`, `legacy_vkd`, and
`v2_visual` when their checkpoints exist. Add a full current joint checkpoint
when available:

```bash
V2_VISUAL_KINEMATIC_CHECKPOINT=checkpoints/velocity_xy_v2_visual_kinematic/best.pth \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

This mode can support quick open-loop accuracy and synthetic input-fault trends,
but not three-seed significance, true physical parameter mismatch, or closed-loop
estimator-controller coupling.

## Individual Commands

Train a legacy baseline:

```bash
python experiments/nuscenes_version_comparison/train_comparison.py \
  --algorithm legacy_kd --run-id debug --seed 42 --smoke
```

Evaluate a frozen checkpoint:

```bash
python experiments/nuscenes_version_comparison/evaluate_comparison.py \
  --algorithm legacy_kd --checkpoint checkpoints/path/best.pth \
  --run-id debug --seed 42 --partition stage1_test
```

Calibration variants are accepted only by V2 algorithms. All perturbations
leave the target tensor unchanged, and scenario metadata is always collected
from the clean batch.

## Interpretation

The report can support claims about open-loop nuScenes accuracy, logged
high-excitation scenarios, sensor perturbations, visual calibration sensitivity,
and Boston-to-Singapore distribution shift. It cannot establish physical vehicle
parameter mismatch robustness, estimator-controller closed-loop stability,
verified center-of-gravity sideslip, or cross-vehicle dynamics generalization.
