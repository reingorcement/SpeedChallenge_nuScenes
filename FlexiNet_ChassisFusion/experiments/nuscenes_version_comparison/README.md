# Self-Contained nuScenes KD/VKD Comparison

This package runs directly from the retained `7de2aad` KD/VKD checkout. The
default `legacy` profile compares:

- `base_velocity`: `[base_vx, 0]` reference.
- `legacy_kd`: kinematic and dynamics-proxy branches.
- `legacy_vkd`: visual, kinematic, and dynamics-proxy branches.

V2 is optional. Missing V2 modules do not block legacy imports, tests, quick
evaluation, or the three-seed pipeline.

## Server Prerequisites

- Linux with Bash 4+ and a CUDA-capable PyTorch environment.
- nuScenes `v1.0-trainval` and the CAN bus expansion.
- Dependencies from `../FlexiNet-main/requirements.txt`.
- The tracked KD/VKD checkpoints under `checkpoints/`.

From `FlexiNet_ChassisFusion`:

```bash
pip install -r ../FlexiNet-main/requirements.txt
python experiments/nuscenes_version_comparison/compatibility_check.py \
  --project-root . --profile legacy
```

The experiment directory is not ignored by Git. Ensure it is included in the
commit used on the server:

```bash
git status --short experiments
git add experiments/nuscenes_version_comparison
```

The scripts do not create or push a Git commit.

## Quick Existing-Checkpoint Evaluation

Start with 32 samples:

```bash
COMPARISON_RUN_ID=quick_smoke \
NUSCENES_ROOT=/path/to/nuscenes \
QUICK_MAX_SAMPLES=32 \
QUICK_VARIANTS="clean can_missing" \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

Then run the default 512-sample screen:

```bash
COMPARISON_RUN_ID=quick_512 \
NUSCENES_ROOT=/path/to/nuscenes \
QUICK_MAX_SAMPLES=512 \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

Default variants are `clean`, `can_missing`, `yaw_bias_pos`,
`wheel_scale_low`, and `camera_missing`. The split manifest is generated
automatically with seed 42. Existing frame caches are used when present;
otherwise images are loaded from the raw nuScenes files.

Output:

```text
outputs/nuscenes_version_comparison/<run_id>/quick_existing_eval/quick_report.md
```

## Three-Seed Legacy Pipeline

```bash
EXPERIMENT_PROFILE=legacy \
COMPARISON_RUN_ID=legacy_three_seed \
NUSCENES_ROOT=/path/to/nuscenes \
NUM_WORKERS=8 \
LEGACY_BATCH_SIZE=32 \
bash experiments/nuscenes_version_comparison/run_server_pipeline.sh
```

The pipeline trains seeds `42`, `3407`, and `2026`, evaluates clean and stress
variants on aligned stage-1 scenes, and writes scene-level paired bootstrap
results. Existing checkpoints and predictions are skipped so the run is
resumable.

## Optional V2 Profile

`EXPERIMENT_PROFILE=full` is accepted only when the selected source tree passes
the full compatibility check and contains `train_v2.py`, `dataset/v2_schema.py`,
`dataset/v2_splits.py`, and `model/v2/model.py`. It is not required for KD/VKD.

## Interpretation Boundary

The reports support open-loop nuScenes accuracy, synthetic sensor perturbation
trends, and logged high-excitation subgroup analysis. They do not establish true
physical parameter mismatch, cross-vehicle dynamics generalization, or
closed-loop estimator-controller stability.
