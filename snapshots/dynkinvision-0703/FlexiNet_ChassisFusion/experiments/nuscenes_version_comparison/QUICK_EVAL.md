# Quick Existing-Checkpoint Evaluation

The quick path evaluates the tracked KD/VKD checkpoints without retraining.
It uses causal history-only inputs and aligns every algorithm on the common
`scene_name + center_token` subset.

## Command

```bash
COMPARISON_RUN_ID=quick_existing \
NUSCENES_ROOT=/path/to/nuscenes \
QUICK_MAX_SAMPLES=512 \
bash experiments/nuscenes_version_comparison/run_quick_existing_eval.sh
```

Use `QUICK_MAX_SAMPLES=32` and
`QUICK_VARIANTS="clean can_missing"` for the first server smoke run.

Frame caches are optional. When
`${NUSCENES_ROOT}/precomputed_frames/CAM_FRONT_64x64_gray/manifest.json` is
absent, the evaluator reads raw `CAM_FRONT` images.

## Outputs

```text
outputs/nuscenes_version_comparison/<run_id>/quick_existing_eval/
  quick_report.md
  quick_report.json
  all_predictions.csv
  <algorithm>/predictions.csv
  <algorithm>/metrics.json
```

Negative paired delta means the left algorithm has lower endpoint error. This
is a screening test, not a repeated-seed training comparison.
