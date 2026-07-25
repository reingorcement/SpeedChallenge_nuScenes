# V3 Visual-Only Server Pipeline

From the `FlexiNet_ChassisFusion` directory in the `zy_speedest` environment, run this single detached command:

```bash
nohup env V3_RUN_TAG=v3_visual_only_full_20260726 bash scripts/run_v3_visual_only_pipeline.sh > logs/v3_visual_only_full_20260726.launch.log 2>&1 &
```

The command performs, in order:

1. CUDA/dependency and V3 regression tests.
2. Direct visual pretraining for seeds 42, 3407, and 2026.
3. Fusion training initialized from each matching direct checkpoint.
4. Frozen-checkpoint evaluation on the independent `stage1_test` scene partition.
5. Scene-paired bootstrap analysis of fusion versus direct, with JSON, CSV, Markdown, and figures.

The default dataset paths match the current server. They can be overridden without editing files:

```bash
NUSCENES_ROOT=/path/to/nuscenes \
V3_FLOW_ROOT=/path/to/flows \
V3_FRAME_ROOT=/path/to/frames \
V3_RUN_TAG=my_v3_run \
bash scripts/run_v3_visual_only_pipeline.sh
```

Generated checkpoints are stored under `checkpoints/<run-tag>/`. Predictions and analysis are stored under `outputs/<run-tag>/`, and the complete console log is stored at `logs/<run-tag>.log`.

Progress can be inspected without affecting the run:

```bash
tail -f logs/v3_visual_only_full_20260726.log
```

Reusing the same `V3_RUN_TAG` skips completed stages and resumes an incomplete stage from its `last.pth` checkpoint.
