# V2 Training Acceleration and Recovery

## What Was Fixed

The failed visual run stopped at epoch 1, batch 4325 with a finite loss and finite model outputs. AMP had grown its scale to `262144`; four values in the final visual residual-head gradient overflowed. The trainer now treats this as a recoverable AMP event: it skips the optimizer and scheduler step, lowers the scale, records diagnostics, and continues. Non-finite loss, failed scale backoff, repeated overflow, or excessive overflow rate remain fatal.

## Cached Training Path

The accelerated path preserves the model, loss, batch size, learning rates, seeds, and `ego_reference` output.

- RGB frames are precomputed once as scene-level `72x128 uint8` arrays and memory-mapped during training.
- Deterministic road-plane outputs are precomputed once per sample and stored in one tensor package per scene.
- Geometry cache identity covers the `raft_large` flow backend, geometry resolution, camera height, road ROI, yaw age handling, output reference, solver parameters, and training-statistics hash.
- Training batches omit full-resolution flow and calibration tensors after cache validation.
- Geometry audit, cache verification, holdout evaluation, and flow/CAN counterfactuals continue to use online geometry.

Default cache locations:

```text
/media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_frames/CAM_FRONT_72x128_rgb
/media/ps/AC78E69278E65B1A/zy/nuscenes/precomputed_geometry_v2/CAM_FRONT
```

## Server Run

Do not resume the failed checkpoint. Upload the updated repository, enter `FlexiNet_ChassisFusion`, and start one fresh pipeline:

```bash
RUN_TAG="accelerated_$(date +%Y%m%d_%H%M%S)"
V2_RUN_TAG="${RUN_TAG}" nohup bash scripts/run_v2_server_pipeline.sh \
  > "logs/v2_stage1_${RUN_TAG}.log" 2>&1 &
```

Monitor it with:

```bash
tail -f "logs/v2_stage1_${RUN_TAG}.log"
nvidia-smi dmon -s pucm -d 2
```

Frame generation and CAN statistics are CPU/I/O stages, so GPU utilization can be zero there. Geometry generation, profiling, smoke, diagnostic, and training use CUDA.

The pipeline performs, in order:

1. data and existing RAFT checks;
2. resumable RGB-frame and geometry-cache generation;
3. online geometry audit and 256-sample cache equivalence audit;
4. online-versus-cached throughput profiling and worker selection;
5. maximum-memory smoke and 5000-batch visual diagnostic;
6. visual, kinematic-pretrain, and joint training for seeds `42/3407/2026`;
7. online holdout counterfactuals and final stage-1 analysis.

## Outputs and Gates

The latest run tag is written to `logs/v2_latest_run.txt`. Results and checkpoints are isolated under:

```text
outputs/<run-tag>/
checkpoints/<run-tag>/
```

Important reports:

```text
logs/v2_geometry_cache_verification_<run-tag>.json
logs/v2_throughput_profile_<run-tag>.json
logs/v2_resolved_runtime_<run-tag>.tsv
outputs/<run-tag>/visual_diagnostic/
outputs/<run-tag>/stage1_test_report/
```

Formal training starts only when cached throughput is at least `1.2x` the online baseline, cache fields match online geometry within `1e-5`, the memory smoke passes, and the 5000-batch diagnostic completes. AMP overflow is accepted only below `0.5%` per epoch with no more than three consecutive events.
