# Notes: GPU Training Optimization

## Sources Compared

### Current branch
- `configs/velocity_xy_fair_gray.yaml`
- `configs/velocity_xy_vkd_gated.yaml`
- `configs/smoke_velocity_xy_vkd_gated.yaml`
- `train.py`
- `scripts/profile_training_step.py`
- `速度XY改造使用说明.md`

### Previous worktree
- `../velocity-xy-fusion/FlexiNet_ChassisFusion/configs/velocity_xy_fair_gray.yaml`
- `../velocity-xy-fusion/FlexiNet_ChassisFusion/configs/velocity_xy_fair_rgb_flow_geom.yaml`
- `../velocity-xy-fusion/FlexiNet_ChassisFusion/train.py`
- `../velocity-xy-fusion/FlexiNet_ChassisFusion/scripts/profile_training_step.py`

## Findings

### Already inherited from previous version
- `batch_size: 32` for gray baseline.
- `num_workers: 8`.
- `prefetch_factor: 4`.
- `use_amp: true`.
- `pin_memory` + `non_blocking=True` tensor transfer.
- `persistent_workers=True` when workers > 0.
- `loss.detach()` accumulation to reduce GPU sync.
- `eval_every` and `log_interval` controls.
- Frame cache support via `frame_cache_root`.
- `scripts/profile_training_step.py` for load/transfer/compute timing.

### Current likely bottlenecks
- If `frame_cache_root` is absent or cache is on slow storage, JPEG/frame loading dominates and GPU will starve.
- If `compute` time is much smaller than `load`, increasing batch alone will not help enough.
- `torch.backends.cudnn.deterministic=True` and `benchmark=False` in `utils/seed.py` favors reproducibility over speed.
- Evaluation can be expensive because it runs full val every epoch by default (`eval_every: 1`).
- New `vkd_gated` branch adds some compute but still uses small 64x64 gray inputs, so VRAM may remain underused at `batch_size: 32`.

## Safe config-only levers
- Increase `batch_size` until OOM, then back off 10-20%.
- Increase `num_workers` from 8 to 12 or 16 on servers with enough CPU/RAM.
- Keep `prefetch_factor: 4`; try 6 if CPU memory is sufficient.
- Set `eval_every: 2` or `5` during long training; keep final epoch eval.
- Set `log_interval: 50` or `100` to reduce `.item()` sync.
- Use `velocity_xy_vkd_gated.yaml` rather than smoke config for real training.
- Use RGB/flow/geometry variants only if those caches exist and the goal is higher compute/VRAM, not fastest baseline.

## Optional code levers
- Add `train.channels_last: true` and move image/model tensors to channels-last where valid.
- Add `train.cudnn_benchmark: true` for fixed image shapes, with deterministic disabled.
- Add `train.allow_tf32: true` on Ampere+ GPUs.
- Add `train.torch_compile: true` optionally, but verify first because dynamic branches and Windows/server PyTorch versions can vary.
- Add a batch auto-tuner script or CLI that doubles batch size until OOM, then writes suggested config.

## Recommended first profile commands
```bash
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 32 --num_workers 8
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 64 --num_workers 8
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 96 --num_workers 12
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 128 --num_workers 16
```

## Rule of thumb
- If `load > compute`, raise workers/prefetch/cache quality first.
- If `compute > load` and VRAM is low, raise batch size.
- If OOM occurs, reduce batch to the previous passing value and consider gradient accumulation only if effective batch size matters more than wall-clock speed.
