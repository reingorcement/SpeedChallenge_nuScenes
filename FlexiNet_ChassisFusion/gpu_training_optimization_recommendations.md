# GPU Training Optimization Recommendations

## Summary
The current branch already contains the main acceleration settings from the previous `velocity-xy-fusion` version: frame cache, AMP, pinned memory, non-blocking transfer, persistent workers, prefetching, reduced logging sync, and `eval_every`.

The next gains should come from two tracks:

1. Config-only tuning to increase batch size and reduce evaluation/logging overhead.
2. Optional code switches for cuDNN benchmark, TF32, channels-last, and batch-size autotuning.

## Recommended First Server Config
For `configs/velocity_xy_vkd_gated.yaml`, start with:

```yaml
train:
  batch_size: 96        # try 64 first if GPU memory is unknown
  num_workers: 12
  prefetch_factor: 4
  eval_every: 3
  log_interval: 50
  use_amp: true
```

If VRAM is still low and `profile_training_step.py` shows `compute >= load`, try:

```yaml
train:
  batch_size: 128
  num_workers: 16
  prefetch_factor: 4
  eval_every: 5
  log_interval: 100
```

If `load` remains larger than `compute`, do not keep increasing batch blindly. First verify:
- `frame_cache_root` points to local SSD/NVMe, not slow network storage.
- `num_workers` is high enough for the CPU.
- `prefetch_factor` is 4 or 6.
- The frame cache contains `labels.json`.

## Profiling Commands
Run these on the training server:

```bash
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 32 --num_workers 8
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 64 --num_workers 8
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 96 --num_workers 12
python scripts/profile_training_step.py --config configs/velocity_xy_vkd_gated.yaml --batches 50 --batch_size 128 --num_workers 16
```

Interpretation:
- `load` high: CPU/cache/DataLoader bottleneck.
- `transfer` high: pinned memory or PCIe transfer bottleneck.
- `compute` high and VRAM low: raise batch size.
- OOM: back off to the previous passing batch size.

## Optional Code Changes Worth Applying
These require code/config edits and should be verified with smoke + profiling:

```yaml
train:
  cudnn_benchmark: true
  deterministic: false
  allow_tf32: true
  channels_last: true
```

Rationale:
- `cudnn_benchmark` can speed fixed-size convolution workloads.
- TF32 can speed matmul/conv on Ampere+ GPUs with little practical loss for this task.
- `channels_last` can improve convolution throughput.
- Current `utils/seed.py` sets deterministic cuDNN behavior, which is reproducible but slower.

## What Not To Do First
- Do not switch to RGB + flow only to fill VRAM unless the experiment needs those modalities. It changes the experimental question.
- Do not use gradient accumulation to “fill VRAM”; it improves effective batch size but does not increase per-step GPU occupancy like a real larger batch.
- Do not disable frame cache. JPEG decoding will likely starve GPU.

## Best Next Implementation
Add a `configs/velocity_xy_vkd_gated_fast.yaml` config with larger batch/workers/eval interval, plus optional train flags:
- `cudnn_benchmark`
- `deterministic`
- `allow_tf32`
- `channels_last`

Then run profile commands and select the largest stable batch for the actual server GPU.
