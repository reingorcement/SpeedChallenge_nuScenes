# Runtime Estimate

## Evidence

The repository server log records V2 visual training at batch size 8:

```text
4325 / 16623 training batches in about 22 minutes
```

This projects to roughly 85 minutes for the training portion of one V2 epoch.
Dev evaluation and checkpoint writing add approximately 10-25 minutes, so the
planning value is **1.5-1.8 hours per V2 epoch** on that server.

The same repository logs show approximately:

```text
Legacy VKD: about 8.2 batches/s at batch size 32
Legacy KD: substantially faster after data-loader warm-up
```

The current protocol contains 132,978 `train_core` samples and these maximum
epoch counts:

| Stage | Maximum epochs |
| --- | ---: |
| Legacy KD | 20 |
| Legacy VKD | 20 |
| V2 visual | 15 |
| V2 kinematic pretrain | 15 |
| V2 joint | 10 |

## Expected Wall Time

Assuming one GPU, existing frame/flow/geometry caches, and normal early stopping:

| Work | Expected time |
| --- | ---: |
| Compatibility check and all smoke runs | 10-30 minutes |
| One seed, four learned comparison arms | 34-50 hours |
| Three-seed main training | 4.5-6.5 days |
| Stage-1 clean/stress/calibration evaluation | 8-16 hours |
| Main experiment total | **5-7 days** |
| Boston-to-Singapore three-seed extension | **2.5-5 additional days** |
| Complete default pipeline | **8-12 days** |

If frame and optical-flow caches must be generated, add approximately 4-12
hours depending on storage throughput and GPU. The first run should therefore
use `PREPARE_DATA=1`; later reruns should use `PREPARE_DATA=0` only after cache
verification.

With three comparable GPUs and one seed assigned to each GPU, seed-level work
can run in parallel. The main experiment is then approximately 2-3 days, and the
complete pipeline including location shift is approximately 3-5 days. The
provided shell script is sequential, so multi-GPU execution requires launching
one seed per process or scheduler job rather than sharing one `run_server_pipeline.sh`.

These are planning ranges, not guarantees. The strongest predictor is the first
full V2 epoch on the target server; update the estimate by multiplying its
measured duration by the remaining stage epochs and expected early-stop points.
