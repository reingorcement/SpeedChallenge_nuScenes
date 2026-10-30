# AMP overflow recovery — 2026-10-30

Formal `20261030_s42_r1_R00` failed at 04:27:57 UTC, after 11,159
processed batches, on the next batch in epoch 5 (zero based). The previous
trainer unscaled gradients and immediately called gradient clipping with
`error_if_nonfinite=True`. This aborts before GradScaler can perform its normal
skip-and-reduce-scale handling of an FP16 backward overflow.

The last checkpoint is at batch 11,000; best is at batch 10,390. Both model and
Adam tensors are finite, with loss scale 2,097,152. An isolated GPU1 experiment
reproduced the same failure using a finite forward loss, then demonstrated that
GradScaler skips the update and halves the scale. These observations support
the AMP handling diagnosis; the exact unsaved failing batch was not replayed.

The fix checks the unscaled gradient norm without modifying gradients. With
nonfinite gradients and enabled AMP, GradScaler handles the recorded overflow;
the finite positive loss scale must decrease. Finite gradients with overflowing
norm, nonfinite FP32 gradients and nonfinite forward loss remain fatal. This
does not change the model, loss, six configurations, dataset or cached RGB/flow.
Processed batches, optimizer updates and AMP skips are separately recorded in
logs, status, metrics and complete checkpoints; resume validates their sum.

Validation includes a forced-overflow checkpoint/resume test and an actual
GPU1 check: three skipped updates leave weights unchanged, followed by two
successful Adam updates. Independent source review passed. Evidence is under
`ART/monitoring/incidents/amp_overflow_{reproduction,fix}_gpu1.json`.

Only the r1 coordinator was stopped. Its nine never-launched jobs were marked
cancelled with their previous state retained. Both original forward workers
continued unchanged. R1 R00 remains failed and contributes no completed result.
Recovery uses fresh `20261030_s42_r2_Rxx` run IDs, seed 42, a newly pushed fixed
commit and a new immutable checkout; the failed optimizer state is not resumed.

The first full dev pass also established that the predefined nonzero-lateral
subset contains only 31 windows from five scenes. The predeclared minimum is
100 windows and five scenes: evidence is insufficient, and the threshold is not
lowered after seeing results. Overall metrics remain reportable with this limit.

`ART=/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5`.
