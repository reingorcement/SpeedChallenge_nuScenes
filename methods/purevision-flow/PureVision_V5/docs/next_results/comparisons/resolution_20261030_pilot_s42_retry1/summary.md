# R02 resolution pilot

Actual training commits: R02_512 `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9`; R02_1024 `be327b7f91f5f7e5ba46b70dede116a21c551652`.

Scope: fixed 32 train / 8 dev scene pilot, seed 42, 40 complete epochs, batch 64.

| Run | Status | Dev EPE | MAE vx | MAE vy | Best epoch | Updates / AMP skips | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261030_pilot_s42_R02_512 | completed | 0.946689 | 0.944315 | 0.025546 | 27 | 4600 / 0 | 74.24 |
| 20261030_pilot_s42_retry1_R02_1024 | completed | 0.879164 | 0.877233 | 0.024956 | 38 | 4600 / 0 | 74.38 |

Train-fitted constant vy: -0.029543193 m/s; dev MAE zero=0.042481, median=0.029386.

Comparison: **comparable**. Decision: **pending_user_choice**.
The two inputs have the same 128×72 downstream view; only RAFT generation resolution changes.

High minus low dev EPE: -0.067525 m/s.

- 32 train / 8 dev scenes only; this is not a full-dataset result.
- Single seed 42, dev selection and dev recomputation; no independent held-out test.
- PLANAR_PROXY rear-axle yaw-frame labels; weak lateral evidence may remain insufficient.
- This report verifies small metadata and checkpoint/prediction bytes, not multi-GB flow array bytes.
- The user chooses the frontend before any full-cache generation; no automatic promotion.
