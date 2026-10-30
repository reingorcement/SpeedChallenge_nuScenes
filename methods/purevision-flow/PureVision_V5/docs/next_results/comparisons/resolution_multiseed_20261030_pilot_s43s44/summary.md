# R02 resolution pilot: three paired training seeds

Scene selection remains fixed at seed 42. Training seeds: 42, 43, 44.

| Seed | Resolution | Status | EPE | MAE vx | MAE vy | Best epoch | Updates / AMP skips |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | 512 | completed | 0.946689 | 0.944315 | 0.025546 | 27 | 4600 / 0 |
| 42 | 1024 | completed | 0.879164 | 0.877233 | 0.024956 | 38 | 4600 / 0 |
| 43 | 512 | completed | 1.010065 | 1.007156 | 0.031861 | 18 | 4600 / 0 |
| 43 | 1024 | completed | 0.856311 | 0.854020 | 0.024199 | 37 | 4600 / 0 |
| 44 | 512 | completed | 0.916859 | 0.914420 | 0.024959 | 32 | 4600 / 0 |
| 44 | 1024 | completed | 0.820553 | 0.818504 | 0.024651 | 40 | 4600 / 0 |

Comparison: **comparable**. Decision: **pending_user_choice**.

Mean ± sample SD (n=3, ddof=1), m/s:

| Metric | 512 | 1024 | Paired high − low |
|---|---:|---:|---:|
| epe | 0.957871 ± 0.047599 | 0.852009 ± 0.029541 | -0.105861 ± 0.043902 |
| mae_vx | 0.955297 ± 0.047333 | 0.849919 ± 0.029579 | -0.105378 ± 0.043800 |
| mae_vy | 0.027455 ± 0.003827 | 0.024602 ± 0.000381 | -0.002853 ± 0.004167 |
| rmse_vx | 1.613145 ± 0.103747 | 1.524995 ± 0.041247 | -0.088150 ± 0.102588 |
| rmse_vy | 0.037091 ± 0.004150 | 0.034427 ± 0.000346 | -0.002664 ± 0.004495 |
| bias_vx | -0.218171 ± 0.083925 | -0.237742 ± 0.037266 | -0.019571 ± 0.115884 |
| bias_vy | -0.004318 ± 0.010356 | 0.003080 ± 0.004392 | 0.007398 ± 0.014744 |

Paired EPE deltas by seed 42/43/44: -0.067525, -0.153754, -0.096305.

| Scene | Mean paired EPE delta | Sample SD | Improved seeds |
|---|---:|---:|---:|
| scene-0048 | -0.197866 | 0.082468 | 3 / 3 |
| scene-0072 | -0.048871 | 0.048678 | 3 / 3 |
| scene-0195 | -0.049083 | 0.041754 | 3 / 3 |
| scene-0236 | -0.135268 | 0.065039 | 3 / 3 |
| scene-0402 | -0.150875 | 0.066817 | 3 / 3 |
| scene-0566 | +0.080559 | 0.098937 | 1 / 3 |
| scene-0584 | -0.011528 | 0.091066 | 2 / 3 |
| scene-1013 | -0.344777 | 0.311828 | 3 / 3 |

- Three training seeds reuse exactly the same 32 train / 8 dev scenes; selection seed remains 42.
- Mean and sample SD use n=3 training runs and ddof=1; paired deltas are high minus low within each seed.
- The nonzero-vy subset still has 13 unique windows in one scene, not 39 independent observations.
- PLANAR_PROXY and same-dev checkpoint selection/recomputation; no held-out test or statistical-significance claim.
- No new RAFT, full cache, Flex encoder or automatic frontend selection is authorized by this report.

Actual training commits:
- R02_512_s42: `946ac67a4f16a7d2edcdc973dc1b84c0dee5d8a9`
- R02_1024_s42: `be327b7f91f5f7e5ba46b70dede116a21c551652`
- R02_512_s43: `252fb5f44c290c0d894faf8373329ba27105604e`
- R02_1024_s43: `252fb5f44c290c0d894faf8373329ba27105604e`
- R02_512_s44: `252fb5f44c290c0d894faf8373329ba27105604e`
- R02_1024_s44: `252fb5f44c290c0d894faf8373329ba27105604e`
