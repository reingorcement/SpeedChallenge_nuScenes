# RGB appearance bypass vs difference:18-epoch comparison

Appearance bypass has lower dev EPE within this seed42 common18-epoch trajectory; inspect lateral and train/dev gaps before a route decision.

| Path | Dev EPE | Train EPE | vx MAE | vy MAE | Updates | AMP skips |
|---|---:|---:|---:|---:|---:|---:|
| difference | 0.206749 | 0.129413 | 0.203872 | 0.016939 | 37394 | 10 |
| appearance_difference | 0.205068 | 0.125555 | 0.202105 | 0.016902 | 37395 | 9 |

Full component errors, scene deltas, curves, lateral rule, costs and hashes are in summary.json.

- 18 complete epochs with original40-epoch cosine; old partial epoch19 excluded. Old last is not epoch18 endpoint.
- Single candidate/seed; no40-epoch claim or isolated proof that appearance is inherently useful.
- Dev used for selection; original scene split not log-disjoint; PLANAR_PROXY rear-axle supervision.
- 31 independent lateral windows/5scenes: unchanged100-window/5-scene rule, insufficient_data.
- Frozen/CONT/R02 40-epoch results are historical context, excluded from this primary comparison.
