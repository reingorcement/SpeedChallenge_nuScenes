# R02 BASE/FLEX — full data, 40 epochs, paired seeds42/43/44

FLEX is the user-selected development mainline (accuracy priority); BASE remains the direct reference.

| Run | EPE | vx MAE | vy MAE | vx RMSE | vy RMSE | vx bias | vy bias | Params | Wall s | AMP skips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R02_BASE_s42 | 0.215142 | 0.212016 | 0.017229 | 0.414209 | 0.026374 | -0.018107 | -0.003579 | 74354 | 1215.5 | 31 |
| R02_FLEX_s42 | 0.191630 | 0.188479 | 0.016827 | 0.371284 | 0.026239 | -0.019759 | -0.002414 | 76850 | 1241.9 | 33 |
| R02_BASE_s43 | 0.204535 | 0.201358 | 0.017001 | 0.398368 | 0.026398 | -0.026806 | -0.001353 | 74354 | 1190.4 | 29 |
| R02_BASE_s44 | 0.205732 | 0.202747 | 0.016813 | 0.391140 | 0.026214 | -0.020908 | -0.000265 | 74354 | 1210.1 | 30 |
| R02_FLEX_s43 | 0.183071 | 0.179927 | 0.016669 | 0.323297 | 0.026103 | 0.016359 | -0.002516 | 76850 | 1112.9 | 33 |
| R02_FLEX_s44 | 0.182979 | 0.179780 | 0.016782 | 0.345904 | 0.026102 | -0.014808 | -0.001048 | 76850 | 1152.3 | 33 |

FLEX EPE wins: 3/3 seeds. Improvement of mean EPE: 10.829%.

Nonzero-vy: 31 independent windows / 5 scenes, insufficient_data. Repeated seeds do not add samples.
Three seeds describe initialization sensitivity; no statistical-superiority claim. Costs are recorded, not selection gates.
Full JSON includes paired lateral acceptance, component metrics/bias, per-scene/subset/constant results, hashes and costs.
