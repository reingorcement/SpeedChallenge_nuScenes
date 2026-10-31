# R02_BASE / R02_FLEX — full train/dev, seed42, 40 epochs

Training commit: `a847938681d60ae86e0e293e24ffa7d2e74e20a9`

| Encoder | EPE | vx MAE | vy MAE | vx RMSE | vy RMSE | vx bias | vy bias | Params | Wall seconds | AMP skips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R02_BASE | 0.215142 | 0.212016 | 0.017229 | 0.414209 | 0.026374 | -0.018107 | -0.003579 | 74354 | 1215.5 | 31 |
| R02_FLEX | 0.191630 | 0.188479 | 0.016827 | 0.371284 | 0.026239 | -0.019759 | -0.002414 | 76850 | 1241.9 | 33 |

FLEX is promising for further seeds; retain BASE as the reference until lateral evidence is adequate

非零侧向：31窗口/5场景，insufficient_data。单seed，不宣称显著优越。
完整逐场景、常量基线、峰值显存、吞吐、哈希和初始化证据见同目录 summary.json；逐窗口预测在 experiments/ 对应run目录。
