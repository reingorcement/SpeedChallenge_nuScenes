# Evaluation report (val)

- Total samples: **33148**
- Source eval dir: `outputs/fair_rgb_flow_geom/eval_last`

## Overall metrics
| Metric | Value |
| --- | --- |
| `loss` | 0.0012 |
| `mae_vx_mps` | 0.0688 |
| `mae_vy_mps` | 0.0176 |
| `rmse_vx_mps` | 0.1007 |
| `rmse_vy_mps` | 0.0258 |
| `vector_endpoint_error_mps` | 0.0744 |
| `speed_magnitude_mae_mps` | 0.0684 |
| `r2_vx` | 0.9992 |
| `r2_vy` | 0.5187 |
| `base_mae_vx_mps` | 0.1833 |
| `base_mae_vy_mps` | 0.0382 |
| `gain_vs_base_vx_mps` | 0.1144 |
| `gain_vs_base_vy_mps` | 0.0206 |
| `gain_vs_base_vector_mps` | 0.1190 |
| `low_speed_vector_endpoint_error_mps` | 0.0316 |
| `mid_speed_vector_endpoint_error_mps` | 0.0859 |
| `high_speed_vector_endpoint_error_mps` | 0.0991 |
| `turning_vector_endpoint_error_mps` | 0.0868 |
| `straight_vector_endpoint_error_mps` | 0.0730 |
| `high_lateral_vector_endpoint_error_mps` | 0.1090 |

## Performance by speed bin
| Speed bin (m/s) | Samples | Endpoint err | Base endpoint err | Gain | MAE vx | MAE vy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-2 | 7710 | 0.0316 | 0.0850 | 0.0534 | 0.0298 | 0.0052 |
| 2-5 | 6833 | 0.0779 | 0.2421 | 0.1642 | 0.0730 | 0.0162 |
| 5-10 | 15764 | 0.0894 | 0.2170 | 0.1276 | 0.0822 | 0.0226 |
| 10-15 | 2841 | 0.0991 | 0.2393 | 0.1402 | 0.0904 | 0.0269 |
| 15+ | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Worst 10 scenes (by mean endpoint error)
| Rank | Scene | Samples | Endpoint err | Base endpoint err | Gain |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | scene-0778 | 223 | 0.1986 | 0.2974 | 0.0988 |
| 2 | scene-0092 | 213 | 0.1462 | 0.2232 | 0.0771 |
| 3 | scene-0035 | 217 | 0.1456 | 0.2790 | 0.1334 |
| 4 | scene-0109 | 219 | 0.1321 | 0.1977 | 0.0657 |
| 5 | scene-0633 | 224 | 0.1288 | 0.1833 | 0.0545 |
| 6 | scene-0630 | 225 | 0.1264 | 0.3081 | 0.1817 |
| 7 | scene-0278 | 215 | 0.1257 | 0.3047 | 0.1790 |
| 8 | scene-0783 | 224 | 0.1161 | 0.2823 | 0.1662 |
| 9 | scene-0802 | 225 | 0.1156 | 0.2442 | 0.1285 |
| 10 | scene-0106 | 217 | 0.1147 | 0.2113 | 0.0966 |

## Worst 10 samples (by endpoint error)
| Rank | Scene | Endpoint err | Target speed | Pred (vx, vy) | Target (vx, vy) |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | scene-0778 | 1.4991 | 0.6762 | (0.823, -0.017) | (-0.676, -0.001) |
| 2 | scene-0778 | 1.4726 | 0.7266 | (0.746, -0.013) | (-0.727, 0.002) |
| 3 | scene-0778 | 1.3828 | 0.6672 | (0.716, -0.014) | (-0.667, -0.002) |
| 4 | scene-0778 | 1.3068 | 0.6135 | (0.693, -0.013) | (-0.614, -0.005) |
| 5 | scene-0778 | 1.2922 | 0.5839 | (0.708, -0.012) | (-0.584, 0.001) |
| 6 | scene-0778 | 1.2762 | 0.5881 | (0.688, -0.011) | (-0.588, 0.001) |
| 7 | scene-0778 | 1.2116 | 0.5345 | (0.677, -0.011) | (-0.534, -0.004) |
| 8 | scene-0778 | 1.1245 | 0.4812 | (0.647, 0.003) | (-0.475, 0.074) |
| 9 | scene-0778 | 1.0928 | 0.6241 | (0.471, 0.002) | (-0.619, 0.079) |
| 10 | scene-0778 | 0.9802 | 0.5826 | (0.397, -0.016) | (-0.582, 0.019) |

## Files
- `val_bad_cases_top50.csv` — top-50 bad cases
- `val_per_scene_metrics.csv` — per-scene aggregated metrics
- `val_per_speed_bin_metrics.csv` — per-speed-bin aggregated metrics
