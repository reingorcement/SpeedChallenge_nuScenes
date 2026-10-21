# Statistical Appendix

All differences are fusion minus direct; negative errors favor fusion.
Scenes, rather than adjacent frames, are the resampling units.

| Metric | Difference | 95% CI | Paired Cohen dz | Significant improvement |
|---|---:|---:|---:|:---:|
| mae_vx_mps | -0.219945 | [-0.270559, -0.172680] | -1.0172 | True |
| mae_vy_mps | -0.000464 | [-0.000898, 0.000005] | -0.2305 | False |
| epe_mps | -0.218659 | [-0.268996, -0.171584] | -1.0156 | True |

The paired Cohen dz is computed from per-scene differences averaged across seeds.
This preliminary interval resamples scenes conditional on seed 42 and does not measure training-seed variability.