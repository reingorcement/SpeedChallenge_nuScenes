# Statistical Appendix

All differences are fusion minus direct; negative errors favor fusion.
Scenes, rather than adjacent frames, are the resampling units.

| Metric | Difference | 95% CI | Paired Cohen dz | Significant improvement |
|---|---:|---:|---:|:---:|
| mae_vx_mps | -0.500000 | [-0.500000, -0.500000] | nan | True |
| mae_vy_mps | -0.125000 | [-0.125000, -0.125000] | nan | True |
| epe_mps | -0.515388 | [-0.515388, -0.515388] | nan | True |

The paired Cohen dz is computed from per-scene differences averaged across seeds.
This preliminary interval resamples scenes conditional on seed 42 and does not measure training-seed variability.