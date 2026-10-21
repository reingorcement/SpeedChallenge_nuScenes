# Statistical Appendix

## Integrity Audit

- Rows: 7,680 = 3 algorithms x 5 variants x 512 samples.
- Unique evaluation scenes: 75.
- Pairing key: `scene_name + center_token`.
- Pairing: exact across every algorithm and variant.
- Velocity labels: invariant across algorithms and perturbations.
- Missing algorithm/variant cells: none.

## Methods

Windows from the same scene are correlated, so inferential comparisons first average EPE within each scene. VKD and KD are then compared using two-sided Wilcoxon signed-rank tests over 75 paired scene means. Five variant comparisons form one family and use Holm-adjusted p-values. Rank-biserial correlation is signed so negative values favor VKD. Bootstrap intervals resample the 75 scene-level paired deltas 10,000 times with seed 42.

No seed-level mean or standard deviation is reported because each method has only one retained checkpoint. Statistical significance across scenes must not be interpreted as reproducibility across training seeds.

## Paired Scene Statistics

| variant | scene_count | scene_mean_delta_mps | ci95_low_mps | ci95_high_mps | vkd_scene_win_rate | wilcoxon_p_raw | wilcoxon_p_holm | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 75 | -0.00193 | -0.00438 | 0.00062 | 0.49333 | 0.13646 | 0.44784 | -0.19789 |
| can_missing | 75 | -0.00156 | -0.00191 | -0.00119 | 0.73333 | 5.482e-10 | 2.741e-09 | -0.82456 |
| yaw_bias_pos | 75 | -0.00223 | -0.00465 | 0.00030 | 0.45333 | 0.11196 | 0.44784 | -0.21123 |
| wheel_scale_low | 75 | -0.00242 | -0.00539 | 0.00055 | 0.46667 | 0.14646 | 0.44784 | -0.19298 |
| camera_missing | 75 | -0.00041 | -0.00300 | 0.00235 | 0.41333 | 0.86997 | 0.86997 | -0.02175 |

## Interpretation Rules

- Negative VKD-KD delta favors VKD.
- A CI crossing zero blocks a directional mean-effect claim.
- Holm-adjusted p-values address the five reported variant contrasts.
- Statistical detectability under CAN removal does not imply practical importance; both absolute errors are catastrophic.
