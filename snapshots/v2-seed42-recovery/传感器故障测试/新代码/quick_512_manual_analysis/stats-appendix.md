# Statistical Appendix

## Integrity Audit

- Rows: 12,800 = 5 algorithms x 5 variants x 512 samples.
- Unique scenes: 75.
- Pairing key: `scene_name + center_token`.
- Sample keys and target velocity labels: exactly aligned.
- Scenario metadata aligned across adapters: False.
- Current visual-to-joint checkpoint comparison valid: False.
- Camera-missing geometry-invalid rows: 512/512.
- Camera-missing fallback-flag rows: 512/512.
- Fallback prediction consistent with base velocity: False.

## Methods

For each V2 joint fault, EPE is first averaged within each scene. Fault-minus-clean differences are then computed over 75 paired scene means. Two-sided Wilcoxon signed-rank tests use one family of four fault contrasts with Holm adjustment. Rank-biserial correlation is positive when the fault worsens EPE. Bootstrap intervals resample paired scene differences 10,000 times with seed 42.

The scene bootstrap quantifies scene-sampling uncertainty conditional on seed 42. It does not quantify training randomness. Tests on the 512 windows as independent observations are prohibited.

## V2 Joint Paired Fault Statistics

| variant | scene_count | scene_mean_degradation_mps | ci95_low_mps | ci95_high_mps | scene_worsening_rate | wilcoxon_p_raw | wilcoxon_p_holm | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| can_missing | 75 | 0.48762 | 0.33367 | 0.65216 | 0.85333 | 1.435e-09 | 4.305e-09 | 0.80421 |
| yaw_bias_pos | 75 | 0.00228 | 0.00084 | 0.00403 | 0.68000 | 0.00110 | 0.00220 | 0.43368 |
| wheel_scale_low | 75 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 1.00000 | 0.00000 |
| camera_missing | 75 | 7.28556 | 6.55545 | 8.05051 | 1.00000 | 5.280e-14 | 2.112e-13 | 1.00000 |

## Quarantined Statistics

- Current V2 visual versus joint: wrong visual checkpoint.
- Cross-algorithm turning/high-steering/high-acceleration groups: adapter CAN metadata differs for identical tokens.
- Wheel-scale-low robustness: perturbation is masked by the model's lower scale saturation.
