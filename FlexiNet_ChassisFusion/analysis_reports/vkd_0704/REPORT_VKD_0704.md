# VKD 0704 Experiment Analysis

## Scope

This report compares the four completed velocity-xy experiments from the 0704 run:

- `vkd_gated`: Vision + Kinematics + Dynamics gated
- `kd_only`: Kinematics + Dynamics only
- `vision_only`: Vision only
- `vision_kinematics`: Vision + Kinematics

Inputs were read from each method's `best_eval` artifacts. Alignment check:

- Reference method: `vkd_gated`
- Samples: `33148`
- Max target difference across aligned methods: `0`

This is a single-seed descriptive comparison. It should not be worded as a statistically significant result.

## Main Result

Best method by `vector_endpoint_error_mps`: `vkd_gated`.

`vkd_gated` improves vector endpoint error over:

- `kd_only` by 0.00121 m/s (1.60% lower than kd_only)
- `vision_kinematics` by 0.00187 m/s (2.46% lower than vision_kinematics)
- `vision_only` by 1.01475 m/s (93.17% lower than vision_only)

| method | vector EE | MAE vx | MAE vy | R2 vx | R2 vy | gain vs base EE |
| --- | --- | --- | --- | --- | --- | --- |
| vkd_gated | 0.07438 | 0.06875 | 0.01782 | 0.99919 | 0.50826 | 0.11898 |
| kd_only | 0.07559 | 0.06995 | 0.01784 | 0.99916 | 0.50384 | 0.11777 |
| vision_only | 1.08914 | 1.08815 | 0.02150 | 0.81050 | 0.31053 | -0.89578 |
| vision_kinematics | 0.07625 | 0.07049 | 0.01815 | 0.99913 | 0.47718 | 0.11710 |

## Scenario Breakdown

These scenarios are computed from the aligned validation samples using the same masks for all methods.

| scenario | n | vkd EE | kd EE | vision+k EE | vision EE | vkd vs vision+k |
| --- | --- | --- | --- | --- | --- | --- |
| turning | 3433 | 0.08652 | 0.08759 | 0.08959 | 1.05679 | 0.00307 |
| high_lateral | 3316 | 0.10875 | 0.10891 | 0.11442 | 1.64905 | 0.00567 |
| wheel_inconsistent | 3315 | 0.08503 | 0.08576 | 0.08815 | 1.13683 | 0.00312 |
| hard_accel | 3315 | 0.08783 | 0.09194 | 0.09194 | 1.02069 | 0.00411 |
| high_lat_accel | 3315 | 0.08660 | 0.08776 | 0.08828 | 1.17717 | 0.00168 |

## Gate Analysis

`vkd_gated` gate weights show how the confidence-gated fusion distributes trust across branches.

| scenario | gate vision | gate kinematic | gate dynamics |
| --- | --- | --- | --- |
| turning | 0.6008 | 0.2321 | 0.1671 |
| high_lateral | 0.5816 | 0.2202 | 0.1982 |
| wheel_inconsistent | 0.5973 | 0.2342 | 0.1685 |
| hard_accel | 0.6124 | 0.1981 | 0.1894 |
| high_lat_accel | 0.6008 | 0.2297 | 0.1695 |

## Generated Figures

- Figure generation was skipped or unavailable in this environment.

## Output Files

- `comparison_summary.csv`: main metric comparison.
- `scenario_comparison.csv`: per-scenario comparison with common masks.
- `gate_summary.csv`: gate weights by scenario for methods that expose gate columns.
- `error_percentiles.csv`: endpoint-error tail behavior.
- `bad_case_overlap.csv`: overlap of top bad cases across methods.

## Interpretation Guardrails

- Use `vector_endpoint_error_mps` as the primary ranking metric.
- Use `mae_vx_mps` and `mae_vy_mps` to explain whether gains come from longitudinal or lateral velocity.
- Use `turning`, `high_lateral`, `wheel_inconsistent`, `hard_accel`, and `high_lat_accel` for mechanism-oriented claims.
- Do not claim significance without repeated seeds.
