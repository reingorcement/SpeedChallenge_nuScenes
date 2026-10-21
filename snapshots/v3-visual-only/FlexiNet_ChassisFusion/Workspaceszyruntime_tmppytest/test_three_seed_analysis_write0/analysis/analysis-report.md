# V3 Visual-Only Analysis Report

## Analysis Question
Does pure-visual V3 fusion improve velocity estimation over its direct visual expert on the independent `stage1_test` split?

## Evidence
- Analysis mode: Formal paired multi-seed analysis
- Seeds: 42, 3407, 2026
- Experimental unit: scene
- Paired scenes: 2
- Bootstrap samples: 200
- Fusion minus direct EPE: -0.515371 m/s
- 95% scene-bootstrap CI: [-0.515371, -0.515371] m/s

## Interpretation
Fusion improves scene-level EPE over direct visual estimation.

## Claim Candidate
- Allowed wording: Fusion reduced EPE versus the paired direct baseline on this split.
- Forbidden stronger wording: the method is generally superior on nuScenes or other datasets.
- Uncertainty: Paired training seeds quantify limited seed variability; the interval resamples scenes on this fixed split.
- Decision: keep

## Limitations
- This report compares fusion only with its strict direct visual baseline.
- Smoke results are excluded from the analysis.
- No CAN, wheel-speed, IMU, kinematic, or dynamic input is evaluated.
- The paired seeds provide limited, not exhaustive, initialization evidence.
