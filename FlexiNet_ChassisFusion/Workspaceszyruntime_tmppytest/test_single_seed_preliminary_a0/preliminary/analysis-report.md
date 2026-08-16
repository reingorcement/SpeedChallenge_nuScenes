# V3 Visual-Only Analysis Report

## Analysis Question
Does pure-visual V3 fusion improve velocity estimation over its direct visual expert on the independent `stage1_test` split?

## Evidence
- Analysis mode: Preliminary single-seed validation
- Seeds: 42
- Experimental unit: scene
- Paired scenes: 2
- Bootstrap samples: 200
- Fusion minus direct EPE: -0.515388 m/s
- 95% scene-bootstrap CI: [-0.515388, -0.515388] m/s

## Interpretation
For this seed, fusion improves scene-level EPE over direct visual estimation on this split.

## Claim Candidate
- Allowed wording: For seed 42 on this fixed split, fusion reduced EPE versus its paired direct baseline.
- Forbidden stronger wording: the method is generally superior on nuScenes or other datasets.
- Uncertainty: One training seed cannot quantify training-seed variability; the interval only resamples scenes conditional on seed 42.
- Decision: preliminary_keep

## Limitations
- This report compares fusion only with its strict direct visual baseline.
- Smoke results are excluded from the analysis.
- No CAN, wheel-speed, IMU, kinematic, or dynamic input is evaluated.
- This single-seed result cannot establish robustness to training initialization.
