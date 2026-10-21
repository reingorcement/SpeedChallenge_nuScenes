# Notes: nuScenes Stage-1 Evaluation

## Experimental Design
- Independent unit: scene.
- Train/development scenes come from official nuScenes train.
- Stage-1 test and sealed final-system test scenes come from official nuScenes val.
- Frame-level metrics remain descriptive; uncertainty and paired comparisons resample scenes.

## Leakage Controls
- Training statistics fit on train_core only.
- Augmentation and weighted sampling apply to train_core only.
- Early stopping and checkpoint selection use dev only.
- Stage-1 test is evaluated after training; final_system_test is inaccessible to phase-1 commands.

## Planned Outputs
- Frozen split manifest with checksum.
- One predictions CSV per seed/mode/ablation.
- Metrics and per-scene CSV files.
- Scene-level paired bootstrap confidence intervals.
- Machine-readable acceptance JSON and human-readable Markdown report.

## Implemented Verification
- Deterministic manifest, checksum tamper detection, disjoint partitions, and sealed final split tests.
- Exact scene-name dataset selection and missing-scene failure tests.
- Prediction alignment and reproducible paired scene-bootstrap tests.
- Inference ablation behavior and missing-CAN mask tests.
- End-to-end synthetic three-seed stage-1 acceptance analysis.
- Local result: 97 tests passed; Python compilation and Bash syntax checks passed.
