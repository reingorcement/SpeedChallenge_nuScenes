# Notes: nuScenes Version Comparison

## Repository Findings

- `train_v2.py` exposes reusable config, dataset, model, move, and forward helpers.
- `NuScenesChassisFusionDataset` accepts explicit `scene_names`, causal windows, RGB images, and frame caches.
- V2 aliases `train`/`val` to `train_core`/`dev` and prevents access to sealed `final_system_test`.
- Legacy `train.py` has no split-manifest CLI, but its model, criterion, and batch-forward helpers are importable.
- Current V2 joint training requires visual training followed by kinematic pretraining before the final joint stage.
- V2 kinematic values are normalized; sensor perturbations must update raw and normalized values consistently.
- Existing stage-1 analysis already establishes scene-level paired bootstrap as the repository convention.
- Boston-only V2 runs use Boston-only normalization statistics. They use online geometry because the existing cache contract is tied to the global statistics hash.
- A pure `7de2aad` checkout lacks `train_v2.py`, `dataset/v2_*`, `model/v2/`, explicit scene selection, and causal current-index forwarding. The experiment folder can live there only when it points to a current source tree.

## Experiment Skeleton

### Goal
- Method: Legacy KD/VKD versus V2 visual/visual-kinematic velocity estimation.
- Hypothesis: V2 reduces clean and stressed scene-level endpoint error.
- Dataset: nuScenes v1.0-trainval with the checked V2 split manifest.
- Metric: Vector endpoint error with scene-level paired bootstrap.

### Verification
- Tiny smoke train/eval for each algorithm.
- Shape and token-alignment checks.
- Metric and deterministic perturbation sanity checks.
- Reproducible commands with seeds 42, 3407, and 2026.
