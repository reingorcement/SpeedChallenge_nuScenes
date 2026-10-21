# Task Plan: GPU Training Optimization

## Goal
Find practical settings and code-level options to increase GPU utilization and VRAM usage for `FlexiNet_ChassisFusion` training without changing model semantics.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Compare current and previous worktree training settings
- [x] Phase 3: Identify safe config-only accelerations
- [x] Phase 4: Identify optional code changes and verification commands
- [x] Phase 5: Deliver recommendations and wait for user confirmation before edits

## Key Questions
1. Which previous-version settings already improved throughput?
2. Is the current bottleneck likely GPU compute, CPU/DataLoader, image decoding, or evaluation frequency?
3. Which changes can safely increase VRAM usage without changing data format or training objective?

## Decisions Made
- Start with read-only comparison and recommendations.
- Do not modify training code or configs until the user confirms which optimization tier to apply.
- The previous worktree acceleration settings are mostly already present in the current branch.
- Recommend starting with a fast VKD config and profiling batch/workers before code changes.
- Added full-server nuScenes VKD usage documentation, including dynamics/EOVM run conditions.

## Errors Encountered
- None yet.

## Status
**Completed** - Recommendations are saved in `gpu_training_optimization_recommendations.md`; full server usage documentation is saved in `服务器全量nuScenes_VKD运行说明.md`.
