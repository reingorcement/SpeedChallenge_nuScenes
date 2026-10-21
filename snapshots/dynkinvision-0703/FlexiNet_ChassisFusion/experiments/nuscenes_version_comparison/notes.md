# Notes: Self-Contained nuScenes KD/VKD Comparison

## Repository Findings

- The active checkout is `7de2aad`; it has KD/VKD source and checkpoints but no V2 modules.
- The original experiment package imported V2 modules at startup, so every CLI failed in this checkout.
- Original `online_mode` made only the chassis window causal; image windows still targeted their center frame.
- KD-only has no visual backbone, so optimizer construction must support a missing backbone.
- Experiment code was written against newer model, criterion, and loss-call interfaces; the legacy path now uses only interfaces present in this checkout.
- Frame caches cannot be assumed on the server. Legacy evaluation can read raw nuScenes images.
- The experiment directory is untracked and must be included in the eventual Git commit.

## Decisions

- Default algorithms are `base_velocity`, `legacy_kd`, and `legacy_vkd`.
- V2 remains optional and is imported only under an explicit full profile.
- The experimental unit and bootstrap unit are scenes, not overlapping frame windows.
- The generated `final_system_test` partition remains sealed.
- Shared perturbations never modify the pose-derived target tensor.

## Verification

- The current checkout passes the legacy compatibility profile without an external source root.
- KD/VKD checkpoints load strictly and complete a synthetic forward/loss pass.
- Related experiment and legacy pipeline tests pass together.
- Python compile and all CLI import checks pass.
- Shell scripts are LF-only; native `bash -n` remains a Linux acceptance check because WSL is unavailable locally.
