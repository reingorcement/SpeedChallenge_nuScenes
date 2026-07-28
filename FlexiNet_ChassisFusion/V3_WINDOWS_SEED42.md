# Windows V3 Seed-42 Validation

This run validates the strict visual-only V3 pipeline on the Windows RTX 5090 server. It trains one paired initialization only: seed 42. Its holdout report is therefore preliminary and must not be presented as evidence of training-seed stability.

## What the Two Stages Do

`direct_pretrain` trains the direct visual expert from RGB frame sequences only. It learns to predict longitudinal and lateral ego velocity without CAN, wheel speed, IMU, kinematic, or dynamic inputs. It also avoids loading optical-flow packages, so it is the cheaper first stage.

`fusion` starts from the matching direct checkpoint. It retains the direct expert and learns when to correct it with optical-flow-based geometric motion estimates. Fusion needs the cached flow packages and trains the reliability, correction, gate, and uncertainty heads. The paired comparison on `stage1_test` answers whether these geometric cues help this one direct initialization.

## Server Setup

The server inputs for this run are:

- Conda environment: `zy_speedest`
- Dataset root: `D:\Workspaces\zy\nuscenes`
- Branch: `feature/v3-visual-only`
- Repository directory: `D:\Workspaces\zy\code\zhangyu\09_SpeedChallenge_nuScenes_worktrees\v3-visual-only`

The screenshots show Anaconda Prompt (`cmd.exe`), not PowerShell. The commands below therefore use Command Prompt syntax. The target directory is currently empty; clone the branch into that folder with:

```batch
cd /d D:\Workspaces\zy\code\zhangyu\09_SpeedChallenge_nuScenes_worktrees\v3-visual-only
git clone --branch feature/v3-visual-only --single-branch git@github.com:reingorcement/SpeedChallenge_nuScenes.git .
```

Activate the training environment and verify its CUDA and test dependencies:

```batch
conda activate zy_speedest
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest --version
```

If the final command reports that `pytest` is absent, install it into `zy_speedest` once:

```batch
python -m pip install pytest
```

Before launching, the dataset must contain both cache paths below. The script validates them before it trains:

```text
D:\Workspaces\zy\nuscenes\precomputed_flow_v2\CAM_FRONT
D:\Workspaces\zy\nuscenes\precomputed_frames\CAM_FRONT_72x128_rgb
```

## Start the Validation

Run this from Anaconda Prompt after activating `zy_speedest`:

```batch
cd /d D:\Workspaces\zy\code\zhangyu\09_SpeedChallenge_nuScenes_worktrees\v3-visual-only\FlexiNet_ChassisFusion
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_v3_visual_only_seed42.ps1 -DataRoot D:\Workspaces\zy\nuscenes -RunTag v3_visual_only_seed42_20260728
```

The command first runs V3 unit tests and a two-batch direct-to-fusion smoke chain. It then runs the full 20-epoch direct stage, the full 20-epoch fusion stage, frozen `stage1_test` evaluation, and a scene-bootstrap comparison conditional on seed 42.

For a separate minimized training window, use this from the same activated Anaconda Prompt:

```batch
start "V3 seed42" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_v3_visual_only_seed42.ps1 -DataRoot D:\Workspaces\zy\nuscenes -RunTag v3_visual_only_seed42_20260728
```

Use a unique run tag for a new experiment. To continue an interrupted experiment, use exactly the same run tag. The script skips completed stages and resumes an incomplete stage from its `last.pth` checkpoint.

## Monitor and Read Results

```batch
powershell -NoProfile -Command "Get-Content .\logs\v3_visual_only_seed42_20260728.log -Wait -Tail 40"
```

The main outputs are ignored by Git and remain on the server:

```text
checkpoints\v3_visual_only_seed42_20260728\direct_seed_42\best.pth
checkpoints\v3_visual_only_seed42_20260728\fusion_seed_42\best.pth
outputs\v3_visual_only_seed42_20260728\stage1_test\direct_seed_42\metrics.json
outputs\v3_visual_only_seed42_20260728\stage1_test\fusion_seed_42\metrics.json
outputs\v3_visual_only_seed42_20260728\stage1_test_analysis_preliminary\analysis-report.md
```

The preliminary report compares fusion with direct on the independent `stage1_test` scenes. Its scene-bootstrap interval resamples scenes only; it does not measure variation across random training seeds. Run seeds 3407 and 2026 before making a formal multi-seed claim.
