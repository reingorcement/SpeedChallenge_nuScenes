[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DataRoot,

    [ValidateNotNullOrEmpty()]
    [string]$RunTag = "v3_visual_only_seed42_$(Get-Date -Format 'yyyyMMdd_HHmmss')",

    [ValidateRange(1, 1024)]
    [int]$DirectBatchSize = 16,

    [ValidateRange(1, 1024)]
    [int]$FusionBatchSize = 4,

    [ValidateRange(0, 64)]
    [int]$NumWorkers = 8,

    [ValidateRange(1, 16)]
    [int]$PrefetchFactor = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

$Seed = 42
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([IO.Path]::GetFileName($RunTag) -ne $RunTag) {
    throw "RunTag must be a directory name, not a path: $RunTag"
}

$Config = "configs\velocity_xy_v3_visual_only.yaml"
$FlowRoot = Join-Path $DataRoot "precomputed_flow_v2\CAM_FRONT"
$FrameRoot = Join-Path $DataRoot "precomputed_frames\CAM_FRONT_72x128_rgb"
$RunRoot = Join-Path "outputs" $RunTag
$CheckpointRoot = Join-Path "checkpoints" $RunTag
$LogPath = Join-Path "logs" "$RunTag.log"
$AnalysisRoot = Join-Path $RunRoot "stage1_test_analysis_preliminary"

function Assert-PathExists {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path does not exist: $Path"
    }
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: python $($Arguments -join ' ')"
    }
}

function Test-CheckpointComplete {
    param([Parameter(Mandatory = $true)][string]$Checkpoint)

    $checkOutput = & python "scripts\v3_pipeline_checks.py" checkpoint `
        --checkpoint $Checkpoint --config $Config 2>&1
    if ($LASTEXITCODE -eq 0) {
        return $true
    }
    if ($LASTEXITCODE -eq 3) {
        return $false
    }
    throw "Could not inspect checkpoint $Checkpoint`: $($checkOutput -join [Environment]::NewLine)"
}

function Invoke-TrainingStage {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct_pretrain", "fusion")]
        [string]$Stage,
        [Parameter(Mandatory = $true)][string]$SaveDir,
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [Parameter(Mandatory = $true)][int]$BatchSize,
        [string]$InitialCheckpoint = ""
    )

    $lastCheckpoint = Join-Path $SaveDir "last.pth"
    $bestCheckpoint = Join-Path $SaveDir "best.pth"
    $arguments = @(
        "train_v3.py",
        "--config", $Config,
        "--stage", $Stage,
        "--seed", "$Seed",
        "--save_dir", $SaveDir,
        "--output_dir", $OutputDir,
        "--dataroot", $DataRoot,
        "--flow_root", $FlowRoot,
        "--frame_cache_root", $FrameRoot,
        "--batch_size", "$BatchSize",
        "--num_workers", "$NumWorkers",
        "--prefetch_factor", "$PrefetchFactor"
    )
    if ($InitialCheckpoint) {
        $arguments += @("--init_checkpoint", $InitialCheckpoint)
    }
    if (Test-Path -LiteralPath $lastCheckpoint) {
        if (Test-CheckpointComplete -Checkpoint $lastCheckpoint) {
            Assert-PathExists -Path $bestCheckpoint
            Write-Host "Skipping completed $Stage seed ${Seed}: $bestCheckpoint"
            return
        }
        $arguments += @("--resume", $lastCheckpoint)
        Write-Host "Resuming $Stage seed $Seed from $lastCheckpoint"
    }
    Invoke-Python -Arguments $arguments
    Assert-PathExists -Path $bestCheckpoint
}

New-Item -ItemType Directory -Force -Path "logs", $RunRoot, $CheckpointRoot | Out-Null
$transcriptStarted = $false
try {
    Start-Transcript -Path $LogPath -Append | Out-Null
    $transcriptStarted = $true

    Write-Host "Run tag: $RunTag"
    Write-Host "Dataset: $DataRoot"
    Write-Host "Flow cache: $FlowRoot"
    Write-Host "Frame cache: $FrameRoot"
    Write-Host "Runtime: direct_batch=$DirectBatchSize, fusion_batch=$FusionBatchSize, workers=$NumWorkers, prefetch=$PrefetchFactor"

    foreach ($requiredPath in @(
        $DataRoot,
        $FlowRoot,
        $FrameRoot,
        $Config,
        "logs\v2_split_manifest.json"
    )) {
        Assert-PathExists -Path $requiredPath
    }

    Invoke-Python -Arguments @(
        "scripts\v3_pipeline_checks.py", "environment",
        "--expected_conda_env", "zy_speedest"
    )
    Invoke-Python -Arguments @("-m", "pytest", "tests\test_v3_visual_only.py", "-q")

    $smokeRoot = Join-Path $RunRoot "smoke"
    $smokeCheckpointRoot = Join-Path $CheckpointRoot "smoke"
    $smokeMarker = Join-Path $smokeRoot "completed.json"
    if (-not (Test-Path -LiteralPath $smokeMarker)) {
        $smokeDirectSave = Join-Path $smokeCheckpointRoot "direct_seed_42"
        $smokeDirectOutput = Join-Path $smokeRoot "direct_seed_42"
        $smokeFusionSave = Join-Path $smokeCheckpointRoot "fusion_seed_42"
        $smokeFusionOutput = Join-Path $smokeRoot "fusion_seed_42"
        Invoke-Python -Arguments @(
            "train_v3.py", "--config", $Config, "--smoke",
            "--stage", "direct_pretrain", "--seed", "$Seed",
            "--save_dir", $smokeDirectSave, "--output_dir", $smokeDirectOutput,
            "--dataroot", $DataRoot, "--flow_root", $FlowRoot,
            "--frame_cache_root", $FrameRoot, "--batch_size", "$DirectBatchSize",
            "--num_workers", "$NumWorkers", "--prefetch_factor", "$PrefetchFactor"
        )
        $smokeDirectBest = Join-Path $smokeDirectSave "best.pth"
        Assert-PathExists -Path $smokeDirectBest
        Invoke-Python -Arguments @(
            "train_v3.py", "--config", $Config, "--smoke",
            "--stage", "fusion", "--seed", "$Seed",
            "--init_checkpoint", $smokeDirectBest,
            "--save_dir", $smokeFusionSave, "--output_dir", $smokeFusionOutput,
            "--dataroot", $DataRoot, "--flow_root", $FlowRoot,
            "--frame_cache_root", $FrameRoot, "--batch_size", "$FusionBatchSize",
            "--num_workers", "$NumWorkers", "--prefetch_factor", "$PrefetchFactor"
        )
        Assert-PathExists -Path (Join-Path $smokeFusionSave "best.pth")
        New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null
        @{ seed = $Seed; completed_at = (Get-Date).ToString("o") } |
            ConvertTo-Json | Set-Content -LiteralPath $smokeMarker -Encoding UTF8
    }
    else {
        Write-Host "Skipping completed seed-42 smoke chain: $smokeMarker"
    }

    $directCheckpointDir = Join-Path $CheckpointRoot "direct_seed_42"
    $directOutputDir = Join-Path $RunRoot "dev\direct_seed_42"
    $fusionCheckpointDir = Join-Path $CheckpointRoot "fusion_seed_42"
    $fusionOutputDir = Join-Path $RunRoot "dev\fusion_seed_42"
    Invoke-TrainingStage -Stage "direct_pretrain" -SaveDir $directCheckpointDir `
        -OutputDir $directOutputDir -BatchSize $DirectBatchSize
    $directBest = Join-Path $directCheckpointDir "best.pth"
    Invoke-TrainingStage -Stage "fusion" -SaveDir $fusionCheckpointDir `
        -OutputDir $fusionOutputDir -BatchSize $FusionBatchSize `
        -InitialCheckpoint $directBest
    $fusionBest = Join-Path $fusionCheckpointDir "best.pth"

    $stage1Root = Join-Path $RunRoot "stage1_test"
    Invoke-Python -Arguments @(
        "scripts\evaluate_v3_holdout.py", "--config", $Config,
        "--stage", "geometry_audit", "--seed", "$Seed",
        "--output_dir", (Join-Path $stage1Root "geometry"),
        "--dataroot", $DataRoot, "--flow_root", $FlowRoot,
        "--frame_cache_root", $FrameRoot
    )
    $directTestDir = Join-Path $stage1Root "direct_seed_42"
    Invoke-Python -Arguments @(
        "scripts\evaluate_v3_holdout.py", "--config", $Config,
        "--stage", "direct_pretrain", "--seed", "$Seed",
        "--checkpoint", $directBest, "--output_dir", $directTestDir,
        "--dataroot", $DataRoot, "--flow_root", $FlowRoot,
        "--frame_cache_root", $FrameRoot
    )
    $fusionTestDir = Join-Path $stage1Root "fusion_seed_42"
    Invoke-Python -Arguments @(
        "scripts\evaluate_v3_holdout.py", "--config", $Config,
        "--stage", "fusion", "--seed", "$Seed",
        "--checkpoint", $fusionBest, "--output_dir", $fusionTestDir,
        "--dataroot", $DataRoot, "--flow_root", $FlowRoot,
        "--frame_cache_root", $FrameRoot
    )
    Invoke-Python -Arguments @(
        "scripts\analyze_v3_results.py", "--preliminary", "--seeds", "$Seed",
        "--direct", (Join-Path $directTestDir "predictions.csv"),
        "--fusion", (Join-Path $fusionTestDir "predictions.csv"),
        "--samples", "10000", "--output_dir", $AnalysisRoot
    )

    Write-Host "V3 seed-42 validation completed successfully."
    Write-Host "Checkpoints: $CheckpointRoot"
    Write-Host "Holdout predictions: $stage1Root"
    Write-Host "Preliminary report: $(Join-Path $AnalysisRoot 'analysis-report.md')"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
