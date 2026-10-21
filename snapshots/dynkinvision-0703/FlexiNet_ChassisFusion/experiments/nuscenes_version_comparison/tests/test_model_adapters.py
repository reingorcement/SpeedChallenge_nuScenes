from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from torch import nn

from experiments.nuscenes_version_comparison.adapters.legacy_adapter import (
    BaseVelocityAdapter,
    LegacyAdapter,
    build_legacy_adapter,
)
from experiments.nuscenes_version_comparison.analyze_comparison import analyze, paired_scene_bootstrap
from experiments.nuscenes_version_comparison.bootstrap import (
    SOURCE_PROJECT_ROOT,
    SOURCE_ROOT_ENV,
    compatibility_report,
)
from experiments.nuscenes_version_comparison.config import load_resolved_config
from experiments.nuscenes_version_comparison.train_comparison import _criterion


class FakeLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_gate_weights = None
        self.last_gate_names = []

    def forward(self, images, chassis, base_velocity=None, **_kwargs):
        self.last_gate_weights = images.new_tensor([[0.4, 0.6]]).expand(images.shape[0], -1)
        self.last_gate_names = ["kinematic", "dynamics"]
        return base_velocity + self.anchor


def test_legacy_and_reference_adapters_share_prediction_contract() -> None:
    batch = {
        "images": torch.zeros(2, 5, 3, 8, 8),
        "chassis": torch.zeros(2, 21, 16),
        "base_velocity": torch.tensor([[3.0, 0.0], [4.0, 0.0]]),
        "chassis_current_index": torch.tensor([20, 20]),
    }
    legacy_prediction, diagnostics = LegacyAdapter(FakeLegacy(), "legacy_kd")(batch)
    base_prediction, _ = BaseVelocityAdapter()(batch)
    torch.testing.assert_close(legacy_prediction, base_prediction)
    assert diagnostics["gate_names"] == ["kinematic", "dynamics"]


@pytest.mark.parametrize("algorithm", ["legacy_kd", "legacy_vkd"])
def test_algorithm_configs_resolve_to_causal_fair_inputs(algorithm: str) -> None:
    config, _, _ = load_resolved_config(algorithm)
    assert config["data"]["online_mode"] is True
    assert config["data"]["seq_len"] == 5
    assert config["data"].get("chassis_mode", "fair_chassis") == "fair_chassis"


@pytest.mark.parametrize(
    ("algorithm", "relative"),
    [
        ("legacy_kd", "checkpoints/velocity_xy_kd_only/best.pth"),
        ("legacy_vkd", "checkpoints/velocity_xy_vkd_gated/best.pth"),
    ],
)
def test_existing_legacy_checkpoints_load_strictly(algorithm: str, relative: str) -> None:
    path = SOURCE_PROJECT_ROOT / relative
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    config["_comparison"] = {"name": algorithm}
    adapter = build_legacy_adapter(config)
    adapter.load_checkpoint(path)
    channels = 3 if config["data"].get("image_mode") == "rgb" else 1
    batch = {
        "images": torch.zeros(
            1,
            int(config["data"]["seq_len"]),
            channels,
            int(config["data"]["img_height"]),
            int(config["data"]["img_width"]),
        ),
        "chassis": torch.zeros(1, 21, 16),
        "base_velocity": torch.zeros(1, 2),
        "chassis_current_index": torch.tensor([20]),
    }
    prediction, _ = adapter(batch)
    assert prediction.shape == (1, 2)
    assert _criterion(config) is not None


def test_scene_bootstrap_uses_negative_delta_as_improvement() -> None:
    result = paired_scene_bootstrap({"a": -0.2, "b": -0.1, "c": -0.3}, samples=1000, seed=7)
    assert result["ci95_high_mps"] < 0.0
    assert result["conclusion"] == "clear_improvement"


def test_analysis_runs_end_to_end_for_four_algorithms_and_three_seeds() -> None:
    errors = {
        "legacy_kd": 0.30,
        "legacy_vkd": 0.25,
        "v2_visual": 0.22,
        "v2_visual_kinematic": 0.18,
    }
    rows = []
    for algorithm, clean_error in errors.items():
        for seed in (42, 3407, 2026):
            for variant, increment in (("clean", 0.0), ("can_missing", 0.1)):
                for scene_index in range(2):
                    error = clean_error + increment + scene_index * 0.01
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "seed": str(seed),
                            "variant": variant,
                            "scene_name": f"scene-{scene_index}",
                            "center_token": f"token-{scene_index}",
                            "target_vx_mps": "5.0",
                            "target_vy_mps": "0.1",
                            "pred_vx_mps": str(5.0 + error),
                            "pred_vy_mps": "0.1",
                            "error_vx_mps": str(error),
                            "error_vy_mps": "0.0",
                            "endpoint_error_mps": str(error),
                            "yaw_rate_radps": "0.2",
                            "steering_feedback": "0.3",
                            "longitudinal_accel_mps2": "1.0",
                            "lateral_accel_mps2": "0.5",
                            "throttle": "0.2",
                            "brake": "0.0",
                            "wheel_relative_std": "0.02",
                        }
                    )
    thresholds = {
        "turning_yaw_rate_radps": 0.1,
        "high_lateral_velocity_mps": 0.05,
        "wheel_inconsistent_relative_std": 0.01,
        "hard_accel_mps2": 0.8,
        "high_lateral_accel_mps2": 0.4,
        "high_steering_feedback": 0.2,
        "high_throttle": 0.1,
        "braking": 0.1,
    }
    report, metrics, stress, deltas, bad_cases = analyze(
        rows,
        thresholds,
        {},
        bootstrap_samples=100,
        bootstrap_seed=42,
    )
    assert len(report["aggregate_clean"]) == 4
    assert report["paired_scene_bootstrap"][
        "v2_visual_kinematic_minus_legacy_vkd:clean"
    ]["conclusion"] == "clear_improvement"
    assert report["paired_scene_bootstrap"][
        "v2_visual_minus_legacy_vkd:clean"
    ]["conclusion"] == "clear_improvement"
    assert report["stress_robustness"]
    assert metrics and stress and deltas and bad_cases


def test_compatibility_profiles_accept_legacy_and_reject_missing_v2(tmp_path) -> None:
    files = {
        "train.py": "",
        "dataset/nuscenes_chassis.py": "scene_names: chassis_current_index",
        "model/fusion_model.py": "chassis_current_index class FlexiNetChassisFusion",
        "scripts/path_sanity_check.py": "",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert compatibility_report(tmp_path, profile="legacy")["compatible"] is True
    report = compatibility_report(tmp_path, profile="full")
    assert report["compatible"] is False
    assert "train_v2.py" in report["missing_files"]


@pytest.mark.parametrize(
    "entrypoint",
    ["train_comparison.py", "evaluate_comparison.py", "analyze_comparison.py", "quick_evaluate_existing.py"],
)
def test_entrypoints_import_when_package_is_copied_to_legacy_host(
    tmp_path: Path, entrypoint: str
) -> None:
    source_package = Path(__file__).resolve().parents[1]
    legacy_package = tmp_path / "experiments" / "nuscenes_version_comparison"
    shutil.copytree(
        source_package,
        legacy_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    environment = dict(os.environ)
    environment[SOURCE_ROOT_ENV] = str(SOURCE_PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, str(legacy_package / entrypoint), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
