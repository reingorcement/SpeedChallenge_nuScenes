from __future__ import annotations

from types import SimpleNamespace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from torch import nn

from experiments.nuscenes_version_comparison.adapters.legacy_adapter import LegacyAdapter
from experiments.nuscenes_version_comparison.adapters.v2_adapter import (
    BaseVelocityAdapter,
    V2Adapter,
)
from experiments.nuscenes_version_comparison.analyze_comparison import analyze, paired_scene_bootstrap
from experiments.nuscenes_version_comparison.bootstrap import (
    SOURCE_PROJECT_ROOT,
    SOURCE_ROOT_ENV,
    compatibility_report,
)
from experiments.nuscenes_version_comparison.config import load_resolved_config


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


class FakeV2(nn.Module):
    def forward(self, images, **_kwargs):
        batch = images.shape[0]
        velocity = images.new_zeros(batch, 2)
        scalar = images.new_zeros(batch)
        vector = images.new_zeros(batch, 2)
        return SimpleNamespace(
            predicted_velocity=velocity,
            visual_velocity=velocity,
            geometry_velocity=velocity,
            geometry_valid=torch.ones(batch, dtype=torch.bool),
            geometry_condition_number=scalar,
            fallback_used=torch.zeros(batch, dtype=torch.bool),
            wheel_scale=images.new_ones(batch, 1),
            residual_gate=images.new_zeros(batch, 1),
            kinematic_gate=vector,
            kinematic_applied_correction=vector,
        )


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


def test_v2_adapter_exposes_diagnostics() -> None:
    batch = {
        "images": torch.zeros(2, 5, 3, 8, 8),
        "angular_rate_camera": torch.zeros(2, 3),
        "angular_rate_valid": torch.ones(2, 3, dtype=torch.bool),
        "angular_rate_age_sec": torch.zeros(2),
        "wheel_speed": torch.zeros(2, 4),
        "wheel_speed_valid": torch.ones(2, 4, dtype=torch.bool),
        "wheel_speed_age_sec": torch.zeros(2, 4),
        "plane_normal_camera": torch.zeros(2, 3),
        "plane_offset": torch.ones(2),
        "rotation_ego_from_camera": torch.eye(3).repeat(2, 1, 1),
        "kinematic_values": torch.zeros(2, 21, 12),
        "kinematic_valid": torch.ones(2, 21, 12, dtype=torch.bool),
        "kinematic_age": torch.zeros(2, 21, 12),
    }
    prediction, diagnostics = V2Adapter(FakeV2(), "v2_visual")(batch)
    assert prediction.shape == (2, 2)
    assert diagnostics["geometry_valid"].dtype == torch.bool


@pytest.mark.parametrize("algorithm", ["legacy_kd", "legacy_vkd", "v2_visual", "v2_visual_kinematic"])
def test_algorithm_configs_resolve_to_causal_fair_inputs(algorithm: str) -> None:
    config, _, _ = load_resolved_config(algorithm)
    assert config["data"]["online_mode"] is True
    assert config["data"]["seq_len"] == 5
    assert config["data"].get("chassis_mode", "fair_chassis") == "fair_chassis"


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


def test_compatibility_report_rejects_a_pure_legacy_shape(tmp_path) -> None:
    for relative in ("train.py", "dataset/nuscenes_chassis.py", "model/fusion_model.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    report = compatibility_report(tmp_path)
    assert report["compatible"] is False
    assert "train_v2.py" in report["missing_files"]
    assert "model/v2/model.py" in report["missing_files"]


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
