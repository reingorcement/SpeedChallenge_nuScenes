from __future__ import annotations

import math
import subprocess
import sys

import torch

from dataset.nuscenes_chassis import (
    FAIR_CHASSIS_DIM,
    NuScenesCanBusAdapter,
    NuScenesChassisFusionDataset,
)
from model.fusion_model import FlexiNetChassisFusion
from utils.evaluation import compute_metrics


class FakeNuScenes:
    def __init__(self) -> None:
        self.records = {
            ("sample_data", "prev_sd"): {
                "ego_pose_token": "prev_pose",
                "timestamp": 0,
            },
            ("sample_data", "center_sd"): {
                "prev": "prev_sd",
                "next": "next_sd",
                "ego_pose_token": "center_pose",
                "timestamp": 1_000_000,
            },
            ("sample_data", "next_sd"): {
                "ego_pose_token": "next_pose",
                "timestamp": 2_000_000,
            },
            ("ego_pose", "prev_pose"): {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 0,
            },
            ("ego_pose", "center_pose"): {
                "translation": [0.0, 1.0, 0.0],
                "rotation": [
                    math.cos(math.pi / 4.0),
                    0.0,
                    0.0,
                    math.sin(math.pi / 4.0),
                ],
                "timestamp": 1_000_000,
            },
            ("ego_pose", "next_pose"): {
                "translation": [0.0, 2.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 2_000_000,
            },
        }

    def get(self, table: str, token: str) -> dict:
        return self.records[(table, token)]


def test_velocity_label_uses_center_difference_in_ego_frame() -> None:
    dataset = object.__new__(NuScenesChassisFusionDataset)
    dataset.nusc = FakeNuScenes()

    velocity = dataset._compute_velocity_label("center_sd")

    torch.testing.assert_close(velocity, torch.tensor([1.0, 0.0]), atol=1.0e-6, rtol=1.0e-6)


def test_fair_chassis_excludes_pose_derived_values(tmp_path) -> None:
    adapter = NuScenesCanBusAdapter(tmp_path)
    records = {
        "vehicle_monitor": {
            "vehicle_speed": 36.0,
            "yaw_rate": 5.0,
            "throttle": 100.0,
            "brake": 0.0,
            "steering": 10.0,
        },
        "zoe_veh_info": {
            "FL_wheel_speed": 10_000.0,
            "FR_wheel_speed": 10_100.0,
            "RL_wheel_speed": 9_900.0,
            "RR_wheel_speed": 10_050.0,
            "odom_speed": 35.0,
            "longitudinal_accel": 0.2,
            "transversal_accel": 0.1,
        },
        "steeranglefeedback": {"value": 7.7},
        "pose": {
            "vel": [99.0, 99.0, 0.0],
            "accel": [88.0, 88.0, 0.0],
            "rotation_rate": [0.0, 0.0, 77.0],
        },
    }
    adapter._select_latest_before = lambda _scene, message, _timestamp: records.get(message)  # type: ignore[method-assign]

    feature_vec, base_vx = adapter.extract_feature_vector(
        "scene-0001",
        1_000_000,
        chassis_mode="fair_chassis",
    )

    assert feature_vec.shape == (FAIR_CHASSIS_DIM,)
    assert base_vx == 10.0
    assert 77.0 not in feature_vec.tolist()
    assert 88.0 not in feature_vec.tolist()
    assert 99.0 not in feature_vec.tolist()


def test_fusion_model_outputs_two_dimensional_velocity_with_optional_branches() -> None:
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=8,
        hidden_dim=32,
        output_dim=2,
        fusion_mode="residual",
        chassis_sequence_model="gru",
        flow_input_channels=2,
        flow_feature_dim=8,
        geometry_input_dim=4,
        geometry_feature_dim=4,
    )
    model.eval()

    images = torch.randn(2, 3, 1, 32, 32)
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)
    base_velocity = torch.tensor([[5.0, 0.0], [2.0, 0.0]])
    flow = torch.randn(2, 2, 2, 32, 32)
    geometry = torch.randn(2, 4)

    with torch.no_grad():
        pred = model(
            images,
            chassis,
            base_velocity=base_velocity,
            flow=flow,
            geometry=geometry,
        )

    assert pred.shape == (2, 2)


def test_compute_metrics_reports_vector_velocity_errors() -> None:
    metrics = compute_metrics(
        preds=[[1.0, 0.2], [2.0, -0.1]],
        targets=[[1.5, 0.0], [1.5, 0.1]],
        base_velocities=[[1.0, 0.0], [2.0, 0.0]],
        loss=0.3,
    )

    assert metrics["mae_vx_mps"] == 0.5
    assert metrics["mae_vy_mps"] == 0.2
    assert metrics["base_mae_vy_mps"] == 0.05
    assert metrics["loss"] == 0.3
    assert "vector_endpoint_error_mps" in metrics


def test_pose_assisted_geometry_prior_uses_base_vx_feature() -> None:
    dataset = object.__new__(NuScenesChassisFusionDataset)
    dataset.geometry_mode = "basic"
    center = torch.zeros(22)
    center[2] = 0.1
    center[5] = 0.2
    center[6] = 0.3
    center[15] = 12.0
    center[-1] = 99.0

    geometry = dataset._compute_geometry_prior(center.unsqueeze(0))

    torch.testing.assert_close(geometry, torch.tensor([0.1, 0.2, 0.3, 12.0]), atol=1.0e-6, rtol=1.0e-6)


def test_eovm_scripts_can_show_help_without_package_install() -> None:
    for script in ["scripts/audit_eovm_inputs.py", "scripts/precompute_eovm_base_velocity.py"]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
