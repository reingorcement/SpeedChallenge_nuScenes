from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from dataset.nuscenes_chassis import (
    DYNAMICS_PROXY_FEATURE_NAMES,
    FAIR_CHASSIS_DIM,
    NuScenesCanBusAdapter,
    NuScenesChassisFusionDataset,
)
from model.fusion_model import FlexiNetChassisFusion
from scripts.path_sanity_check import build_model_from_config, load_config
from utils.evaluation import EvaluationResult, compute_metrics, save_predictions_csv


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


def test_direct_vision_only_model_does_not_require_kinematic_branch() -> None:
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        hidden_dim=32,
        output_dim=2,
        fusion_mode="direct",
        use_vision_branch=True,
        use_kinematic_branch=False,
        use_dynamics_branch=False,
    )
    model.eval()

    images = torch.randn(2, 3, 1, 32, 32)
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)

    with torch.no_grad():
        pred = model(images, chassis)

    assert pred.shape == (2, 2)


def test_residual_gated_model_tracks_active_branch_gate_weights() -> None:
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=8,
        hidden_dim=32,
        output_dim=2,
        fusion_mode="residual_gated",
        use_vision_branch=True,
        use_kinematic_branch=True,
        use_dynamics_branch=True,
        branch_feature_dim=16,
        dynamics_feature_dim=8,
        save_gate_weights=True,
    )
    model.eval()

    images = torch.randn(2, 3, 1, 32, 32)
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)
    base_velocity = torch.tensor([[5.0, 0.1], [2.0, -0.2]])

    with torch.no_grad():
        pred = model(images, chassis, base_velocity=base_velocity)

    assert pred.shape == (2, 2)
    assert model.last_gate_names == ["vision", "kinematic", "dynamics"]
    assert model.last_gate_weights is not None
    assert model.last_gate_weights.shape == (2, 3)
    torch.testing.assert_close(
        model.last_gate_weights.sum(dim=1),
        torch.ones(2),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_residual_gated_kd_only_omits_inactive_vision_branch() -> None:
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=8,
        hidden_dim=32,
        output_dim=2,
        fusion_mode="residual_gated",
        use_vision_branch=False,
        use_kinematic_branch=True,
        use_dynamics_branch=True,
        branch_feature_dim=16,
        dynamics_feature_dim=8,
        save_gate_weights=True,
    )
    model.eval()

    images = torch.randn(2, 3, 1, 32, 32)
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)
    base_velocity = torch.tensor([[5.0, 0.1], [2.0, -0.2]])

    with torch.no_grad():
        pred = model(images, chassis, base_velocity=base_velocity)

    assert pred.shape == (2, 2)
    assert model.last_gate_names == ["kinematic", "dynamics"]
    assert model.last_gate_weights is not None
    assert model.last_gate_weights.shape == (2, 2)


def test_residual_gated_respects_save_gate_weights_flag() -> None:
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=8,
        hidden_dim=32,
        output_dim=2,
        fusion_mode="residual_gated",
        use_vision_branch=False,
        use_kinematic_branch=True,
        use_dynamics_branch=True,
        branch_feature_dim=16,
        dynamics_feature_dim=8,
        save_gate_weights=False,
    )
    model.eval()

    images = torch.randn(2, 3, 1, 32, 32)
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)
    base_velocity = torch.tensor([[5.0, 0.1], [2.0, -0.2]])

    with torch.no_grad():
        pred = model(images, chassis, base_velocity=base_velocity)

    assert pred.shape == (2, 2)
    assert model.last_gate_weights is None
    assert model.last_gate_names == []


def test_dynamics_proxy_feature_names_match_model_input_order() -> None:
    assert DYNAMICS_PROXY_FEATURE_NAMES == [
        "base_vx",
        "base_vy",
        "yaw_rate",
        "steering",
        "steering_feedback",
        "longitudinal_accel",
        "transversal_accel",
        "wheel_speed_std",
    ]


def test_config_model_builder_passes_branch_switches() -> None:
    cfg = {
        "data": {
            "chassis_mode": "fair_chassis",
            "image_mode": "gray",
            "flow_mode": "none",
            "geometry_mode": "none",
        },
        "model": {
            "fusion_mode": "residual_gated",
            "output_dim": 2,
            "chassis_feature_dim": 8,
            "chassis_sequence_model": "gru",
            "hidden_dim": 32,
            "dropout": 0.3,
            "use_vision_branch": False,
            "use_kinematic_branch": True,
            "use_dynamics_branch": True,
            "branch_feature_dim": 16,
            "dynamics_feature_dim": 8,
            "save_gate_weights": True,
        },
    }

    model = build_model_from_config(cfg)

    assert model.fusion_mode == "residual_gated"
    assert model.use_vision_branch is False
    assert model.use_kinematic_branch is True
    assert model.use_dynamics_branch is True
    assert model.save_gate_weights is True


def test_vkd_ablation_configs_build_models() -> None:
    config_names = [
        "smoke_velocity_xy_vkd_gated.yaml",
        "velocity_xy_vision_only.yaml",
        "velocity_xy_kd_only.yaml",
        "velocity_xy_vision_kinematics.yaml",
        "velocity_xy_vkd_gated.yaml",
    ]

    for config_name in config_names:
        cfg = load_config(Path("configs") / config_name)
        model = build_model_from_config(cfg)
        assert model.output_dim == 2


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


def test_compute_metrics_reports_proxy_scenario_groups() -> None:
    metrics = compute_metrics(
        preds=[[1.0, 0.0], [1.8, 0.4], [8.0, 0.2], [12.0, -0.5]],
        targets=[[1.1, 0.1], [1.5, 0.6], [8.5, 0.1], [11.0, -0.8]],
        base_velocities=[[1.0, 0.0], [1.5, 0.0], [8.0, 0.0], [12.0, 0.0]],
        yaw_rates=[0.0, 0.2, 0.05, -0.3],
        wheel_speed_stds=[0.01, 0.2, 0.8, 1.2],
        longitudinal_accels=[0.1, 0.2, 3.0, -4.0],
        transversal_accels=[0.0, 0.4, 0.2, -1.5],
    )

    assert metrics["turning_sample_count"] == 2
    assert metrics["wheel_inconsistent_sample_count"] == 2
    assert metrics["hard_accel_sample_count"] == 2
    assert metrics["high_lat_accel_sample_count"] == 2
    assert "wheel_inconsistent_mae_vx_mps" in metrics
    assert "low_speed_lateral_vector_endpoint_error_mps" in metrics


def test_predictions_csv_writes_proxy_and_gate_columns(tmp_path) -> None:
    result = EvaluationResult(
        loss=0.1,
        preds=[[1.2, 0.3]],
        targets=[[1.0, 0.1]],
        base_velocities=[[1.0, 0.0]],
        scene_names=["scene-0001"],
        center_tokens=["token"],
        center_timestamps=[123],
        yaw_rates=[0.2],
        wheel_speed_stds=[0.4],
        longitudinal_accels=[1.5],
        transversal_accels=[-0.3],
        gate_weights=[[0.2, 0.3, 0.5]],
        gate_names=["vision", "kinematic", "dynamics"],
    )
    output_path = tmp_path / "predictions.csv"

    save_predictions_csv(result, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["yaw_rate_radps"] == "0.2"
    assert row["wheel_speed_std_mps"] == "0.4"
    assert row["longitudinal_accel_mps2"] == "1.5"
    assert row["transversal_accel_mps2"] == "-0.3"
    assert row["gate_vision"] == "0.2"
    assert row["gate_kinematic"] == "0.3"
    assert row["gate_dynamics"] == "0.5"


def test_postprocess_reports_per_scenario_metrics() -> None:
    from analysis.postprocess import per_scenario_metrics

    rows = [
        {
            "target_vx_mps": 1.0,
            "target_vy_mps": 0.6,
            "target_speed_mps": 1.2,
            "error_vx_mps": 0.1,
            "error_vy_mps": 0.2,
            "endpoint_error_mps": 0.3,
            "base_endpoint_error_mps": 0.5,
            "yaw_rate_radps": 0.2,
            "wheel_speed_std_mps": 0.8,
            "longitudinal_accel_mps2": 0.1,
            "transversal_accel_mps2": 1.2,
        },
        {
            "target_vx_mps": 10.0,
            "target_vy_mps": 0.1,
            "target_speed_mps": 10.0,
            "error_vx_mps": -0.2,
            "error_vy_mps": 0.1,
            "endpoint_error_mps": 0.4,
            "base_endpoint_error_mps": 0.6,
            "yaw_rate_radps": 0.0,
            "wheel_speed_std_mps": 0.1,
            "longitudinal_accel_mps2": 3.0,
            "transversal_accel_mps2": 0.1,
        },
    ]

    scenario_rows = per_scenario_metrics(rows)
    scenario_names = {row.scenario for row in scenario_rows}
    assert "turning" in scenario_names
    assert "wheel_inconsistent" in scenario_names
    assert "hard_accel" in scenario_names


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


def _make_frame_cache(tmp_path, *, with_labels: bool = True):
    frames = np.array(
        [
            [[0, 255], [128, 64]],
            [[10, 20], [30, 40]],
            [[1, 2], [3, 4]],
        ],
        dtype=np.uint8,
    )
    np.save(tmp_path / "scene-0001.npy", frames)
    manifest = {
        "version": "v1.0-mini",
        "cam_channel": "CAM_FRONT",
        "img_height": 2,
        "img_width": 2,
        "image_mode": "gray",
        "scenes": {"scene-0001": ["t0", "t1", "t2"]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if with_labels:
        (tmp_path / "labels.json").write_text(json.dumps({"t1": [1.5, -0.2]}), encoding="utf-8")


def _make_cache_dataset(tmp_path, *, can_bus_adapter):
    dataset = object.__new__(NuScenesChassisFusionDataset)
    dataset.frame_cache_root = tmp_path
    dataset._frame_token_to_loc = {}
    dataset._frame_scene_arrays = {}
    dataset._label_cache = None
    dataset.img_size = (2, 2)
    dataset.image_mode = "gray"
    dataset.version = "v1.0-mini"
    dataset.cam_channel = "CAM_FRONT"
    dataset.split = "train"
    dataset.can_bus_adapter = can_bus_adapter
    dataset.nusc = "sentinel"
    dataset.samples = [{"tokens": ["t0", "t1", "t2"], "center_token": "t1"}]
    dataset._load_frame_cache_manifest()
    return dataset


def test_frame_cache_normalization_matches_totensor_normalize(tmp_path) -> None:
    _make_frame_cache(tmp_path)
    dataset = _make_cache_dataset(tmp_path, can_bus_adapter=None)

    frame = dataset._load_frame_from_cache("t0")

    assert frame.shape == (1, 2, 2)
    torch.testing.assert_close(frame[0, 0, 0], torch.tensor(0.0 / 255.0 * 2.0 - 1.0))
    torch.testing.assert_close(frame[0, 0, 1], torch.tensor(255.0 / 255.0 * 2.0 - 1.0))
    torch.testing.assert_close(frame[0, 1, 0], torch.tensor(128.0 / 255.0 * 2.0 - 1.0))


def test_frame_cache_uses_label_cache_and_releases_nuscenes(tmp_path) -> None:
    _make_frame_cache(tmp_path)
    dataset = _make_cache_dataset(tmp_path, can_bus_adapter=object())

    label = dataset._compute_velocity_label("t1")

    torch.testing.assert_close(label, torch.tensor([1.5, -0.2]))
    assert dataset.nusc is None


def test_frame_cache_rejects_mismatched_manifest(tmp_path) -> None:
    _make_frame_cache(tmp_path)
    dataset = object.__new__(NuScenesChassisFusionDataset)
    dataset.frame_cache_root = tmp_path
    dataset._frame_token_to_loc = {}
    dataset._frame_scene_arrays = {}
    dataset._label_cache = None
    dataset.img_size = (64, 64)  # mismatch
    dataset.image_mode = "gray"
    dataset.version = "v1.0-mini"
    dataset.cam_channel = "CAM_FRONT"
    dataset.split = "train"
    dataset.can_bus_adapter = None
    dataset.nusc = "sentinel"
    dataset.samples = [{"tokens": ["t0", "t1", "t2"], "center_token": "t1"}]

    try:
        dataset._load_frame_cache_manifest()
    except ValueError as exc:
        assert "img_height" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched manifest")


def test_eovm_scripts_can_show_help_without_package_install() -> None:
    for script in [
        "scripts/audit_eovm_inputs.py",
        "scripts/precompute_eovm_base_velocity.py",
        "scripts/precompute_frames.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
