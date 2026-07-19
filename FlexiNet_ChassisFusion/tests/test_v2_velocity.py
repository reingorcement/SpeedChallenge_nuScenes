from __future__ import annotations

import math
import json
import random
import sys
import types

import numpy as np
import pytest
import torch

from dataset.nuscenes_chassis import NuScenesCanBusAdapter
from dataset.v2_schema import (
    KINEMATIC_VALUE_DIM,
    KINEMATIC_VALUE_FEATURES,
    V2NuScenesDatasetAdapter,
    apply_sequence_photometric_augmentation,
    build_kinematic_step,
    normalize_kinematic_values,
)
from dataset.v2_splits import (
    create_split_manifest,
    load_split_manifest,
    partition_scene_names,
)
from model.v2.ground_geometry import (
    RobustRoadPlaneVelocitySolver,
    road_plane_depth,
    translation_flow_from_velocity,
)
from model.v2.lever_arm import correct_reference_to_cg
from model.v2.kinematics import WeakKinematicEncoder
from model.v2.losses import V2VelocityLoss
from model.v2.model import V2NuScenesVelocityModel
from model.v2.residual import BoundedFiLMResidual, MonotonicUncertaintyGate
from model.v2.rotation_flow import rotational_flow, scale_intrinsics
from model.v2.types import TimedSignal, V2VelocityOutput
from scripts.prepare_v2_training_stats import resolve_field_age_limits
from scripts.compare_v2_multiseed import compare_multiseed
from scripts.analyze_v2_stage1 import (
    analyze,
    align_prediction_rows,
    paired_scene_bootstrap,
)
from scripts.evaluate_v2_holdout import _modified_batch, ablation_prediction
from train_v2 import (
    _accumulation_steps,
    _accumulation_divisor,
    _checkpoint,
    _configure_trainable_modules,
    _load_resume,
    _load_initial_weights,
    load_config,
    _make_scheduler,
    _next_no_improve,
    _parameter_groups,
    _selection_result,
    _set_epoch_training_mode,
)


def camera_setup(batch_size: int = 1, height: int = 32, width: int = 48):
    intrinsics = torch.tensor(
        [[100.0, 0.0, width / 2], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
    ).repeat(batch_size, 1, 1)
    normal = torch.tensor([[0.0, -1.0, 0.0]]).repeat(batch_size, 1)
    offset = torch.full((batch_size,), 1.5)
    rotation = torch.eye(3).repeat(batch_size, 1, 1)
    return intrinsics, normal, offset, rotation


def synthetic_flow(velocity: torch.Tensor, height: int = 32, width: int = 48):
    batch_size = velocity.shape[0]
    intrinsics, normal, offset, rotation = camera_setup(batch_size, height, width)
    depth, valid = road_plane_depth(intrinsics, normal, offset, height, width)
    dt = torch.full((batch_size,), 0.1)
    flow = translation_flow_from_velocity(velocity, intrinsics, depth, dt)
    confidence = valid[:, None].float()
    return flow, confidence, intrinsics, dt, normal, offset, rotation, valid[:, None]


def solve(flow, confidence, intrinsics, dt, normal, offset, rotation, road_mask=None):
    solver = RobustRoadPlaneVelocitySolver(min_points=24, max_condition_number=1.0e8)
    batch_size = flow.shape[0]
    return solver(
        flow=flow,
        confidence=confidence,
        intrinsics=intrinsics,
        dt=dt,
        angular_rate_camera=torch.zeros(batch_size, 3),
        angular_rate_valid=torch.ones(batch_size, 3, dtype=torch.bool),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        road_mask=road_mask,
    )


def test_intrinsics_scaling_preserves_normalized_coordinates() -> None:
    intrinsics = torch.tensor([[800.0, 0.0, 800.0], [0.0, 800.0, 450.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(intrinsics, (900, 1600), (288, 512))
    original_x = (1200.0 - intrinsics[0, 2]) / intrinsics[0, 0]
    resized_x = (1200.0 * 512.0 / 1600.0 - scaled[0, 2]) / scaled[0, 0]
    torch.testing.assert_close(original_x, resized_x)


def test_zero_angular_rate_produces_zero_rotation_flow() -> None:
    intrinsics, _, _, _ = camera_setup()
    flow = rotational_flow(intrinsics, torch.zeros(1, 3), 16, 24, torch.tensor([0.1]))
    assert torch.count_nonzero(flow) == 0


def test_positive_camera_z_rotation_moves_right_pixel_up() -> None:
    intrinsics, _, _, _ = camera_setup(height=21, width=41)
    flow = rotational_flow(intrinsics, torch.tensor([[0.0, 0.0, 1.0]]), 21, 41, torch.tensor([0.1]))
    center_y = 10
    right_x = 30
    assert abs(float(flow[0, 0, center_y, right_x])) < 1.0e-5
    assert float(flow[0, 1, center_y, right_x]) < 0.0


def test_yaw_only_mask_still_removes_available_rotation() -> None:
    expected = torch.tensor([[0.3, 0.0, 6.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(expected)
    yaw_rate = torch.tensor([[0.0, 0.0, 0.2]])
    flow = flow + rotational_flow(intrinsics, yaw_rate, flow.shape[-2], flow.shape[-1], dt)
    solver = RobustRoadPlaneVelocitySolver(min_points=24, max_condition_number=1.0e8)
    estimate = solver(
        flow=flow,
        confidence=confidence,
        intrinsics=intrinsics,
        dt=dt,
        angular_rate_camera=yaw_rate,
        angular_rate_valid=torch.tensor([[False, False, True]]),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        road_mask=mask,
    )
    torch.testing.assert_close(estimate.camera_velocity, expected, atol=2.0e-3, rtol=2.0e-3)


def test_irls_recovers_synthetic_planar_translation() -> None:
    expected = torch.tensor([[0.4, -0.1, 8.0]])
    args = synthetic_flow(expected)
    estimate = solve(*args[:-1], road_mask=args[-1])
    assert bool(estimate.valid[0])
    torch.testing.assert_close(estimate.camera_velocity, expected, atol=2.0e-3, rtol=2.0e-3)


def test_irls_downweights_large_flow_outliers() -> None:
    expected = torch.tensor([[0.4, 0.0, 8.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(expected)
    corrupted = flow.clone()
    corrupted[:, :, 20:24, 20:26] += 80.0
    estimate = solve(corrupted, confidence, intrinsics, dt, normal, offset, rotation, mask)
    assert bool(estimate.valid[0])
    torch.testing.assert_close(estimate.camera_velocity, expected, atol=0.15, rtol=0.04)


def test_degenerate_support_is_rejected_without_nan() -> None:
    expected = torch.tensor([[0.0, 0.0, 5.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(expected)
    confidence.zero_()
    confidence[:, :, -2:, -2:] = 1.0
    estimate = solve(flow, confidence, intrinsics, dt, normal, offset, rotation, mask)
    assert not bool(estimate.valid[0])
    assert torch.isfinite(estimate.camera_velocity).all()
    assert torch.isfinite(estimate.covariance).all()


@pytest.mark.parametrize("bad_dt", [0.0, float("nan")])
def test_invalid_frame_interval_is_rejected_without_nan(bad_dt: float) -> None:
    expected = torch.tensor([[0.0, 0.0, 5.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(expected)
    dt.fill_(bad_dt)
    flow[:, :, -1, -1] = float("nan")
    confidence[:, :, -1, -1] = float("nan")
    estimate = solve(flow, confidence, intrinsics, dt, normal, offset, rotation, mask)
    assert not bool(estimate.valid[0])
    assert torch.isfinite(estimate.camera_velocity).all()
    assert torch.isfinite(estimate.covariance).all()


def test_timed_signal_distinguishes_valid_zero_from_missing_zero() -> None:
    value = torch.zeros(2, 3)
    signal = TimedSignal(
        value=value,
        valid=torch.tensor([[True, True, True], [False, False, False]]),
        age_sec=torch.zeros(2),
    )
    assert signal.usable(0.25).squeeze(-1).tolist() == [True, False]


def test_can_lookup_returns_true_message_age(tmp_path) -> None:
    path = tmp_path / "scene-0001_vehicle_monitor.json"
    path.write_text(json.dumps([{"utime": 900_000, "yaw_rate": 1.0}]), encoding="utf-8")
    adapter = NuScenesCanBusAdapter(tmp_path, max_age_us=250_000)

    record, age = adapter.select_latest_before_with_age("scene-0001", "vehicle_monitor", 1_000_000)

    assert record is not None
    assert age == pytest.approx(0.1)


def test_kinematic_timestamps_are_exactly_21_steps_and_strictly_causal() -> None:
    center_timestamp_us = 10_000_000

    timestamps = NuScenesCanBusAdapter.sample_timestamps(
        center_timestamp_us=center_timestamp_us,
        window_sec=1.0,
        hz=20.0,
        online_mode=True,
    )

    assert len(timestamps) == 21
    assert timestamps[0] == center_timestamp_us - 1_000_000
    assert timestamps[-1] == center_timestamp_us
    assert all(current <= center_timestamp_us for current in timestamps)
    assert all(later - earlier == 50_000 for earlier, later in zip(timestamps, timestamps[1:]))


def test_kinematic_schema_excludes_pose_and_absolute_wheel_speed() -> None:
    assert KINEMATIC_VALUE_FEATURES == (
        "yaw_rate",
        "steering_command",
        "steering_feedback",
        "longitudinal_accel",
        "lateral_accel",
        "throttle",
        "brake",
        "wheel_residual_fl",
        "wheel_residual_fr",
        "wheel_residual_rl",
        "wheel_residual_rr",
        "wheel_speed_relative_std",
    )
    assert not any(
        "pose" in name or "absolute" in name or "wheel_mean" in name
        for name in KINEMATIC_VALUE_FEATURES
    )


def test_kinematic_wheel_residuals_are_invariant_to_common_scale() -> None:
    monitor = {"yaw_rate": 2.0, "steering": 3.0, "throttle": 4.0, "brake": 5.0}
    steer = {"value": 1.0}
    wheel = {
        "longitudinal_accel": 0.1,
        "transversal_accel": 0.2,
        "FL_wheel_speed": 9_000.0,
        "FR_wheel_speed": 10_000.0,
        "RL_wheel_speed": 11_000.0,
        "RR_wheel_speed": 10_000.0,
    }
    scaled_wheel = {
        key: value * 2.0 if key.endswith("wheel_speed") else value
        for key, value in wheel.items()
    }
    first = build_kinematic_step(monitor, 0.01, wheel, 0.02, steer, 0.03, 0.25)
    second = build_kinematic_step(monitor, 0.01, scaled_wheel, 0.02, steer, 0.03, 0.25)

    torch.testing.assert_close(first[0][7:11], second[0][7:11])
    torch.testing.assert_close(first[0][11], second[0][11])
    assert first[0].shape == (KINEMATIC_VALUE_DIM,)
    assert first[2].min() >= 0.0 and first[2].max() <= 1.0


def test_missing_kinematic_fields_use_mask_and_maximum_age() -> None:
    values, valid, age, *_ = build_kinematic_step(None, 0.25, None, 0.25, None, 0.25, 0.25)
    assert torch.count_nonzero(values) == 0
    assert not bool(valid.any())
    torch.testing.assert_close(age, torch.ones_like(age))


def test_v2_adapter_rejects_noncausal_dataset(tmp_path) -> None:
    base = type("Base", (), {"online_mode": False, "seq_len": 5})()
    with pytest.raises(ValueError, match="online_mode=True"):
        V2NuScenesDatasetAdapter(base, tmp_path, camera_height_m=1.5)


def test_stage1_manifest_is_deterministic_disjoint_and_seals_final_test(tmp_path) -> None:
    train = [f"scene-{index:04d}" for index in range(20)]
    val = [f"scene-{index:04d}" for index in range(20, 30)]
    metadata = {
        name: {
            "location": "boston" if index % 2 else "singapore",
            "lighting": "night" if index % 3 == 0 else "day",
            "weather": "rain" if index % 5 == 0 else "dry",
            "motion_regime": "turning" if index % 4 == 0 else "straight",
        }
        for index, name in enumerate(train + val)
    }
    first = create_split_manifest(train, val, metadata, seed=42)
    second = create_split_manifest(reversed(train), reversed(val), metadata, seed=42)
    assert first == second
    partitions = first["partitions"]
    assert len(partitions["train_core"]) == 17
    assert len(partitions["dev"]) == 3
    assert len(partitions["stage1_test"]) == 5
    assert len(partitions["final_system_test"]) == 5
    sets = [set(values) for values in partitions.values()]
    assert all(not left & right for index, left in enumerate(sets) for right in sets[index + 1 :])

    path = tmp_path / "split.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    assert load_split_manifest(path) == first
    assert partition_scene_names(path, "stage1_test") == partitions["stage1_test"]
    with pytest.raises(PermissionError, match="sealed"):
        partition_scene_names(path, "final_system_test")
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["partitions"]["dev"].reverse()
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_split_manifest(path)


def test_dataset_scene_names_select_exact_official_scenes(monkeypatch, tmp_path) -> None:
    class FakeNuScenes:
        def __init__(self, **_kwargs):
            self.scene = [
                {"name": name, "first_sample_token": f"sample-{name}"}
                for name in ("scene-a", "scene-b", "scene-c")
            ]

        def get(self, table, token):
            if table == "sample":
                scene = token.removeprefix("sample-")
                return {"data": {"CAM_FRONT": f"{scene}-0"}}
            if table == "sample_data":
                scene, index_text = token.rsplit("-", 1)
                index = int(index_text)
                return {
                    "prev": f"{scene}-{index - 1}" if index > 0 else "",
                    "next": f"{scene}-{index + 1}" if index < 2 else "",
                    "timestamp": index + 1,
                }
            raise KeyError(table)

    nuscenes_package = types.ModuleType("nuscenes")
    nuscenes_module = types.ModuleType("nuscenes.nuscenes")
    nuscenes_module.NuScenes = FakeNuScenes
    utils_module = types.ModuleType("nuscenes.utils")
    splits_module = types.ModuleType("nuscenes.utils.splits")
    splits_module.create_splits_scenes = lambda: {
        "mini_train": [],
        "mini_val": [],
        "train": ["scene-a", "scene-b"],
        "val": ["scene-c"],
        "test": [],
    }
    monkeypatch.setitem(sys.modules, "nuscenes", nuscenes_package)
    monkeypatch.setitem(sys.modules, "nuscenes.nuscenes", nuscenes_module)
    monkeypatch.setitem(sys.modules, "nuscenes.utils", utils_module)
    monkeypatch.setitem(sys.modules, "nuscenes.utils.splits", splits_module)
    from dataset.nuscenes_chassis import NuScenesChassisFusionDataset

    dataset = NuScenesChassisFusionDataset(
        tmp_path,
        version="v1.0-trainval",
        split="train",
        seq_len=1,
        scene_names=["scene-b"],
    )
    assert dataset.scene_names == ["scene-b"]
    assert {sample["scene_name"] for sample in dataset.samples} == {"scene-b"}
    with pytest.raises(ValueError, match="outside official"):
        NuScenesChassisFusionDataset(
            tmp_path,
            version="v1.0-trainval",
            split="train",
            seq_len=1,
            scene_names=["scene-c"],
        )


def test_holdout_alignment_and_scene_bootstrap_are_reproducible() -> None:
    left = [
        {
            "scene_name": scene,
            "center_token": f"token-{index}",
            "target_vx_mps": "5.0",
            "target_vy_mps": "0.5",
        }
        for index, scene in enumerate(("scene-a", "scene-b", "scene-c"))
    ]
    assert len(align_prediction_rows(left, list(reversed(left)))) == 3
    altered = [dict(row) for row in left]
    altered[0]["target_vx_mps"] = "6.0"
    with pytest.raises(ValueError, match="Target mismatch"):
        align_prediction_rows(left, altered)
    deltas = {"scene-a": -0.2, "scene-b": 0.1, "scene-c": -0.1}
    first = paired_scene_bootstrap(deltas, samples=500, seed=42)
    second = paired_scene_bootstrap(deltas, samples=500, seed=42)
    assert first == second
    assert first["scene_count"] == 3
    assert first["ci95_low"] <= first["mean_delta"] <= first["ci95_high"]


def test_holdout_ablation_postprocessing_and_missing_can_masks() -> None:
    output = make_output(torch.tensor([[5.4, 0.3], [4.0, -0.1]]), torch.Size((2, 2)))
    object.__setattr__(output, "visual_velocity", torch.tensor([[5.2, 0.2], [3.9, -0.1]]))
    object.__setattr__(output, "geometry_velocity", torch.tensor([[5.0, 0.0], [4.0, 0.0]]))
    object.__setattr__(output, "wheel_scale", torch.tensor([[1.04], [1.0]]))
    object.__setattr__(output, "residual", torch.tensor([[0.2, 0.2], [-0.1, -0.1]]))
    object.__setattr__(output, "residual_gate", torch.ones(2, 1))
    object.__setattr__(output, "kinematic_applied_correction", torch.tensor([[0.2, 0.1], [0.1, 0.0]]))
    _, no_kinematic = ablation_prediction(None, {}, output, "no_kinematic_correction")
    _, no_visual = ablation_prediction(None, {}, output, "no_visual_residual")
    _, no_wheel = ablation_prediction(None, {}, output, "no_wheel_scaling")
    torch.testing.assert_close(no_kinematic, output.visual_velocity)
    torch.testing.assert_close(no_visual, output.predicted_velocity - output.residual)
    assert float(no_wheel[0, 0]) == pytest.approx(5.2)

    batch = {
        "images": torch.randn(2, 3, 3, 4, 4),
        "kinematic_values": torch.randn(2, 21, 12),
        "kinematic_valid": torch.ones(2, 21, 12, dtype=torch.bool),
        "kinematic_age": torch.zeros(2, 21, 12),
        "wheel_speed": torch.ones(2, 4),
        "wheel_speed_valid": torch.ones(2, 4, dtype=torch.bool),
        "wheel_speed_age_sec": torch.zeros(2, 4),
        "angular_rate_camera": torch.ones(2, 3),
        "angular_rate_valid": torch.ones(2, 3, dtype=torch.bool),
        "angular_rate_age_sec": torch.zeros(2),
    }
    missing = _modified_batch(batch, "missing_can")
    assert not bool(missing["kinematic_valid"].any())
    assert not bool(missing["wheel_speed_valid"].any())
    assert not bool(missing["angular_rate_valid"].any())
    assert torch.count_nonzero(missing["kinematic_values"]) == 0


def test_stage1_analyzer_accepts_consistent_three_seed_improvement(tmp_path) -> None:
    def write_predictions(path, prediction_offset, correction=0.0):
        rows = []
        for index, (scene, target_vy, yaw_rate) in enumerate(
            (("scene-a", 0.0, 0.0), ("scene-b", 1.0, 0.1), ("scene-c", -1.0, -0.1))
        ):
            rows.append(
                {
                    "variant": "baseline",
                    "seed": 42,
                    "scene_name": scene,
                    "center_token": f"token-{index}",
                    "center_timestamp_us": index,
                    "target_vx_mps": 5.0,
                    "target_vy_mps": target_vy,
                    "pred_vx_mps": 5.0 + prediction_offset,
                    "pred_vy_mps": target_vy + prediction_offset,
                    "visual_vx_mps": 5.0 + prediction_offset,
                    "visual_vy_mps": target_vy + prediction_offset,
                    "geometry_vx_mps": 5.0,
                    "geometry_vy_mps": 0.0,
                    "geometry_valid": 1,
                    "geometry_condition_number": 100.0,
                    "geometry_residual_rms": 0.1,
                    "geometry_support_count": 100,
                    "fallback_used": 0,
                    "wheel_scale": 1.0,
                    "visual_residual_vx_mps": 0.0,
                    "visual_residual_vy_mps": 0.0,
                    "visual_residual_gate": 1.0,
                    "kinematic_residual_vx_mps": correction,
                    "kinematic_residual_vy_mps": correction,
                    "kinematic_correction_vx_mps": correction,
                    "kinematic_correction_vy_mps": correction,
                    "kinematic_gate_vx": 0.5,
                    "kinematic_gate_vy": 0.5,
                    "kinematic_log_variance_vx": 0.0,
                    "kinematic_log_variance_vy": 0.0,
                    "kinematic_coverage": 1.0,
                    "kinematic_available": 1,
                    "yaw_rate_radps": yaw_rate,
                    "wheel_relative_std": 0.0,
                }
            )
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    geometry = tmp_path / "geometry.csv"
    visual = [tmp_path / f"visual-{seed}.csv" for seed in (42, 3407, 2026)]
    joint = [tmp_path / f"joint-{seed}.csv" for seed in (42, 3407, 2026)]
    write_predictions(geometry, 0.4)
    for path in visual:
        write_predictions(path, 0.2)
    for path in joint:
        write_predictions(path, 0.1, correction=0.1)
    report, per_scene, ablations, bad_cases = analyze(
        [42, 3407, 2026],
        str(geometry),
        [str(path) for path in visual],
        [str(path) for path in joint],
        high_lateral_threshold=0.5,
        bootstrap_samples=200,
    )
    assert report["acceptance"]["stage1_pass"]
    assert report["group_improved_seed_count"] == 3
    assert per_scene and ablations and bad_cases


def test_no_cg_correction_without_explicit_lever_arm() -> None:
    velocity = torch.tensor([[5.0, 0.2, 0.0]])
    corrected, applied = correct_reference_to_cg(velocity, torch.tensor([[0.0, 0.0, 0.5]]), None)
    assert not applied
    torch.testing.assert_close(corrected, velocity)


def make_output(predicted: torch.Tensor, target_shape: torch.Size) -> V2VelocityOutput:
    batch_size = target_shape[0]
    return V2VelocityOutput(
        geometry_velocity=predicted,
        geometry_covariance=torch.eye(3).repeat(batch_size, 1, 1),
        geometry_condition_number=torch.ones(batch_size),
        geometry_residual_rms=torch.zeros(batch_size),
        geometry_valid=torch.ones(batch_size, dtype=torch.bool),
        geometry_support_count=torch.full((batch_size,), 100),
        visual_velocity=predicted,
        wheel_scale=torch.ones(batch_size, 1),
        fallback_used=torch.zeros(batch_size, dtype=torch.bool),
        residual=torch.zeros_like(predicted),
        residual_gate=torch.ones(batch_size, 1),
        kinematic_residual=torch.zeros_like(predicted),
        kinematic_applied_correction=torch.zeros_like(predicted),
        kinematic_gate=torch.zeros_like(predicted),
        kinematic_log_variance=torch.zeros_like(predicted),
        kinematic_coverage=torch.zeros(batch_size, 1),
        kinematic_available=torch.zeros(batch_size, dtype=torch.bool),
        predicted_velocity=predicted,
        output_reference="ego_reference",
    )


def test_beta_loss_uses_target_angle_and_masks_low_speed() -> None:
    criterion = V2VelocityLoss(beta_weight=1.0, geometry_weight=0.0, residual_weight=0.0)
    target = torch.tensor([[5.0, 0.0], [0.1, 0.0]])
    predicted = torch.tensor([[0.0, 5.0], [0.0, 0.1]])
    terms = criterion(make_output(predicted, target.shape), target)
    assert float(terms.beta) > 0.5
    low_only = criterion(make_output(predicted[1:], target[1:].shape), target[1:])
    assert float(low_only.beta) == 0.0


def test_beta_loss_has_finite_gradient_at_zero_prediction() -> None:
    criterion = V2VelocityLoss(beta_weight=1.0, geometry_weight=0.0, residual_weight=0.0)
    target = torch.tensor([[5.0, 0.5]])
    predicted = torch.zeros(1, 2, requires_grad=True)

    terms = criterion(make_output(predicted, target.shape), target)
    terms.total.backward()

    assert torch.isfinite(terms.total)
    assert torch.isfinite(predicted.grad).all()
    assert float(terms.beta) == 0.0


def test_film_residual_is_bounded_per_axis() -> None:
    model = BoundedFiLMResidual(condition_dim=4, residual_limits=(2.0, 0.5))
    residual = model(torch.randn(3, 256, 2, 4, 4) * 100.0, torch.randn(3, 4) * 100.0)
    assert torch.all(residual[:, 0].abs() <= 2.0)
    assert torch.all(residual[:, 1].abs() <= 0.5)


def test_uncertainty_gate_sanitizes_nonfinite_covariance() -> None:
    gate = MonotonicUncertaintyGate()
    covariance = torch.eye(3).repeat(3, 1, 1)
    covariance[0, 0, 0] = float("nan")
    covariance[1, 2, 2] = float("inf")
    covariance[2, 0, 0] = -float("inf")

    result = gate(covariance, torch.ones(3, dtype=torch.bool))

    assert torch.isfinite(result).all()
    assert torch.all((result >= 0.0) & (result <= 1.0))


def test_weak_kinematic_encoder_is_bounded_and_uncertainty_gate_is_monotonic() -> None:
    encoder = WeakKinematicEncoder()
    values = torch.randn(2, 21, 12)
    valid = torch.ones_like(values, dtype=torch.bool)
    ages = torch.zeros_like(values)
    with torch.no_grad():
        encoder.log_variance_head.weight.zero_()
        encoder.log_variance_head.bias.fill_(-2.0)
    low_uncertainty = encoder(values, valid, ages)
    with torch.no_grad():
        encoder.log_variance_head.bias.fill_(2.0)
    high_uncertainty = encoder(values, valid, ages)

    assert torch.all(low_uncertainty.residual[:, 0].abs() <= 1.0)
    assert torch.all(low_uncertainty.residual[:, 1].abs() <= 0.5)
    assert torch.all(low_uncertainty.gate > high_uncertainty.gate)


def test_weak_kinematic_encoder_handles_all_missing_sequence() -> None:
    encoder = WeakKinematicEncoder()
    values = torch.zeros(2, 21, 12)
    valid = torch.zeros_like(values, dtype=torch.bool)
    ages = torch.ones_like(values)

    estimate = encoder(values, valid, ages)

    assert not bool(estimate.available.any())
    assert torch.count_nonzero(estimate.gate) == 0
    assert torch.isfinite(estimate.residual).all()
    assert torch.isfinite(estimate.log_variance).all()


def test_training_stage_freeze_policy_is_explicit() -> None:
    visual = V2NuScenesVelocityModel(mode="visual")
    _configure_trainable_modules(visual, "visual")
    assert any(parameter.requires_grad for parameter in visual.residual_model.parameters())
    assert not any(parameter.requires_grad for parameter in visual.kinematic_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in visual.backbone.parameters())

    kinematic = V2NuScenesVelocityModel(mode="visual_kinematic")
    _configure_trainable_modules(kinematic, "kinematic_pretrain")
    assert not any(parameter.requires_grad for parameter in kinematic.residual_model.parameters())
    assert any(parameter.requires_grad for parameter in kinematic.kinematic_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in kinematic.backbone.parameters())

    joint = V2NuScenesVelocityModel(mode="visual_kinematic")
    _configure_trainable_modules(joint, "joint")
    assert any(parameter.requires_grad for parameter in joint.residual_model.parameters())
    assert any(parameter.requires_grad for parameter in joint.kinematic_encoder.parameters())
    assert any(parameter.requires_grad for parameter in joint.backbone.dig.parameters())
    assert not any(parameter.requires_grad for parameter in joint.backbone.cma_block.parameters())


def test_frozen_batch_norm_statistics_do_not_drift() -> None:
    model = V2NuScenesVelocityModel(mode="visual")
    _configure_trainable_modules(model, "visual")
    _set_epoch_training_mode(model, "visual", epoch=0, freeze_backbone_epochs=5)
    frozen_bn = next(
        module for module in model.backbone.modules() if isinstance(module, torch.nn.BatchNorm3d)
    )
    before = frozen_bn.running_mean.clone()
    model.backbone.forward_feature_map(torch.randn(2, 3, 3, 8, 8))
    torch.testing.assert_close(frozen_bn.running_mean, before)
    assert not model.backbone.training
    assert model.residual_model.training


def test_visual_tail_batch_norm_updates_only_after_unfreeze() -> None:
    model = V2NuScenesVelocityModel(mode="visual")
    _configure_trainable_modules(model, "visual")
    _set_epoch_training_mode(model, "visual", epoch=5, freeze_backbone_epochs=5)
    frozen_bn = next(
        module for module in model.backbone.cma_block.modules()
        if isinstance(module, torch.nn.BatchNorm3d)
    )
    tail_bn = next(
        module for module in model.backbone.sfe_module.modules()
        if isinstance(module, torch.nn.BatchNorm3d)
    )
    frozen_before = frozen_bn.running_mean.clone()
    tail_before = tail_bn.running_mean.clone()
    model.backbone.forward_feature_map(torch.randn(2, 3, 3, 8, 8))
    torch.testing.assert_close(frozen_bn.running_mean, frozen_before)
    assert not torch.equal(tail_bn.running_mean, tail_before)
    assert not model.backbone.cma_block.training
    assert model.backbone.sfe_module.training


def test_kinematic_pretrain_keeps_visual_modules_in_eval_mode() -> None:
    model = V2NuScenesVelocityModel(mode="visual_kinematic")
    _configure_trainable_modules(model, "kinematic_pretrain")
    _set_epoch_training_mode(model, "kinematic_pretrain", epoch=0, freeze_backbone_epochs=0)
    assert not model.backbone.training
    assert not model.residual_model.training
    assert model.kinematic_encoder.training


def test_residual_branches_start_as_neutral_corrections() -> None:
    visual = BoundedFiLMResidual(condition_dim=4)
    visual_residual = visual(torch.randn(2, 256, 3, 4, 5), torch.randn(2, 4))
    torch.testing.assert_close(visual_residual, torch.zeros_like(visual_residual))

    kinematic = WeakKinematicEncoder()
    estimate = kinematic(
        torch.randn(2, 21, 12),
        torch.ones(2, 21, 12, dtype=torch.bool),
        torch.zeros(2, 21, 12),
    )
    torch.testing.assert_close(estimate.residual, torch.zeros_like(estimate.residual))
    torch.testing.assert_close(estimate.log_variance, torch.zeros_like(estimate.log_variance))


def test_kinematic_normalization_masks_clips_and_uses_supplied_train_stats() -> None:
    values = torch.tensor([[[12.0] * 12, [3.0] * 12]])
    valid = torch.tensor([[[True] * 12, [False] * 12]])
    normalized = normalize_kinematic_values(
        values,
        valid,
        mean=torch.ones(12),
        std=torch.full((12,), 2.0),
    )
    torch.testing.assert_close(normalized[:, 0], torch.full((1, 12), 5.0))
    assert torch.count_nonzero(normalized[:, 1]) == 0


def test_sequence_photometric_augmentation_uses_identical_frame_parameters() -> None:
    frame = torch.linspace(-1.0, 1.0, 3 * 6 * 8).reshape(3, 6, 8)
    sequence = frame.repeat(5, 1, 1, 1)
    augmented = apply_sequence_photometric_augmentation(
        sequence, brightness=1.05, contrast=0.95, saturation=1.02, hue=0.01, gamma=0.97
    )
    for index in range(1, sequence.shape[0]):
        torch.testing.assert_close(augmented[index], augmented[0])
    assert torch.isfinite(augmented).all()


def test_can_interval_audit_resolves_field_specific_age_limits() -> None:
    audit = {
        "can_channels": {
            "vehicle_monitor": {"p95_interval_sec": 0.04},
            "zoe_veh_info": {"p95_interval_sec": 0.08},
            "steeranglefeedback": {"p95_interval_sec": 0.20},
        }
    }
    limits = resolve_field_age_limits(audit)
    assert limits[0] == pytest.approx(0.06)
    assert limits[2] == pytest.approx(0.15)
    assert limits[3] == pytest.approx(0.10)
    assert limits[5] == pytest.approx(0.06)


def test_scenario_weights_follow_train_statistics_contract() -> None:
    adapter = object.__new__(V2NuScenesDatasetAdapter)
    adapter._scenario_weights = {"a": 1.0, "b": 2.0, "c": 3.0}
    adapter.base_dataset = type(
        "Base", (), {"samples": [{"center_token": "a"}, {"center_token": "b"}, {"center_token": "c"}]}
    )()
    torch.testing.assert_close(adapter.scenario_weights(), torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))


def test_recommended_stage_configs_lock_training_defaults() -> None:
    visual = load_config("configs/velocity_xy_v2_visual.yaml")
    kinematic = load_config("configs/velocity_xy_v2_kinematic.yaml")
    joint = load_config("configs/velocity_xy_v2_visual_kinematic.yaml")
    assert visual["loss"]["sigma_vy"] == 1.0
    assert visual["loss"]["beta_min_speed"] == 2.0
    assert visual["loss"]["geometry_weight"] == 0.0
    assert visual["train"]["epochs"] == 15
    assert visual["train"]["freeze_backbone_epochs"] == 5
    assert kinematic["train"]["scenario_sampling"] is True
    assert joint["train"]["head_lr"] == pytest.approx(1.0e-4)
    assert joint["train"]["kinematic_lr"] == pytest.approx(1.0e-4)
    assert joint["train"]["effective_batch_size"] == 8


def test_optimizer_groups_exclude_bias_and_norm_from_weight_decay() -> None:
    model = V2NuScenesVelocityModel(mode="visual")
    cfg = {
        "train": {
            "head_lr": 3.0e-4,
            "kinematic_lr": 3.0e-4,
            "backbone_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
        }
    }
    groups = _parameter_groups(model, cfg, "visual")
    assert {group["weight_decay"] for group in groups} == {0.0, 1.0e-4}
    no_decay = [group for group in groups if group["weight_decay"] == 0.0]
    assert no_decay
    assert all(parameter.ndim <= 1 for group in no_decay for parameter in group["params"])


def test_warmup_cosine_scheduler_reaches_minimum_ratio() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = _make_scheduler(optimizer, total_steps=10, warmup_fraction=0.2, min_lr_ratio=0.1)
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] < 1.0
    assert max(learning_rates) == pytest.approx(1.0)
    assert learning_rates[-1] == pytest.approx(0.1)


def test_effective_batch_requires_integral_accumulation() -> None:
    assert _accumulation_steps(8, 8) == 1
    assert _accumulation_steps(4, 8) == 2
    with pytest.raises(ValueError, match="positive multiple"):
        _accumulation_steps(6, 8)
    assert [_accumulation_divisor(step, 5, 2) for step in range(5)] == [2, 2, 2, 2, 1]


def test_stage_checkpoint_selection_matches_acceptance_constraint() -> None:
    baseline = {"final_endpoint_error_mps": 1.0}
    eligible, score = _selection_result(
        {
            "final_endpoint_error_mps": 1.01,
            "turning_mae_vy_mps": 0.2,
            "high_lateral_mae_vy_mps": 0.4,
        },
        "joint",
        baseline,
    )
    assert eligible
    assert score == pytest.approx(0.3)
    rejected, _ = _selection_result(
        {
            "final_endpoint_error_mps": 1.021,
            "turning_mae_vy_mps": 0.1,
            "high_lateral_mae_vy_mps": 0.1,
        },
        "joint",
        baseline,
    )
    assert not rejected


def test_visual_early_stopping_starts_after_backbone_unfreezes() -> None:
    no_improve = 0
    for epoch in range(5):
        no_improve = _next_no_improve(
            no_improve,
            improved=False,
            stage="visual",
            epoch=epoch,
            freeze_backbone_epochs=5,
        )
    assert no_improve == 0
    assert _next_no_improve(
        no_improve,
        improved=False,
        stage="visual",
        epoch=5,
        freeze_backbone_epochs=5,
    ) == 1
    assert _next_no_improve(
        3,
        improved=True,
        stage="visual",
        epoch=6,
        freeze_backbone_epochs=5,
    ) == 0


def test_training_checkpoint_restores_complete_resume_state(tmp_path) -> None:
    model = V2NuScenesVelocityModel(mode="visual")
    cfg = {
        "train": {
            "stage": "visual",
            "head_lr": 3.0e-4,
            "kinematic_lr": 3.0e-4,
            "backbone_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
        }
    }
    optimizer = torch.optim.AdamW(_parameter_groups(model, cfg, "visual"), weight_decay=0.0)
    scheduler = _make_scheduler(optimizer, 4, 0.25, 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    original = next(model.parameters()).detach().clone()
    package = _checkpoint(
        2,
        model,
        optimizer,
        scheduler,
        scaler,
        cfg,
        {"final_endpoint_error_mps": 1.0},
        [{"epoch": 0}],
        1.0,
        2,
        3,
        1,
    )
    expected_random = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(()))
    path = tmp_path / "resume.pth"
    torch.save(package, path)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    state = _load_resume(path, model, optimizer, scheduler, scaler, "visual")
    torch.testing.assert_close(next(model.parameters()), original)
    assert state == (3, [{"epoch": 0}], 1.0, 2, 3, 1)
    assert random.random() == pytest.approx(expected_random)
    assert float(np.random.rand()) == pytest.approx(expected_numpy)
    assert float(torch.rand(())) == pytest.approx(expected_torch)


def test_kinematic_stage_does_not_load_random_visual_checkpoint_head(tmp_path) -> None:
    source = V2NuScenesVelocityModel(mode="visual")
    with torch.no_grad():
        source.kinematic_encoder.residual_head[-1].weight.fill_(1.0)
        source.kinematic_encoder.residual_head[-1].bias.fill_(1.0)
    path = tmp_path / "visual.pth"
    torch.save({"model": source.state_dict()}, path)
    target = V2NuScenesVelocityModel(mode="visual_kinematic")
    _load_initial_weights(target, {"model": {"init_checkpoint": str(path)}}, "kinematic_pretrain")
    assert torch.count_nonzero(target.kinematic_encoder.residual_head[-1].weight) == 0
    assert torch.count_nonzero(target.kinematic_encoder.residual_head[-1].bias) == 0


def test_multiseed_acceptance_requires_average_epe_and_two_group_improvements() -> None:
    visual = [
        {"final_endpoint_error_mps": 1.0, "turning_mae_vy_mps": 0.4, "high_lateral_mae_vy_mps": 0.5}
        for _ in range(3)
    ]
    joint = [
        {"final_endpoint_error_mps": 1.01, "turning_mae_vy_mps": 0.3, "high_lateral_mae_vy_mps": 0.5},
        {"final_endpoint_error_mps": 1.02, "turning_mae_vy_mps": 0.4, "high_lateral_mae_vy_mps": 0.4},
        {"final_endpoint_error_mps": 1.00, "turning_mae_vy_mps": 0.4, "high_lateral_mae_vy_mps": 0.5},
    ]
    report = compare_multiseed([42, 3407, 2026], visual, joint)
    assert report["mean_epe_within_2_percent"]
    assert report["group_improved_seed_count"] == 2
    assert report["kinematics_claim_allowed"]


def test_weak_kinematic_encoder_overfits_fixed_tiny_batch() -> None:
    torch.manual_seed(42)
    encoder = WeakKinematicEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1.0e-2)
    values = torch.randn(8, 5, 12)
    valid = torch.ones_like(values, dtype=torch.bool)
    ages = torch.zeros_like(values)
    target = torch.tensor([0.4, -0.2]).repeat(8, 1)
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(encoder(values, valid, ages).residual, target)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        estimate = encoder(values, valid, ages)
        loss = torch.nn.functional.mse_loss(estimate.residual, target)
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        )
        optimizer.step()
    final = torch.nn.functional.mse_loss(encoder(values, valid, ages).residual, target)
    assert float(final.detach()) <= 0.2 * float(initial)


def test_geometry_mode_supports_independent_image_and_flow_sizes() -> None:
    expected = torch.tensor([[0.2, 0.0, 5.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(
        expected, 32, 48
    )
    model = V2NuScenesVelocityModel(mode="geometry_only")
    output = model(
        images=torch.randn(1, 3, 3, 12, 20),
        flow=flow,
        flow_confidence=confidence,
        intrinsics=intrinsics,
        frame_dt=dt,
        angular_rate_camera=TimedSignal(
            torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1)
        ),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        road_mask=mask,
    )
    assert output.predicted_velocity.shape == (1, 2)
    assert torch.count_nonzero(output.residual) == 0


def test_visual_mode_returns_finite_two_axis_velocity() -> None:
    expected = torch.tensor([[0.2, 0.0, 5.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(
        expected, 24, 32
    )
    model = V2NuScenesVelocityModel(mode="visual")

    output = model(
        images=torch.randn(1, 3, 3, 12, 16),
        flow=flow,
        flow_confidence=confidence,
        intrinsics=intrinsics,
        frame_dt=dt,
        angular_rate_camera=TimedSignal(
            torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1)
        ),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        road_mask=mask,
    )

    assert output.predicted_velocity.shape == (1, 2)
    assert torch.isfinite(output.predicted_velocity).all()
    assert torch.count_nonzero(output.kinematic_applied_correction) == 0


def test_invalid_geometry_uses_wheel_fallback() -> None:
    model = V2NuScenesVelocityModel(mode="geometry_only")
    images = torch.randn(1, 3, 3, 12, 20)
    intrinsics, normal, offset, rotation = camera_setup(height=16, width=24)
    output = model(
        images=images,
        flow=torch.zeros(1, 2, 16, 24),
        flow_confidence=torch.zeros(1, 1, 16, 24),
        intrinsics=intrinsics,
        frame_dt=torch.tensor([0.1]),
        angular_rate_camera=TimedSignal(
            torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1)
        ),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        wheel_speed=TimedSignal(
            torch.tensor([[8.0, 8.1, 7.9, 8.0]]),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1),
        ),
    )
    assert bool(output.fallback_used[0])
    assert float(output.predicted_velocity[0, 0]) == pytest.approx(8.0, abs=1.0e-5)


def test_grayscale_pretrained_stem_is_adapted_to_rgb(tmp_path) -> None:
    source = V2NuScenesVelocityModel(input_channels=1, mode="geometry_only")
    checkpoint = tmp_path / "gray.pth"
    torch.save({"state_dict": source.backbone.state_dict()}, checkpoint)
    target = V2NuScenesVelocityModel(input_channels=3, mode="geometry_only")

    report = target.load_flexinet_backbone(checkpoint)

    expected = source.backbone.cma_block.conv1.weight.repeat(1, 3, 1, 1, 1) / 3.0
    torch.testing.assert_close(target.backbone.cma_block.conv1.weight, expected)
    assert report["adapted_tensors"] == 1


def test_v2_forward_backward_has_finite_gradients() -> None:
    batch_size, height, width = 2, 24, 32
    expected = torch.tensor([[0.2, 0.0, 5.0], [0.1, -0.05, 4.0]])
    flow, confidence, intrinsics, dt, normal, offset, rotation, mask = synthetic_flow(
        expected, height, width
    )
    model = V2NuScenesVelocityModel(mode="visual_kinematic")
    images = torch.randn(batch_size, 3, 3, height, width)
    output = model(
        images=images,
        flow=flow,
        flow_confidence=confidence,
        intrinsics=intrinsics,
        frame_dt=dt,
        angular_rate_camera=TimedSignal(
            torch.zeros(batch_size, 3),
            torch.ones(batch_size, 3, dtype=torch.bool),
            torch.zeros(batch_size),
        ),
        plane_normal_camera=normal,
        plane_offset=offset,
        rotation_ego_from_camera=rotation,
        kinematic_values=torch.randn(batch_size, 21, 12),
        kinematic_valid=torch.ones(batch_size, 21, 12, dtype=torch.bool),
        kinematic_age=torch.zeros(batch_size, 21, 12),
        road_mask=mask,
    )
    target = expected[:, :2]
    terms = V2VelocityLoss()(output, target)
    assert torch.isfinite(terms.total)
    assert torch.isfinite(terms.kinematic_nll)
    terms.total.backward()
    gradients = [parameter.grad for parameter in model.residual_model.parameters() if parameter.grad is not None]
    gradients.extend(
        parameter.grad
        for parameter in model.kinematic_encoder.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
