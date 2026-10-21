from __future__ import annotations

import csv
import json

import torch

from dataset.nuscenes_chassis import (
    FAIR_CHASSIS_DIM,
    NuScenesCanBusAdapter,
    PrecomputedPhysicsExtraLookup,
    _current_sequence_index,
)
from dataset.physics_schema import (
    PHYSICS_EXTRA_FEATURES,
    PHYSICS_FEATURE_DIM,
    PHYSICS_INPUT_DIM,
    PHYSICS_INPUT_INDEX,
)
from model.fusion_model import FlexiNetChassisFusion
from model.physics_velocity import CausalPhysicsVelocityEstimator
from train import KnowledgeDrivenVelocityLoss, validate_physics_config
from utils.evaluation import EvaluationResult, compute_metrics, save_predictions_csv


def _fair_physics_batch(batch_size: int = 2, steps: int = 8, speed: float = 10.0):
    inputs = torch.zeros(batch_size, steps, PHYSICS_INPUT_DIM)
    valid = torch.zeros_like(inputs, dtype=torch.bool)
    inputs[:, :, PHYSICS_INPUT_INDEX["base_vx"]] = speed
    inputs[:, :, PHYSICS_INPUT_INDEX["wheel_speed_fl"] : PHYSICS_INPUT_INDEX["wheel_speed_rr"] + 1] = speed
    valid[:, :, :10] = True
    dt = torch.full((batch_size, steps), 0.05)
    dt[:, 0] = 0.0
    return inputs, valid, dt


def test_physics_schema_dimensions_are_stable() -> None:
    assert PHYSICS_INPUT_DIM == 25
    assert PHYSICS_FEATURE_DIM == 18
    assert len(PHYSICS_EXTRA_FEATURES) == 15


def test_constant_straight_speed_is_preserved() -> None:
    inputs, valid, dt = _fair_physics_batch(speed=10.0)
    estimate = CausalPhysicsVelocityEstimator()(inputs, valid, dt)

    torch.testing.assert_close(estimate.velocity[:, 0], torch.full((2,), 10.0), atol=2.0e-2, rtol=0.0)
    torch.testing.assert_close(estimate.velocity[:, 1], torch.zeros(2), atol=1.0e-5, rtol=0.0)
    assert estimate.features.shape == (2, 8, PHYSICS_FEATURE_DIM)


def test_acceleration_prediction_works_without_wheel_measurement() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=5, speed=5.0)
    valid[:, :, 6:10] = False
    inputs[:, :, PHYSICS_INPUT_INDEX["longitudinal_accel"]] = 2.0

    estimate = CausalPhysicsVelocityEstimator()(inputs, valid, dt)

    torch.testing.assert_close(estimate.velocity[0, 0], torch.tensor(5.4), atol=1.0e-5, rtol=0.0)
    assert estimate.confidence[0, 0] == 0.0


def test_single_overspeed_wheel_is_downweighted() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=5, speed=10.0)
    inputs[:, :, PHYSICS_INPUT_INDEX["wheel_speed_fl"]] = 30.0

    estimate = CausalPhysicsVelocityEstimator()(inputs, valid, dt)

    assert estimate.velocity[0, 0] < 11.0
    assert estimate.features[0, -1, 13] > 0.1


def test_full_signal_and_reverse_paths_are_finite() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=5, speed=8.0)
    valid[:] = True
    inputs[:, :, 10:14] = 100.0
    inputs[:, :, 14:18] = 500.0
    inputs[:, :, 18:22] = 7_500.0
    inputs[:, :, PHYSICS_INPUT_INDEX["road_friction"]] = 0.8
    forward = CausalPhysicsVelocityEstimator()(inputs, valid, dt)
    assert torch.isfinite(forward.features).all()

    inputs[:, :, PHYSICS_INPUT_INDEX["reverse_enable"]] = 1.0
    reverse = CausalPhysicsVelocityEstimator()(inputs, valid, dt)
    assert reverse.velocity[0, 0] < 0.0


def test_physics_estimator_is_strictly_causal() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=8, speed=10.0)
    estimator = CausalPhysicsVelocityEstimator()
    baseline = estimator(inputs, valid, dt).features
    changed = inputs.clone()
    changed[:, 5:, :] = 100.0
    changed_result = estimator(changed, valid, dt).features

    torch.testing.assert_close(baseline[:, :5], changed_result[:, :5], atol=0.0, rtol=0.0)


def test_constrained_physics_parameters_and_gradients_are_finite() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=4, speed=7.0)
    estimator = CausalPhysicsVelocityEstimator()
    loss = estimator(inputs, valid, dt).velocity.square().mean()
    loss.backward()

    assert 20_000.0 <= float(estimator.cornering_front.detach()) <= 140_000.0
    assert 20_000.0 <= float(estimator.cornering_rear.detach()) <= 140_000.0
    assert 0.2 <= float(estimator.learned_friction.detach()) <= 1.2
    assert 0.0 <= float(estimator.observer_gain.detach()) <= 2.0
    assert float(estimator.process_noise.detach()) > 0.0
    assert float(estimator.measurement_noise.detach()) > 0.0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in estimator.parameters())


def test_hybrid_model_and_physics_losses_backpropagate() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=2, steps=5, speed=8.0)
    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=8,
        hidden_dim=32,
        fusion_mode="physics_residual_gated",
        use_vision_branch=False,
        use_kinematic_branch=True,
        use_dynamics_branch=False,
    )
    chassis = torch.randn(2, 5, FAIR_CHASSIS_DIM)
    pred = model(
        torch.randn(2, 3, 1, 32, 32),
        chassis,
        physics_inputs=inputs,
        physics_valid_mask=valid,
        physics_dt=dt,
    )
    criterion = KnowledgeDrivenVelocityLoss(
        sigma_vx=5.0,
        sigma_vy=0.5,
        beta_loss_weight=0.05,
        physics_prior_loss_weight=0.20,
        physics_correction_loss_weight=0.01,
    )
    target = torch.tensor([[8.1, 0.1], [7.9, -0.1]])
    loss = criterion(
        pred,
        target,
        chassis=chassis,
        physics_prior=model.last_physics_prior,
        physics_confidence=model.last_physics_confidence,
        correction_gate=model.last_correction_gate,
    )
    loss.backward()

    assert pred.shape == (2, 2)
    assert model.last_physics_prior is not None
    assert model.last_physics_confidence is not None
    assert model.last_correction_gate is not None
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_precomputed_full_physics_lookup_preserves_missing_mask(tmp_path) -> None:
    path = tmp_path / "scene-0001.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_us", "wheel_torque_fl", "road_friction"])
        writer.writeheader()
        writer.writerow({"timestamp_us": 100, "wheel_torque_fl": 12.0, "road_friction": 0.7})
    lookup = PrecomputedPhysicsExtraLookup(tmp_path)

    values, valid = lookup.lookup("scene-0001", 150)

    assert values[PHYSICS_EXTRA_FEATURES.index("wheel_torque_fl")] == 12.0
    assert values[PHYSICS_EXTRA_FEATURES.index("road_friction")] == 0.7
    assert valid[PHYSICS_EXTRA_FEATURES.index("wheel_torque_fl")]
    assert not valid[PHYSICS_EXTRA_FEATURES.index("wheel_torque_fr")]


def test_fair_adapter_marks_unavailable_physics_signals_invalid(tmp_path) -> None:
    adapter = NuScenesCanBusAdapter(tmp_path)
    records = {
        "vehicle_monitor": {"vehicle_speed": 36.0, "yaw_rate": 5.0, "steering": 10.0},
        "zoe_veh_info": {
            "FL_wheel_speed": 10_000.0,
            "FR_wheel_speed": 10_000.0,
            "RL_wheel_speed": 10_000.0,
            "RR_wheel_speed": 10_000.0,
            "longitudinal_accel": 0.2,
            "transversal_accel": 0.1,
        },
        "steeranglefeedback": {"value": 7.7},
    }
    adapter._select_latest_before = lambda _scene, message, _timestamp: records.get(message)  # type: ignore[method-assign]

    inputs, valid, dt, timestamps = adapter.extract_physics_sequence(
        "scene-0001", 1_000_000, window_sec=0.1, hz=10.0, online_mode=True
    )

    assert inputs.shape == (2, PHYSICS_INPUT_DIM)
    assert valid[:, :10].all()
    assert not valid[:, 10:].any()
    torch.testing.assert_close(dt, torch.tensor([0.0, 0.1]))
    assert timestamps == [900_000, 1_000_000]


def test_physics_metrics_and_csv_diagnostics_are_reported(tmp_path) -> None:
    metrics = compute_metrics(
        preds=[[5.0, 0.2]],
        targets=[[5.2, 0.1]],
        base_velocities=[[5.0, 0.0]],
        physics_priors=[[5.1, 0.05]],
    )
    assert "sideslip_mae_rad" in metrics
    assert metrics["physics_mae_vx_mps"] > 0.0
    assert "gain_vs_physics_vector_mps" in metrics

    result = EvaluationResult(
        loss=0.1,
        preds=[[5.0, 0.2]],
        targets=[[5.2, 0.1]],
        base_velocities=[[5.0, 0.0]],
        scene_names=["scene-0001"],
        center_tokens=["token"],
        center_timestamps=[123],
        physics_priors=[[5.1, 0.05]],
        physics_confidences=[[0.8, 0.6]],
        correction_gates=[[0.2, 0.4]],
    )
    output = tmp_path / "predictions.csv"
    save_predictions_csv(result, output)
    with output.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["physics_vx_mps"] == "5.1"
    assert row["physics_confidence_vy"] == "0.6"
    assert row["correction_gate_vx"] == "0.2"


def test_physics_config_requires_explicit_unit_audit() -> None:
    cfg = {"data": {"physics_units_audited": False}, "model": {"fusion_mode": "physics"}}
    try:
        validate_physics_config(cfg)
    except ValueError as exc:
        assert "physics_units_audited=true" in str(exc)
    else:
        raise AssertionError("Expected unaudited physics config to be rejected")

    cfg["data"]["physics_units_audited"] = True
    validate_physics_config(cfg)


def test_online_sequence_uses_last_frame_as_current() -> None:
    assert _current_sequence_index(13, online_mode=True) == 12
    assert _current_sequence_index(13, online_mode=False) == 6


def test_partial_can_record_uses_field_level_validity(tmp_path) -> None:
    adapter = NuScenesCanBusAdapter(tmp_path)
    records = {
        "vehicle_monitor": {"vehicle_speed": 36.0, "yaw_rate": float("nan")},
        "zoe_veh_info": {"FL_wheel_speed": 10_000.0, "longitudinal_accel": 0.2},
        "steeranglefeedback": {"value": None},
    }
    adapter._select_latest_before = lambda _scene, message, _timestamp: records.get(message)  # type: ignore[method-assign]

    inputs, valid, _, _ = adapter.extract_physics_sequence(
        "scene-0001", 1_000_000, window_sec=0.0, hz=20.0, online_mode=True
    )

    assert valid[0, PHYSICS_INPUT_INDEX["base_vx"]]
    assert not valid[0, PHYSICS_INPUT_INDEX["yaw_rate"]]
    assert valid[0, PHYSICS_INPUT_INDEX["longitudinal_accel"]]
    assert not valid[0, PHYSICS_INPUT_INDEX["lateral_accel"]]
    assert valid[0, PHYSICS_INPUT_INDEX["wheel_speed_fl"]]
    assert not valid[0, PHYSICS_INPUT_INDEX["wheel_speed_fr"]]
    assert not valid[0, PHYSICS_INPUT_INDEX["steering_feedback"]]
    assert torch.isfinite(inputs).all()


def test_stale_can_and_full_physics_records_become_invalid(tmp_path) -> None:
    can_row = [{"utime": 100, "vehicle_speed": 36.0}]
    (tmp_path / "scene-0001_vehicle_monitor.json").write_text(json.dumps(can_row), encoding="utf-8")
    adapter = NuScenesCanBusAdapter(tmp_path, max_age_us=50)
    assert adapter._select_latest_before("scene-0001", "vehicle_monitor", 200) is None

    with (tmp_path / "scene-0001.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_us", "road_friction"])
        writer.writeheader()
        writer.writerow({"timestamp_us": 100, "road_friction": 0.8})
    lookup = PrecomputedPhysicsExtraLookup(tmp_path, max_age_us=50)
    _, valid = lookup.lookup("scene-0001", 200)
    assert not any(valid)


def test_invalid_physics_time_axis_and_measurements_are_rejected() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=3)
    estimator = CausalPhysicsVelocityEstimator()
    for invalid_dt in [torch.tensor([[0.05, 0.05, 0.05]]), torch.tensor([[0.0, -0.05, 0.05]])]:
        try:
            estimator(inputs, valid, invalid_dt)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid dt to be rejected")

    invalid_inputs = inputs.clone()
    invalid_inputs[0, 1, PHYSICS_INPUT_INDEX["yaw_rate"]] = float("nan")
    try:
        estimator(invalid_inputs, valid, dt)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite valid measurement to be rejected")


def test_rear_drive_flag_changes_full_signal_wheel_selection() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=1, steps=2, speed=5.0)
    inputs[:, :, 6:10] = torch.tensor([5.0, 5.0, 10.0, 10.0])
    valid[:, :, 10:25] = True
    inputs[:, :, 14:18] = torch.tensor([100.0, 100.0, -10.0, -10.0])
    inputs[:, :, 18:22] = 7_500.0
    inputs[:, :, PHYSICS_INPUT_INDEX["road_friction"]] = 0.8
    estimator = CausalPhysicsVelocityEstimator()
    all_wheel_observation = estimator(inputs, valid, dt).features[0, -1, 1]

    inputs[:, :, PHYSICS_INPUT_INDEX["rear_drive_enable"]] = 1.0
    rear_drive_observation = estimator(inputs, valid, dt).features[0, -1, 1]

    assert rear_drive_observation > all_wheel_observation + 1.0


def test_explicit_current_index_selects_online_chassis_frame() -> None:
    chassis = torch.zeros(2, 3, FAIR_CHASSIS_DIM)
    chassis[:, 0, 0] = 1.0
    chassis[:, 1, 0] = 2.0
    chassis[:, 2, 0] = 3.0
    selected = FlexiNetChassisFusion._select_chassis_frame(chassis, torch.tensor([2, 2]))
    torch.testing.assert_close(selected[:, 0], torch.tensor([3.0, 3.0]))


def test_physics_feature_only_branch_runs_without_vision_or_kinematics() -> None:
    inputs, valid, dt = _fair_physics_batch(batch_size=2, steps=4, speed=6.0)
    model = FlexiNetChassisFusion(
        chassis_input_dim=FAIR_CHASSIS_DIM,
        hidden_dim=24,
        dropout=0.0,
        fusion_mode="physics_residual_gated",
        use_vision_branch=False,
        use_kinematic_branch=False,
        use_dynamics_branch=False,
        branch_feature_dim=16,
    )
    pred = model(
        torch.zeros(2, 1, 1, 8, 8),
        torch.zeros(2, 4, FAIR_CHASSIS_DIM),
        physics_inputs=inputs,
        physics_valid_mask=valid,
        physics_dt=dt,
        chassis_current_index=torch.tensor([3, 3]),
    )
    pred.square().mean().backward()

    assert pred.shape == (2, 2)
    assert model.physics_encoder is not None
    assert any(parameter.grad is not None for parameter in model.physics_encoder.parameters())
