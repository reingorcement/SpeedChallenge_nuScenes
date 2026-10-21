from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.physics_schema import PHYSICS_FEATURE_DIM, PHYSICS_INPUT_DIM, PHYSICS_INPUT_INDEX


@dataclass
class PhysicsEstimate:
    velocity: torch.Tensor
    confidence: torch.Tensor
    features: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class CausalPhysicsVelocityEstimator(nn.Module):
    """Differentiable causal velocity observer derived from the Simulink estimator.

    Inputs use SI units. Wheel speed is linear speed at each wheel in m/s.
    Missing measurements must be marked false in ``valid_mask``.
    """

    def __init__(
        self,
        mass: float = 3052.0,
        wheelbase: float = 3.128,
        lf: float = 1.293,
        front_track: float = 1.600,
        rear_track: float = 1.740,
    ) -> None:
        super().__init__()
        if min(mass, wheelbase, lf, front_track, rear_track) <= 0.0:
            raise ValueError("Vehicle geometry parameters must be positive.")
        if lf >= wheelbase:
            raise ValueError("lf must be smaller than wheelbase.")

        self.mass = float(mass)
        self.wheelbase = float(wheelbase)
        self.lf = float(lf)
        self.lr = float(wheelbase - lf)
        self.front_track = float(front_track)
        self.rear_track = float(rear_track)

        self.raw_cornering_front = nn.Parameter(torch.tensor(0.0))
        self.raw_cornering_rear = nn.Parameter(torch.tensor(math.log(7.0 / 5.0)))
        self.raw_observer_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_friction = nn.Parameter(torch.tensor(1.4))
        self.raw_process_noise = nn.Parameter(torch.tensor(_inverse_softplus(1.0e-2)))
        self.raw_measurement_noise = nn.Parameter(torch.tensor(_inverse_softplus(7.7574e-5)))

    @property
    def cornering_front(self) -> torch.Tensor:
        return 20_000.0 + 120_000.0 * torch.sigmoid(self.raw_cornering_front)

    @property
    def cornering_rear(self) -> torch.Tensor:
        return 20_000.0 + 120_000.0 * torch.sigmoid(self.raw_cornering_rear)

    @property
    def observer_gain(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_observer_gain)

    @property
    def learned_friction(self) -> torch.Tensor:
        return 0.2 + 1.0 * torch.sigmoid(self.raw_friction)

    @property
    def process_noise(self) -> torch.Tensor:
        return F.softplus(self.raw_process_noise).clamp_min(1.0e-7)

    @property
    def measurement_noise(self) -> torch.Tensor:
        return F.softplus(self.raw_measurement_noise).clamp_min(1.0e-7)

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        dt: torch.Tensor,
    ) -> PhysicsEstimate:
        if inputs.dim() != 3 or inputs.shape[-1] != PHYSICS_INPUT_DIM:
            raise ValueError(f"inputs must have shape [B,K,{PHYSICS_INPUT_DIM}].")
        if inputs.shape[1] == 0:
            raise ValueError("inputs must contain at least one time step.")
        if valid_mask.shape != inputs.shape:
            raise ValueError("valid_mask must have the same shape as inputs.")
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(inputs.shape[0], -1)
        if dt.shape != inputs.shape[:2]:
            raise ValueError("dt must have shape [K] or [B,K].")

        valid_mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
        dt = dt.to(device=inputs.device, dtype=inputs.dtype)
        if not torch.isfinite(dt).all():
            raise ValueError("dt must contain only finite values.")
        if (dt < 0.0).any():
            raise ValueError("dt must be non-negative.")
        if not torch.allclose(dt[:, 0], torch.zeros_like(dt[:, 0]), atol=1.0e-8, rtol=0.0):
            raise ValueError("dt[:, 0] must be zero because the first sample initializes the state.")
        if not torch.isfinite(torch.where(valid_mask, inputs, torch.zeros_like(inputs))).all():
            raise ValueError("Valid physics measurements must contain only finite values.")
        inputs = torch.where(valid_mask, inputs, torch.zeros_like(inputs))
        batch_size, steps, _ = inputs.shape
        vx = torch.zeros(batch_size, device=inputs.device, dtype=inputs.dtype)
        vy_dynamic = torch.zeros_like(vx)
        covariance = torch.ones_like(vx)
        previous_wheels = torch.zeros(batch_size, 4, device=inputs.device, dtype=inputs.dtype)
        previous_wheel_valid = torch.zeros(batch_size, 4, device=inputs.device, dtype=torch.bool)
        features = []

        for step in range(steps):
            current = inputs[:, step]
            current_valid = valid_mask[:, step]
            step_dt = dt[:, step]
            yaw_rate = self._value(current, current_valid, "yaw_rate")
            ax = self._value(current, current_valid, "longitudinal_accel")
            ay = self._value(current, current_valid, "lateral_accel")
            steering = self._steering(current, current_valid)
            base_vx = self._value(current, current_valid, "base_vx")
            base_valid = current_valid[:, PHYSICS_INPUT_INDEX["base_vx"]]

            wheels, wheel_valid = self._wheel_values(current, current_valid, yaw_rate)
            wheel_observation, wheel_mean, wheel_std, slip_score = self._wheel_observation(
                current=current,
                valid=current_valid,
                wheels=wheels,
                wheel_valid=wheel_valid,
                previous_wheels=previous_wheels,
                previous_valid=previous_wheel_valid,
                dt=step_dt,
                ax=ax,
            )
            wheel_measurement_valid = wheel_valid.any(dim=1)
            initial_vx = torch.where(wheel_measurement_valid, wheel_observation, base_vx)
            if step == 0:
                vx = torch.where(wheel_measurement_valid | base_valid, initial_vx, vx)

            vx_predict = vx + step_dt * (ax + yaw_rate * vy_dynamic)
            process_variance = self.process_noise * (1.0 + 4.0 * ax.abs() + 2.0 * yaw_rate.abs())
            predicted_covariance = covariance + process_variance * step_dt
            lateral_motion = (ay.abs() / 1.5 + yaw_rate.abs() / 0.1 + steering.abs() / 0.05).clamp(0.0, 10.0)
            measurement_variance = self.measurement_noise * (
                1.0 + 100.0 * slip_score + 10.0 * lateral_motion
            )
            kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance)
            kalman_gain = torch.where(wheel_measurement_valid, kalman_gain, torch.zeros_like(kalman_gain))
            vx_innovation = wheel_observation - vx_predict
            vx = (vx_predict + kalman_gain * vx_innovation).clamp(-80.0, 80.0)
            covariance = ((1.0 - kalman_gain) * predicted_covariance).clamp(1.0e-8, 1.0e4)

            vy_kinematic, vy_dynamic, vy, sideslip, lateral_excitation, confidence_vy = self._lateral_step(
                current=current,
                valid=current_valid,
                vx=vx,
                previous_vy_dynamic=vy_dynamic,
                yaw_rate=yaw_rate,
                steering=steering,
                ay=ay,
                dt=step_dt,
            )
            confidence_vx = (
                (1.0 - slip_score)
                * torch.exp(-vx_innovation.abs() / 3.0)
                * wheel_measurement_valid.to(inputs.dtype)
            ).clamp(0.0, 1.0)

            features.append(
                torch.stack(
                    [
                        vx_predict,
                        wheel_observation,
                        vx,
                        vy_kinematic,
                        vy_dynamic,
                        vy,
                        sideslip,
                        yaw_rate,
                        steering,
                        ax,
                        ay,
                        wheel_mean,
                        wheel_std,
                        slip_score,
                        vx_innovation,
                        lateral_excitation,
                        confidence_vx,
                        confidence_vy,
                    ],
                    dim=1,
                )
            )
            previous_wheels = wheels
            previous_wheel_valid = wheel_valid

        feature_sequence = torch.stack(features, dim=1)
        if feature_sequence.shape[-1] != PHYSICS_FEATURE_DIM:
            raise RuntimeError("Physics feature dimension mismatch.")
        return PhysicsEstimate(
            velocity=feature_sequence[:, -1, [2, 5]],
            confidence=feature_sequence[:, -1, [16, 17]],
            features=feature_sequence,
        )

    @staticmethod
    def _value(current: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
        index = PHYSICS_INPUT_INDEX[name]
        return torch.where(valid[:, index], current[:, index], torch.zeros_like(current[:, index]))

    def _steering(self, current: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        feedback_index = PHYSICS_INPUT_INDEX["steering_feedback"]
        command_index = PHYSICS_INPUT_INDEX["steering_command"]
        steering = torch.where(
            valid[:, feedback_index],
            current[:, feedback_index],
            torch.where(valid[:, command_index], current[:, command_index], torch.zeros_like(current[:, command_index])),
        )
        return steering.clamp(-0.7, 0.7)

    def _wheel_values(
        self,
        current: torch.Tensor,
        valid: torch.Tensor,
        yaw_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indexes = [PHYSICS_INPUT_INDEX[f"wheel_speed_{corner}"] for corner in ("fl", "fr", "rl", "rr")]
        wheels = current[:, indexes]
        wheel_valid = valid[:, indexes]
        offsets = wheels.new_tensor(
            [self.front_track / 2.0, -self.front_track / 2.0, self.rear_track / 2.0, -self.rear_track / 2.0]
        )
        wheels = wheels + yaw_rate.unsqueeze(1) * offsets.unsqueeze(0)
        reverse_index = PHYSICS_INPUT_INDEX["reverse_enable"]
        reverse_sign = torch.where(
            valid[:, reverse_index] & (current[:, reverse_index] > 0.5),
            -torch.ones_like(yaw_rate),
            torch.ones_like(yaw_rate),
        )
        return wheels * reverse_sign.unsqueeze(1), wheel_valid

    def _wheel_observation(
        self,
        current: torch.Tensor,
        valid: torch.Tensor,
        wheels: torch.Tensor,
        wheel_valid: torch.Tensor,
        previous_wheels: torch.Tensor,
        previous_valid: torch.Tensor,
        dt: torch.Tensor,
        ax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_float = wheel_valid.to(wheels.dtype)
        count = valid_float.sum(dim=1).clamp_min(1.0)
        wheel_mean = (wheels * valid_float).sum(dim=1) / count
        centered = (wheels - wheel_mean.unsqueeze(1)) * valid_float
        wheel_std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0.0) + 1.0e-8)
        deviation = centered.abs()
        acceleration = (wheels - previous_wheels) / dt.clamp_min(1.0e-6).unsqueeze(1)
        acceleration_error = (acceleration - ax.unsqueeze(1)).abs()
        acceleration_valid = wheel_valid & previous_valid & (dt > 0.0).unsqueeze(1)
        slip_per_wheel = torch.sigmoid((deviation - 0.5) / 0.15)
        slip_per_wheel = slip_per_wheel + acceleration_valid.to(wheels.dtype) * torch.sigmoid(
            (acceleration_error - 2.5) / 0.5
        )
        slip_per_wheel = (0.5 * slip_per_wheel).clamp(0.0, 1.0)
        robust_weights = torch.exp(-deviation / 0.5) * (1.0 - slip_per_wheel) * valid_float

        force_indexes = [PHYSICS_INPUT_INDEX[f"tire_fx_{corner}"] for corner in ("fl", "fr", "rl", "rr")]
        torque_indexes = [PHYSICS_INPUT_INDEX[f"wheel_torque_{corner}"] for corner in ("fl", "fr", "rl", "rr")]
        force_valid = valid[:, force_indexes]
        torque_valid = valid[:, torque_indexes]
        signed_effort = torch.where(force_valid, current[:, force_indexes], current[:, torque_indexes])
        effort_valid = force_valid | torque_valid
        rear_drive_index = PHYSICS_INPUT_INDEX["rear_drive_enable"]
        rear_drive_active = valid[:, rear_drive_index] & (current[:, rear_drive_index] > 0.5)
        rear_axle_mask = torch.tensor([False, False, True, True], device=valid.device).unsqueeze(0)
        drivetrain_mask = torch.where(
            rear_drive_active.unsqueeze(1),
            rear_axle_mask.expand_as(effort_valid),
            torch.ones_like(effort_valid),
        )
        effort_valid = effort_valid & drivetrain_mask
        effort_count = effort_valid.to(wheels.dtype).sum(dim=1).clamp_min(1.0)
        mean_effort = (signed_effort * effort_valid.to(wheels.dtype)).sum(dim=1) / effort_count
        selection_logits = -torch.sign(mean_effort).unsqueeze(1) * wheels / 0.5
        selection_weights = torch.softmax(selection_logits.masked_fill(~wheel_valid, -1.0e4), dim=1)
        full_available = effort_valid.any(dim=1, keepdim=True)
        weights = torch.where(full_available, robust_weights * selection_weights, robust_weights)
        weight_sum = weights.sum(dim=1)
        fallback = wheel_mean
        observation = torch.where(
            weight_sum > 1.0e-6,
            (weights * wheels).sum(dim=1) / weight_sum.clamp_min(1.0e-6),
            fallback,
        )
        slip_score = torch.where(
            wheel_valid.any(dim=1),
            (slip_per_wheel * valid_float).sum(dim=1) / count,
            torch.ones_like(wheel_mean),
        )
        return observation, wheel_mean, wheel_std, slip_score

    def _lateral_step(
        self,
        current: torch.Tensor,
        valid: torch.Tensor,
        vx: torch.Tensor,
        previous_vy_dynamic: torch.Tensor,
        yaw_rate: torch.Tensor,
        steering: torch.Tensor,
        ay: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stable_vx = torch.where(vx >= 0.0, vx.clamp_min(0.5), vx.clamp_max(-0.5))
        beta_kinematic = torch.atan((self.lr / self.wheelbase) * torch.tan(steering))
        vy_kinematic = vx * torch.tan(beta_kinematic)
        alpha_front = (previous_vy_dynamic + self.lf * yaw_rate) / stable_vx - steering
        alpha_rear = (previous_vy_dynamic - self.lr * yaw_rate) / stable_vx

        load_indexes = [PHYSICS_INPUT_INDEX[f"normal_load_{corner}"] for corner in ("fl", "fr", "rl", "rr")]
        load_valid = valid[:, load_indexes]
        measured_loads = current[:, load_indexes].clamp_min(0.0)
        static_front = self.mass * 9.81 * self.lr / self.wheelbase
        static_rear = self.mass * 9.81 * self.lf / self.wheelbase
        static_loads = current.new_tensor([static_front / 2.0, static_front / 2.0, static_rear / 2.0, static_rear / 2.0])
        loads = torch.where(load_valid, measured_loads, static_loads.unsqueeze(0))
        front_load = loads[:, :2].sum(dim=1)
        rear_load = loads[:, 2:].sum(dim=1)

        friction_index = PHYSICS_INPUT_INDEX["road_friction"]
        friction = torch.where(
            valid[:, friction_index],
            current[:, friction_index].clamp(0.1, 1.5),
            self.learned_friction.expand_as(vx),
        )
        linear_front = torch.clamp(-self.cornering_front * alpha_front, -friction * front_load, friction * front_load)
        linear_rear = torch.clamp(-self.cornering_rear * alpha_rear, -friction * rear_load, friction * rear_load)
        brush_front = self._brush_axle_force(alpha_front, loads[:, :2], friction)
        brush_rear = self._brush_axle_force(alpha_rear, loads[:, 2:], friction)
        full_load_available = load_valid.all(dim=1)
        front_force = torch.where(full_load_available, brush_front, linear_front)
        rear_force = torch.where(full_load_available, brush_rear, linear_rear)
        fx_indexes = [PHYSICS_INPUT_INDEX[f"tire_fx_{corner}"] for corner in ("fl", "fr")]
        fx_valid = valid[:, fx_indexes]
        front_fx = (current[:, fx_indexes] * fx_valid.to(current.dtype)).sum(dim=1)
        modeled_ay = (front_force * torch.cos(steering) + rear_force + front_fx * torch.sin(steering)) / self.mass
        lateral_innovation = ay - modeled_ay
        vy_derivative = (ay - yaw_rate * vx + self.observer_gain * lateral_innovation).clamp(-20.0, 20.0)
        vy_dynamic = (previous_vy_dynamic + dt * vy_derivative).clamp(-20.0, 20.0)

        lateral_excitation = (
            0.45 * (yaw_rate.abs() / 0.1)
            + 0.35 * (ay.abs() / 1.5)
            + 0.20 * (steering.abs() / 0.05)
        ).clamp(0.0, 1.0)
        dynamic_weight = torch.sigmoid((vx.abs() - 1.0) / 0.3) * (0.25 + 0.75 * lateral_excitation)
        vy = vy_kinematic * (1.0 - dynamic_weight) + vy_dynamic * dynamic_weight
        sideslip = torch.atan2(vy, stable_vx)
        ay_valid = valid[:, PHYSICS_INPUT_INDEX["lateral_accel"]]
        yaw_valid = valid[:, PHYSICS_INPUT_INDEX["yaw_rate"]]
        confidence_vy = (
            dynamic_weight
            * torch.exp(-lateral_innovation.abs() / 3.0)
            * (ay_valid & yaw_valid).to(current.dtype)
        ).clamp(0.0, 1.0)
        return vy_kinematic, vy_dynamic, vy, sideslip, lateral_excitation, confidence_vy

    @staticmethod
    def _brush_axle_force(alpha: torch.Tensor, loads: torch.Tensor, friction: torch.Tensor) -> torch.Tensor:
        # Parameters match Brush = [28.93, 1.496, 0.9486, 0.06054] in Init.m.
        a, b, c, d = 28.93, 1.496, 0.9486, 0.06054
        safe_loads = loads.clamp_min(1.0)
        abs_alpha = alpha.abs().unsqueeze(1)
        mu = friction.clamp_min(0.1).unsqueeze(1)
        normalized_slip = a * abs_alpha / (3.0 * mu * safe_loads.pow(d))
        adhesion = (
            b
            * safe_loads.pow(c - d)
            * a
            * abs_alpha
            * (1.0 - normalized_slip + normalized_slip.square() / 3.0)
        )
        saturated = b * mu * safe_loads.pow(c)
        magnitude = torch.where(normalized_slip < 1.0, adhesion, saturated)
        return (-torch.sign(alpha).unsqueeze(1) * magnitude).sum(dim=1)
