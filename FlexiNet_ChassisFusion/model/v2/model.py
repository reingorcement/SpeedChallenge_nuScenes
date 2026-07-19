from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..flexinet_backbone import FlexiNetBackbone
from .ground_geometry import RobustRoadPlaneVelocitySolver
from .kinematics import WeakKinematicEncoder
from .lever_arm import camera_to_reference_velocity, correct_reference_to_cg
from .residual import BoundedFiLMResidual, MonotonicUncertaintyGate
from .types import TimedSignal, V2VelocityOutput


class V2NuScenesVelocityModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        mode: str = "visual_kinematic",
        residual_limits: tuple[float, float] = (3.0, 1.5),
        kinematic_residual_limits: tuple[float, float] = (1.0, 0.5),
        wheel_scale_limit: float = 0.15,
        wheel_scale_jitter: float = 0.0,
        kinematic_dropout: float = 0.0,
        sensor_max_age_sec: float = 0.25,
        output_reference: str = "ego_reference",
    ) -> None:
        super().__init__()
        if mode not in {"geometry_only", "visual", "visual_kinematic"}:
            raise ValueError("mode must be 'geometry_only', 'visual', or 'visual_kinematic'.")
        if output_reference not in {"ego_reference", "cg"}:
            raise ValueError("output_reference must be 'ego_reference' or 'cg'.")
        self.mode = mode
        self.backbone = FlexiNetBackbone(input_channels=input_channels)
        self.geometry_solver = RobustRoadPlaneVelocitySolver()
        self.residual_model = BoundedFiLMResidual(
            condition_dim=4,
            residual_limits=residual_limits,
        )
        self.kinematic_encoder = WeakKinematicEncoder(
            value_dim=12,
            residual_limits=kinematic_residual_limits,
        )
        self.uncertainty_gate = MonotonicUncertaintyGate()
        self.wheel_scale_limit = float(wheel_scale_limit)
        self.wheel_scale_jitter = float(wheel_scale_jitter)
        self.kinematic_dropout = float(kinematic_dropout)
        self.sensor_max_age_sec = float(sensor_max_age_sec)
        self.output_reference = output_reference

    def load_flexinet_backbone(self, checkpoint_path: str | Path, strict: bool = False) -> dict[str, int]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {}
        target_state = self.backbone.state_dict()
        adapted_tensors = 0
        for key, value in state_dict.items():
            key = key.removeprefix("module.").removeprefix("backbone.")
            if key not in target_state:
                continue
            target = target_state[key]
            if value.shape == target.shape:
                cleaned[key] = value
            elif (
                key == "cma_block.conv1.weight"
                and value.ndim == 5
                and value.shape[1] == 1
                and target.shape[1] == 3
                and value.shape[0] == target.shape[0]
                and value.shape[2:] == target.shape[2:]
            ):
                cleaned[key] = value.repeat(1, 3, 1, 1, 1) / 3.0
                adapted_tensors += 1
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Backbone mismatch: missing={missing}, unexpected={unexpected}")
        return {
            "loaded_tensors": len(cleaned),
            "loaded_parameters": sum(int(value.numel()) for value in cleaned.values()),
            "adapted_tensors": adapted_tensors,
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
        }

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def freeze_visual(self) -> None:
        self.freeze_backbone()
        for parameter in self.residual_model.parameters():
            parameter.requires_grad = False

    def unfreeze_visual_tail(self) -> None:
        self.freeze_backbone()
        for module in (self.backbone.sfe_module, self.backbone.mfe_module, self.backbone.dig):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def set_residual_trainable(self, trainable: bool) -> None:
        for parameter in self.residual_model.parameters():
            parameter.requires_grad = trainable

    def set_kinematics_trainable(self, trainable: bool) -> None:
        for parameter in self.kinematic_encoder.parameters():
            parameter.requires_grad = trainable

    def set_stage_training_mode(
        self,
        stage: str,
        visual_tail_trainable: bool = False,
    ) -> None:
        """Set module modes without allowing frozen BatchNorm statistics to drift."""

        self.train()
        if stage == "visual":
            self.kinematic_encoder.eval()
            if visual_tail_trainable:
                self.backbone.cma_block.eval()
                self.backbone.aft_block.eval()
            else:
                self.backbone.eval()
        elif stage == "kinematic_pretrain":
            self.backbone.eval()
            self.residual_model.eval()
            self.kinematic_encoder.train()
        elif stage == "joint":
            self.backbone.cma_block.eval()
            self.backbone.aft_block.eval()
        elif stage == "geometry_audit":
            self.eval()
        else:
            raise ValueError(f"Unsupported training stage: {stage}")

    def forward(
        self,
        images: torch.Tensor,
        flow: torch.Tensor,
        flow_confidence: torch.Tensor,
        intrinsics: torch.Tensor,
        frame_dt: torch.Tensor,
        angular_rate_camera: TimedSignal,
        plane_normal_camera: torch.Tensor,
        plane_offset: torch.Tensor,
        rotation_ego_from_camera: torch.Tensor,
        kinematic_values: Optional[torch.Tensor] = None,
        kinematic_valid: Optional[torch.Tensor] = None,
        kinematic_age: Optional[torch.Tensor] = None,
        wheel_speed: Optional[TimedSignal] = None,
        road_mask: Optional[torch.Tensor] = None,
        reference_lever_arm_from_cg: Optional[torch.Tensor] = None,
    ) -> V2VelocityOutput:
        gyro_valid = angular_rate_camera.valid_fields(self.sensor_max_age_sec)
        gyro_value = torch.where(gyro_valid, angular_rate_camera.value, torch.zeros_like(angular_rate_camera.value))
        with torch.autocast(device_type=flow.device.type, enabled=False):
            geometry = self.geometry_solver(
                flow=flow.float(),
                confidence=flow_confidence.float(),
                intrinsics=intrinsics.float(),
                dt=frame_dt.float(),
                angular_rate_camera=gyro_value.float(),
                angular_rate_valid=gyro_valid,
                plane_normal_camera=plane_normal_camera.float(),
                plane_offset=plane_offset.float(),
                rotation_ego_from_camera=rotation_ego_from_camera.float(),
                road_mask=road_mask,
            )
        ego_3d = camera_to_reference_velocity(geometry.camera_velocity, rotation_ego_from_camera)
        angular_rate_ego = torch.matmul(
            rotation_ego_from_camera, gyro_value.unsqueeze(-1)
        ).squeeze(-1)
        corrected_3d, cg_applied = correct_reference_to_cg(
            ego_3d, angular_rate_ego, reference_lever_arm_from_cg
        )
        if self.output_reference == "cg" and not cg_applied:
            raise ValueError("CG output requires reference_lever_arm_from_cg calibration.")
        geometry_velocity = corrected_3d[:, :2]
        wheel_scale, wheel_fallback, fallback_used = self._wheel_scale_and_fallback(
            geometry_velocity, geometry.valid, wheel_speed
        )
        visual_base = geometry_velocity.clone()
        visual_base[:, 0] = visual_base[:, 0] * wheel_scale[:, 0]
        visual_base = torch.where(geometry.valid[:, None], visual_base, wheel_fallback)

        covariance_trace = geometry.covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        covariance_trace = torch.nan_to_num(
            covariance_trace, nan=1.0e6, posinf=1.0e6, neginf=0.0
        ).clamp(0.0, 1.0e6)
        condition = torch.nan_to_num(
            geometry.condition_number, nan=1.0e12, posinf=1.0e12, neginf=1.0e12
        ).clamp(0.0, 1.0e12)
        residual_rms = torch.nan_to_num(
            geometry.residual_rms, nan=100.0, posinf=100.0, neginf=0.0
        ).clamp(0.0, 100.0)
        geometry_context = torch.stack(
            (
                torch.log1p(covariance_trace),
                torch.log1p(condition),
                residual_rms,
                geometry.valid.to(images.dtype),
            ),
            dim=-1,
        )
        residual = images.new_zeros(images.shape[0], 2)
        residual_gate = images.new_zeros(images.shape[0], 1)
        visual_velocity = visual_base
        if self.mode in {"visual", "visual_kinematic"}:
            feature_map = self.backbone.forward_feature_map(images)
            residual = self.residual_model(feature_map, geometry_context)
            residual_gate = self.uncertainty_gate(geometry.covariance, geometry.valid)
            visual_velocity = visual_base + residual_gate * residual

        kinematic_residual = images.new_zeros(images.shape[0], 2)
        kinematic_correction = images.new_zeros(images.shape[0], 2)
        kinematic_gate = images.new_zeros(images.shape[0], 2)
        kinematic_log_variance = images.new_zeros(images.shape[0], 2)
        kinematic_coverage = images.new_zeros(images.shape[0], 1)
        kinematic_available = torch.zeros(images.shape[0], device=images.device, dtype=torch.bool)
        if self.mode == "visual_kinematic":
            if kinematic_values is None or kinematic_valid is None or kinematic_age is None:
                raise ValueError("visual_kinematic mode requires values, validity, and age sequences.")
            kinematic = self.kinematic_encoder(
                kinematic_values,
                kinematic_valid,
                kinematic_age,
                dropout_probability=self.kinematic_dropout,
            )
            kinematic_residual = kinematic.residual
            kinematic_gate = kinematic.gate
            kinematic_log_variance = kinematic.log_variance
            kinematic_coverage = kinematic.coverage
            kinematic_available = kinematic.available
            limits = self.kinematic_encoder.residual_limits.to(kinematic_residual)
            kinematic_correction = limits * torch.tanh(
                kinematic_gate * kinematic_residual / limits.clamp_min(1.0e-6)
            )
        predicted = visual_velocity + kinematic_correction
        return V2VelocityOutput(
            geometry_velocity=geometry_velocity,
            geometry_covariance=geometry.covariance,
            geometry_condition_number=geometry.condition_number,
            geometry_residual_rms=geometry.residual_rms,
            geometry_valid=geometry.valid,
            geometry_support_count=geometry.support_count,
            visual_velocity=visual_velocity,
            wheel_scale=wheel_scale,
            fallback_used=fallback_used,
            residual=residual,
            residual_gate=residual_gate,
            kinematic_residual=kinematic_residual,
            kinematic_applied_correction=kinematic_correction,
            kinematic_gate=kinematic_gate,
            kinematic_log_variance=kinematic_log_variance,
            kinematic_coverage=kinematic_coverage,
            kinematic_available=kinematic_available,
            predicted_velocity=predicted,
            output_reference=self.output_reference,
        )

    def _wheel_scale_and_fallback(
        self,
        geometry_velocity: torch.Tensor,
        geometry_valid: torch.Tensor,
        wheel_speed: Optional[TimedSignal],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = geometry_velocity.new_ones(geometry_velocity.shape[0], 1)
        fallback = geometry_velocity.new_zeros(geometry_velocity.shape[0], 2)
        fallback_used = torch.zeros(
            geometry_velocity.shape[0], device=geometry_velocity.device, dtype=torch.bool
        )
        if wheel_speed is None:
            return scale, fallback, fallback_used
        field_valid = wheel_speed.valid_fields(self.sensor_max_age_sec)
        weights = field_valid.to(geometry_velocity.dtype)
        wheel_mean = (torch.nan_to_num(wheel_speed.value) * weights).sum(dim=-1)
        wheel_mean = wheel_mean / weights.sum(dim=-1).clamp_min(1.0)
        ratio = wheel_mean / geometry_velocity[:, 0].abs().clamp_min(0.5)
        if self.training and self.wheel_scale_jitter > 0.0:
            jitter = 1.0 + (2.0 * torch.rand_like(ratio) - 1.0) * self.wheel_scale_jitter
            ratio = ratio * jitter
        ratio = ratio.clamp(1.0 - self.wheel_scale_limit, 1.0 + self.wheel_scale_limit)
        apply = geometry_valid & (weights.sum(dim=-1) > 0) & (geometry_velocity[:, 0].abs() >= 0.5)
        scale = torch.where(apply[:, None], ratio[:, None], scale)
        fallback_available = (~geometry_valid) & (weights.sum(dim=-1) >= 2)
        fallback[:, 0] = torch.where(fallback_available, wheel_mean.clamp_min(0.0), 0.0)
        fallback_used = fallback_available
        return scale, fallback, fallback_used
