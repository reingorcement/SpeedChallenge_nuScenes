from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flexinet_backbone import FlexiNetBackbone
from .geometry import (
    RobustVisualTwistSolver,
    fuse_twist_estimates,
    transform_camera_twist_to_ego,
)
from .types import V3VelocityOutput


class V3VisualOnlyVelocityModel(nn.Module):
    """Causal visual metric velocity model with no chassis-shaped input surface."""

    REQUIRED_INPUTS = frozenset(
        {
            "images",
            "flows",
            "flow_confidence",
            "pair_dt",
            "intrinsics",
            "plane_normal_camera",
            "plane_offset",
            "rotation_ego_from_camera",
            "translation_ego_from_camera",
        }
    )

    def __init__(
        self,
        mode: str = "fusion",
        pair_count: int = 3,
        twist_dof: int = 6,
        support_mode: str = "learned",
        pooling: str = "regions",
        uncertainty: bool = True,
        direct_expert: bool = True,
        residual_limits: tuple[float, float] = (3.0, 1.5),
        fixed_roi_top_fraction: float = 0.45,
    ) -> None:
        super().__init__()
        if mode not in {"geometry_audit", "direct_pretrain", "fusion"}:
            raise ValueError("mode must be geometry_audit, direct_pretrain, or fusion.")
        if pair_count not in {1, 3}:
            raise ValueError("pair_count must be 1 or 3.")
        if support_mode not in {"fixed_roi", "fb_only", "learned"}:
            raise ValueError("support_mode must be fixed_roi, fb_only, or learned.")
        if pooling not in {"global", "regions"}:
            raise ValueError("pooling must be global or regions.")
        self.mode = mode
        self.pair_count = pair_count
        self.support_mode = support_mode
        self.pooling = pooling
        self.uncertainty_enabled = bool(uncertainty)
        self.direct_expert_enabled = bool(direct_expert)
        self.fixed_roi_top_fraction = float(fixed_roi_top_fraction)

        self.backbone = FlexiNetBackbone(input_channels=3)
        self.reliability_head = nn.Conv3d(256, 1, kernel_size=1)
        self.region_projection = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True)
        )
        descriptor_dim = 384 if pooling == "regions" else 64
        diagnostic_dim = 14
        hidden_dim = 192
        self.direct_head = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        fusion_dim = descriptor_dim + diagnostic_dim
        self.correction_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.geometry_solver = RobustVisualTwistSolver(twist_dof=twist_dof)
        self.register_buffer(
            "residual_limits", torch.tensor(residual_limits, dtype=torch.float32)
        )

    def load_direct_checkpoint(self, checkpoint_path: str | Path) -> dict[str, int]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        own_state = self.state_dict()
        compatible = {
            key.removeprefix("module."): value
            for key, value in state.items()
            if key.removeprefix("module.") in own_state
            and own_state[key.removeprefix("module.")].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return {
            "loaded_tensors": len(compatible),
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
        }

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_visual_tail(self) -> None:
        self.freeze_backbone()
        for module in (
            self.backbone.sfe_module,
            self.backbone.mfe_module,
            self.backbone.dig,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def set_stage_training_mode(self, freeze_backbone: bool) -> None:
        self.train()
        if freeze_backbone:
            self.freeze_backbone()
            self.backbone.eval()
        else:
            self.unfreeze_visual_tail()
            self.backbone.cma_block.eval()
            self.backbone.aft_block.eval()

    def forward(self, visual: Mapping[str, torch.Tensor]) -> V3VelocityOutput:
        missing = self.REQUIRED_INPUTS - visual.keys()
        if missing:
            raise ValueError(f"Missing V3 visual inputs: {sorted(missing)}")
        images = visual["images"]
        flows = visual["flows"][:, -self.pair_count :]
        confidence = visual["flow_confidence"][:, -self.pair_count :]
        pair_dt = visual["pair_dt"][:, -self.pair_count :]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W].")
        if flows.ndim != 5 or flows.shape[2] != 2:
            raise ValueError("flows must have shape [B,P,2,Hg,Wg].")

        feature_map = self.backbone.forward_feature_map(images)
        descriptor = self._descriptor(feature_map)
        if self.mode == "direct_pretrain":
            return self._direct_only_output(descriptor, flows)
        reliability = self._reliability(feature_map, flows.shape[-2:])
        if self.support_mode != "learned":
            reliability = torch.ones_like(reliability)
        support_mask = self._support_mask(flows)

        pair_estimates = []
        for pair_index in range(self.pair_count):
            pair_estimates.append(
                self.geometry_solver(
                    flow=flows[:, pair_index].float(),
                    confidence=confidence[:, pair_index].float(),
                    intrinsics=visual["intrinsics"].float(),
                    dt=pair_dt[:, pair_index].float(),
                    plane_normal_camera=visual["plane_normal_camera"].float(),
                    plane_offset=visual["plane_offset"].float(),
                    reliability=reliability[:, pair_index].float(),
                    support_mask=support_mask[:, pair_index]
                    if support_mask is not None
                    else None,
                )
            )

        pair_twist_camera = torch.stack(
            [item.twist_camera for item in pair_estimates], dim=1
        )
        pair_covariance_camera = torch.stack(
            [item.covariance_camera for item in pair_estimates], dim=1
        )
        pair_twist_ego, pair_covariance_ego = transform_camera_twist_to_ego(
            pair_twist_camera,
            pair_covariance_camera,
            visual["rotation_ego_from_camera"].float(),
            visual["translation_ego_from_camera"].float(),
        )
        pair_valid = torch.stack([item.valid for item in pair_estimates], dim=1)
        fused_twist, fused_covariance, geometry_valid, dispersion = (
            fuse_twist_estimates(pair_twist_ego, pair_covariance_ego, pair_valid)
        )
        geometry_velocity = fused_twist[:, :2]
        geometry_covariance = fused_covariance[:, :2, :2]
        pair_condition = torch.stack(
            [item.condition_number for item in pair_estimates], dim=1
        )
        pair_residual = torch.stack(
            [item.residual_rms_px for item in pair_estimates], dim=1
        )
        pair_support = torch.stack(
            [item.support_fraction for item in pair_estimates], dim=1
        )
        diagnostics = self._diagnostics(
            geometry_velocity,
            geometry_covariance,
            pair_condition,
            pair_residual,
            pair_support,
            pair_valid,
            dispersion,
            confidence,
        )
        fused_features = torch.cat((descriptor, diagnostics), dim=-1)
        direct_velocity = self.direct_head(descriptor)
        if not self.direct_expert_enabled:
            direct_velocity = torch.zeros_like(direct_velocity)
        correction = self.residual_limits.to(fused_features) * torch.tanh(
            self.correction_head(fused_features)
        )
        geometry_corrected = geometry_velocity + correction
        gate = torch.sigmoid(self.gate_head(fused_features))
        gate = gate * geometry_valid[:, None].to(gate.dtype)

        if self.mode == "geometry_audit":
            predicted = geometry_velocity
            gate = geometry_valid[:, None].to(predicted.dtype).expand(-1, 2)
        else:
            predicted = gate * geometry_corrected + (1.0 - gate) * direct_velocity
        log_variance = self.uncertainty_head(
            torch.cat((fused_features, predicted), dim=-1)
        )
        log_variance = (
            log_variance.clamp(-6.0, 6.0)
            if self.uncertainty_enabled
            else torch.zeros_like(log_variance)
        )
        valid_weight = pair_valid.to(pair_residual.dtype)
        valid_denominator = valid_weight.sum(dim=1).clamp_min(1.0)

        return V3VelocityOutput(
            predicted_velocity=predicted,
            predicted_log_variance=log_variance,
            geometry_velocity=geometry_velocity,
            geometry_corrected_velocity=geometry_corrected,
            direct_velocity=direct_velocity,
            visual_correction=correction,
            geometry_gate=gate,
            geometry_valid=geometry_valid,
            geometry_covariance=geometry_covariance,
            geometry_condition_number=(
                torch.nan_to_num(pair_condition, posinf=1.0e12) * valid_weight
            ).sum(dim=1)
            / valid_denominator,
            geometry_residual_rms_px=(pair_residual * valid_weight).sum(dim=1)
            / valid_denominator,
            geometry_support_fraction=(pair_support * valid_weight).sum(dim=1)
            / valid_denominator,
            pair_twist_camera=pair_twist_camera,
            pair_twist_ego=pair_twist_ego,
            pair_covariance_camera=pair_covariance_camera,
            pair_covariance_ego=pair_covariance_ego,
            pair_condition_number=pair_condition,
            pair_residual_rms_px=pair_residual,
            pair_support_fraction=pair_support,
            pair_valid=pair_valid,
            pair_predicted_flow=torch.stack(
                [item.predicted_flow for item in pair_estimates], dim=1
            ),
            pair_dispersion=dispersion,
            reliability=reliability,
            candidate_mask=torch.stack(
                [item.candidate_mask for item in pair_estimates], dim=1
            ),
        )

    def _descriptor(self, feature_map: torch.Tensor) -> torch.Tensor:
        size = (1, 2, 3) if self.pooling == "regions" else (1, 1, 1)
        pooled = F.adaptive_avg_pool3d(feature_map, size)
        if self.pooling == "regions":
            pooled = pooled.flip(dims=(3,))  # near-to-far, then left-to-right
        tokens = (
            pooled.squeeze(2).permute(0, 2, 3, 1).reshape(feature_map.shape[0], -1, 256)
        )
        return self.region_projection(tokens).flatten(1)

    def _direct_only_output(
        self, descriptor: torch.Tensor, flows: torch.Tensor
    ) -> V3VelocityOutput:
        batch_size, pair_count, _, height, width = flows.shape
        direct_velocity = self.direct_head(descriptor)
        if not self.direct_expert_enabled:
            direct_velocity = torch.zeros_like(direct_velocity)
        diagnostics = descriptor.new_zeros(batch_size, 14)
        fused_features = torch.cat((descriptor, diagnostics), dim=-1)
        log_variance = self.uncertainty_head(
            torch.cat((fused_features, direct_velocity), dim=-1)
        )
        log_variance = (
            log_variance.clamp(-6.0, 6.0)
            if self.uncertainty_enabled
            else torch.zeros_like(log_variance)
        )
        pair_twist = descriptor.new_zeros(batch_size, pair_count, 6)
        pair_covariance = (
            torch.eye(6, device=descriptor.device, dtype=descriptor.dtype)[
                None, None
            ].expand(batch_size, pair_count, -1, -1)
            * 1.0e6
        )
        pair_scalar = descriptor.new_zeros(batch_size, pair_count)
        pair_valid = torch.zeros(
            batch_size, pair_count, device=descriptor.device, dtype=torch.bool
        )
        return V3VelocityOutput(
            predicted_velocity=direct_velocity,
            predicted_log_variance=log_variance,
            geometry_velocity=descriptor.new_zeros(batch_size, 2),
            geometry_corrected_velocity=descriptor.new_zeros(batch_size, 2),
            direct_velocity=direct_velocity,
            visual_correction=descriptor.new_zeros(batch_size, 2),
            geometry_gate=descriptor.new_zeros(batch_size, 2),
            geometry_valid=torch.zeros(
                batch_size, device=descriptor.device, dtype=torch.bool
            ),
            geometry_covariance=(
                torch.eye(2, device=descriptor.device, dtype=descriptor.dtype)[
                    None
                ].expand(batch_size, -1, -1)
                * 1.0e6
            ),
            geometry_condition_number=descriptor.new_full((batch_size,), 1.0e12),
            geometry_residual_rms_px=descriptor.new_zeros(batch_size),
            geometry_support_fraction=descriptor.new_zeros(batch_size),
            pair_twist_camera=pair_twist,
            pair_twist_ego=pair_twist,
            pair_covariance_camera=pair_covariance,
            pair_covariance_ego=pair_covariance,
            pair_condition_number=descriptor.new_full((batch_size, pair_count), 1.0e12),
            pair_residual_rms_px=pair_scalar,
            pair_support_fraction=pair_scalar,
            pair_valid=pair_valid,
            pair_predicted_flow=torch.zeros_like(flows),
            pair_dispersion=descriptor.new_zeros(batch_size),
            reliability=descriptor.new_ones(batch_size, pair_count, 1, height, width),
            candidate_mask=torch.zeros(
                batch_size,
                pair_count,
                1,
                height,
                width,
                device=descriptor.device,
                dtype=torch.bool,
            ),
        )

    def _reliability(
        self, feature_map: torch.Tensor, geometry_size: tuple[int, int]
    ) -> torch.Tensor:
        logits = self.reliability_head(feature_map)[:, :, -self.pair_count :]
        batch_size = logits.shape[0]
        logits = logits.permute(0, 2, 1, 3, 4).reshape(
            batch_size * self.pair_count, 1, *logits.shape[-2:]
        )
        weights = F.interpolate(
            logits, geometry_size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(weights).reshape(
            batch_size, self.pair_count, 1, *geometry_size
        )

    def _support_mask(self, flows: torch.Tensor) -> torch.Tensor | None:
        if self.support_mode != "fixed_roi":
            return None
        mask = torch.zeros_like(flows[:, :, :1], dtype=torch.bool)
        top = int(round(flows.shape[-2] * self.fixed_roi_top_fraction))
        mask[..., top:, :] = True
        return mask

    @staticmethod
    def _diagnostics(
        geometry_velocity: torch.Tensor,
        covariance: torch.Tensor,
        condition: torch.Tensor,
        residual: torch.Tensor,
        support: torch.Tensor,
        valid: torch.Tensor,
        dispersion: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        valid_float = valid.to(geometry_velocity.dtype)
        denominator = valid_float.sum(dim=1).clamp_min(1.0)
        mean_condition = (
            torch.log1p(torch.nan_to_num(condition, posinf=1.0e12).clamp(0.0, 1.0e12))
            * valid_float
        ).sum(dim=1) / denominator
        mean_residual = (torch.nan_to_num(residual, nan=100.0) * valid_float).sum(
            dim=1
        ) / denominator
        mean_support = (support * valid_float).sum(dim=1) / denominator
        diagonal = covariance.diagonal(dim1=-2, dim2=-1)
        pair_pattern = F.pad(valid_float, (0, 3 - valid.shape[1]))
        return torch.cat(
            (
                geometry_velocity / geometry_velocity.new_tensor([5.0, 1.0]),
                torch.log1p(diagonal.clamp(0.0, 1.0e6)),
                mean_condition[:, None],
                mean_residual.clamp(0.0, 100.0)[:, None] / 10.0,
                mean_support[:, None],
                valid_float.mean(dim=1, keepdim=True),
                dispersion.clamp(0.0, 20.0)[:, None] / 5.0,
                confidence.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None],
                pair_pattern,
                (valid_float.sum(dim=1) > 0).to(valid_float.dtype)[:, None],
            ),
            dim=-1,
        )
