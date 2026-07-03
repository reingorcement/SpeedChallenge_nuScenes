from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from dataset.nuscenes_chassis import FAIR_CHASSIS_INDEX

from .flexinet_backbone import FlexiNetBackbone


class ChassisEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChassisSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 16,
        dropout: float = 0.3,
        model_type: str = "gru",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        if model_type == "gru":
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(inplace=True),
            )
        elif model_type in {"mean", "none"}:
            self.encoder = ChassisEncoder(input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        else:
            raise ValueError("chassis_sequence_model must be 'gru', 'mean', or 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.model_type == "gru":
            _, hidden = self.gru(x)
            return self.proj(hidden[-1])
        return self.encoder(x.mean(dim=1))


class FlowEncoder(nn.Module):
    def __init__(self, input_channels: int = 2, output_dim: int = 16) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.dim() != 5:
            raise ValueError("flow must have shape [B,T,C,H,W].")
        batch_size, steps, channels, height, width = flow.shape
        encoded = self.net(flow.reshape(batch_size * steps, channels, height, width)).view(batch_size, steps, 32)
        return self.proj(encoded.mean(dim=1))


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsProxyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConfidenceGatedFusionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        branch_count: int,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if branch_count <= 0:
            raise ValueError("branch_count must be positive.")
        self.branch_count = branch_count
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * branch_count, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, branch_count),
        )
        self.pred_head = FusionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
        )

    def forward(self, branch_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(branch_features) != self.branch_count:
            raise ValueError(f"Expected {self.branch_count} branches, got {len(branch_features)}.")
        stacked = torch.stack(branch_features, dim=1)
        gate_logits = self.gate_mlp(stacked.flatten(1))
        gate = torch.softmax(gate_logits, dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        return self.pred_head(fused), gate


class FlexiNetChassisFusion(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        chassis_input_dim: int = 16,
        chassis_feature_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        fusion_mode: str = "residual",
        output_dim: int = 2,
        chassis_sequence_model: str = "gru",
        flow_input_channels: int = 0,
        flow_feature_dim: int = 0,
        geometry_input_dim: int = 0,
        geometry_feature_dim: int = 0,
        use_vision_branch: bool = True,
        use_kinematic_branch: bool = True,
        use_dynamics_branch: bool = False,
        branch_feature_dim: int = 64,
        dynamics_feature_dim: int = 8,
        save_gate_weights: bool = False,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"residual", "direct", "residual_gated"}:
            raise ValueError("fusion_mode must be 'residual', 'direct', or 'residual_gated'.")
        if not any([use_vision_branch, use_kinematic_branch, use_dynamics_branch]):
            raise ValueError("At least one of vision, kinematic, or dynamics branches must be enabled.")
        if use_dynamics_branch and dynamics_feature_dim != 8:
            raise ValueError("dynamics_feature_dim must be 8 for the current dynamics proxy.")

        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        self.use_vision_branch = bool(use_vision_branch)
        self.use_kinematic_branch = bool(use_kinematic_branch)
        self.use_dynamics_branch = bool(use_dynamics_branch)
        self.branch_feature_dim = int(branch_feature_dim)
        self.dynamics_feature_dim = int(dynamics_feature_dim)
        self.save_gate_weights = bool(save_gate_weights)
        self.last_gate_weights: Optional[torch.Tensor] = None
        self.last_gate_names: list[str] = []
        self.backbone = FlexiNetBackbone(input_channels=input_channels) if self.use_vision_branch else None

        self.chassis_encoder = ChassisSequenceEncoder(
            input_dim=chassis_input_dim,
            hidden_dim=hidden_dim,
            output_dim=chassis_feature_dim,
            dropout=dropout,
            model_type=chassis_sequence_model,
        )
        self.flow_encoder = (
            FlowEncoder(input_channels=flow_input_channels, output_dim=flow_feature_dim)
            if self.use_vision_branch and flow_input_channels > 0 and flow_feature_dim > 0
            else None
        )
        self.geometry_encoder = (
            GeometryEncoder(
                input_dim=geometry_input_dim,
                output_dim=geometry_feature_dim,
                hidden_dim=max(hidden_dim // 2, geometry_feature_dim, 8),
                dropout=dropout,
            )
            if self.use_vision_branch and geometry_input_dim > 0 and geometry_feature_dim > 0
            else None
        )

        self.vision_input_dim = 256
        if self.flow_encoder is not None:
            self.vision_input_dim += flow_feature_dim
        if self.geometry_encoder is not None:
            self.vision_input_dim += geometry_feature_dim

        fusion_input_dim = 0
        if self.use_vision_branch:
            fusion_input_dim += self.vision_input_dim
        if self.use_kinematic_branch:
            fusion_input_dim += chassis_feature_dim

        self.dynamics_encoder = (
            DynamicsProxyEncoder(
                input_dim=dynamics_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=self.branch_feature_dim,
                dropout=dropout,
            )
            if self.use_dynamics_branch
            else None
        )
        if self.dynamics_encoder is not None:
            fusion_input_dim += self.branch_feature_dim

        self.residual_speed_head = FusionHead(
            input_dim=fusion_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
        )
        self.direct_speed_head = FusionHead(
            input_dim=fusion_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
        )
        self.vision_projection = (
            ProjectionBlock(self.vision_input_dim, self.branch_feature_dim, dropout=dropout)
            if self.fusion_mode == "residual_gated" and self.use_vision_branch
            else None
        )
        self.kinematic_projection = (
            ProjectionBlock(chassis_feature_dim, self.branch_feature_dim, dropout=dropout)
            if self.fusion_mode == "residual_gated" and self.use_kinematic_branch
            else None
        )
        active_branch_count = sum(
            [self.use_vision_branch, self.use_kinematic_branch, self.use_dynamics_branch]
        )
        self.gated_fusion_head = (
            ConfidenceGatedFusionHead(
                feature_dim=self.branch_feature_dim,
                branch_count=active_branch_count,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            if self.fusion_mode == "residual_gated"
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        chassis_vec: torch.Tensor,
        base_speed: Optional[torch.Tensor] = None,
        base_velocity: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_gate_weights = None
        self.last_gate_names = []

        if self.fusion_mode == "residual_gated":
            return self._forward_residual_gated(
                images=images,
                chassis_vec=chassis_vec,
                base_speed=base_speed,
                base_velocity=base_velocity,
                flow=flow,
                geometry=geometry,
            )

        features = []
        if self.use_vision_branch:
            features.append(self._encode_vision(images, flow=flow, geometry=geometry))
        if self.use_kinematic_branch:
            features.append(self.chassis_encoder(chassis_vec))
        if self.dynamics_encoder is not None:
            dynamics_input = self._build_dynamics_proxy(
                chassis_vec=chassis_vec,
                base_speed=base_speed,
                base_velocity=base_velocity,
                batch_size=images.shape[0],
                device=images.device,
                dtype=images.dtype,
            )
            features.append(self.dynamics_encoder(dynamics_input))
        fused = torch.cat(features, dim=1)

        if self.fusion_mode == "residual":
            delta = self.residual_speed_head(fused)
            return self._resolve_base_velocity(
                delta=delta,
                base_speed=base_speed,
                base_velocity=base_velocity,
            ) + delta
        return self.direct_speed_head(fused)

    def _encode_vision(
        self,
        images: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.backbone is None:
            raise RuntimeError("Vision branch is disabled, so the FlexiNet backbone is not available.")
        visual_feat = self.backbone.forward_features(images)
        features = [visual_feat]
        if self.flow_encoder is not None:
            if flow is None or flow.numel() == 0:
                features.append(torch.zeros(images.shape[0], self.flow_encoder.output_dim, device=images.device))
            else:
                features.append(self.flow_encoder(flow))
        if self.geometry_encoder is not None:
            if geometry is None or geometry.numel() == 0:
                features.append(torch.zeros(images.shape[0], self.geometry_encoder.output_dim, device=images.device))
            else:
                features.append(self.geometry_encoder(geometry))
        return torch.cat(features, dim=1)

    def _forward_residual_gated(
        self,
        images: torch.Tensor,
        chassis_vec: torch.Tensor,
        base_speed: Optional[torch.Tensor] = None,
        base_velocity: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gated_fusion_head is None:
            raise RuntimeError("residual_gated mode requires gated_fusion_head.")
        branches: list[torch.Tensor] = []
        branch_names: list[str] = []

        if self.use_vision_branch:
            if self.vision_projection is None:
                raise RuntimeError("vision_projection is required when vision branch is active.")
            branches.append(self.vision_projection(self._encode_vision(images, flow=flow, geometry=geometry)))
            branch_names.append("vision")
        if self.use_kinematic_branch:
            if self.kinematic_projection is None:
                raise RuntimeError("kinematic_projection is required when kinematic branch is active.")
            branches.append(self.kinematic_projection(self.chassis_encoder(chassis_vec)))
            branch_names.append("kinematic")
        if self.dynamics_encoder is not None:
            dynamics_input = self._build_dynamics_proxy(
                chassis_vec=chassis_vec,
                base_speed=base_speed,
                base_velocity=base_velocity,
                batch_size=images.shape[0],
                device=images.device,
                dtype=images.dtype,
            )
            branches.append(self.dynamics_encoder(dynamics_input))
            branch_names.append("dynamics")

        delta, gate = self.gated_fusion_head(branches)
        if self.save_gate_weights:
            self.last_gate_weights = gate.detach()
            self.last_gate_names = branch_names
        return self._resolve_base_velocity(
            delta=delta,
            base_speed=base_speed,
            base_velocity=base_velocity,
        ) + delta

    def _build_dynamics_proxy(
        self,
        chassis_vec: torch.Tensor,
        base_speed: Optional[torch.Tensor],
        base_velocity: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if chassis_vec.dim() == 3:
            center_vec = chassis_vec[:, chassis_vec.shape[1] // 2, :]
        elif chassis_vec.dim() == 2:
            center_vec = chassis_vec
        else:
            raise ValueError("chassis_vec must have shape [B,C] or [B,K,C].")
        if center_vec.shape[1] <= FAIR_CHASSIS_INDEX["base_vx"]:
            raise ValueError("chassis_vec does not contain the required fair chassis proxy features.")

        resolved_base_velocity = self._resolve_base_velocity(
            delta=torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype),
            base_speed=base_speed,
            base_velocity=base_velocity,
        )
        return torch.stack(
            [
                resolved_base_velocity[:, 0],
                resolved_base_velocity[:, 1],
                center_vec[:, FAIR_CHASSIS_INDEX["yaw_rate"]],
                center_vec[:, FAIR_CHASSIS_INDEX["steering"]],
                center_vec[:, FAIR_CHASSIS_INDEX["steering_feedback"]],
                center_vec[:, FAIR_CHASSIS_INDEX["longitudinal_accel"]],
                center_vec[:, FAIR_CHASSIS_INDEX["transversal_accel"]],
                center_vec[:, FAIR_CHASSIS_INDEX["wheel_speed_std"]],
            ],
            dim=1,
        ).to(device=device, dtype=dtype)

    def _resolve_base_velocity(
        self,
        delta: torch.Tensor,
        base_speed: Optional[torch.Tensor] = None,
        base_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_velocity is not None:
            return base_velocity.to(device=delta.device, dtype=delta.dtype)
        if base_speed is not None and self.output_dim == 2:
            base_speed = base_speed.to(device=delta.device, dtype=delta.dtype)
            return torch.cat([base_speed, torch.zeros_like(base_speed)], dim=1)
        if base_speed is not None:
            return base_speed.to(device=delta.device, dtype=delta.dtype)
        return torch.zeros_like(delta)

    def load_flexinet_backbone(self, checkpoint_path: str | Path, strict: bool = False) -> None:
        if self.backbone is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            if key.startswith("cma_block.") or key.startswith("aft_block.") or key.startswith("sfe_module.") or key.startswith("mfe_module.") or key.startswith("dig.") or key.startswith("global_pool."):
                cleaned[key] = value
        backbone_state = self.backbone.state_dict()
        backbone_state.update(
            {
                k: v
                for k, v in cleaned.items()
                if k in backbone_state and backbone_state[k].shape == v.shape
            }
        )
        self.backbone.load_state_dict(backbone_state, strict=strict)

    def freeze_backbone(self) -> None:
        if self.backbone is None:
            return
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        if self.backbone is None:
            return
        for param in self.backbone.parameters():
            param.requires_grad = True

