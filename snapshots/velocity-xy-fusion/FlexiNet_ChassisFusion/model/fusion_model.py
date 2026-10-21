from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

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
    ) -> None:
        super().__init__()
        self.backbone = FlexiNetBackbone(input_channels=input_channels)
        self.output_dim = output_dim
        self.chassis_encoder = ChassisSequenceEncoder(
            input_dim=chassis_input_dim,
            hidden_dim=hidden_dim,
            output_dim=chassis_feature_dim,
            dropout=dropout,
            model_type=chassis_sequence_model,
        )
        self.flow_encoder = (
            FlowEncoder(input_channels=flow_input_channels, output_dim=flow_feature_dim)
            if flow_input_channels > 0 and flow_feature_dim > 0
            else None
        )
        self.geometry_encoder = (
            GeometryEncoder(
                input_dim=geometry_input_dim,
                output_dim=geometry_feature_dim,
                hidden_dim=max(hidden_dim // 2, geometry_feature_dim, 8),
                dropout=dropout,
            )
            if geometry_input_dim > 0 and geometry_feature_dim > 0
            else None
        )
        self.fusion_mode = fusion_mode
        fusion_input_dim = 256 + chassis_feature_dim
        if self.flow_encoder is not None:
            fusion_input_dim += flow_feature_dim
        if self.geometry_encoder is not None:
            fusion_input_dim += geometry_feature_dim
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

    def forward(
        self,
        images: torch.Tensor,
        chassis_vec: torch.Tensor,
        base_speed: Optional[torch.Tensor] = None,
        base_velocity: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visual_feat = self.backbone.forward_features(images)
        chassis_feat = self.chassis_encoder(chassis_vec)
        features = [visual_feat, chassis_feat]
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
        fused = torch.cat(features, dim=1)

        if self.fusion_mode == "residual":
            delta = self.residual_speed_head(fused)
            if base_velocity is None:
                if base_speed is not None and self.output_dim == 2:
                    base_velocity = torch.cat([base_speed, torch.zeros_like(base_speed)], dim=1)
                elif base_speed is not None:
                    base_velocity = base_speed
                else:
                    base_velocity = torch.zeros_like(delta)
            return base_velocity + delta
        return self.direct_speed_head(fused)

    def load_flexinet_backbone(self, checkpoint_path: str | Path, strict: bool = False) -> None:
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
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

