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
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
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
    ) -> None:
        super().__init__()
        self.backbone = FlexiNetBackbone(input_channels=input_channels)
        self.chassis_encoder = ChassisEncoder(
            input_dim=chassis_input_dim,
            hidden_dim=hidden_dim,
            output_dim=chassis_feature_dim,
            dropout=dropout,
        )
        self.fusion_mode = fusion_mode
        self.residual_speed_head = FusionHead(
            input_dim=256 + chassis_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.direct_speed_head = FusionHead(
            input_dim=256 + chassis_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, chassis_vec: torch.Tensor, base_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        visual_feat = self.backbone.forward_features(images)
        chassis_feat = self.chassis_encoder(chassis_vec)
        fused = torch.cat([visual_feat, chassis_feat], dim=1)

        if self.fusion_mode == "residual":
            delta = self.residual_speed_head(fused)
            if base_speed is None:
                base_speed = torch.zeros_like(delta)
            return base_speed + delta
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
        backbone_state.update({k: v for k, v in cleaned.items() if k in backbone_state})
        self.backbone.load_state_dict(backbone_state, strict=strict)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

