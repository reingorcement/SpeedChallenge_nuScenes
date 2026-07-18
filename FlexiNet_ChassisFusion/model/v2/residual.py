from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedFiLMResidual(nn.Module):
    """Condition visual maps without admitting wheel speed to the residual branch."""

    def __init__(
        self,
        feature_channels: int = 256,
        condition_dim: int = 12,
        hidden_dim: int = 128,
        residual_limits: tuple[float, float] = (3.0, 1.5),
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_channels + self.condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.register_buffer("residual_limits", torch.tensor(residual_limits, dtype=torch.float32))
        self.reset_output_head()

    def reset_output_head(self) -> None:
        output = self.head[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, feature_map: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 5:
            raise ValueError("feature_map must have shape [B,C,T,H,W].")
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition must have shape [B,{self.condition_dim}].")
        gamma, beta = self.condition_encoder(condition).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)[:, :, None, None, None]
        beta = 0.1 * torch.tanh(beta)[:, :, None, None, None]
        modulated = feature_map * (1.0 + gamma) + beta
        pooled = F.adaptive_avg_pool3d(modulated, 1).flatten(1)
        raw = self.head(torch.cat((pooled, condition), dim=-1))
        return torch.tanh(raw) * self.residual_limits.to(raw)


class MonotonicUncertaintyGate(nn.Module):
    """Allow more bounded correction as geometry uncertainty increases."""

    def __init__(self, log_variance_threshold: float = 0.0, sharpness: float = 2.0) -> None:
        super().__init__()
        self.log_variance_threshold = float(log_variance_threshold)
        self.sharpness = float(sharpness)

    def forward(self, covariance: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        planar_variance = covariance[:, 0, 0] + covariance[:, 2, 2]
        log_variance = torch.log1p(planar_variance.clamp_min(0.0))
        gate = torch.sigmoid(self.sharpness * (log_variance - self.log_variance_threshold))
        gate = torch.where(valid, gate, torch.ones_like(gate))
        return gate[:, None]
