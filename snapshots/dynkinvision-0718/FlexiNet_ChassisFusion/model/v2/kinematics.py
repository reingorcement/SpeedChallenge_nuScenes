from __future__ import annotations

import torch
import torch.nn as nn

from .types import KinematicEstimate


class WeakKinematicEncoder(nn.Module):
    """Encode deployable relative-motion CAN history without absolute wheel speed."""

    def __init__(
        self,
        value_dim: int = 12,
        step_dim: int = 64,
        hidden_dim: int = 128,
        residual_limits: tuple[float, float] = (1.0, 0.5),
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.input_dim = self.value_dim * 3
        self.step_encoder = nn.Sequential(
            nn.Linear(self.input_dim, step_dim),
            nn.ReLU(inplace=True),
            nn.Linear(step_dim, step_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(step_dim, hidden_dim, batch_first=True)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.log_variance_head = nn.Linear(hidden_dim, 2)
        self.register_buffer("residual_limits", torch.tensor(residual_limits, dtype=torch.float32))
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        output = self.residual_head[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        normalized_age: torch.Tensor,
        dropout_probability: float = 0.0,
    ) -> KinematicEstimate:
        if values.ndim != 3 or values.shape[-1] != self.value_dim:
            raise ValueError(f"values must have shape [B,K,{self.value_dim}].")
        if valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor matching values.")
        if normalized_age.shape != values.shape:
            raise ValueError("normalized_age must match values.")

        finite = torch.isfinite(values) & torch.isfinite(normalized_age)
        field_valid = valid & finite
        safe_values = torch.where(field_valid, values, torch.zeros_like(values))
        safe_age = torch.where(
            field_valid,
            normalized_age.clamp(0.0, 1.0),
            torch.ones_like(normalized_age),
        )
        encoded_input = torch.cat((safe_values, field_valid.to(values.dtype), safe_age), dim=-1)
        encoded_steps = self.step_encoder(encoded_input)
        _, hidden = self.gru(encoded_steps)
        latent = hidden[-1]
        residual = torch.tanh(self.residual_head(latent)) * self.residual_limits.to(latent)
        log_variance = self.log_variance_head(latent).clamp(-6.0, 4.0)
        coverage = field_valid.to(values.dtype).mean(dim=(1, 2), keepdim=False)[:, None]
        gate = coverage * torch.sigmoid(-log_variance)

        dropped = torch.zeros(values.shape[0], 1, device=values.device, dtype=torch.bool)
        if self.training and dropout_probability > 0.0:
            dropped = torch.rand(values.shape[0], 1, device=values.device) < dropout_probability
            gate = torch.where(dropped, torch.zeros_like(gate), gate)

        available = field_valid.any(dim=(1, 2)) & ~dropped[:, 0]
        return KinematicEstimate(
            residual=residual,
            log_variance=log_variance,
            gate=gate,
            coverage=coverage,
            available=available,
            latent=latent,
        )
