from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from train_v2 import build_model, forward_batch


class BaseVelocityAdapter(nn.Module):
    family = "reference"
    algorithm = "base_velocity"

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        return batch["base_velocity"], {}

    def load_checkpoint(self, path: str | Path) -> None:
        raise ValueError("base_velocity does not use a checkpoint.")


class V2Adapter(nn.Module):
    family = "v2"

    def __init__(self, model: nn.Module, algorithm: str) -> None:
        super().__init__()
        self.model = model
        self.algorithm = algorithm

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        output = forward_batch(self.model, batch)
        diagnostics = {
            "visual_velocity": output.visual_velocity,
            "geometry_velocity": output.geometry_velocity,
            "geometry_valid": output.geometry_valid,
            "geometry_condition_number": output.geometry_condition_number,
            "fallback_used": output.fallback_used,
            "wheel_scale": output.wheel_scale,
            "visual_residual_gate": output.residual_gate,
            "kinematic_gate": output.kinematic_gate,
            "kinematic_correction": output.kinematic_applied_correction,
        }
        return output.predicted_velocity, diagnostics

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)


def build_v2_adapter(config: dict[str, Any]) -> V2Adapter:
    return V2Adapter(build_model(config), str(config["_comparison"]["name"]))
