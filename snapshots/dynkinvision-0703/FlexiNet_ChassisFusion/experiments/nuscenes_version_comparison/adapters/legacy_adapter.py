from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.path_sanity_check import build_model_from_config


def _optional(batch: dict[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    return value if isinstance(value, torch.Tensor) and value.numel() else None


class BaseVelocityAdapter(nn.Module):
    family = "reference"
    algorithm = "base_velocity"

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        return batch["base_velocity"], {}

    def load_checkpoint(self, path: str | Path) -> None:
        raise ValueError("base_velocity does not use a checkpoint.")


class LegacyAdapter(nn.Module):
    family = "legacy"

    def __init__(self, model: nn.Module, algorithm: str) -> None:
        super().__init__()
        self.model = model
        self.algorithm = algorithm

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        prediction = self.model(
            batch["images"],
            batch["chassis"],
            base_velocity=batch["base_velocity"],
            flow=_optional(batch, "flow"),
            geometry=_optional(batch, "geometry"),
            chassis_current_index=_optional(batch, "chassis_current_index"),
        )
        diagnostics: dict[str, Any] = {}
        gate = getattr(self.model, "last_gate_weights", None)
        names = getattr(self.model, "last_gate_names", None)
        if isinstance(gate, torch.Tensor) and names:
            diagnostics["gate_weights"] = gate
            diagnostics["gate_names"] = list(names)
        return prediction, diagnostics

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)


def _resolve_pretrained(path_value: str | None, project_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [project_root / path, Path.cwd() / path]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def build_legacy_adapter(
    config: dict[str, Any], *, load_pretrained: bool = False, project_root: Path | None = None
) -> LegacyAdapter:
    model = build_model_from_config(config)
    if load_pretrained and bool(config["model"].get("use_vision_branch", True)):
        root = project_root or Path(__file__).resolve().parents[3]
        pretrained = _resolve_pretrained(config["model"].get("pretrained_path"), root)
        if pretrained is None:
            raise FileNotFoundError(
                f"Legacy visual pretrained checkpoint not found: {config['model'].get('pretrained_path')}"
            )
        model.load_flexinet_backbone(pretrained, strict=False)
    if bool(config["model"].get("use_vision_branch", True)) and bool(
        config["model"].get("freeze_backbone", True)
    ):
        model.freeze_backbone()
    return LegacyAdapter(model, str(config["_comparison"]["name"]))
