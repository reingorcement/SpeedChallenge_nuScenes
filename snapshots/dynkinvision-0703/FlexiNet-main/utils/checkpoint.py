from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state[key] = value

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(cleaned_state, strict=True)
    return model
