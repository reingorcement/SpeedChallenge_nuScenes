"""Reusable inference helper for analysis scripts.

This wraps the same dataset/model/checkpoint construction that ``test.py``
performs, but exposes it as a function so multiple CLIs (single-checkpoint
evaluation, checkpoint comparison, batch ablation summaries) can share it
without duplicating the boilerplate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.nuscenes_chassis import (  # noqa: E402  (path setup must come first)
    FAIR_CHASSIS_DIM,
    POSE_ASSISTED_CHASSIS_DIM,
    NuScenesChassisFusionDataset,
)
from model.fusion_model import FlexiNetChassisFusion  # noqa: E402
from train import build_criterion, run_eval  # noqa: E402
from utils.evaluation import EvaluationResult  # noqa: E402


@dataclass
class InferenceArtifacts:
    """Bundle returned by :func:`run_inference_from_checkpoint`."""

    result: EvaluationResult
    config: dict
    split: str
    checkpoint_path: Path
    dataset_size: int
    scene_count: int


def _build_dataset(cfg: dict, split: str) -> NuScenesChassisFusionDataset:
    return NuScenesChassisFusionDataset(
        dataroot=cfg["data"]["dataroot"],
        version=cfg["data"]["version"],
        split=split,
        img_size=(cfg["data"]["img_height"], cfg["data"]["img_width"]),
        seq_len=int(cfg["data"]["seq_len"]),
        cam_channel=cfg["data"]["cam_channel"],
        can_bus_root=cfg["data"]["can_bus_root"],
        allow_pose_fallback=bool(cfg["data"]["allow_pose_fallback"]),
        image_cache_size=int(cfg["data"].get("image_cache_size", 0)),
        chassis_mode=str(cfg["data"].get("chassis_mode", "fair_chassis")),
        chassis_window_sec=float(cfg["data"].get("chassis_window_sec", 0.0)),
        chassis_hz=float(cfg["data"].get("chassis_hz", 1.0)),
        online_mode=bool(cfg["data"].get("online_mode", False)),
        image_mode=str(cfg["data"].get("image_mode", "gray")),
        flow_mode=str(cfg["data"].get("flow_mode", "none")),
        flow_root=cfg["data"].get("flow_root"),
        geometry_mode=str(cfg["data"].get("geometry_mode", "basic")),
        base_velocity_source=str(cfg["data"].get("base_velocity_source", "zero_lateral")),
        eovm_base_root=cfg["data"].get("eovm_base_root"),
        frame_cache_root=cfg["data"].get("frame_cache_root"),
        max_scenes=cfg["data"].get("max_scenes"),
    )


def _build_model(cfg: dict) -> FlexiNetChassisFusion:
    chassis_mode = str(cfg["data"].get("chassis_mode", "fair_chassis"))
    chassis_input_dim = POSE_ASSISTED_CHASSIS_DIM if chassis_mode == "pose_assisted" else FAIR_CHASSIS_DIM
    input_channels = 3 if str(cfg["data"].get("image_mode", "gray")) == "rgb" else 1
    geometry_input_dim = 4 if str(cfg["data"].get("geometry_mode", "basic")) == "basic" else 0
    flow_enabled = str(cfg["data"].get("flow_mode", "none")) == "precomputed"
    return FlexiNetChassisFusion(
        input_channels=input_channels,
        chassis_input_dim=int(cfg["model"].get("chassis_input_dim", chassis_input_dim)),
        chassis_feature_dim=int(cfg["model"]["chassis_feature_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        fusion_mode=str(cfg["model"]["fusion_mode"]),
        output_dim=int(cfg["model"].get("output_dim", 2)),
        chassis_sequence_model=str(cfg["model"].get("chassis_sequence_model", "gru")),
        flow_input_channels=2 if flow_enabled else 0,
        flow_feature_dim=int(cfg["model"].get("flow_feature_dim", 0)) if flow_enabled else 0,
        geometry_input_dim=int(cfg["model"].get("geometry_input_dim", geometry_input_dim)),
        geometry_feature_dim=int(cfg["model"].get("geometry_feature_dim", 0)),
        use_vision_branch=bool(cfg["model"].get("use_vision_branch", True)),
        use_kinematic_branch=bool(cfg["model"].get("use_kinematic_branch", True)),
        use_dynamics_branch=bool(cfg["model"].get("use_dynamics_branch", False)),
        branch_feature_dim=int(cfg["model"].get("branch_feature_dim", 64)),
        dynamics_feature_dim=int(cfg["model"].get("dynamics_feature_dim", 8)),
        save_gate_weights=bool(cfg["model"].get("save_gate_weights", False)),
    )


def run_inference_from_checkpoint(
    cfg: dict,
    checkpoint_path: str | Path,
    split: str = "val",
    *,
    allow_pose_fallback: bool = False,
    device: Optional[torch.device] = None,
    max_eval_batches: int = 0,
) -> InferenceArtifacts:
    """Run the standard ``run_eval`` pipeline for a saved checkpoint.

    Parameters
    ----------
    cfg
        Loaded configuration dict (matches the YAML structure under ``configs/``).
    checkpoint_path
        Path to the ``.pth`` file saved by ``train.py``.
    split
        Dataset split to evaluate (``"val"``, ``"test"`` or anything supported
        by :class:`NuScenesChassisFusionDataset`).
    allow_pose_fallback
        Mirrors ``test.py --allow_pose_fallback``: lets the dataset fall back to
        ego-pose when CAN bus data is missing.
    device
        Override device. Defaults to ``cuda`` if available.
    max_eval_batches
        If > 0, stop after this many batches (useful for smoke-style runs).
    """

    if allow_pose_fallback:
        cfg["data"]["allow_pose_fallback"] = True

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _build_dataset(cfg, split=split)
    num_workers = int(cfg["train"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = _build_model(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    criterion = build_criterion(
        str(cfg["train"]["loss"]),
        sigma_vx=float(cfg["train"].get("sigma_vx", 1.0)),
        sigma_vy=float(cfg["train"].get("sigma_vy", 1.0)),
        speed_weight=float(cfg["train"].get("speed_loss_weight", 0.0)),
    )
    result = run_eval(model, loader, criterion, device, max_batches=max_eval_batches)

    return InferenceArtifacts(
        result=result,
        config=cfg,
        split=split,
        checkpoint_path=Path(checkpoint_path),
        dataset_size=len(dataset),
        scene_count=len(dataset.scene_names),
    )
