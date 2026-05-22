from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset
from model.fusion_model import FlexiNetChassisFusion
from train import build_criterion, load_config_from_yaml, run_eval
from utils.evaluation import copy_small_results_for_git, write_evaluation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="FlexiNet chassis fusion testing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/FlexiNet_ChassisFusion/eval")
    parser.add_argument("--results_dir", type=str, default="results_git/FlexiNet_ChassisFusion/test_eval")
    parser.add_argument("--allow_pose_fallback", action="store_true")
    args = parser.parse_args()

    cfg = load_config_from_yaml(Path(args.config))
    if args.allow_pose_fallback:
        cfg["data"]["allow_pose_fallback"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NuScenesChassisFusionDataset(
        dataroot=cfg["data"]["dataroot"],
        version=cfg["data"]["version"],
        split="val",
        img_size=(cfg["data"]["img_height"], cfg["data"]["img_width"]),
        seq_len=int(cfg["data"]["seq_len"]),
        cam_channel=cfg["data"]["cam_channel"],
        can_bus_root=cfg["data"]["can_bus_root"],
        allow_pose_fallback=bool(cfg["data"]["allow_pose_fallback"]),
        image_cache_size=int(cfg["data"].get("image_cache_size", 0)),
    )
    print(f"Eval scenes: {len(dataset.scene_names)} | Eval samples: {len(dataset)}")
    num_workers = int(cfg["train"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=16,
        chassis_feature_dim=int(cfg["model"]["chassis_feature_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        fusion_mode=str(cfg["model"]["fusion_mode"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    criterion = build_criterion(str(cfg["train"]["loss"]))
    result = run_eval(model, loader, criterion, device)
    metrics = write_evaluation_artifacts(result, args.output_dir, prefix="val")
    copy_small_results_for_git(args.output_dir, args.results_dir, include_predictions=False)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"MAE: {metrics['mae_mps']:.4f} m/s ({metrics['mae_kmh']:.4f} km/h)")
    print(f"RMSE: {metrics['rmse_mps']:.4f} m/s ({metrics['rmse_kmh']:.4f} km/h)")
    print(f"Base MAE: {metrics['base_mae_mps']:.4f} m/s ({metrics['base_mae_kmh']:.4f} km/h)")
    print(f"Base RMSE: {metrics['base_rmse_mps']:.4f} m/s ({metrics['base_rmse_kmh']:.4f} km/h)")
    print(f"MAE gain vs base: {metrics['mae_gain_vs_base_mps']:.4f} m/s")
    print(f"Saved evaluation artifacts to: {Path(args.output_dir)}")
    print(f"Copied small analysis artifacts to: {Path(args.results_dir)}")


if __name__ == "__main__":
    main()
