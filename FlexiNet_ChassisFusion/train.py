from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset
from model.fusion_model import FlexiNetChassisFusion
from utils.evaluation import (
    EvaluationResult,
    compute_metrics,
    copy_small_results_for_git,
    plot_training_history,
    save_history_csv,
    save_history_json,
    write_evaluation_artifacts,
)
from utils.seed import set_seed


def load_config_from_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_criterion(name: str) -> nn.Module:
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name in {"smoothl1", "huber"}:
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss: {name}")


def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EvaluationResult:
    model.eval()
    losses = []
    preds: list[float] = []
    targets: list[float] = []
    base_speeds: list[float] = []
    scene_names: list[str] = []
    center_tokens: list[str] = []
    center_timestamps: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["images"].to(device)
            chassis = batch["chassis"].to(device)
            base_speed = batch["base_speed"].to(device)
            speed = batch["speed"].to(device)
            pred = model(images, chassis, base_speed=base_speed)
            loss = criterion(pred, speed)
            losses.append(loss.item())
            preds.extend(pred.squeeze(1).detach().cpu().tolist())
            targets.extend(speed.squeeze(1).detach().cpu().tolist())
            base_speeds.extend(base_speed.squeeze(1).detach().cpu().tolist())

            if "scene_name" in batch:
                scene_names.extend(list(batch["scene_name"]))
            if "center_token" in batch:
                center_tokens.extend(list(batch["center_token"]))
            if "center_timestamp" in batch:
                timestamps = batch["center_timestamp"]
                if isinstance(timestamps, torch.Tensor):
                    center_timestamps.extend([int(v) for v in timestamps.detach().cpu().tolist()])
                else:
                    center_timestamps.extend([int(v) for v in timestamps])
    avg_loss = sum(losses) / max(len(losses), 1)
    return EvaluationResult(
        loss=avg_loss,
        preds=preds,
        targets=targets,
        base_speeds=base_speeds,
        scene_names=scene_names,
        center_tokens=center_tokens,
        center_timestamps=center_timestamps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FlexiNet chassis fusion training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--allow_pose_fallback", action="store_true", help="Use ego-pose fallback when CAN bus is unavailable")
    args = parser.parse_args()

    cfg = load_config_from_yaml(Path(args.config))
    if args.allow_pose_fallback:
        cfg["data"]["allow_pose_fallback"] = True

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = NuScenesChassisFusionDataset(
        dataroot=cfg["data"]["dataroot"],
        version=cfg["data"]["version"],
        split="train",
        img_size=(cfg["data"]["img_height"], cfg["data"]["img_width"]),
        seq_len=int(cfg["data"]["seq_len"]),
        cam_channel=cfg["data"]["cam_channel"],
        can_bus_root=cfg["data"]["can_bus_root"],
        allow_pose_fallback=bool(cfg["data"]["allow_pose_fallback"]),
    )
    val_set = NuScenesChassisFusionDataset(
        dataroot=cfg["data"]["dataroot"],
        version=cfg["data"]["version"],
        split="val",
        img_size=(cfg["data"]["img_height"], cfg["data"]["img_width"]),
        seq_len=int(cfg["data"]["seq_len"]),
        cam_channel=cfg["data"]["cam_channel"],
        can_bus_root=cfg["data"]["can_bus_root"],
        allow_pose_fallback=bool(cfg["data"]["allow_pose_fallback"]),
    )
    print(f"Train scenes: {len(train_set.scene_names)} | Train samples: {len(train_set)}")
    print(f"Val scenes: {len(val_set.scene_names)} | Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    val_loader = DataLoader(val_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]))

    model = FlexiNetChassisFusion(
        input_channels=1,
        chassis_input_dim=16,
        chassis_feature_dim=int(cfg["model"]["chassis_feature_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        fusion_mode=str(cfg["model"]["fusion_mode"]),
    )
    model.load_flexinet_backbone(cfg["model"]["pretrained_path"], strict=False)
    if bool(cfg["model"]["freeze_backbone"]):
        model.freeze_backbone()
    model.to(device)

    criterion = build_criterion(str(cfg["train"]["loss"]))
    head_params = list(model.chassis_encoder.parameters()) + list(model.residual_speed_head.parameters()) + list(model.direct_speed_head.parameters())
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": float(cfg["train"]["learning_rate_head"])},
            {"params": backbone_params, "lr": float(cfg["train"]["learning_rate_backbone"])},
        ],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["train"]["epochs"]), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if bool(cfg["train"]["use_amp"]) and device.type == "cuda" else None

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["train"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["train"].get("results_dir", "results_git/FlexiNet_ChassisFusion"))
    results_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        running = 0.0
        train_preds: list[float] = []
        train_targets: list[float] = []
        train_base_speeds: list[float] = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in bar:
            images = batch["images"].to(device)
            chassis = batch["chassis"].to(device)
            base_speed = batch["base_speed"].to(device)
            speed = batch["speed"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                pred = model(images, chassis, base_speed=base_speed)
                loss = criterion(pred, speed)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                optimizer.step()

            running += loss.item()
            train_preds.extend(pred.squeeze(1).detach().cpu().tolist())
            train_targets.extend(speed.squeeze(1).detach().cpu().tolist())
            train_base_speeds.extend(base_speed.squeeze(1).detach().cpu().tolist())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_loader), 1)
        val_result = run_eval(model, val_loader, criterion, device)
        train_metrics = compute_metrics(train_preds, train_targets, train_base_speeds, loss=train_loss)
        val_metrics = compute_metrics(val_result.preds, val_result.targets, val_result.base_speeds, loss=val_result.loss)
        scheduler.step()
        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} val_loss={val_result.loss:.4f} "
            f"val_mae={val_metrics['mae_mps']:.4f}m/s "
            f"val_rmse={val_metrics['rmse_mps']:.4f}m/s "
            f"base_mae={val_metrics['base_mae_mps']:.4f}m/s"
        )

        history_row = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "train_mae_mps": train_metrics["mae_mps"],
            "train_rmse_mps": train_metrics["rmse_mps"],
            "train_base_mae_mps": train_metrics["base_mae_mps"],
            "val_loss": val_result.loss,
            "val_mae_mps": val_metrics["mae_mps"],
            "val_rmse_mps": val_metrics["rmse_mps"],
            "base_mae_mps": val_metrics["base_mae_mps"],
            "base_rmse_mps": val_metrics["base_rmse_mps"],
            "mae_gain_vs_base_mps": val_metrics["mae_gain_vs_base_mps"],
            "rmse_gain_vs_base_mps": val_metrics["rmse_gain_vs_base_mps"],
        }
        history.append(history_row)
        save_history_csv(history, log_dir / "training_history.csv")
        save_history_json(history, log_dir / "training_history.json")
        plot_training_history(history, log_dir)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": history,
            "config": cfg,
        }
        torch.save(checkpoint, save_dir / "last.pth")
        write_evaluation_artifacts(val_result, log_dir / "last_eval", prefix="val")
        if val_result.loss < best_val:
            best_val = val_result.loss
            torch.save(checkpoint, save_dir / "best.pth")
            write_evaluation_artifacts(val_result, log_dir / "best_eval", prefix="val")
            print(f"Saved new best checkpoint to {save_dir / 'best.pth'}")
        copy_small_results_for_git(log_dir, results_dir, include_predictions=False)


if __name__ == "__main__":
    main()
