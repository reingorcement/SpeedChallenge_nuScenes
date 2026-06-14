from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.nuscenes_chassis import FAIR_CHASSIS_DIM, POSE_ASSISTED_CHASSIS_DIM, NuScenesChassisFusionDataset
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


class VelocitySmoothL1Loss(nn.Module):
    def __init__(self, sigma_vx: float = 1.0, sigma_vy: float = 1.0, speed_weight: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("sigma", torch.tensor([sigma_vx, sigma_vy], dtype=torch.float32))
        self.speed_weight = float(speed_weight)
        self.base_loss = nn.SmoothL1Loss(beta=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.to(device=pred.device, dtype=pred.dtype).clamp_min(1.0e-6)
        loss = self.base_loss(pred / sigma, target / sigma)
        if self.speed_weight > 0.0:
            pred_speed = torch.linalg.vector_norm(pred, dim=1)
            target_speed = torch.linalg.vector_norm(target, dim=1)
            loss = loss + self.speed_weight * self.base_loss(pred_speed, target_speed)
        return loss


def build_criterion(
    name: str,
    sigma_vx: float = 1.0,
    sigma_vy: float = 1.0,
    speed_weight: float = 0.0,
) -> nn.Module:
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name in {"smoothl1", "huber", "velocity_smoothl1"}:
        return VelocitySmoothL1Loss(sigma_vx=sigma_vx, sigma_vy=sigma_vy, speed_weight=speed_weight)
    raise ValueError(f"Unsupported loss: {name}")


def _optional_batch_tensor(batch: dict, key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return value.to(device)


def _model_forward_from_batch(model: nn.Module, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device)
    chassis = batch["chassis"].to(device)
    base_velocity = batch["base_velocity"].to(device)
    velocity = batch["velocity"].to(device)
    pred = model(
        images,
        chassis,
        base_velocity=base_velocity,
        flow=_optional_batch_tensor(batch, "flow", device),
        geometry=_optional_batch_tensor(batch, "geometry", device),
    )
    return pred, velocity, base_velocity


def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EvaluationResult:
    model.eval()
    losses = []
    preds: list[list[float]] = []
    targets: list[list[float]] = []
    base_velocities: list[list[float]] = []
    scene_names: list[str] = []
    center_tokens: list[str] = []
    center_timestamps: list[int] = []
    yaw_rates: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            pred, velocity, base_velocity = _model_forward_from_batch(model, batch, device)
            loss = criterion(pred, velocity)
            losses.append(loss.item())
            preds.extend(pred.detach().cpu().tolist())
            targets.extend(velocity.detach().cpu().tolist())
            base_velocities.extend(base_velocity.detach().cpu().tolist())
            geometry = batch.get("geometry")
            if isinstance(geometry, torch.Tensor) and geometry.numel() > 0:
                yaw_rates.extend([float(v[0]) for v in geometry.detach().cpu().tolist()])

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
        base_velocities=base_velocities,
        scene_names=scene_names,
        center_tokens=center_tokens,
        center_timestamps=center_timestamps,
        yaw_rates=yaw_rates if yaw_rates else None,
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
    )
    print(f"Train scenes: {len(train_set.scene_names)} | Train samples: {len(train_set)}")
    print(f"Val scenes: {len(val_set.scene_names)} | Val samples: {len(val_set)}")

    num_workers = int(cfg["train"]["num_workers"])
    use_persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=bool(cfg["train"].get("shuffle_train", True)),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_persistent_workers,
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)) if use_persistent_workers else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_persistent_workers,
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)) if use_persistent_workers else None,
    )

    chassis_mode = str(cfg["data"].get("chassis_mode", "fair_chassis"))
    chassis_input_dim = (
        POSE_ASSISTED_CHASSIS_DIM if chassis_mode == "pose_assisted" else FAIR_CHASSIS_DIM
    )
    input_channels = 3 if str(cfg["data"].get("image_mode", "gray")) == "rgb" else 1
    geometry_input_dim = 4 if str(cfg["data"].get("geometry_mode", "basic")) == "basic" else 0
    flow_enabled = str(cfg["data"].get("flow_mode", "none")) == "precomputed"
    model = FlexiNetChassisFusion(
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
    )
    model.load_flexinet_backbone(cfg["model"]["pretrained_path"], strict=False)
    if bool(cfg["model"]["freeze_backbone"]):
        model.freeze_backbone()
    model.to(device)

    criterion = build_criterion(
        str(cfg["train"]["loss"]),
        sigma_vx=float(cfg["train"].get("sigma_vx", 1.0)),
        sigma_vy=float(cfg["train"].get("sigma_vy", 1.0)),
        speed_weight=float(cfg["train"].get("speed_loss_weight", 0.0)),
    )
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(param) for param in backbone_params}
    head_params = [param for param in model.parameters() if id(param) not in backbone_param_ids]
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
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                pred, velocity, base_velocity = _model_forward_from_batch(model, batch, device)
                loss = criterion(pred, velocity)

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
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_loader), 1)
        val_result = run_eval(model, val_loader, criterion, device)
        val_metrics = compute_metrics(
            val_result.preds,
            val_result.targets,
            val_result.base_velocities,
            loss=val_result.loss,
            yaw_rates=val_result.yaw_rates,
        )
        scheduler.step()
        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} val_loss={val_result.loss:.4f} "
            f"val_mae_vx={val_metrics['mae_vx_mps']:.4f}m/s "
            f"val_mae_vy={val_metrics['mae_vy_mps']:.4f}m/s "
            f"base_mae_vx={val_metrics.get('base_mae_vx_mps', 0.0):.4f}m/s"
        )

        history_row = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_mae_vx_mps": val_metrics["mae_vx_mps"],
            "val_mae_vy_mps": val_metrics["mae_vy_mps"],
            "val_vector_endpoint_error_mps": val_metrics["vector_endpoint_error_mps"],
            "base_mae_vx_mps": val_metrics.get("base_mae_vx_mps", 0.0),
            "base_mae_vy_mps": val_metrics.get("base_mae_vy_mps", 0.0),
            "gain_vs_base_vx_mps": val_metrics.get("gain_vs_base_vx_mps", 0.0),
            "gain_vs_base_vy_mps": val_metrics.get("gain_vs_base_vy_mps", 0.0),
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
            "val_metrics": val_metrics,
            "history": history,
            "config": cfg,
        }
        torch.save(checkpoint, save_dir / "last.pth")
        write_evaluation_artifacts(val_result, log_dir / "last_eval", prefix="val")
        selection_metric = val_metrics["selection_metric_mps"]
        if selection_metric < best_val:
            best_val = selection_metric
            torch.save(checkpoint, save_dir / "best.pth")
            write_evaluation_artifacts(val_result, log_dir / "best_eval", prefix="val")
            print(f"Saved new best checkpoint to {save_dir / 'best.pth'}")
        copy_small_results_for_git(log_dir, results_dir, include_predictions=False)


if __name__ == "__main__":
    main()
