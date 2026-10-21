from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset
from dataset.v2_splits import partition_scene_names
from dataset.v3_schema import V3NuScenesVisualDataset
from model.v3 import V3VelocityLoss, V3VisualOnlyVelocityModel
from utils.seed import set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_smoke_dataset_limit(cfg: dict[str, Any]) -> None:
    smoke_max_scenes = int(cfg["train"].get("smoke_max_scenes", 2))
    if smoke_max_scenes <= 0:
        raise ValueError("train.smoke_max_scenes must be positive.")
    configured_max = cfg["data"].get("max_scenes")
    cfg["data"]["max_scenes"] = (
        smoke_max_scenes
        if configured_max is None
        else min(int(configured_max), smoke_max_scenes)
    )


def build_dataset(cfg: dict[str, Any], split: str) -> V3NuScenesVisualDataset:
    data = cfg["data"]
    if not bool(data.get("online_mode", False)):
        raise ValueError("V3 requires data.online_mode=true.")
    partition = {"train": "train_core", "val": "dev"}.get(split, split)
    manifest_path = data.get("split_manifest_path")
    if manifest_path:
        scene_names = partition_scene_names(
            manifest_path, partition, expected_version=str(data["version"])
        )
        official_split = "train" if partition in {"train_core", "dev"} else "val"
    else:
        if partition not in {"train_core", "dev"}:
            raise ValueError("stage1_test requires data.split_manifest_path.")
        scene_names = None
        official_split = "train" if partition == "train_core" else "val"
    base = NuScenesChassisFusionDataset(
        dataroot=data["dataroot"],
        version=data["version"],
        split=official_split,
        img_size=(int(data["visual_height"]), int(data["visual_width"])),
        seq_len=int(data.get("seq_len", 5)),
        cam_channel=data.get("cam_channel", "CAM_FRONT"),
        can_bus_root=None,
        allow_pose_fallback=False,
        scene_names=scene_names,
        image_cache_size=int(data.get("image_cache_size", 0)),
        online_mode=True,
        image_mode="rgb",
        flow_mode="none",
        geometry_mode="none",
        frame_cache_root=data.get("frame_cache_root"),
        max_scenes=data.get("max_scenes"),
        retain_nusc_metadata=True,
    )
    augmentation = (
        data.get("photometric_augmentation") if partition == "train_core" else None
    )
    return V3NuScenesVisualDataset(
        base,
        flow_root=data["flow_root"],
        camera_height_m=float(data["camera_height_m"]),
        pair_count=3,
        geometry_size=(int(data["geometry_height"]), int(data["geometry_width"])),
        flow_backend=data.get("flow_backend"),
        photometric_augmentation=augmentation,
        load_flows=str(cfg["model"].get("mode", "fusion")) != "direct_pretrain",
    )


def build_model(cfg: dict[str, Any], **overrides: Any) -> V3VisualOnlyVelocityModel:
    model = dict(cfg["model"])
    model.update(overrides)
    return V3VisualOnlyVelocityModel(
        mode=str(model.get("mode", "fusion")),
        pair_count=int(model.get("pair_count", 3)),
        twist_dof=int(model.get("twist_dof", 6)),
        support_mode=str(model.get("support_mode", "learned")),
        pooling=str(model.get("pooling", "regions")),
        uncertainty=bool(model.get("uncertainty", True)),
        direct_expert=bool(model.get("direct_expert", True)),
        residual_limits=tuple(model.get("residual_limits", [3.0, 1.5])),
        fixed_roi_top_fraction=float(model.get("fixed_roi_top_fraction", 0.45)),
    )


def move_batch(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    return batch


def _regression_calibration(
    error: torch.Tensor, log_variance: torch.Tensor, bins: int = 10
) -> float:
    axis_sigma = torch.exp(0.5 * log_variance) * error.new_tensor([5.0, 1.0])
    uncertainty = torch.linalg.vector_norm(axis_sigma, dim=-1)
    magnitude = torch.linalg.vector_norm(error, dim=-1)
    order = uncertainty.argsort()
    chunks = torch.tensor_split(order, bins)
    differences = []
    for chunk in chunks:
        if chunk.numel():
            predicted = uncertainty[chunk].mean()
            empirical = magnitude[chunk].square().mean().sqrt()
            differences.append((predicted - empirical).abs())
    return float(torch.stack(differences).mean()) if differences else float("nan")


def _risk_coverage(error: torch.Tensor, log_variance: torch.Tensor) -> float:
    risk = torch.linalg.vector_norm(error, dim=-1)
    axis_sigma = torch.exp(0.5 * log_variance) * error.new_tensor([5.0, 1.0])
    uncertainty = torch.linalg.vector_norm(axis_sigma, dim=-1)
    ordered_risk = risk[uncertainty.argsort()]
    coverage_risk = ordered_risk.cumsum(0) / torch.arange(
        1, ordered_risk.numel() + 1, device=ordered_risk.device
    )
    return float(coverage_risk.mean())


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    geometry_valid: torch.Tensor,
) -> dict[str, float]:
    error = prediction - target
    endpoint = torch.linalg.vector_norm(error, dim=-1)
    speed_error = (
        torch.linalg.vector_norm(prediction, dim=-1)
        - torch.linalg.vector_norm(target, dim=-1)
    ).abs()
    beta_mask = torch.linalg.vector_norm(target, dim=-1) >= 2.0
    beta_error = prediction.new_empty(0)
    if bool(beta_mask.any()):
        difference = torch.atan2(
            prediction[beta_mask, 1], prediction[beta_mask, 0]
        ) - torch.atan2(target[beta_mask, 1], target[beta_mask, 0])
        beta_error = torch.atan2(torch.sin(difference), torch.cos(difference)).abs()
    normalized = error / error.new_tensor([5.0, 1.0])
    nll = 0.5 * (torch.exp(-log_variance) * normalized.square() + log_variance)
    return {
        "mae_vx_mps": float(error[:, 0].abs().mean()),
        "mae_vy_mps": float(error[:, 1].abs().mean()),
        "epe_mps": float(endpoint.mean()),
        "p95_epe_mps": float(torch.quantile(endpoint, 0.95)),
        "speed_mae_mps": float(speed_error.mean()),
        "sideslip_mae_deg": float(torch.rad2deg(beta_error).mean())
        if beta_error.numel()
        else float("nan"),
        "nll": float(nll.mean()),
        "regression_calibration_error": _regression_calibration(error, log_variance),
        "risk_coverage_auc": _risk_coverage(error, log_variance),
        "geometry_valid_rate": float(geometry_valid.float().mean()),
        "visual_fallback_rate": float((~geometry_valid).float().mean()),
    }


@torch.no_grad()
def evaluate(
    model: V3VisualOnlyVelocityModel,
    loader: DataLoader,
    criterion: V3VelocityLoss,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[dict[str, float], list[dict[str, str | float]]]:
    model.eval()
    predictions, targets, variances, valid_flags = [], [], [], []
    rows: list[dict[str, str | float]] = []
    total_loss = 0.0
    sample_count = 0
    for step, raw_batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        output = model(batch["visual"])
        terms = criterion(output, batch["target"], batch["visual"])
        batch_size = output.predicted_velocity.shape[0]
        total_loss += float(terms.total) * batch_size
        sample_count += batch_size
        predictions.append(output.predicted_velocity.cpu())
        targets.append(batch["target"]["velocity"].cpu())
        variances.append(output.predicted_log_variance.cpu())
        valid_flags.append(output.geometry_valid.cpu())
        for index, token in enumerate(raw_batch["center_token"]):
            rows.append(
                {
                    "center_token": token,
                    "scene_name": raw_batch["scene_name"][index],
                    "target_vx_mps": float(batch["target"]["velocity"][index, 0]),
                    "target_vy_mps": float(batch["target"]["velocity"][index, 1]),
                    "pred_vx_mps": float(output.predicted_velocity[index, 0]),
                    "pred_vy_mps": float(output.predicted_velocity[index, 1]),
                    "log_variance_vx": float(output.predicted_log_variance[index, 0]),
                    "log_variance_vy": float(output.predicted_log_variance[index, 1]),
                    "geometry_valid": float(output.geometry_valid[index]),
                    "geometry_gate_vx": float(output.geometry_gate[index, 0]),
                    "geometry_gate_vy": float(output.geometry_gate[index, 1]),
                }
            )
    if not predictions:
        raise RuntimeError("V3 evaluation loader produced no batches.")
    metrics = compute_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(variances),
        torch.cat(valid_flags),
    )
    metrics["loss"] = total_loss / sample_count
    return metrics, rows


def write_evaluation(
    metrics: dict[str, float], rows: list[dict[str, str | float]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    if rows:
        with (output_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build_loader(
    dataset: V3NuScenesVisualDataset,
    cfg: dict[str, Any],
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    train = cfg["train"]
    workers = int(train.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_size": int(train.get("batch_size", 4)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(train.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def configure_stage(
    model: V3VisualOnlyVelocityModel, stage: str, epoch: int, freeze_epochs: int
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "direct_pretrain":
        for module in (
            model.backbone,
            model.region_projection,
            model.direct_head,
            model.uncertainty_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.train()
        return
    for module in (
        model.reliability_head,
        model.region_projection,
        model.direct_head,
        model.correction_head,
        model.gate_head,
        model.uncertainty_head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.set_stage_training_mode(freeze_backbone=epoch < freeze_epochs)


def optimizer_state_matches(
    optimizer: torch.optim.Optimizer, state: dict[str, Any]
) -> bool:
    current = [len(group["params"]) for group in optimizer.param_groups]
    saved = [len(group["params"]) for group in state.get("param_groups", [])]
    return current == saved


def next_early_stopping_count(
    current: int,
    improved: bool,
    stage: str,
    epoch: int,
    freeze_backbone_epochs: int,
) -> int:
    if improved:
        return 0
    if stage == "fusion" and epoch < freeze_backbone_epochs:
        return current
    return current + 1


def clip_gradient_norm(
    parameters: Iterable[torch.Tensor], max_norm: float
) -> torch.Tensor:
    """Return a gradient norm so AMP can recover from transient overflows."""
    return torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=False
    )


def geometry_audit(
    cfg: dict[str, Any],
    loader: DataLoader,
    criterion: V3VelocityLoss,
    device: torch.device,
    output_dir: Path,
    max_batches: int,
) -> None:
    variants = {
        "single_pair_3dof": {"pair_count": 1, "twist_dof": 3},
        "single_pair_6dof": {"pair_count": 1, "twist_dof": 6},
        "three_pair_6dof": {"pair_count": 3, "twist_dof": 6},
    }
    summary = {}
    for name, overrides in variants.items():
        model = build_model(
            cfg, mode="geometry_audit", support_mode="fb_only", **overrides
        ).to(device)
        metrics, rows = evaluate(model, loader, criterion, device, max_batches)
        write_evaluation(metrics, rows, output_dir / name)
        summary[name] = metrics
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or audit strict visual-only V3."
    )
    parser.add_argument("--config", default="configs/velocity_xy_v3_visual_only.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--stage", choices=["geometry_audit", "direct_pretrain", "fusion"]
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--save_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--dataroot")
    parser.add_argument("--flow_root")
    parser.add_argument("--frame_cache_root")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--early_stopping_min_delta", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ("dataroot", "flow_root", "frame_cache_root"):
        value = getattr(args, key)
        if value:
            cfg["data"][key] = value
    if args.stage:
        cfg["model"]["mode"] = args.stage
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.init_checkpoint:
        cfg["model"]["init_checkpoint"] = args.init_checkpoint
    if args.save_dir:
        cfg["train"]["save_dir"] = args.save_dir
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    for key in (
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "early_stopping_patience",
        "early_stopping_min_delta",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg["train"][key] = value
    if args.smoke:
        apply_smoke_dataset_limit(cfg)
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = build_dataset(cfg, "train_core")
    val_set = build_dataset(cfg, "dev")
    train_loader = build_loader(train_set, cfg, device, shuffle=True)
    val_loader = build_loader(val_set, cfg, device, shuffle=False)
    criterion = V3VelocityLoss(**cfg.get("loss", {})).to(device)
    stage = str(cfg["model"].get("mode", "fusion"))
    if stage == "direct_pretrain":
        for name in (
            "geometry",
            "rotation",
            "reprojection",
            "residual",
            "support",
            "tv",
        ):
            criterion.weights[name] = 0.0
    output_dir = Path(cfg["train"]["output_dir"])
    save_dir = Path(cfg["train"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    eval_limit = int(cfg["train"].get("smoke_val_batches", 2)) if args.smoke else 0
    if stage == "geometry_audit":
        geometry_audit(cfg, val_loader, criterion, device, output_dir, eval_limit)
        return

    model = build_model(cfg).to(device)
    initial_checkpoint = cfg["model"].get("init_checkpoint")
    if stage == "fusion" and initial_checkpoint and not args.resume:
        model.load_direct_checkpoint(initial_checkpoint)
    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 5))
    early_stopping_patience = int(cfg["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(
        cfg["train"].get("early_stopping_min_delta", 1.0e-4)
    )
    if early_stopping_patience < 0:
        raise ValueError("train.early_stopping_patience must be non-negative.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("train.early_stopping_min_delta must be non-negative.")
    start_epoch = 0
    best_epe = math.inf
    epochs_without_improvement = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_epe = float(resume_checkpoint.get("best_epe", math.inf))
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
    configure_stage(model, stage, start_epoch, freeze_epochs)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["train"].get("learning_rate", 1.0e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1.0e-4)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(cfg["train"].get("use_amp", True)),
    )
    if resume_checkpoint is not None:
        optimizer_state = resume_checkpoint["optimizer"]
        if optimizer_state_matches(optimizer, optimizer_state):
            optimizer.load_state_dict(optimizer_state)
        else:
            print(
                "Optimizer parameter set changed at the visual unfreeze boundary; "
                "resuming model weights with a fresh optimizer."
            )
        scaler.load_state_dict(resume_checkpoint["scaler"])
    epochs = 1 if args.smoke else int(cfg["train"].get("epochs", 20))
    train_limit = int(cfg["train"].get("smoke_train_batches", 2)) if args.smoke else 0
    grad_clip = float(cfg["train"].get("grad_clip", 2.0))
    consecutive_amp_overflows = 0
    max_consecutive_amp_overflows = 8
    for epoch in range(start_epoch, epochs):
        previous_trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        configure_stage(model, stage, epoch, freeze_epochs)
        current_trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        if current_trainable != previous_trainable:
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=float(cfg["train"].get("learning_rate", 1.0e-4)),
                weight_decay=float(cfg["train"].get("weight_decay", 1.0e-4)),
            )
        bar = tqdm(train_loader, desc=f"v3 {stage} {epoch + 1}/{epochs}")
        for step, raw_batch in enumerate(bar):
            if train_limit and step >= train_limit:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                output = model(batch["visual"])
                terms = criterion(output, batch["target"], batch["visual"])
            if not bool(torch.isfinite(terms.total)):
                raise RuntimeError(
                    f"Non-finite V3 loss at epoch={epoch + 1}, batch={step}."
                )
            scaler.scale(terms.total).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_gradient_norm(model.parameters(), grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                if not scaler.is_enabled():
                    raise RuntimeError(
                        "Non-finite V3 gradients without AMP at "
                        f"epoch={epoch + 1}, batch={step}."
                    )
                scaler.step(optimizer)
                scaler.update()
                consecutive_amp_overflows += 1
                print(
                    "Skipped AMP overflow at "
                    f"epoch={epoch + 1}, batch={step}; "
                    f"scale={scaler.get_scale():.1f}, streak={consecutive_amp_overflows}."
                )
                if consecutive_amp_overflows >= max_consecutive_amp_overflows:
                    raise RuntimeError(
                        "V3 AMP did not recover from consecutive non-finite gradients "
                        f"at epoch={epoch + 1}, batch={step}."
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
            consecutive_amp_overflows = 0
            bar.set_postfix(loss=f"{float(terms.total.detach()):.4f}", refresh=False)
        metrics, rows = evaluate(model, val_loader, criterion, device, eval_limit)
        write_evaluation(metrics, rows, output_dir / "last_eval")
        current_epe = float(metrics["epe_mps"])
        improved = current_epe < best_epe - early_stopping_min_delta
        if improved:
            best_epe = current_epe
        epochs_without_improvement = next_early_stopping_count(
            epochs_without_improvement,
            improved,
            stage,
            epoch,
            freeze_epochs,
        )
        early_stopped = (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "metrics": metrics,
            "best_epe": best_epe,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopped": early_stopped,
        }
        torch.save(checkpoint, save_dir / "last.pth")
        if improved:
            torch.save(checkpoint, save_dir / "best.pth")
            write_evaluation(metrics, rows, output_dir / "best_eval")
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    **metrics,
                    "epochs_without_improvement": epochs_without_improvement,
                    "early_stopping_patience": early_stopping_patience,
                },
                allow_nan=True,
            )
        )
        if early_stopped:
            print(
                "Early stopping after "
                f"{epochs_without_improvement} epochs without dev EPE improvement."
            )
            break


if __name__ == "__main__":
    main()
