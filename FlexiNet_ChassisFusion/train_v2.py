from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Iterable

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset
from dataset.v2_schema import KINEMATIC_VALUE_DIM, V2NuScenesDatasetAdapter
from dataset.v2_splits import partition_scene_names
from model.v2 import TimedSignal, V2NuScenesVelocityModel, V2VelocityLoss
from utils.seed import set_seed


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(cfg: dict, split: str) -> V2NuScenesDatasetAdapter:
    data = cfg["data"]
    if not bool(data.get("online_mode", False)):
        raise ValueError("V2 configuration must set data.online_mode=true.")
    split_aliases = {"train": "train_core", "val": "dev"}
    partition = split_aliases.get(split, split)
    manifest_path = data.get("split_manifest_path")
    if manifest_path:
        scene_names = partition_scene_names(
            manifest_path,
            partition,
            expected_version=str(data["version"]),
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
        seq_len=int(data["seq_len"]),
        cam_channel=data["cam_channel"],
        can_bus_root=data["can_bus_root"],
        allow_pose_fallback=False,
        scene_names=scene_names,
        image_cache_size=int(data.get("image_cache_size", 0)),
        chassis_mode="fair_chassis",
        chassis_window_sec=float(data.get("chassis_window_sec", 1.0)),
        chassis_hz=float(data.get("chassis_hz", 20.0)),
        online_mode=True,
        image_mode="rgb",
        flow_mode="none",
        geometry_mode="none",
        frame_cache_root=data.get("frame_cache_root"),
        max_scenes=data.get("max_scenes"),
        retain_nusc_metadata=True,
    )
    augmentation = data.get("photometric_augmentation") if partition == "train_core" else None
    return V2NuScenesDatasetAdapter(
        base,
        flow_root=data["flow_root"],
        camera_height_m=float(data["camera_height_m"]),
        road_mask_top_fraction=float(data.get("road_mask_top_fraction", 0.45)),
        geometry_size=(int(data["geometry_height"]), int(data["geometry_width"])),
        kinematic_stats_path=data.get("kinematic_stats_path"),
        field_max_age_sec=data.get("field_max_age_sec"),
        photometric_augmentation=augmentation,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: V2NuScenesVelocityModel, batch: dict):
    gyro = TimedSignal(
        value=batch["angular_rate_camera"],
        valid=batch["angular_rate_valid"],
        age_sec=batch["angular_rate_age_sec"],
    )
    wheel = TimedSignal(
        value=batch["wheel_speed"],
        valid=batch["wheel_speed_valid"],
        age_sec=batch["wheel_speed_age_sec"],
    )
    return model(
        images=batch["images"],
        flow=batch["flow_v2"],
        flow_confidence=batch["flow_confidence"],
        intrinsics=batch["intrinsics"],
        frame_dt=batch["frame_dt"],
        angular_rate_camera=gyro,
        plane_normal_camera=batch["plane_normal_camera"],
        plane_offset=batch["plane_offset"],
        rotation_ego_from_camera=batch["rotation_ego_from_camera"],
        kinematic_values=batch["kinematic_values"],
        kinematic_valid=batch["kinematic_valid"],
        kinematic_age=batch["kinematic_age"],
        wheel_speed=wheel,
        road_mask=batch["road_mask"],
    )


def build_model(cfg: dict) -> V2NuScenesVelocityModel:
    model_cfg = cfg["model"]
    return V2NuScenesVelocityModel(
        input_channels=3,
        mode=str(model_cfg["mode"]),
        residual_limits=tuple(model_cfg.get("residual_limits", [3.0, 1.5])),
        kinematic_residual_limits=tuple(
            model_cfg.get("kinematic_residual_limits", [1.0, 0.5])
        ),
        wheel_scale_limit=float(model_cfg.get("wheel_scale_limit", 0.15)),
        wheel_scale_jitter=float(model_cfg.get("wheel_scale_jitter", 0.0)),
        kinematic_dropout=float(model_cfg.get("kinematic_dropout", 0.0)),
        sensor_max_age_sec=float(model_cfg.get("sensor_max_age_sec", 0.25)),
        output_reference=str(model_cfg.get("output_reference", "ego_reference")),
    )


def _axis_metrics(prefix: str, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.numel() == 0:
        return {}
    error = prediction - target
    endpoint = torch.linalg.vector_norm(error, dim=-1)
    return {
        f"{prefix}_mae_vx_mps": float(error[:, 0].abs().mean()),
        f"{prefix}_rmse_vx_mps": float(error[:, 0].square().mean().sqrt()),
        f"{prefix}_mae_vy_mps": float(error[:, 1].abs().mean()),
        f"{prefix}_rmse_vy_mps": float(error[:, 1].square().mean().sqrt()),
        f"{prefix}_endpoint_error_mps": float(endpoint.mean()),
    }


def _masked_axis_metrics(
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    return _axis_metrics(prefix, prediction[mask], target[mask]) if bool(mask.any()) else {}


@torch.no_grad()
def evaluate(
    model: V2NuScenesVelocityModel,
    loader: DataLoader,
    criterion: V2VelocityLoss,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    keys = (
        "target",
        "prediction",
        "visual",
        "geometry",
        "geometry_valid",
        "condition",
        "geometry_residual",
        "support",
        "fallback",
        "kinematic_gate",
        "kinematic_correction",
        "yaw_rate",
        "wheel_std",
    )
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    rows: list[dict[str, float | str]] = []
    total_loss = 0.0
    total_count = 0
    for step, raw_batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        output = forward_batch(model, batch)
        terms = criterion(output, batch["velocity"])
        batch_size = batch["velocity"].shape[0]
        total_loss += float(terms.total) * batch_size
        total_count += batch_size
        raw_kinematics = batch.get("kinematic_values_raw", batch["kinematic_values"])
        tensors = {
            "target": batch["velocity"],
            "prediction": output.predicted_velocity,
            "visual": output.visual_velocity,
            "geometry": output.geometry_velocity,
            "geometry_valid": output.geometry_valid,
            "condition": output.geometry_condition_number,
            "geometry_residual": output.geometry_residual_rms,
            "support": output.geometry_support_count,
            "fallback": output.fallback_used,
            "kinematic_gate": output.kinematic_gate,
            "kinematic_correction": output.kinematic_applied_correction,
            "yaw_rate": raw_kinematics[:, -1, 0],
            "wheel_std": raw_kinematics[:, -1, KINEMATIC_VALUE_DIM - 1],
        }
        for key, value in tensors.items():
            collected[key].append(value.detach().cpu())
        for index in range(batch_size):
            rows.append(
                {
                    "scene_name": raw_batch["scene_name"][index],
                    "center_token": raw_batch["center_token"][index],
                    "target_vx_mps": float(batch["velocity"][index, 0]),
                    "target_vy_mps": float(batch["velocity"][index, 1]),
                    "pred_vx_mps": float(output.predicted_velocity[index, 0]),
                    "pred_vy_mps": float(output.predicted_velocity[index, 1]),
                    "visual_vx_mps": float(output.visual_velocity[index, 0]),
                    "visual_vy_mps": float(output.visual_velocity[index, 1]),
                    "geometry_valid": int(output.geometry_valid[index]),
                    "fallback_used": int(output.fallback_used[index]),
                    "kinematic_gate_vx": float(output.kinematic_gate[index, 0]),
                    "kinematic_gate_vy": float(output.kinematic_gate[index, 1]),
                }
            )
    if total_count == 0:
        raise RuntimeError("Evaluation produced no samples.")
    data = {key: torch.cat(values) for key, values in collected.items()}
    target = data["target"]
    metrics = {"loss": total_loss / total_count, "sample_count": float(total_count)}
    metrics.update(_axis_metrics("final", data["prediction"], target))
    metrics.update(_axis_metrics("visual", data["visual"], target))
    metrics.update(
        _masked_axis_metrics("geometry_valid", data["geometry"], target, data["geometry_valid"])
    )
    metrics["selection_metric_mps"] = metrics["final_endpoint_error_mps"]
    beta_mask = torch.linalg.vector_norm(target, dim=-1) >= criterion.beta_min_speed
    if bool(beta_mask.any()):
        predicted_beta = torch.atan2(data["prediction"][:, 1], data["prediction"][:, 0])
        target_beta = torch.atan2(target[:, 1], target[:, 0])
        beta_error = torch.atan2(
            torch.sin(predicted_beta - target_beta), torch.cos(predicted_beta - target_beta)
        )
        metrics["beta_mae_rad"] = float(beta_error[beta_mask].abs().mean())
    metrics["geometry_valid_rate"] = float(data["geometry_valid"].float().mean())
    valid_condition = data["condition"][data["geometry_valid"] & torch.isfinite(data["condition"])]
    metrics["geometry_condition_median"] = (
        float(valid_condition.median()) if valid_condition.numel() else 1.0e12
    )
    metrics["geometry_residual_median"] = float(data["geometry_residual"].median())
    metrics["geometry_support_median"] = float(data["support"].float().median())
    metrics["fallback_rate"] = float(data["fallback"].float().mean())
    metrics["kinematic_gate_vx_mean"] = float(data["kinematic_gate"][:, 0].mean())
    metrics["kinematic_gate_vy_mean"] = float(data["kinematic_gate"][:, 1].mean())
    metrics["kinematic_correction_norm_mean"] = float(
        torch.linalg.vector_norm(data["kinematic_correction"], dim=-1).mean()
    )
    straight = data["yaw_rate"].abs() < 0.05
    metrics.update(_masked_axis_metrics("straight", data["prediction"], target, straight))
    metrics.update(_masked_axis_metrics("turning", data["prediction"], target, ~straight))
    lateral_threshold = torch.quantile(target[:, 1].abs(), 0.9)
    high_lateral = target[:, 1].abs() >= lateral_threshold
    metrics.update(_masked_axis_metrics("high_lateral", data["prediction"], target, high_lateral))
    wheel_inconsistent = data["wheel_std"] >= 0.05
    metrics.update(
        _masked_axis_metrics("wheel_inconsistent", data["prediction"], target, wheel_inconsistent)
    )
    sign_mask = straight & data["geometry_valid"] & (target[:, 0].abs() >= 1.0)
    metrics["geometry_straight_vx_sign_agreement"] = (
        float(
            (torch.sign(data["geometry"][sign_mask, 0]) == torch.sign(target[sign_mask, 0]))
            .float()
            .mean()
        )
        if bool(sign_mask.any())
        else 0.0
    )
    return metrics, rows


def _write_evaluation(metrics: dict[str, float], rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    if rows:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _configure_trainable_modules(model: V2NuScenesVelocityModel, stage: str) -> None:
    model.freeze_backbone()
    model.set_residual_trainable(False)
    model.set_kinematics_trainable(False)
    if stage == "visual":
        model.set_residual_trainable(True)
    elif stage == "kinematic_pretrain":
        model.set_kinematics_trainable(True)
    elif stage == "joint":
        model.unfreeze_visual_tail()
        model.set_residual_trainable(True)
        model.set_kinematics_trainable(True)
    elif stage != "geometry_audit":
        raise ValueError(f"Unsupported train.stage: {stage}")


def _set_epoch_training_mode(
    model: V2NuScenesVelocityModel,
    stage: str,
    epoch: int,
    freeze_backbone_epochs: int,
) -> None:
    visual_tail_trainable = stage == "joint" or (
        stage == "visual" and epoch >= freeze_backbone_epochs
    )
    if stage == "visual" and visual_tail_trainable:
        model.unfreeze_visual_tail()
    model.set_stage_training_mode(stage, visual_tail_trainable=visual_tail_trainable)


def _add_parameter_groups(
    groups: list[dict],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
    label: str,
) -> None:
    decay = []
    no_decay = []
    for name, parameter in named_parameters:
        target = no_decay if parameter.ndim <= 1 or name.endswith("bias") else decay
        target.append(parameter)
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": label})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{label}_no_decay"})


def _parameter_groups(model: V2NuScenesVelocityModel, cfg: dict, stage: str) -> list[dict]:
    train_cfg = cfg["train"]
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))
    groups: list[dict] = []
    if stage in {"visual", "joint"}:
        _add_parameter_groups(
            groups,
            model.residual_model.named_parameters(),
            float(train_cfg["head_lr"]),
            weight_decay,
            "visual_head",
        )
    if stage in {"kinematic_pretrain", "joint"}:
        _add_parameter_groups(
            groups,
            model.kinematic_encoder.named_parameters(),
            float(train_cfg["kinematic_lr"]),
            weight_decay,
            "kinematic",
        )
    if stage in {"visual", "joint"}:
        tail_named_parameters = []
        for module_name, module in (
            ("sfe", model.backbone.sfe_module),
            ("mfe", model.backbone.mfe_module),
            ("dig", model.backbone.dig),
        ):
            tail_named_parameters.extend(
                (f"{module_name}.{name}", parameter)
                for name, parameter in module.named_parameters()
            )
        _add_parameter_groups(
            groups,
            tail_named_parameters,
            float(train_cfg["backbone_lr"]),
            weight_decay,
            "visual_tail",
        )
    return groups


def _load_initial_weights(model: V2NuScenesVelocityModel, cfg: dict, stage: str) -> None:
    model_cfg = cfg["model"]
    init_checkpoint = model_cfg.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if stage == "kinematic_pretrain":
            state = {key: value for key, value in state.items() if not key.startswith("kinematic_encoder.")}
        incompatible = model.load_state_dict(state, strict=False)
        if stage == "kinematic_pretrain":
            model.kinematic_encoder.reset_output_heads()
        print(
            f"Loaded stage checkpoint {init_checkpoint}: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
        return
    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        report = model.load_flexinet_backbone(pretrained, strict=False)
        print(f"FlexiNet load report: {report}")


def _build_loader(
    dataset: V2NuScenesDatasetAdapter,
    cfg: dict,
    device: torch.device,
    shuffle: bool,
    stage: str,
) -> DataLoader:
    workers = int(cfg["train"].get("num_workers", 8))
    sampler = None
    if shuffle and stage in {"kinematic_pretrain", "joint"} and bool(
        cfg["train"].get("scenario_sampling", True)
    ):
        weights = dataset.scenario_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)) if workers > 0 else None,
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _accumulation_steps(micro_batch: int, effective_batch: int) -> int:
    if micro_batch <= 0 or effective_batch < micro_batch or effective_batch % micro_batch != 0:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size.")
    return effective_batch // micro_batch


def _accumulation_divisor(step: int, batch_count: int, accumulation_steps: int) -> int:
    remainder = batch_count % accumulation_steps
    if remainder and step >= batch_count - remainder:
        return remainder
    return accumulation_steps


def _selection_result(
    metrics: dict[str, float],
    stage: str,
    visual_baseline: dict[str, float] | None,
    constrained: bool = True,
) -> tuple[bool, float]:
    epe = float(metrics["final_endpoint_error_mps"])
    if stage == "visual" or not constrained:
        return True, epe
    if stage not in {"kinematic_pretrain", "joint"}:
        return True, epe
    if visual_baseline is None:
        raise ValueError(f"{stage} requires train.visual_baseline_metrics.")
    eligible = epe <= 1.02 * float(visual_baseline["final_endpoint_error_mps"])
    group_values = [
        float(metrics[key])
        for key in ("turning_mae_vy_mps", "high_lateral_mae_vy_mps")
        if key in metrics
    ]
    score = sum(group_values) / len(group_values) if group_values else math.inf
    return eligible, score


def _checkpoint(
    epoch: int,
    model: V2NuScenesVelocityModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    metrics: dict,
    history: list[dict],
    best_score: float,
    no_improve: int,
    optimizer_steps: int,
    clipped_steps: int,
) -> dict:
    numpy_state = np.random.get_state()
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "metrics": metrics,
        "history": history,
        "best_score": best_score,
        "no_improve": no_improve,
        "optimizer_steps": optimizer_steps,
        "clipped_steps": clipped_steps,
        "rng_state": {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "state": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _load_resume(
    path: str | Path,
    model: V2NuScenesVelocityModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    stage: str,
) -> tuple[int, list[dict], float, int, int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_stage = checkpoint.get("config", {}).get("train", {}).get("stage")
    if checkpoint_stage != stage:
        raise ValueError(f"Resume stage mismatch: checkpoint={checkpoint_stage}, requested={stage}.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        numpy_state = rng_state["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["state"].cpu().numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return (
        int(checkpoint["epoch"]) + 1,
        list(checkpoint.get("history", [])),
        float(checkpoint.get("best_score", math.inf)),
        int(checkpoint.get("no_improve", 0)),
        int(checkpoint.get("optimizer_steps", 0)),
        int(checkpoint.get("clipped_steps", 0)),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    overrides = {
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value not in (None, 0, ""):
            cfg["train"][key] = value
    if args.init_checkpoint:
        cfg["model"]["init_checkpoint"] = args.init_checkpoint
    if args.visual_baseline_metrics:
        cfg["train"]["visual_baseline_metrics"] = args.visual_baseline_metrics
    data_overrides = {
        "dataroot": args.dataroot,
        "can_bus_root": args.can_bus_root,
        "flow_root": args.flow_root,
        "split_manifest_path": args.split_manifest,
    }
    for key, value in data_overrides.items():
        if value:
            cfg["data"][key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or audit the causal V2 nuScenes model.")
    parser.add_argument("--config", default="configs/velocity_xy_v2_visual_kinematic.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--effective_batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--save_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--visual_baseline_metrics")
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--split_manifest")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V2 nuScenes stages are server-only and require CUDA.")

    stage = str(cfg["train"]["stage"])
    train_set = build_dataset(cfg, "train_core")
    val_set = build_dataset(cfg, "dev")
    train_loader = _build_loader(train_set, cfg, device, shuffle=True, stage=stage)
    val_loader = _build_loader(val_set, cfg, device, shuffle=False, stage=stage)
    model = build_model(cfg)
    if not args.resume:
        _load_initial_weights(model, cfg, stage)
    _configure_trainable_modules(model, stage)
    model.to(device)
    criterion = V2VelocityLoss(**cfg["loss"]).to(device)
    output_dir = Path(cfg["train"]["output_dir"])
    save_dir = Path(cfg["train"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    max_eval_batches = int(cfg["train"].get("smoke_val_batches", 2)) if args.smoke else 0
    if stage == "geometry_audit":
        metrics, rows = evaluate(model, val_loader, criterion, device, max_eval_batches)
        _write_evaluation(metrics, rows, output_dir / "geometry_audit")
        print(json.dumps(metrics, indent=2))
        if not args.smoke and bool(cfg["train"].get("enforce_geometry_acceptance", True)):
            if metrics["geometry_valid_rate"] < 0.60:
                raise RuntimeError("Geometry valid rate is below the required 60%.")
            if metrics["geometry_condition_median"] >= 1.0e5:
                raise RuntimeError("Median geometry condition number is at or above 1e5.")
            if metrics["geometry_straight_vx_sign_agreement"] < 0.95:
                raise RuntimeError("Straight-driving geometry vx sign agreement is below 95%.")
        return

    groups = _parameter_groups(model, cfg, stage)
    if not groups:
        raise RuntimeError("No trainable parameter groups were configured.")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("use_amp", True)))
    epochs = 1 if args.smoke else int(cfg["train"]["epochs"])
    train_limit = int(cfg["train"].get("smoke_train_batches", 2)) if args.smoke else 0
    batches_per_epoch = min(len(train_loader), train_limit) if train_limit else len(train_loader)
    micro_batch = int(cfg["train"]["batch_size"])
    effective_batch = int(cfg["train"].get("effective_batch_size", micro_batch))
    accumulation_steps = _accumulation_steps(micro_batch, effective_batch)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    scheduler = _make_scheduler(
        optimizer,
        total_steps=max(optimizer_steps_per_epoch * epochs, 1),
        warmup_fraction=float(cfg["train"].get("warmup_fraction", 0.05)),
        min_lr_ratio=float(cfg["train"].get("min_lr_ratio", 0.1)),
    )
    visual_baseline_path = cfg["train"].get("visual_baseline_metrics")
    visual_baseline = (
        json.loads(Path(visual_baseline_path).read_text(encoding="utf-8"))
        if visual_baseline_path
        else None
    )
    start_epoch = 0
    history: list[dict] = []
    best_score = math.inf
    no_improve = 0
    optimizer_steps = 0
    clipped_steps = 0
    if args.resume:
        (
            start_epoch,
            history,
            best_score,
            no_improve,
            optimizer_steps,
            clipped_steps,
        ) = _load_resume(args.resume, model, optimizer, scheduler, scaler, stage)

    if not args.resume:
        initial_metrics, initial_rows = evaluate(model, val_loader, criterion, device, max_eval_batches)
        initial_eligible, initial_score = _selection_result(
            initial_metrics, stage, visual_baseline, constrained=not args.smoke
        )
        initial_metrics.update(
            {"checkpoint_eligible": float(initial_eligible), "checkpoint_score": initial_score}
        )
        history.append({"epoch": 0, "train_loss": None, **initial_metrics})
        _write_evaluation(initial_metrics, initial_rows, output_dir / "initial_eval")
        initial_checkpoint = _checkpoint(
            -1,
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            initial_metrics,
            history,
            initial_score if initial_eligible else math.inf,
            0,
            0,
            0,
        )
        torch.save(initial_checkpoint, save_dir / "initial.pth")
        if initial_eligible:
            best_score = initial_score
            torch.save(initial_checkpoint, save_dir / "best.pth")
            _write_evaluation(initial_metrics, initial_rows, output_dir / "best_eval")

    freeze_backbone_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    min_delta = float(cfg["train"].get("early_stopping_min_delta", 1.0e-4))
    grad_clip = float(cfg["train"].get("grad_clip", 2.0))
    max_clip_fraction = float(cfg["train"].get("max_clip_fraction", 0.2))
    run_metadata = {
        "git_commit": _git_commit(),
        "seed": int(cfg["train"].get("seed", 42)),
        "micro_batch_size": micro_batch,
        "effective_batch_size": effective_batch,
        "accumulation_steps": accumulation_steps,
        "stage": stage,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, epochs):
        _set_epoch_training_mode(model, stage, epoch, freeze_backbone_epochs)
        running_loss = 0.0
        batch_count = 0
        epoch_optimizer_steps = 0
        epoch_clipped_steps = 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"{stage} {epoch + 1}/{epochs}")
        for step, raw_batch in enumerate(bar):
            if step >= batches_per_epoch:
                break
            batch = move_batch(raw_batch, device)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                output = forward_batch(model, batch)
                terms = criterion(output, batch["velocity"])
                divisor = _accumulation_divisor(step, batches_per_epoch, accumulation_steps)
                scaled_loss = terms.total / divisor
            scaler.scale(scaled_loss).backward()
            running_loss += float(terms.total.detach())
            batch_count += 1
            boundary = (step + 1) % accumulation_steps == 0 or step + 1 == batches_per_epoch
            if boundary:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("Encountered non-finite gradient norm.")
                clipped = float(grad_norm) > grad_clip
                epoch_clipped_steps += int(clipped)
                clipped_steps += int(clipped)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_optimizer_steps += 1
                optimizer_steps += 1
            bar.set_postfix(loss=f"{float(terms.total.detach()):.4f}")

        clip_fraction = epoch_clipped_steps / max(epoch_optimizer_steps, 1)
        if clip_fraction > max_clip_fraction:
            raise RuntimeError(
                f"Gradient clipping fraction {clip_fraction:.1%} exceeds {max_clip_fraction:.1%}."
            )
        metrics, rows = evaluate(model, val_loader, criterion, device, max_eval_batches)
        eligible, score = _selection_result(
            metrics, stage, visual_baseline, constrained=not args.smoke
        )
        metrics.update(
            {
                "checkpoint_eligible": float(eligible),
                "checkpoint_score": score,
                "gradient_clip_fraction": clip_fraction,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        train_loss = running_loss / max(batch_count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        improved = eligible and score < best_score - min_delta
        if improved:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
        checkpoint = _checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            metrics,
            history,
            best_score,
            no_improve,
            optimizer_steps,
            clipped_steps,
        )
        torch.save(checkpoint, save_dir / "last.pth")
        _write_evaluation(metrics, rows, output_dir / "last_eval")
        if improved:
            torch.save(checkpoint, save_dir / "best.pth")
            _write_evaluation(metrics, rows, output_dir / "best_eval")
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, allow_nan=False)
        print(
            f"epoch={epoch + 1} train={train_loss:.5f} val={metrics['loss']:.5f} "
            f"eligible={eligible} score={score:.5f} lr={metrics['learning_rate']:.3e}"
        )
        if patience > 0 and no_improve >= patience:
            print(f"Early stopping after {no_improve} epochs without an eligible improvement.")
            break
    if not (save_dir / "best.pth").is_file():
        raise RuntimeError("Training produced no checkpoint satisfying the stage acceptance constraint.")


if __name__ == "__main__":
    main()
