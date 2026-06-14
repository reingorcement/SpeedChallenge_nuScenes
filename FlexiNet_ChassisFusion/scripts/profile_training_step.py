from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.nuscenes_chassis import FAIR_CHASSIS_DIM, POSE_ASSISTED_CHASSIS_DIM, NuScenesChassisFusionDataset
from model.fusion_model import FlexiNetChassisFusion
from train import build_criterion, load_config_from_yaml
from utils.seed import set_seed


def _optional_batch_tensor(batch: dict, key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return value.to(device, non_blocking=True)


def build_dataset(cfg: dict, split: str) -> NuScenesChassisFusionDataset:
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
    )


def build_model(cfg: dict) -> FlexiNetChassisFusion:
    chassis_mode = str(cfg["data"].get("chassis_mode", "fair_chassis"))
    chassis_input_dim = POSE_ASSISTED_CHASSIS_DIM if chassis_mode == "pose_assisted" else FAIR_CHASSIS_DIM
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
    return model


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile data loading and one training step.")
    parser.add_argument("--config", type=str, default="configs/velocity_xy_fair_gray.yaml")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--frame_cache_root", type=str, default=None, help="Override data.frame_cache_root for A/B profiling")
    args = parser.parse_args()

    cfg = load_config_from_yaml(Path(args.config))
    if args.frame_cache_root is not None:
        cfg["data"]["frame_cache_root"] = args.frame_cache_root or None
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(cfg, "train")
    num_workers = int(args.num_workers if args.num_workers is not None else cfg["train"]["num_workers"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(cfg["train"].get("shuffle_train", True)),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)) if num_workers > 0 else None,
    )

    model = build_model(cfg).to(device)
    criterion = build_criterion(
        str(cfg["train"]["loss"]),
        sigma_vx=float(cfg["train"].get("sigma_vx", 1.0)),
        sigma_vy=float(cfg["train"].get("sigma_vy", 1.0)),
        speed_weight=float(cfg["train"].get("speed_loss_weight", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg["train"]["learning_rate_head"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda") if bool(cfg["train"]["use_amp"]) and device.type == "cuda" else None

    load_times: list[float] = []
    transfer_times: list[float] = []
    compute_times: list[float] = []
    total_times: list[float] = []
    iterator = iter(loader)

    for index in range(args.batches):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        batch = next(iterator)
        load_end = time.perf_counter()

        transfer_start = time.perf_counter()
        images = batch["images"].to(device, non_blocking=True)
        chassis = batch["chassis"].to(device, non_blocking=True)
        base_velocity = batch["base_velocity"].to(device, non_blocking=True)
        velocity = batch["velocity"].to(device, non_blocking=True)
        flow = _optional_batch_tensor(batch, "flow", device)
        geometry = _optional_batch_tensor(batch, "geometry", device)
        cuda_sync(device)
        transfer_end = time.perf_counter()

        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            pred = model(images, chassis, base_velocity=base_velocity, flow=flow, geometry=geometry)
            loss = criterion(pred, velocity)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        cuda_sync(device)
        compute_end = time.perf_counter()

        total_end = time.perf_counter()
        load_times.append(load_end - load_start)
        transfer_times.append(transfer_end - transfer_start)
        compute_times.append(compute_end - compute_start)
        total_times.append(total_end - total_start)
        print(
            f"batch={index + 1:03d} "
            f"load={load_times[-1]:.3f}s "
            f"transfer={transfer_times[-1]:.3f}s "
            f"compute={compute_times[-1]:.3f}s "
            f"total={total_times[-1]:.3f}s "
            f"loss={float(loss.item()):.6f}",
            flush=True,
        )

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    print("---- average ----")
    print(f"load:     {mean(load_times):.3f}s")
    print(f"transfer: {mean(transfer_times):.3f}s")
    print(f"compute:  {mean(compute_times):.3f}s")
    print(f"total:    {mean(total_times):.3f}s")


if __name__ == "__main__":
    main()
