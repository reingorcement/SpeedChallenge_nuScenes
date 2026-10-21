from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.nuscenes_chassis import FAIR_CHASSIS_DIM, POSE_ASSISTED_CHASSIS_DIM, NuScenesChassisFusionDataset
from model.fusion_model import FlexiNetChassisFusion
from train import KnowledgeDrivenVelocityLoss, build_criterion, validate_physics_config


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_existing_path(raw_value: str | None, *, prefer_file: bool = False) -> tuple[Path | None, list[Path]]:
    if not raw_value:
        return None, []

    raw_path = Path(raw_value)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((PROJECT_ROOT / raw_path).resolve())

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if prefer_file and candidate.is_file():
            return candidate, unique_candidates
        if not prefer_file and candidate.exists():
            return candidate, unique_candidates
    return None, unique_candidates


def check_required_path(label: str, raw_value: str | None, *, prefer_file: bool = False) -> bool:
    resolved, candidates = resolve_existing_path(raw_value, prefer_file=prefer_file)
    if resolved is not None:
        kind = "file" if prefer_file else "dir"
        print(f"[OK] {label}: {resolved} ({kind})")
        return True
    if not raw_value:
        print(f"[SKIP] {label}: not configured")
        return False
    print(f"[MISS] {label}: {raw_value}")
    for candidate in candidates:
        print(f"       tried: {candidate}")
    return False


def check_nuscenes_root(root: Path) -> bool:
    required = ["samples", "sweeps", "maps"]
    optional = ["v1.0-trainval", "v1.0-mini", "v1.0-test"]
    ok = True
    for name in required:
        path = root / name
        if path.exists():
            print(f"[OK] dataroot subdir: {path}")
        else:
            print(f"[MISS] dataroot subdir: {path}")
            ok = False
    if any((root / name).exists() for name in optional):
        existing = [name for name in optional if (root / name).exists()]
        print(f"[OK] dataroot version dir(s): {', '.join(existing)}")
    else:
        print(f"[WARN] dataroot has no version dir among: {', '.join(optional)}")
    return ok


def build_model_from_config(cfg: dict) -> FlexiNetChassisFusion:
    chassis_mode = str(cfg["data"].get("chassis_mode", "fair_chassis"))
    input_channels = 3 if str(cfg["data"].get("image_mode", "gray")) == "rgb" else 1
    chassis_input_dim = POSE_ASSISTED_CHASSIS_DIM if chassis_mode == "pose_assisted" else FAIR_CHASSIS_DIM
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
        physics_mass=float(cfg["model"].get("physics_mass", 3052.0)),
        physics_wheelbase=float(cfg["model"].get("physics_wheelbase", 3.128)),
        physics_lf=float(cfg["model"].get("physics_lf", 1.293)),
        physics_front_track=float(cfg["model"].get("physics_front_track", 1.600)),
        physics_rear_track=float(cfg["model"].get("physics_rear_track", 1.740)),
    )


def make_dummy_inputs(cfg: dict, sample: dict | None = None) -> dict[str, torch.Tensor | None]:
    seq_len = int(cfg["data"]["seq_len"])
    img_h = int(cfg["data"]["img_height"])
    img_w = int(cfg["data"]["img_width"])
    channels = 3 if str(cfg["data"].get("image_mode", "gray")) == "rgb" else 1
    chassis_len = sample["chassis"].shape[0] if sample is not None else max(int(round(float(cfg["data"].get("chassis_window_sec", 0.0)) * float(cfg["data"].get("chassis_hz", 1.0)))) + 1, 1)
    chassis_dim = sample["chassis"].shape[-1] if sample is not None else (POSE_ASSISTED_CHASSIS_DIM if str(cfg["data"].get("chassis_mode", "fair_chassis")) == "pose_assisted" else FAIR_CHASSIS_DIM)

    images = torch.randn(1, seq_len, channels, img_h, img_w)
    chassis = torch.randn(1, chassis_len, chassis_dim)
    base_velocity = torch.zeros(1, 2)
    base_velocity[:, 0] = 10.0
    flow = None
    if str(cfg["data"].get("flow_mode", "none")) == "precomputed":
        flow = torch.randn(1, seq_len - 1, 2, img_h, img_w)
    geometry = None
    if str(cfg["data"].get("geometry_mode", "basic")) == "basic":
        geometry = torch.randn(1, 4)
    physics_inputs = sample["physics_inputs"].unsqueeze(0) if sample is not None else torch.zeros(1, chassis_len, 25)
    physics_valid_mask = sample["physics_valid_mask"].unsqueeze(0) if sample is not None else torch.zeros(1, chassis_len, 25, dtype=torch.bool)
    physics_dt = sample["physics_dt"].unsqueeze(0) if sample is not None else torch.full((1, chassis_len), 1.0 / max(float(cfg["data"].get("chassis_hz", 1.0)), 1.0))
    if sample is None:
        physics_inputs[:, :, 0] = 10.0
        physics_inputs[:, :, 6:10] = 10.0
        physics_valid_mask[:, :, :10] = True
        physics_dt[:, 0] = 0.0
    return {
        "images": images,
        "chassis": chassis,
        "base_velocity": base_velocity,
        "flow": flow,
        "geometry": geometry,
        "physics_inputs": physics_inputs,
        "physics_valid_mask": physics_valid_mask,
        "physics_dt": physics_dt,
        "chassis_current_index": (
            sample["chassis_current_index"].reshape(1)
            if sample is not None
            else torch.tensor([chassis_len - 1 if bool(cfg["data"].get("online_mode", False)) else chassis_len // 2])
        ),
    }


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    backbone = getattr(model, "backbone", None)
    backbone_params = list(backbone.parameters()) if backbone is not None else []
    backbone_param_ids = {id(param) for param in backbone_params}
    head_params = [param for param in model.parameters() if id(param) not in backbone_param_ids]
    groups = [{"params": head_params, "lr": float(cfg["train"]["learning_rate_head"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg["train"]["learning_rate_backbone"])})
    return torch.optim.AdamW(groups, weight_decay=float(cfg["train"]["weight_decay"]))


def run_tiny_training_step(cfg: dict, dataset: NuScenesChassisFusionDataset, model: FlexiNetChassisFusion) -> bool:
    device = next(model.parameters()).device
    batch_size = min(int(cfg["train"]["batch_size"]), 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = build_criterion(
        str(cfg["train"]["loss"]),
        sigma_vx=float(cfg["train"].get("sigma_vx", 1.0)),
        sigma_vy=float(cfg["train"].get("sigma_vy", 1.0)),
        speed_weight=float(cfg["train"].get("speed_loss_weight", 0.0)),
        beta_loss_weight=float(cfg["train"].get("beta_loss_weight", 0.0)),
        physics_prior_loss_weight=float(cfg["train"].get("physics_prior_loss_weight", 0.0)),
        physics_correction_loss_weight=float(cfg["train"].get("physics_correction_loss_weight", 0.0)),
    )
    optimizer = build_optimizer(cfg, model)
    clip_value = float(cfg["train"].get("grad_clip", 0.0))
    use_amp = bool(cfg["train"].get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    batch = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)

    images = batch["images"].to(device)
    chassis = batch["chassis"].to(device)
    base_velocity = batch["base_velocity"].to(device)
    velocity = batch["velocity"].to(device)
    flow = batch.get("flow")
    geometry = batch.get("geometry")
    flow_tensor = flow.to(device) if isinstance(flow, torch.Tensor) and flow.numel() > 0 else None
    geometry_tensor = geometry.to(device) if isinstance(geometry, torch.Tensor) and geometry.numel() > 0 else None
    physics_inputs = batch["physics_inputs"].to(device)
    physics_valid_mask = batch["physics_valid_mask"].to(device)
    physics_dt = batch["physics_dt"].to(device)

    with torch.autocast(device_type="cuda", enabled=scaler is not None):
        pred = model(
            images,
            chassis,
            base_velocity=base_velocity,
            flow=flow_tensor,
            geometry=geometry_tensor,
            physics_inputs=physics_inputs,
            physics_valid_mask=physics_valid_mask,
            physics_dt=physics_dt,
            chassis_current_index=batch["chassis_current_index"].to(device),
        )
        if isinstance(criterion, KnowledgeDrivenVelocityLoss):
            loss = criterion(
                pred,
                velocity,
                chassis=chassis,
                physics_prior=model.last_physics_prior,
                physics_confidence=model.last_physics_confidence,
                correction_gate=model.last_correction_gate,
                chassis_current_index=batch["chassis_current_index"].to(device),
            )
        else:
            loss = criterion(pred, velocity)

    if not torch.isfinite(loss):
        print("[FAIL] training loss is not finite")
        return False

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else torch.tensor(0.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else torch.tensor(0.0)
        optimizer.step()

    trainable_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"[OK] tiny train step loss: {float(loss.item()):.6f}")
    print(f"[OK] tiny train step grad_norm: {float(grad_norm):.6f}")
    print(f"[OK] tiny train step params_with_grad: {trainable_grads}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick path sanity check for FlexiNet_ChassisFusion")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "trainval", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-train-step", action="store_true")
    parser.add_argument("--allow_pose_fallback", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[FAIL] config file not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    if args.allow_pose_fallback:
        cfg["data"]["allow_pose_fallback"] = True
    try:
        validate_physics_config(cfg)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"Config: {config_path.resolve()}")
    print(f"Working dir: {Path.cwd()}")

    ok = True
    ok &= check_required_path("dataroot", cfg["data"].get("dataroot"))
    ok &= check_required_path("can_bus_root", cfg["data"].get("can_bus_root"))
    use_vision_branch = bool(cfg["model"].get("use_vision_branch", True))
    if use_vision_branch:
        ok &= check_required_path("pretrained_path", cfg["model"].get("pretrained_path"), prefer_file=True)
    else:
        check_required_path("pretrained_path", cfg["model"].get("pretrained_path"), prefer_file=True)

    if str(cfg["data"].get("flow_mode", "none")) == "precomputed":
        ok &= check_required_path("flow_root", cfg["data"].get("flow_root"))
    else:
        check_required_path("flow_root", cfg["data"].get("flow_root"))

    if str(cfg["data"].get("base_velocity_source", "zero_lateral")) == "eovm_ekf":
        ok &= check_required_path("eovm_base_root", cfg["data"].get("eovm_base_root"))
    else:
        check_required_path("eovm_base_root", cfg["data"].get("eovm_base_root"))

    dataroot_raw = cfg["data"].get("dataroot")
    dataroot_path, _ = resolve_existing_path(dataroot_raw)
    if dataroot_path is not None:
        ok &= check_nuscenes_root(dataroot_path)

    sample: dict | None = None
    dataset: NuScenesChassisFusionDataset | None = None
    if not args.skip_dataset:
        try:
            dataset = NuScenesChassisFusionDataset(
                dataroot=cfg["data"]["dataroot"],
                version=cfg["data"]["version"],
                split=args.split,
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
                physics_extra_root=cfg["data"].get("physics_extra_root"),
                physics_units_audited=bool(cfg["data"].get("physics_units_audited", False)),
                can_max_age_sec=float(cfg["data"].get("can_max_age_sec", 0.25)),
                physics_extra_max_age_sec=float(cfg["data"].get("physics_extra_max_age_sec", 0.5)),
            )
            print(f"[OK] dataset init: scenes={len(dataset.scene_names)} samples={len(dataset)}")
            if len(dataset) == 0:
                print("[FAIL] dataset is empty")
                ok = False
            else:
                sample_index = min(max(args.sample_index, 0), len(dataset) - 1)
                sample = dataset[sample_index]
                print(f"[OK] sample[{sample_index}] images={tuple(sample['images'].shape)}")
                print(f"[OK] sample[{sample_index}] chassis={tuple(sample['chassis'].shape)}")
                print(f"[OK] sample[{sample_index}] base_velocity={tuple(sample['base_velocity'].shape)}")
                print(f"[OK] sample[{sample_index}] velocity={tuple(sample['velocity'].shape)}")
                print(f"[OK] sample[{sample_index}] flow={tuple(sample['flow'].shape)}")
                print(f"[OK] sample[{sample_index}] geometry={tuple(sample['geometry'].shape)}")
        except Exception as exc:
            print(f"[FAIL] dataset probe failed: {exc}")
            ok = False

    if not args.skip_model:
        try:
            model = build_model_from_config(cfg)
            if use_vision_branch:
                pretrained_raw = cfg["model"].get("pretrained_path")
                pretrained_path, _ = resolve_existing_path(pretrained_raw, prefer_file=True)
                assert pretrained_path is not None
                model.load_flexinet_backbone(pretrained_path, strict=False)
            model.eval()
            dummy = make_dummy_inputs(cfg, sample)
            with torch.no_grad():
                output = model(
                    dummy["images"],
                    dummy["chassis"],
                    base_velocity=dummy["base_velocity"],
                    flow=dummy["flow"],
                    geometry=dummy["geometry"],
                    physics_inputs=dummy["physics_inputs"],
                    physics_valid_mask=dummy["physics_valid_mask"],
                    physics_dt=dummy["physics_dt"],
                    chassis_current_index=dummy["chassis_current_index"],
                )
            print(f"[OK] model forward output shape: {tuple(output.shape)}")
        except Exception as exc:
            print(f"[FAIL] model probe failed: {exc}")
            ok = False

    if not args.skip_train_step:
        if dataset is None:
            try:
                dataset = NuScenesChassisFusionDataset(
                    dataroot=cfg["data"]["dataroot"],
                    version=cfg["data"]["version"],
                    split=args.split,
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
                    physics_extra_root=cfg["data"].get("physics_extra_root"),
                    physics_units_audited=bool(cfg["data"].get("physics_units_audited", False)),
                    can_max_age_sec=float(cfg["data"].get("can_max_age_sec", 0.25)),
                    physics_extra_max_age_sec=float(cfg["data"].get("physics_extra_max_age_sec", 0.5)),
                )
            except Exception as exc:
                print(f"[FAIL] dataset rebuild for train step failed: {exc}")
                ok = False
                dataset = None
        if dataset is not None:
            try:
                model = build_model_from_config(cfg)
                if use_vision_branch:
                    pretrained_raw = cfg["model"].get("pretrained_path")
                    pretrained_path, _ = resolve_existing_path(pretrained_raw, prefer_file=True)
                    assert pretrained_path is not None
                    model.load_flexinet_backbone(pretrained_path, strict=False)
                if use_vision_branch and bool(cfg["model"].get("freeze_backbone", True)):
                    model.freeze_backbone()
                model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                if run_tiny_training_step(cfg, dataset, model):
                    print("[OK] tiny algorithm smoke test passed")
                else:
                    ok = False
            except Exception as exc:
                print(f"[FAIL] tiny train step failed: {exc}")
                ok = False

    print("[DONE] path sanity check passed" if ok else "[DONE] path sanity check found problems")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
