from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.nuscenes_chassis import FAIR_CHASSIS_DIM, POSE_ASSISTED_CHASSIS_DIM, NuScenesChassisFusionDataset
from model.fusion_model import FlexiNetChassisFusion
from train import build_criterion, load_config_from_yaml, run_eval, validate_physics_config
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
    validate_physics_config(cfg)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    criterion = build_criterion(
        str(cfg["train"]["loss"]),
        sigma_vx=float(cfg["train"].get("sigma_vx", 1.0)),
        sigma_vy=float(cfg["train"].get("sigma_vy", 1.0)),
        speed_weight=float(cfg["train"].get("speed_loss_weight", 0.0)),
        beta_loss_weight=float(cfg["train"].get("beta_loss_weight", 0.0)),
        prior_loss_weight=float(cfg["train"].get("prior_loss_weight", 0.0)),
        smoothness_loss_weight=float(cfg["train"].get("smoothness_loss_weight", 0.0)),
        consistency_loss_weight=float(cfg["train"].get("consistency_loss_weight", 0.0)),
        physics_prior_loss_weight=float(cfg["train"].get("physics_prior_loss_weight", 0.0)),
        physics_correction_loss_weight=float(cfg["train"].get("physics_correction_loss_weight", 0.0)),
        beta_vx_eps=float(cfg["train"].get("beta_vx_eps", 0.5)),
    )
    result = run_eval(model, loader, criterion, device)
    metrics = write_evaluation_artifacts(result, args.output_dir, prefix="val")
    copy_small_results_for_git(args.output_dir, args.results_dir, include_predictions=False)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"MAE vx: {metrics['mae_vx_mps']:.4f} m/s")
    print(f"MAE vy: {metrics['mae_vy_mps']:.4f} m/s")
    print(f"Vector endpoint error: {metrics['vector_endpoint_error_mps']:.4f} m/s")
    print(f"Base MAE vx: {metrics.get('base_mae_vx_mps', 0.0):.4f} m/s")
    print(f"Base MAE vy: {metrics.get('base_mae_vy_mps', 0.0):.4f} m/s")
    print(f"Gain vs base vx: {metrics.get('gain_vs_base_vx_mps', 0.0):.4f} m/s")
    print(f"Gain vs base vy: {metrics.get('gain_vs_base_vy_mps', 0.0):.4f} m/s")
    print(f"Saved evaluation artifacts to: {Path(args.output_dir)}")
    print(f"Copied small analysis artifacts to: {Path(args.results_dir)}")


if __name__ == "__main__":
    main()
