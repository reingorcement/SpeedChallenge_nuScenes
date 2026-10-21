"""Test existing V3 weights only: RGB expert, flow geometry, full fusion."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.v3 import V3VisualOnlyVelocityModel
from model.v3.geometry import fuse_twist_estimates, transform_camera_twist_to_ego
# Import helpers only. Never invoke the training entry point.
from train_v3 import build_dataset, build_loader, build_model, load_config, move_batch, write_evaluation
from utils.sideslip import sideslip_values
from scripts.summarize_v3_ablation import summarize

MODES = ("rgb_only", "flow_geometry_only", "fusion")


def load_existing_model(
    path: Path, fallback_config: dict[str, Any] | None = None, expected_mode: str = "fusion",
) -> tuple[V3VisualOnlyVelocityModel, dict[str, Any]]:
    package = torch.load(path, map_location="cpu")
    config = copy.deepcopy(package.get("config", fallback_config))
    if config is None:
        raise ValueError("Checkpoint has no saved config; supply the matching --config.")
    mode = config["model"].get("mode", "fusion")
    if mode != expected_mode:
        raise ValueError(f"Expected {expected_mode} checkpoint, got {mode}: {path}")
    model = build_model(config)
    model.load_state_dict(package.get("model", package.get("state_dict", package)), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, config


@torch.inference_mode()
def predict_mode(
    model: V3VisualOnlyVelocityModel, visual: Mapping[str, torch.Tensor], mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Evaluation requires eval() and frozen parameters.")
    if mode == "fusion":
        output = model(visual)
        return output.predicted_velocity, output.geometry_valid
    if mode == "rgb_only":
        if not model.direct_expert_enabled:
            raise ValueError("Checkpoint has no enabled RGB expert.")
        descriptor = model._descriptor(model.backbone.forward_feature_map(visual["images"]))
        velocity = model.direct_head(descriptor)
        return velocity, torch.zeros(velocity.shape[0], dtype=torch.bool, device=velocity.device)
    if mode != "flow_geometry_only":
        raise ValueError(f"Unknown evaluation mode: {mode}")
    # No RGB/backbone/learned reliability/correction/gate/direct-head access.
    flows = visual["flows"][:, -model.pair_count :].float()
    confidence = visual["flow_confidence"][:, -model.pair_count :].float()
    dt = visual["pair_dt"][:, -model.pair_count :].float()
    support = model._support_mask(flows)
    estimates = [model.geometry_solver(
        flow=flows[:, i], confidence=confidence[:, i],
        intrinsics=visual["intrinsics"].float(), dt=dt[:, i],
        plane_normal_camera=visual["plane_normal_camera"].float(),
        plane_offset=visual["plane_offset"].float(),
        reliability=torch.ones_like(confidence[:, i]),
        support_mask=support[:, i] if support is not None else None,
    ) for i in range(model.pair_count)]
    twists, covariances = transform_camera_twist_to_ego(
        torch.stack([item.twist_camera for item in estimates], dim=1),
        torch.stack([item.covariance_camera for item in estimates], dim=1),
        visual["rotation_ego_from_camera"].float(), visual["translation_ego_from_camera"].float(),
    )
    fused, _, valid, _ = fuse_twist_estimates(
        twists, covariances, torch.stack([item.valid for item in estimates], dim=1)
    )
    return fused[:, :2], valid


def prediction_rows(
    batch: dict[str, Any], prediction: torch.Tensor, valid: torch.Tensor, min_speed: float,
) -> list[dict[str, Any]]:
    target, prediction = batch["target"]["velocity"].cpu(), prediction.cpu()
    beta = sideslip_values(prediction, target, min_speed)
    return [{
        "center_token": token, "scene_name": batch["scene_name"][i],
        "center_timestamp": int(batch["center_timestamp"][i]),
        "target_vx_mps": float(target[i, 0]), "target_vy_mps": float(target[i, 1]),
        "pred_vx_mps": float(prediction[i, 0]), "pred_vy_mps": float(prediction[i, 1]),
        "geometry_valid": int(valid[i]),
        **{key: float(value[i]) for key, value in beta.items()},
    } for i, token in enumerate(batch["center_token"])]


def row_metrics(rows: list[dict[str, Any]], min_speed: float) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Evaluation produced no samples.")
    pred = torch.tensor([[r["pred_vx_mps"], r["pred_vy_mps"]] for r in rows], dtype=torch.float64)
    target = torch.tensor([[r["target_vx_mps"], r["target_vy_mps"]] for r in rows], dtype=torch.float64)
    error = pred - target
    beta = torch.tensor([r["beta_error_deg"] for r in rows if r["beta_eval_valid"]], dtype=torch.float64)
    return {
        "sample_count": len(rows), "scene_count": len({r["scene_name"] for r in rows}),
        "mae_vx_mps": float(error[:, 0].abs().mean()), "mae_vy_mps": float(error[:, 1].abs().mean()),
        "rmse_vx_mps": float(error[:, 0].square().mean().sqrt()),
        "rmse_vy_mps": float(error[:, 1].square().mean().sqrt()),
        "epe_mps": float(torch.linalg.vector_norm(error, dim=-1).mean()),
        "sideslip_min_speed_mps": min_speed, "sideslip_sample_count": beta.numel(),
        "sideslip_undefined_prediction_count": sum(bool(r["beta_eval_valid"]) and not bool(r["pred_beta_defined"]) for r in rows),
        "sideslip_mae_deg": float(beta.abs().mean()) if beta.numel() else float("nan"),
        "sideslip_rmse_deg": float(beta.square().mean().sqrt()) if beta.numel() else float("nan"),
        "geometry_valid_rate": sum(float(r["geometry_valid"]) for r in rows) / len(rows),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Existing fusion checkpoint.")
    parser.add_argument("--rgb_checkpoint", type=Path, help="Optional existing direct_pretrain checkpoint.")
    parser.add_argument("--config", help="Fallback if fusion checkpoint has no config.")
    parser.add_argument("--rgb_config", help="Fallback if RGB checkpoint has no config.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--flow_root", type=Path)
    parser.add_argument("--frame_cache_root", type=Path)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = full holdout; positive = partial TEST.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_batches < 0:
        parser.error("Invalid batch size, worker count, or batch limit.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a small functional test.")
    model, cfg = load_existing_model(args.checkpoint, load_config(args.config) if args.config else None)
    rgb_model = model
    if args.rgb_checkpoint:
        rgb_model, rgb_cfg = load_existing_model(
            args.rgb_checkpoint, load_config(args.rgb_config) if args.rgb_config else None,
            expected_mode="direct_pretrain",
        )
        for key in ("version", "seq_len", "visual_height", "visual_width", "cam_channel"):
            if rgb_cfg["data"].get(key) != cfg["data"].get(key):
                raise ValueError(f"RGB/fusion data contract differs: {key}")
    device = torch.device(args.device)
    model.to(device)
    rgb_model.to(device)
    data_root = args.dataroot.resolve()
    cfg["data"].update(
        dataroot=str(data_root), max_scenes=None,
        flow_root=str((args.flow_root or data_root / "precomputed_flow_v2/CAM_FRONT").resolve()),
        frame_cache_root=str((args.frame_cache_root or data_root / "precomputed_frames/CAM_FRONT_72x128_rgb").resolve()),
    )
    manifest = args.split_manifest or Path(cfg["data"].get("split_manifest_path") or "logs/v2_split_manifest.json")
    cfg["data"]["split_manifest_path"] = str(manifest.resolve())
    cfg["train"].update(batch_size=args.batch_size, num_workers=args.num_workers)
    for path in (data_root, Path(cfg["data"]["flow_root"]), Path(cfg["data"]["frame_cache_root"]), manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    min_speed = float(cfg.get("loss", {}).get("beta_min_speed", 2.0))
    dataset = build_dataset(cfg, "stage1_test")
    loader = build_loader(dataset, cfg, device, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "test_only": True, "partition": "stage1_test", "partial_test": args.max_batches > 0,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "rgb_source": "existing_direct_checkpoint" if args.rgb_checkpoint else "fusion_checkpoint_rgb_expert",
        "rgb_checkpoint": str(args.rgb_checkpoint.resolve()) if args.rgb_checkpoint else None,
        "rgb_checkpoint_sha256": file_sha256(args.rgb_checkpoint) if args.rgb_checkpoint else None,
        "split_manifest_sha256": file_sha256(manifest), "max_batches": args.max_batches,
        "beta_min_speed_mps": min_speed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (args.output_dir / "split_manifest.json").write_bytes(manifest.read_bytes())
    rows = {mode: [] for mode in MODES}
    for index, raw in enumerate(tqdm(loader, desc="Frozen V3 test (no training)")):
        if args.max_batches and index >= args.max_batches:
            break
        batch = move_batch(raw, device)
        for mode in MODES:
            prediction, valid = predict_mode(rgb_model if mode == "rgb_only" else model, batch["visual"], mode)
            rows[mode].extend(prediction_rows(batch, prediction, valid, min_speed))
    for mode in MODES:
        write_evaluation(row_metrics(rows[mode], min_speed), rows[mode], args.output_dir / mode)
    summarize(args.output_dir)
    print(f"Test complete; weights unchanged. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
