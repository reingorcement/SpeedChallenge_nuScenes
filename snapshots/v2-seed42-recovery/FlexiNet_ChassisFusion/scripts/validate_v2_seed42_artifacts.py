from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_geometry_cache import (  # noqa: E402
    GEOMETRY_CACHE_VERSION,
    file_sha256,
    geometry_cache_fingerprint,
)
from dataset.v2_splits import load_split_manifest  # noqa: E402
from train_v2 import (  # noqa: E402
    _config_fingerprint,
    _resume_contract,
    build_geometry_cache_contract,
    build_model,
    load_config,
)


def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a V2 training checkpoint with model state: {path}")
    return checkpoint


def _validate_checkpoint_config(checkpoint: dict, stage: str) -> None:
    cfg = checkpoint.get("config", {})
    if cfg.get("train", {}).get("stage") != stage:
        raise ValueError(f"Expected {stage} checkpoint, got {cfg.get('train', {}).get('stage')}.")
    if int(cfg.get("train", {}).get("seed", -1)) != 42:
        raise ValueError("Seed-42 continuation requires a seed 42 checkpoint.")
    if cfg.get("model", {}).get("output_reference") != "ego_reference":
        raise ValueError("Checkpoint output reference must be ego_reference.")


def _validate_joint_config(cfg: dict) -> None:
    expected = {
        "stage": "joint",
        "seed": 42,
        "epochs": 10,
        "batch_size": 8,
        "effective_batch_size": 8,
        "head_lr": 1.0e-4,
        "kinematic_lr": 1.0e-4,
        "backbone_lr": 5.0e-6,
        "grad_clip": 2.0,
        "early_stopping_patience": 4,
    }
    train = cfg["train"]
    for key, value in expected.items():
        actual = train.get(key)
        matches = math.isclose(float(actual), float(value)) if isinstance(value, float) else actual == value
        if not matches:
            raise ValueError(f"Joint config {key}={actual!r}, expected {value!r}.")
    if cfg["model"].get("mode") != "visual_kinematic":
        raise ValueError("Joint model mode must be visual_kinematic.")
    if cfg["model"].get("output_reference") != "ego_reference":
        raise ValueError("Joint output reference must be ego_reference.")


def _validate_frame_cache(root: Path, cfg: dict) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frame cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "img_height": int(cfg["data"]["visual_height"]),
        "img_width": int(cfg["data"]["visual_width"]),
        "image_mode": "rgb",
        "version": cfg["data"]["version"],
        "cam_channel": cfg["data"]["cam_channel"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frame cache {key}={manifest.get(key)!r}, expected {value!r}.")
    return manifest


def _validate_flow_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"RAFT flow root not found: {root}")
    if not any(path.suffix in {".pt", ".npy"} for path in root.iterdir() if path.is_file()):
        raise FileNotFoundError(f"RAFT flow root contains no .pt or .npy files: {root}")


def _validate_geometry_cache(root: Path, cfg: dict) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Geometry cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = build_geometry_cache_contract(cfg)
    if int(manifest.get("version", -1)) != GEOMETRY_CACHE_VERSION:
        raise ValueError("Geometry cache version mismatch.")
    if manifest.get("contract") != contract:
        raise ValueError("Geometry cache contract does not match the resolved joint config.")
    if manifest.get("fingerprint") != geometry_cache_fingerprint(contract):
        raise ValueError("Geometry cache fingerprint mismatch.")
    return manifest


def _validate_completion(
    marker_path: Path,
    expected_checkpoint: Path,
    expected_cfg: dict | None = None,
    visual_metrics_path: Path | None = None,
) -> dict:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("stage") != "joint":
        raise ValueError("Joint completion marker is not complete.")
    if int(marker.get("seed", -1)) != 42:
        raise ValueError("Joint completion marker is not for seed 42.")
    resolved_path = marker_path.parent / "resolved_config.yaml"
    resolved = load_config(resolved_path)
    if marker.get("config_fingerprint") != _config_fingerprint(resolved):
        raise ValueError("Joint completion marker config fingerprint mismatch.")
    if expected_cfg is not None and _resume_contract(resolved) != _resume_contract(expected_cfg):
        raise ValueError("Completed joint training contract does not match current settings.")
    if marker.get("best_checkpoint_sha256") != file_sha256(expected_checkpoint):
        raise ValueError("Joint best checkpoint hash does not match its completion marker.")
    if visual_metrics_path is not None and marker.get(
        "visual_baseline_metrics_sha256"
    ) != file_sha256(visual_metrics_path):
        raise ValueError("Visual baseline metrics hash does not match the completion marker.")
    return marker


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate seed-42 continuation artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--visual_checkpoint", required=True)
    parser.add_argument("--kinematic_checkpoint", required=True)
    parser.add_argument("--visual_metrics", required=True)
    parser.add_argument("--kinematic_run_metadata", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--training_stats", required=True)
    parser.add_argument("--frame_cache_root", required=True)
    parser.add_argument("--flow_root", required=True)
    parser.add_argument("--geometry_cache_root", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--can_bus_root", required=True)
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--completion_marker")
    parser.add_argument("--joint_checkpoint")
    parser.add_argument("--expected_source_commit", default="9e0dc23")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "visual_checkpoint": Path(args.visual_checkpoint),
        "kinematic_checkpoint": Path(args.kinematic_checkpoint),
        "visual_metrics": Path(args.visual_metrics),
        "kinematic_run_metadata": Path(args.kinematic_run_metadata),
        "split_manifest": Path(args.split_manifest),
        "training_stats": Path(args.training_stats),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    cfg = load_config(args.config)
    cfg["data"].update(
        {
            "dataroot": args.dataroot,
            "can_bus_root": args.can_bus_root,
            "flow_root": args.flow_root,
            "split_manifest_path": args.split_manifest,
            "frame_cache_root": args.frame_cache_root,
            "geometry_cache_root": args.geometry_cache_root,
            "geometry_cache_mode": "required",
            "kinematic_stats_path": args.training_stats,
        }
    )
    cfg["model"]["init_checkpoint"] = args.kinematic_checkpoint
    cfg["train"]["visual_baseline_metrics"] = args.visual_metrics
    _validate_joint_config(cfg)

    visual_checkpoint = _load_checkpoint(paths["visual_checkpoint"])
    _validate_checkpoint_config(visual_checkpoint, "visual")
    visual_model = build_model({**cfg, "model": {**cfg["model"], "mode": "visual"}})
    visual_model.load_state_dict(visual_checkpoint["model"], strict=True)

    kinematic_checkpoint = _load_checkpoint(paths["kinematic_checkpoint"])
    _validate_checkpoint_config(kinematic_checkpoint, "kinematic_pretrain")
    joint_model = build_model(cfg)
    joint_model.load_state_dict(kinematic_checkpoint["model"], strict=True)

    visual_metrics = json.loads(paths["visual_metrics"].read_text(encoding="utf-8"))
    if not math.isfinite(float(visual_metrics["final_endpoint_error_mps"])):
        raise ValueError("Visual baseline EPE must be finite.")
    source_metadata = json.loads(paths["kinematic_run_metadata"].read_text(encoding="utf-8"))
    source_commit = str(source_metadata.get("git_commit", ""))
    if not source_commit.startswith(args.expected_source_commit):
        raise ValueError(
            f"Kinematic source commit {source_commit!r} does not match "
            f"{args.expected_source_commit!r}."
        )

    split_manifest = load_split_manifest(paths["split_manifest"])
    json.loads(paths["training_stats"].read_text(encoding="utf-8"))
    frame_manifest = _validate_frame_cache(Path(args.frame_cache_root), cfg)
    _validate_flow_root(Path(args.flow_root))
    geometry_manifest = _validate_geometry_cache(Path(args.geometry_cache_root), cfg)

    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = _load_checkpoint(Path(args.resume_checkpoint))
        _validate_checkpoint_config(resume_checkpoint, "joint")
        if _resume_contract(resume_checkpoint["config"]) != _resume_contract(cfg):
            raise ValueError("Joint resume checkpoint contract does not match current settings.")

    completion = None
    if args.completion_marker:
        if not args.joint_checkpoint:
            parser.error("--joint_checkpoint is required with --completion_marker.")
        completion = _validate_completion(
            Path(args.completion_marker),
            Path(args.joint_checkpoint),
            expected_cfg=cfg,
            visual_metrics_path=paths["visual_metrics"],
        )

    report = {
        "status": "valid",
        "seed": 42,
        "source_git_commit": source_commit,
        "current_git_commit": _git_commit(),
        "output_reference": cfg["model"]["output_reference"],
        "visual_checkpoint_sha256": file_sha256(paths["visual_checkpoint"]),
        "kinematic_checkpoint_sha256": file_sha256(paths["kinematic_checkpoint"]),
        "resume_checkpoint_sha256": (
            file_sha256(args.resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "split_manifest_sha256": file_sha256(paths["split_manifest"]),
        "split_manifest_checksum": split_manifest["checksum_sha256"],
        "training_stats_sha256": file_sha256(paths["training_stats"]),
        "visual_metrics_sha256": file_sha256(paths["visual_metrics"]),
        "frame_cache_manifest_sha256": file_sha256(
            Path(args.frame_cache_root) / "manifest.json"
        ),
        "frame_cache_scene_count": len(frame_manifest.get("scenes", {})),
        "flow_backend": cfg["data"]["flow_backend"],
        "geometry_cache_fingerprint": geometry_manifest["fingerprint"],
        "joint_completion_validated": completion is not None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
