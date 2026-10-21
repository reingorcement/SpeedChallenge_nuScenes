from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nuscenes_version_comparison.bootstrap import compatibility_report  # noqa: E402


DEPENDENCIES = ("torch", "torchvision", "numpy", "PIL", "yaml", "tqdm", "nuscenes")


def _check_file(path: Path, label: str, errors: list[str], status: dict[str, Any]) -> None:
    exists = path.is_file()
    status[label] = {"path": str(path), "exists": exists}
    if not exists:
        errors.append(f"{label} not found: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a server before starting KD/VKD experiments.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--profile", choices=["legacy", "full"], default="legacy")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--can-bus-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-root", default="outputs/nuscenes_version_comparison")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-existing-checkpoints", action="store_true")
    parser.add_argument("--legacy-kd-checkpoint", default="checkpoints/velocity_xy_kd_only/best.pth")
    parser.add_argument("--legacy-vkd-checkpoint", default="checkpoints/velocity_xy_vkd_gated/best.pth")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    dataroot = Path(args.dataroot).expanduser().resolve()
    can_root = Path(args.can_bus_root).expanduser().resolve() if args.can_bus_root else dataroot / "can_bus"
    errors: list[str] = []
    status: dict[str, Any] = {
        "compatibility": compatibility_report(root, profile=args.profile),
        "dependencies": {},
        "data": {},
        "checkpoints": {},
    }
    if not status["compatibility"]["compatible"]:
        errors.append("source compatibility check failed")
    for name in DEPENDENCIES:
        try:
            importlib.import_module(name)
            status["dependencies"][name] = True
        except Exception as exc:
            status["dependencies"][name] = False
            errors.append(f"dependency {name!r} is unavailable: {exc}")

    version_root = dataroot / "v1.0-trainval"
    _check_file(version_root / "scene.json", "nuScenes scene metadata", errors, status["data"])
    _check_file(version_root / "sample_data.json", "nuScenes sample_data metadata", errors, status["data"])
    can_files = list(can_root.glob("scene-*_*.json")) if can_root.is_dir() else []
    status["data"]["can_bus"] = {"path": str(can_root), "json_count": len(can_files)}
    if not can_files:
        errors.append(f"no CAN JSON files found: {can_root}")
    if args.split_manifest:
        _check_file(Path(args.split_manifest), "split manifest", errors, status["data"])

    if args.check_existing_checkpoints:
        for name, value in {
            "legacy_kd": args.legacy_kd_checkpoint,
            "legacy_vkd": args.legacy_vkd_checkpoint,
        }.items():
            checkpoint = Path(value)
            _check_file(checkpoint if checkpoint.is_absolute() else root / checkpoint, name, errors, status["checkpoints"])

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        status["cuda"] = {"available": cuda_available, "device_count": torch.cuda.device_count()}
        if args.require_cuda and not cuda_available:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
    except Exception:
        status["cuda"] = {"available": False, "device_count": 0}

    output = Path(args.output_root)
    output_parent = output
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    status["output"] = {"path": str(output), "parent_writable": os.access(output_parent, os.W_OK)}
    if not status["output"]["parent_writable"]:
        errors.append(f"output parent is not writable: {output_parent}")

    status["ready"] = not errors
    status["errors"] = errors
    print(json.dumps(status, indent=2))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
