from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


REQUIRED_MODULES = (
    "cv2",
    "matplotlib",
    "numpy",
    "pytest",
    "torchvision",
    "tqdm",
    "nuscenes.nuscenes",
)


def check_environment(expected_conda_env: str) -> dict[str, Any]:
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
    environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    if (
        environment.lower() != expected_conda_env.lower()
        and expected_conda_env.lower() not in sys.executable.lower()
    ):
        raise RuntimeError(
            f"Expected {expected_conda_env}, got environment={environment!r}, "
            f"python={sys.executable}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V3 training.")
    return {
        "python": sys.executable,
        "conda_environment": environment,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def checkpoint_complete(
    checkpoint_path: str | Path, config_path: str | Path
) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return int(checkpoint["epoch"]) + 1 >= int(config["train"]["epochs"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks used by the V3 server pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--expected_conda_env", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "environment":
        print(json.dumps(check_environment(args.expected_conda_env), indent=2))
        return 0
    return 0 if checkpoint_complete(args.checkpoint, args.config) else 3


if __name__ == "__main__":
    raise SystemExit(main())
