from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
HOST_PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT_ENV = "NUSCENES_COMPARISON_PROJECT_ROOT"
REQUIRED_SOURCE_FILES = (
    "train.py",
    "train_v2.py",
    "dataset/nuscenes_chassis.py",
    "dataset/v2_schema.py",
    "dataset/v2_splits.py",
    "model/fusion_model.py",
    "model/v2/model.py",
    "scripts/path_sanity_check.py",
)
REQUIRED_INTERFACE_MARKERS = {
    "train_v2.py": ("def build_dataset", "def build_model", "def forward_batch"),
    "dataset/nuscenes_chassis.py": ("scene_names:", "retain_nusc_metadata", "chassis_current_index"),
    "dataset/v2_schema.py": ("class V2NuScenesDatasetAdapter", "kinematic_values_raw"),
    "model/fusion_model.py": ("chassis_current_index", "class FlexiNetChassisFusion"),
    "model/v2/model.py": ("class V2NuScenesVelocityModel", "predicted_velocity"),
}


def compatibility_report(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    missing = [relative for relative in REQUIRED_SOURCE_FILES if not (root / relative).is_file()]
    missing_interfaces = []
    for relative, markers in REQUIRED_INTERFACE_MARKERS.items():
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing_interfaces.extend(
            f"{relative}:{marker}" for marker in markers if marker not in text
        )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "host_project_root": str(HOST_PROJECT_ROOT),
        "source_project_root": str(root),
        "source_commit": commit,
        "compatible": not missing and not missing_interfaces,
        "missing_files": missing,
        "missing_interfaces": missing_interfaces,
    }


def resolve_source_project_root(explicit: str | Path | None = None) -> Path:
    configured = explicit or os.environ.get(SOURCE_ROOT_ENV) or HOST_PROJECT_ROOT
    report = compatibility_report(configured)
    if not report["compatible"]:
        missing = ", ".join(report["missing_files"] + report["missing_interfaces"])
        raise RuntimeError(
            f"Comparison source root {report['source_project_root']} is missing: {missing}. "
            f"A pure legacy checkout cannot run V2. Set {SOURCE_ROOT_ENV} to a current "
            "FlexiNet_ChassisFusion directory containing both legacy and V2 modules."
        )
    return Path(report["source_project_root"])


def activate_source_project_root(explicit: str | Path | None = None) -> Path:
    root = resolve_source_project_root(explicit)
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    return root


SOURCE_PROJECT_ROOT = (
    activate_source_project_root() if __package__ else HOST_PROJECT_ROOT
)
