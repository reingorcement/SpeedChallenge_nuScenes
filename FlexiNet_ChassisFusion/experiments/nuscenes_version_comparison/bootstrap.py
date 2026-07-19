from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
HOST_PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT_ENV = "NUSCENES_COMPARISON_PROJECT_ROOT"
LEGACY_REQUIRED_SOURCE_FILES = (
    "train.py",
    "dataset/nuscenes_chassis.py",
    "model/fusion_model.py",
    "scripts/path_sanity_check.py",
)
V2_REQUIRED_SOURCE_FILES = (
    "train_v2.py",
    "dataset/v2_schema.py",
    "dataset/v2_splits.py",
    "model/v2/model.py",
)
LEGACY_INTERFACE_MARKERS = {
    "dataset/nuscenes_chassis.py": ("scene_names:", "chassis_current_index"),
    "model/fusion_model.py": ("chassis_current_index", "class FlexiNetChassisFusion"),
}
V2_INTERFACE_MARKERS = {
    "train_v2.py": ("def build_dataset", "def build_model", "def forward_batch"),
    "dataset/v2_schema.py": ("class V2NuScenesDatasetAdapter", "kinematic_values_raw"),
    "model/v2/model.py": ("class V2NuScenesVelocityModel", "predicted_velocity"),
}


def compatibility_report(project_root: str | Path, profile: str = "legacy") -> dict[str, Any]:
    if profile not in {"legacy", "full"}:
        raise ValueError("profile must be 'legacy' or 'full'.")
    root = Path(project_root).expanduser().resolve()
    required_files = LEGACY_REQUIRED_SOURCE_FILES
    markers = dict(LEGACY_INTERFACE_MARKERS)
    if profile == "full":
        required_files += V2_REQUIRED_SOURCE_FILES
        markers.update(V2_INTERFACE_MARKERS)
    missing = [relative for relative in required_files if not (root / relative).is_file()]
    missing_interfaces = []
    for relative, required_markers in markers.items():
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing_interfaces.extend(
            f"{relative}:{marker}" for marker in required_markers if marker not in text
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
        "profile": profile,
        "compatible": not missing and not missing_interfaces,
        "missing_files": missing,
        "missing_interfaces": missing_interfaces,
    }


def resolve_source_project_root(
    explicit: str | Path | None = None,
    profile: str = "legacy",
) -> Path:
    configured = explicit or os.environ.get(SOURCE_ROOT_ENV) or HOST_PROJECT_ROOT
    report = compatibility_report(configured, profile=profile)
    if not report["compatible"]:
        missing = ", ".join(report["missing_files"] + report["missing_interfaces"])
        raise RuntimeError(
            f"Comparison source root {report['source_project_root']} is missing: {missing}. "
            f"Profile {profile!r} cannot run from this source tree."
        )
    return Path(report["source_project_root"])


def activate_source_project_root(explicit: str | Path | None = None) -> Path:
    root = resolve_source_project_root(explicit)
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    return root


SOURCE_PROJECT_ROOT = activate_source_project_root() if __package__ else HOST_PROJECT_ROOT
