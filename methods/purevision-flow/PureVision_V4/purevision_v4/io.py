"""Explicit input contracts and writes confined to V4 artifact directories."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FOLDERS = {"runs", "training", "prepared", "verification"}
TARGET_VERSION = "body_reference_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def artifact_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.relative_to(ROOT).parts or path.relative_to(ROOT).parts[0] not in ARTIFACT_FOLDERS:
        raise ValueError(f"Outputs must be inside {ROOT}/runs, training, prepared, or verification")
    return path


def write_json(path, value):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "purevision_v4").rglob("*.py"))}


def validate_rotation(value, name):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError(f"{name} must be a proper rotation")
    return r


def calibration(raw):
    """Normalize the old static-CG format to one body-reference convention."""
    size = np.asarray(raw["image_size"])
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 8) or np.any(size != size.astype(int)):
        raise ValueError("image_size must be integer [width,height]")
    k = np.asarray(raw["K"], dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8:
        raise ValueError("K must be finite zero-skew pinhole intrinsics")
    if np.any(np.asarray(raw.get("distortion_coefficients", [0]), float) != 0):
        raise ValueError("Images must be undistorted using the supplied K")
    n = np.asarray(raw["plane_normal_camera"], float)
    height = float(raw["height_m"])
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1, atol=1e-5):
        raise ValueError("plane_normal_camera must be a unit vector")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height_m must be positive")
    if "rotation_body_from_camera" in raw:
        c = validate_rotation(raw["rotation_body_from_camera"], "rotation_body_from_camera")
        lever = np.asarray(raw["camera_lever_body_m"], float)
        reference = raw["reference_point"]
        source = "body_reference_v1"
    else:
        ego = validate_rotation(raw["rotation_ego_from_camera"], "rotation_ego_from_camera")
        mount = validate_rotation(raw["body_mount_rotation"], "body_mount_rotation")
        optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
        c = mount @ optical
        lever = c @ ego.T @ np.asarray(raw["translation_ego_from_camera"], float)
        reference = "static_cg"
        source = "legacy_static_cg"
    if lever.shape != (3,) or not np.isfinite(lever).all() or not isinstance(reference, str) or not reference.strip():
        raise ValueError("Invalid reference point or camera lever")
    result = dict(image_size=size.astype(int).tolist(), K=k.tolist(), height_m=height,
                plane_normal_camera=n.tolist(), rotation_body_from_camera=c.tolist(),
                camera_lever_body_m=lever.tolist(), reference_point=reference, source_format=source)
    if source == "legacy_static_cg":
        result["legacy_rotation_ego_from_camera"] = ego.tolist()
        result["legacy_translation_ego_from_camera"] = np.asarray(raw["translation_ego_from_camera"], float).tolist()
    return result


def intrinsics(cal, size=(288, 512), legacy=False):
    height, width = size
    if min(height, width) <= 0:
        raise ValueError("Target dimensions must be positive")
    sx, sy = width / cal["image_size"][0], height / cal["image_size"][1]
    offset_x, offset_y = (0., 0.) if legacy else ((sx - 1) / 2, (sy - 1) / 2)
    return np.array([[sx, 0, offset_x], [0, sy, offset_y], [0, 0, 1.]]) @ np.asarray(cal["K"], float)


def load_sequence(path, start=0, count=0, image_root=None):
    root = Path(path).resolve()
    images_root = Path(image_root).resolve() if image_root is not None else root
    cal = calibration(read_json(root / "calibration.json"))
    raw = read_json(root / "frames.json")
    if not isinstance(raw, list) or start < 0 or count < 0:
        raise ValueError("Invalid sequence slice")
    raw = raw[start:start + count if count else None]
    if len(raw) < 5:
        raise ValueError("At least five consecutive frames are required")
    frames = []
    for offset, item in enumerate(raw):
        rel = str(item["file"]).replace("\\", "/")
        file = (images_root / rel).resolve()
        if Path(rel).is_absolute() or PureWindowsPath(rel).is_absolute() or not file.is_relative_to(images_root):
            raise ValueError("Image path escapes the explicit image root")
        time = float(item.get("time_s", item.get("solver_time", float("nan"))))
        ident = item.get("image_id", start + offset)
        if not isinstance(ident, int) or isinstance(ident, bool):
            raise ValueError("image_id must be an integer")
        if not math.isfinite(time) or (frames and (time <= frames[-1]["time_s"] or ident <= frames[-1]["image_id"])):
            raise ValueError("Frame IDs and times must be strictly increasing")
        with Image.open(file) as image:
            if list(image.size) != cal["image_size"]:
                raise ValueError(f"Image dimensions disagree with calibration: {file}")
            image.verify()
        # Explicit allowlist: label/state fields never enter the estimator.
        frames.append(dict(image_id=ident, time_s=time, file=str(file), sha256=sha256(file)))
    return frames, cal


def target_spec(cal):
    return dict(version=TARGET_VERSION, axes="forward_left_up", units="m/s",
                reference_point=cal["reference_point"], time="output_frame")


def load_config(path=None):
    cfg = read_json(path or ROOT / "configs/default.json")
    if cfg.get("version") != "4.0.0":
        raise ValueError("A V4 configuration is required")
    for key in ["raft_size", "geometry_size", "rgb_size"]:
        values = cfg["preprocessing"][key]
        if len(values) != 2 or any(not isinstance(v, int) or v < 8 for v in values):
            raise ValueError(f"Invalid {key}")
    if any(v % 8 for v in cfg["preprocessing"]["raft_size"]):
        raise ValueError("RAFT dimensions must be divisible by 8")
    if cfg["preprocessing"]["rgb_size"] != [72, 128]:
        raise ValueError("Frozen V3 RGB features require 72x128")
    if cfg["aggregation"]["mode"] != "mean":
        raise ValueError("Only the validated mean aggregation is enabled")
    for key in ["max_residual_px", "max_condition", "max_nfev", "min_points", "max_points", "covariance_floor"]:
        if not math.isfinite(cfg["finite_solver"][key]) or cfg["finite_solver"][key] <= 0:
            raise ValueError(f"Invalid finite solver setting {key}")
    if cfg["finite_solver"]["max_points"] < cfg["finite_solver"]["min_points"]:
        raise ValueError("max_points must be at least min_points")
    return cfg
