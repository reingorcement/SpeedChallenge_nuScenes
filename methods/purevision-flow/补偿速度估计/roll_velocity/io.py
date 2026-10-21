"""Validated visual inputs and portable artifact I/O. No label readers."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import math

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAL_FIELDS = (
    "image_size", "K", "height_m", "plane_normal_camera",
    "rotation_ego_from_camera", "translation_ego_from_camera", "body_mount_rotation",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def intrinsics(cal, width=512, height=288):
    """Pixel-center transform used by the corrected v2 interface."""
    sx, sy = width / cal["image_size"][0], height / cal["image_size"][1]
    transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    return transform @ np.asarray(cal["K"], float)


def validate_calibration(raw):
    missing = set(CAL_FIELDS) - raw.keys()
    if missing:
        raise ValueError(f"Missing fixed calibration fields: {sorted(missing)}")
    cal = {key: raw[key] for key in CAL_FIELDS}
    size = np.asarray(cal["image_size"])
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 8) or np.any(size != size.astype(int)):
        raise ValueError("image_size must be [width, height] in integer pixels")
    for key, shape in [("K", (3, 3)), ("plane_normal_camera", (3,)),
                       ("rotation_ego_from_camera", (3, 3)),
                       ("translation_ego_from_camera", (3,)), ("body_mount_rotation", (3, 3))]:
        a = np.asarray(cal[key], float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}")
    k = np.asarray(cal["K"], float)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8:
        raise ValueError("K must be a pinhole matrix with zero skew; use undistorted images")
    for key in ["rotation_ego_from_camera", "body_mount_rotation"]:
        r = np.asarray(cal[key], float)
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
            raise ValueError(f"{key} must be a proper rotation")
    if not np.isclose(np.linalg.norm(cal["plane_normal_camera"]), 1, atol=1e-5):
        raise ValueError("plane_normal_camera must be a unit normal")
    if not np.isfinite(cal["height_m"]) or cal["height_m"] <= 0:
        raise ValueError("height_m must be positive meters")
    if "distortion_coefficients" in raw and np.any(np.asarray(raw["distortion_coefficients"]) != 0):
        raise ValueError("Undistort images and supply the corresponding K before inference")
    return cal


def load_sequence(path, start=0, count=0):
    path = Path(path).resolve()
    cal = validate_calibration(read_json(path / "calibration.json"))
    raw = read_json(path / "frames.json")
    if not isinstance(raw, list) or start < 0 or count < 0:
        raise ValueError("frames.json must be a list; start/count must be nonnegative")
    raw = raw[start:start + count if count else None]
    if len(raw) < 5:
        raise ValueError("At least five consecutive images are required (four warmup frames)")
    frames = []
    for offset, frame in enumerate(raw):
        rel = str(frame["file"]).replace("\\", "/")
        file = (path / rel).resolve()
        if PureWindowsPath(rel).is_absolute() or not file.is_relative_to(path):
            raise ValueError(f"Image path must stay inside the input directory: {rel}")
        if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            raise ValueError(f"Unsupported image type: {rel}")
        timestamp = float(frame["solver_time"])
        ident = frame.get("image_id", start + offset)
        if not isinstance(ident, int) or isinstance(ident, bool):
            raise ValueError("image_id must be an integer")
        if not math.isfinite(timestamp) or (frames and timestamp <= frames[-1]["solver_time"]):
            raise ValueError("solver_time must be finite and strictly increasing, in seconds")
        if frames and ident <= frames[-1]["image_id"]:
            raise ValueError("image_id must be strictly increasing")
        with Image.open(file) as image:
            if list(image.size) != cal["image_size"]:
                raise ValueError(f"Image dimensions disagree with calibration: {rel}")
            image.verify()
        frames.append(dict(image_id=ident, file=rel, solver_time=timestamp, sha256=sha256(file)))
    return frames, cal


def code_hashes():
    files = [ROOT / "run.py"]
    for folder in ("roll_velocity", "model"):
        files.extend(sorted((ROOT / folder).rglob("*.py")))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
