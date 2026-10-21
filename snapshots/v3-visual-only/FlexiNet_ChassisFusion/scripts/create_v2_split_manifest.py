from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_splits import create_split_manifest  # noqa: E402


def _yaw_from_quaternion(rotation: list[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _scene_motion_regime(nusc, scene: dict) -> str:
    positions: list[tuple[float, float, int, float]] = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"].get("LIDAR_TOP")
        if lidar_token:
            sample_data = nusc.get("sample_data", lidar_token)
            pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            positions.append(
                (
                    float(pose["translation"][0]),
                    float(pose["translation"][1]),
                    int(pose["timestamp"]),
                    _yaw_from_quaternion(pose["rotation"]),
                )
            )
        sample_token = sample.get("next", "")
    speeds = []
    yaw_rates = []
    for previous, current in zip(positions, positions[1:]):
        dt = (current[2] - previous[2]) / 1_000_000.0
        if dt <= 0.0:
            continue
        speeds.append(math.hypot(current[0] - previous[0], current[1] - previous[1]) / dt)
        delta_yaw = math.atan2(math.sin(current[3] - previous[3]), math.cos(current[3] - previous[3]))
        yaw_rates.append(abs(delta_yaw) / dt)
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    turning_fraction = sum(value >= 0.05 for value in yaw_rates) / len(yaw_rates) if yaw_rates else 0.0
    if turning_fraction >= 0.25:
        return "turning"
    if mean_speed >= 8.0:
        return "high_speed"
    return "low_speed_straight"


def _scene_metadata(nusc, scene: dict, official_split: str) -> dict:
    log = nusc.get("log", scene["log_token"])
    description = str(scene.get("description", "")).lower()
    lighting = "night" if "night" in description else "day"
    weather = "rain" if any(word in description for word in ("rain", "wet")) else "dry_or_unknown"
    return {
        "official_split": official_split,
        "location": str(log.get("location", "unknown")),
        "lighting": lighting,
        "weather": weather,
        "motion_regime": _scene_motion_regime(nusc, scene),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen scene-level V2 experiment split.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--can_bus_root")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/v2_split_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace frozen split manifest {output}; pass --overwrite explicitly."
        )
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    can_bus_root = Path(args.can_bus_root) if args.can_bus_root else Path(args.dataroot) / "can_bus"
    official = create_splits_scenes()
    train_names = set(official["train"])
    val_names = set(official["val"])
    metadata = {}
    excluded_scenes = {}
    eligible_train_names = set()
    eligible_val_names = set()
    for scene in nusc.scene:
        name = scene["name"]
        split = "train" if name in train_names else "val" if name in val_names else "excluded"
        if split != "excluded":
            required_channels = ("vehicle_monitor", "zoe_veh_info", "steeranglefeedback")
            missing_channels = [
                channel
                for channel in required_channels
                if not (can_bus_root / f"{name}_{channel}.json").is_file()
            ]
            first_sample = nusc.get("sample", scene["first_sample_token"])
            if "CAM_FRONT" not in first_sample["data"]:
                excluded_scenes[name] = "missing_CAM_FRONT"
            elif missing_channels:
                excluded_scenes[name] = f"missing_CAN:{','.join(missing_channels)}"
            else:
                metadata[name] = _scene_metadata(nusc, scene, split)
                (eligible_train_names if split == "train" else eligible_val_names).add(name)
    manifest = create_split_manifest(
        eligible_train_names,
        eligible_val_names,
        metadata,
        args.seed,
        args.version,
        excluded_scenes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "output": str(output),
        "checksum_sha256": manifest["checksum_sha256"],
        "partition_sizes": {key: len(value) for key, value in manifest["partitions"].items()},
        "excluded_scene_count": len(excluded_scenes),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
