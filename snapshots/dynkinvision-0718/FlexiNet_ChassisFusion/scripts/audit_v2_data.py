from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_splits import partition_scene_names  # noqa: E402


CHANNEL_FIELDS = {
    "vehicle_monitor": ("yaw_rate", "steering", "throttle", "brake"),
    "zoe_veh_info": (
        "longitudinal_accel",
        "transversal_accel",
        "FL_wheel_speed",
        "FR_wheel_speed",
        "RL_wheel_speed",
        "RR_wheel_speed",
    ),
    "steeranglefeedback": ("value",),
}


def _audit_can_channel(
    root: Path,
    channel: str,
    fields: tuple[str, ...],
    scene_names: set[str] | None = None,
) -> dict:
    files = sorted(root.glob(f"scene-*_{channel}.json"))
    if scene_names is not None:
        files = [path for path in files if path.name[:10] in scene_names]
    field_finite = {field: 0 for field in fields}
    record_count = 0
    intervals = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            continue
        timestamps = sorted(int(row.get("utime", 0)) for row in records if int(row.get("utime", 0)) > 0)
        intervals.extend(
            (timestamps[index] - timestamps[index - 1]) / 1_000_000.0
            for index in range(1, len(timestamps))
            if timestamps[index] > timestamps[index - 1]
        )
        record_count += len(records)
        for row in records:
            for field in fields:
                try:
                    finite = math.isfinite(float(row[field]))
                except (KeyError, TypeError, ValueError):
                    finite = False
                field_finite[field] += int(finite)
    return {
        "scene_count": len(files),
        "record_count": record_count,
        "median_interval_sec": statistics.median(intervals) if intervals else None,
        "p95_interval_sec": (
            sorted(intervals)[min(int(0.95 * len(intervals)), len(intervals) - 1)] if intervals else None
        ),
        "field_finite_rate": {
            field: count / max(record_count, 1) for field, count in field_finite.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit nuScenes inputs required by the V2 velocity estimator.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--can_bus_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--cam_channel", default="CAM_FRONT")
    parser.add_argument("--split_manifest")
    parser.add_argument("--partition", default="train_core")
    parser.add_argument("--output", default="logs/v2_data_audit.json")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    selected_scene_names = (
        set(
            partition_scene_names(
                args.split_manifest,
                args.partition,
                expected_version=args.version,
            )
        )
        if args.split_manifest
        else None
    )
    can_root = Path(args.can_bus_root)
    scenes_with_can = {path.name[:10] for path in can_root.glob("scene-*_vehicle_monitor.json")}
    calibration_tokens = set()
    camera_frames = 0
    timestamp_failures = 0
    for scene in nusc.scene:
        if selected_scene_names is not None and scene["name"] not in selected_scene_names:
            continue
        sample = nusc.get("sample", scene["first_sample_token"])
        token = sample["data"].get(args.cam_channel, "")
        while token:
            sample_data = nusc.get("sample_data", token)
            camera_frames += 1
            calibration_tokens.add(sample_data["calibrated_sensor_token"])
            if int(sample_data.get("timestamp", 0)) <= 0:
                timestamp_failures += 1
            token = sample_data.get("next", "")

    calibrations = [nusc.get("calibrated_sensor", token) for token in calibration_tokens]
    can_audit = {
        channel: _audit_can_channel(can_root, channel, fields, selected_scene_names)
        for channel, fields in CHANNEL_FIELDS.items()
    }
    report = {
        "version": args.version,
        "camera_channel": args.cam_channel,
        "camera_frames": camera_frames,
        "camera_calibration_count": len(calibrations),
        "all_camera_intrinsics_present": all(bool(row.get("camera_intrinsic")) for row in calibrations),
        "all_camera_extrinsics_present": all(bool(row.get("rotation")) and bool(row.get("translation")) for row in calibrations),
        "invalid_camera_timestamps": timestamp_failures,
        "partition": args.partition if args.split_manifest else "all",
        "scene_count": len(selected_scene_names) if selected_scene_names is not None else len(nusc.scene),
        "scenes_with_vehicle_monitor": len(
            scenes_with_can & selected_scene_names if selected_scene_names is not None else scenes_with_can
        ),
        "can_channels": can_audit,
        "full_three_axis_imu_available": False,
        "verified_cg_lever_arm_available": False,
        "label_reference": "nuScenes ego_pose reference point",
        "blocking_gaps": [
            "No deployable full three-axis IMU in the current fair-chassis interface.",
            "No verified camera/IMU/GNSS-to-CG lever-arm calibration.",
            "Camera height is nominal until independently measured.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
