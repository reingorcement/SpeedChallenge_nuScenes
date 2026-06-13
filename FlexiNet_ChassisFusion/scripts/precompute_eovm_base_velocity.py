from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eovm_ekf"))

from eovm_ekf import EovmEkf, SensorSample, SteeringKalmanFilter


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return sorted(data, key=lambda row: int(row.get("utime", 0))) if isinstance(data, list) else []


def _latest_before(records: list[dict], timestamps: list[int], timestamp_us: int) -> dict | None:
    index = bisect_right(timestamps, timestamp_us) - 1
    return records[index] if index >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute EOVM-EKF base velocities from nuScenes CAN records.")
    parser.add_argument("--can_bus_root", required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--wheel_speed_scale", type=float, default=1.0, help="Scale raw wheel fields into rad/s.")
    parser.add_argument("--steering_scale", type=float, default=1.0 / 780.0, help="Scale vehicle_monitor steering into rad-like delta.")
    parser.add_argument("--sampling_time", type=float, default=0.01)
    args = parser.parse_args()

    can_root = Path(args.can_bus_root)
    monitor = _load_json(can_root / f"{args.scene_name}_vehicle_monitor.json")
    wheel = _load_json(can_root / f"{args.scene_name}_zoe_veh_info.json")
    if not monitor or not wheel:
        raise FileNotFoundError("vehicle_monitor and zoe_veh_info messages are required for EOVM precompute.")

    wheel_timestamps = [int(row.get("utime", 0)) for row in wheel]
    estimator = EovmEkf(sampling_time=args.sampling_time)
    steering_filter = SteeringKalmanFilter(sampling_time=args.sampling_time)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_us", "vx", "vy", "gamma", "delta", "delta_dot"])
        writer.writeheader()
        previous_timestamp = None
        for monitor_row in monitor:
            timestamp_us = int(monitor_row.get("utime", 0))
            wheel_row = _latest_before(wheel, wheel_timestamps, timestamp_us)
            if wheel_row is None:
                continue
            if previous_timestamp is not None:
                dt = max((timestamp_us - previous_timestamp) / 1e6, 1.0e-3)
                estimator.sampling_time = dt
                steering_filter.sampling_time = dt
            previous_timestamp = timestamp_us

            steering_angle = float(monitor_row.get("steering", 0.0)) * args.steering_scale
            delta, delta_dot = steering_filter.step(steering_angle)
            sample = SensorSample(
                steering_angle=steering_angle,
                yaw_rate=math.radians(float(monitor_row.get("yaw_rate", 0.0))),
                wheel_speeds=np.array(
                    [
                        float(wheel_row.get("FL_wheel_speed", 0.0)) * args.wheel_speed_scale,
                        float(wheel_row.get("FR_wheel_speed", 0.0)) * args.wheel_speed_scale,
                        float(wheel_row.get("RL_wheel_speed", 0.0)) * args.wheel_speed_scale,
                        float(wheel_row.get("RR_wheel_speed", 0.0)) * args.wheel_speed_scale,
                    ],
                    dtype=float,
                ),
                ax=float(wheel_row.get("longitudinal_accel", 0.0)),
                ay=float(wheel_row.get("transversal_accel", 0.0)),
            )
            result = estimator.step(sample, delta=delta, delta_dot=delta_dot)
            writer.writerow(
                {
                    "timestamp_us": timestamp_us,
                    "vx": result.vx,
                    "vy": result.vy,
                    "gamma": result.gamma,
                    "delta": result.delta,
                    "delta_dot": result.delta_dot,
                }
            )

    print(f"Saved EOVM-EKF base velocity estimates to {output_path}")
    print("Use only after auditing wheel_speed_scale, steering_scale, and vehicle geometry for nuScenes.")


if __name__ == "__main__":
    main()
