from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eovm_ekf"))

from eovm_ekf.eovm import load_transfer_weights
from eovm_ekf.parameters import EstimatorTuning, VehicleGeometry


REQUIRED_MESSAGES = ["vehicle_monitor", "zoe_veh_info", "steeranglefeedback"]
WHEEL_FIELDS = ["FL_wheel_speed", "FR_wheel_speed", "RL_wheel_speed", "RR_wheel_speed"]


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _range(values: list[float]) -> tuple[float, float] | None:
    return (min(values), max(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit nuScenes CAN fields needed by EOVM-EKF and load transfer.")
    parser.add_argument("--can_bus_root", required=True)
    parser.add_argument("--scene_name", required=True, help="Example: scene-0061")
    args = parser.parse_args()

    can_root = Path(args.can_bus_root)
    for message in REQUIRED_MESSAGES:
        records = _load_json(can_root / f"{args.scene_name}_{message}.json")
        print(f"{message}: {len(records)} records")

    wheel_records = _load_json(can_root / f"{args.scene_name}_zoe_veh_info.json")
    monitor_records = _load_json(can_root / f"{args.scene_name}_vehicle_monitor.json")
    steer_records = _load_json(can_root / f"{args.scene_name}_steeranglefeedback.json")

    for field in WHEEL_FIELDS:
        values = [float(row.get(field, 0.0)) for row in wheel_records if field in row]
        print(f"{field} raw range: {_range(values)}")
    for field in ["longitudinal_accel", "transversal_accel", "odom_speed"]:
        values = [float(row.get(field, 0.0)) for row in wheel_records if field in row]
        print(f"{field} range: {_range(values)}")
    for field in ["vehicle_speed", "yaw_rate", "steering"]:
        values = [float(row.get(field, 0.0)) for row in monitor_records if field in row]
        print(f"{field} range: {_range(values)}")
    steer_values = [float(row.get("value", 0.0)) for row in steer_records if "value" in row]
    print(f"steeranglefeedback.value range: {_range(steer_values)}")

    if wheel_records:
        sample = wheel_records[len(wheel_records) // 2]
        ax = float(sample.get("longitudinal_accel", 0.0))
        ay = float(sample.get("transversal_accel", 0.0))
        weights = load_transfer_weights(ax=ax, ay=ay, geometry=VehicleGeometry(), tuning=EstimatorTuning())
        print(f"load-transfer theta for middle sample ax={ax}, ay={ay}: {weights.theta.tolist()}")

    print("Audit note: EOVM still needs wheel-speed unit conversion, steering sign/scale, and vehicle geometry calibration before use as a main baseline.")


if __name__ == "__main__":
    main()
