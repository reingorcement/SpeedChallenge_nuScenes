from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_schema import KINEMATIC_VALUE_FEATURES
from train_v2 import build_dataset, load_config


FEATURE_CHANNEL_AND_CAP = (
    ("vehicle_monitor", 0.10),
    ("vehicle_monitor", 0.15),
    ("steeranglefeedback", 0.15),
    ("zoe_veh_info", 0.10),
    ("zoe_veh_info", 0.10),
    ("vehicle_monitor", 0.25),
    ("vehicle_monitor", 0.25),
    ("zoe_veh_info", 0.10),
    ("zoe_veh_info", 0.10),
    ("zoe_veh_info", 0.10),
    ("zoe_veh_info", 0.10),
    ("zoe_veh_info", 0.10),
)


def resolve_field_age_limits(audit: dict) -> list[float]:
    channels = audit.get("can_channels", {})
    limits = []
    for channel, cap in FEATURE_CHANNEL_AND_CAP:
        interval = channels.get(channel, {}).get("p95_interval_sec")
        if interval is None or not math.isfinite(float(interval)):
            raise ValueError(f"Missing finite p95 CAN interval for {channel}.")
        limits.append(min(cap, max(0.05, 1.5 * float(interval))))
    return limits


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit train-only V2 kinematic statistics.")
    parser.add_argument("--config", default="configs/velocity_xy_v2_visual.yaml")
    parser.add_argument("--audit", default="logs/v2_data_audit.json")
    parser.add_argument("--output", default="logs/v2_training_stats.json")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--split_manifest")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    for key, value in (
        ("dataroot", args.dataroot),
        ("can_bus_root", args.can_bus_root),
        ("flow_root", args.flow_root),
        ("split_manifest_path", args.split_manifest),
    ):
        if value:
            cfg["data"][key] = value
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    age_limits = resolve_field_age_limits(audit)
    cfg["data"]["kinematic_stats_path"] = None
    cfg["data"]["field_max_age_sec"] = age_limits
    cfg["data"]["photometric_augmentation"] = {"enabled": False}
    dataset = build_dataset(cfg, "train_core")

    sums = torch.zeros(len(KINEMATIC_VALUE_FEATURES), dtype=torch.float64)
    square_sums = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    records = []
    sample_count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    for index in range(sample_count):
        sample = dataset.base_dataset.samples[index]
        values, valid, *_ = dataset._build_kinematic_sequence(
            sample["scene_name"], sample["center_timestamp"]
        )
        valid64 = valid.to(torch.float64)
        values64 = values.to(torch.float64)
        sums += (values64 * valid64).sum(dim=0)
        square_sums += (values64.square() * valid64).sum(dim=0)
        counts += valid64.sum(dim=0)
        target = dataset.base_dataset._compute_velocity_label(sample["center_token"])
        records.append(
            {
                "token": str(sample["center_token"]),
                "yaw_rate": float(values[-1, 0]),
                "vy_abs": abs(float(target[1])),
            }
        )
    if bool((counts == 0).any()):
        missing = [name for name, count in zip(KINEMATIC_VALUE_FEATURES, counts) if count == 0]
        raise RuntimeError(f"Cannot fit statistics; fields have zero valid observations: {missing}")
    mean = sums / counts
    variance = (square_sums / counts - mean.square()).clamp_min(1.0e-6)
    std = variance.sqrt()
    vy_values = torch.tensor([record["vy_abs"] for record in records], dtype=torch.float32)
    high_lateral_threshold = float(torch.quantile(vy_values, 0.9))
    sample_weights = {}
    turning_count = 0
    high_lateral_count = 0
    for record in records:
        turning = abs(record["yaw_rate"]) >= 0.05
        high_lateral = record["vy_abs"] >= high_lateral_threshold
        turning_count += int(turning)
        high_lateral_count += int(high_lateral)
        sample_weights[record["token"]] = 3.0 if high_lateral else (2.0 if turning else 1.0)
    report = {
        "source": "nuScenes train_core split only",
        "features": list(KINEMATIC_VALUE_FEATURES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "valid_count": counts.to(torch.long).tolist(),
        "field_max_age_sec": age_limits,
        "high_lateral_threshold_mps": high_lateral_threshold,
        "sample_count": sample_count,
        "turning_fraction": turning_count / max(sample_count, 1),
        "high_lateral_fraction": high_lateral_count / max(sample_count, 1),
        "sample_weights": sample_weights,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sample_weights"}, indent=2))


if __name__ == "__main__":
    main()
