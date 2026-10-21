from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_v2 import build_dataset, build_model, forward_batch, load_config, move_batch  # noqa: E402


DEFAULT_VARIANTS = (
    "baseline",
    "no_kinematic_correction",
    "no_visual_residual",
    "no_wheel_scaling",
    "no_yaw_compensation",
    "missing_can",
    "stale_can",
    "shuffled_images",
    "shuffled_flow",
    "shuffled_can",
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _permutation(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        return torch.arange(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def _modified_batch(batch: dict[str, Any], variant: str) -> dict[str, Any]:
    modified = dict(batch)
    batch_size = int(batch["images"].shape[0])
    permutation = _permutation(batch_size, batch["images"].device)
    if variant == "no_yaw_compensation":
        modified["angular_rate_camera"] = torch.zeros_like(batch["angular_rate_camera"])
        modified["angular_rate_valid"] = torch.zeros_like(batch["angular_rate_valid"])
    elif variant in {"missing_can", "stale_can"}:
        if variant == "missing_can":
            modified["kinematic_values"] = torch.zeros_like(batch["kinematic_values"])
            modified["wheel_speed"] = torch.zeros_like(batch["wheel_speed"])
            modified["angular_rate_camera"] = torch.zeros_like(batch["angular_rate_camera"])
        modified["kinematic_valid"] = torch.zeros_like(batch["kinematic_valid"])
        modified["kinematic_age"] = torch.ones_like(batch["kinematic_age"])
        modified["wheel_speed_valid"] = torch.zeros_like(batch["wheel_speed_valid"])
        modified["wheel_speed_age_sec"] = torch.full_like(batch["wheel_speed_age_sec"], 1.0e6)
        modified["angular_rate_valid"] = torch.zeros_like(batch["angular_rate_valid"])
        modified["angular_rate_age_sec"] = torch.full_like(batch["angular_rate_age_sec"], 1.0e6)
    elif variant == "shuffled_images":
        modified["images"] = batch["images"][permutation]
    elif variant == "shuffled_flow":
        modified["flow_v2"] = batch["flow_v2"][permutation]
        modified["flow_confidence"] = batch["flow_confidence"][permutation]
    elif variant == "shuffled_can":
        for key in (
            "kinematic_values",
            "kinematic_valid",
            "kinematic_age",
            "wheel_speed",
            "wheel_speed_valid",
            "wheel_speed_age_sec",
            "angular_rate_camera",
            "angular_rate_valid",
            "angular_rate_age_sec",
        ):
            modified[key] = batch[key][permutation]
    return modified


def ablation_prediction(model, batch: dict[str, Any], baseline, variant: str):
    if variant == "baseline":
        return baseline, baseline.predicted_velocity
    if variant == "no_kinematic_correction":
        return baseline, baseline.visual_velocity
    if variant == "no_visual_residual":
        correction = baseline.residual_gate * baseline.residual
        return baseline, baseline.predicted_velocity - correction
    if variant == "no_wheel_scaling":
        wheel_delta = torch.zeros_like(baseline.predicted_velocity)
        wheel_delta[:, 0] = torch.where(
            baseline.geometry_valid,
            (baseline.wheel_scale[:, 0] - 1.0) * baseline.geometry_velocity[:, 0],
            torch.zeros_like(baseline.geometry_velocity[:, 0]),
        )
        return baseline, baseline.predicted_velocity - wheel_delta
    if variant not in DEFAULT_VARIANTS:
        raise ValueError(f"Unknown holdout variant {variant!r}.")
    output = forward_batch(model, _modified_batch(batch, variant))
    return output, output.predicted_velocity


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def metrics_from_rows(rows: list[dict[str, Any]], high_lateral_threshold: float) -> dict[str, float | bool]:
    if not rows:
        raise ValueError("Cannot compute holdout metrics from no rows.")
    errors_vx = [float(row["pred_vx_mps"]) - float(row["target_vx_mps"]) for row in rows]
    errors_vy = [float(row["pred_vy_mps"]) - float(row["target_vy_mps"]) for row in rows]
    endpoints = [math.hypot(x, y) for x, y in zip(errors_vx, errors_vy)]
    report: dict[str, float | bool] = {
        "sample_count": float(len(rows)),
        "scene_count": float(len({str(row["scene_name"]) for row in rows})),
        "mae_vx_mps": _mean([abs(value) for value in errors_vx]),
        "rmse_vx_mps": math.sqrt(_mean([value * value for value in errors_vx])),
        "mae_vy_mps": _mean([abs(value) for value in errors_vy]),
        "rmse_vy_mps": math.sqrt(_mean([value * value for value in errors_vy])),
        "endpoint_error_mps": _mean(endpoints),
    }
    subsets = {
        "straight": [abs(float(row["yaw_rate_radps"])) < 0.05 for row in rows],
        "turning": [abs(float(row["yaw_rate_radps"])) >= 0.05 for row in rows],
        "high_lateral": [abs(float(row["target_vy_mps"])) >= high_lateral_threshold for row in rows],
        "wheel_inconsistent": [float(row["wheel_relative_std"]) >= 0.05 for row in rows],
        "geometry_invalid": [not bool(int(row["geometry_valid"])) for row in rows],
        "fallback": [bool(int(row["fallback_used"])) for row in rows],
    }
    for name, mask in subsets.items():
        indices = [index for index, selected in enumerate(mask) if selected]
        report[f"{name}_sample_count"] = float(len(indices))
        if indices:
            report[f"{name}_mae_vy_mps"] = _mean([abs(errors_vy[index]) for index in indices])
            report[f"{name}_endpoint_error_mps"] = _mean([endpoints[index] for index in indices])

    valid_rows = [row for row in rows if bool(int(row["geometry_valid"]))]
    finite_conditions = sorted(
        float(row["geometry_condition_number"])
        for row in valid_rows
        if math.isfinite(float(row["geometry_condition_number"]))
    )
    report["geometry_valid_rate"] = len(valid_rows) / len(rows)
    report["geometry_condition_median"] = (
        statistics.median(finite_conditions) if finite_conditions else 1.0e12
    )
    sign_rows = [
        row
        for row in valid_rows
        if abs(float(row["yaw_rate_radps"])) < 0.05 and abs(float(row["target_vx_mps"])) >= 1.0
    ]
    report["geometry_straight_vx_sign_agreement"] = (
        _mean(
            [
                float(
                    math.copysign(1.0, float(row["geometry_vx_mps"]))
                    == math.copysign(1.0, float(row["target_vx_mps"]))
                )
                for row in sign_rows
            ]
        )
        if sign_rows
        else 0.0
    )
    finite_fields = (
        "target_vx_mps",
        "target_vy_mps",
        "pred_vx_mps",
        "pred_vy_mps",
        "visual_vx_mps",
        "visual_vy_mps",
        "kinematic_correction_vx_mps",
        "kinematic_correction_vy_mps",
    )
    report["all_outputs_finite"] = all(
        math.isfinite(float(row[field])) for row in rows for field in finite_fields
    )
    report["max_abs_kinematic_correction_vx_mps"] = max(
        abs(float(row["kinematic_correction_vx_mps"])) for row in rows
    )
    report["max_abs_kinematic_correction_vy_mps"] = max(
        abs(float(row["kinematic_correction_vy_mps"])) for row in rows
    )
    return report


def _row(raw_batch: dict, batch: dict, output, prediction, index: int, variant: str, seed: int) -> dict:
    raw_kinematics = batch.get("kinematic_values_raw", batch["kinematic_values"])
    return {
        "variant": variant,
        "seed": seed,
        "scene_name": raw_batch["scene_name"][index],
        "center_token": raw_batch["center_token"][index],
        "center_timestamp_us": int(batch["center_timestamp"][index]),
        "target_vx_mps": float(batch["velocity"][index, 0]),
        "target_vy_mps": float(batch["velocity"][index, 1]),
        "pred_vx_mps": float(prediction[index, 0]),
        "pred_vy_mps": float(prediction[index, 1]),
        "visual_vx_mps": float(output.visual_velocity[index, 0]),
        "visual_vy_mps": float(output.visual_velocity[index, 1]),
        "geometry_vx_mps": float(output.geometry_velocity[index, 0]),
        "geometry_vy_mps": float(output.geometry_velocity[index, 1]),
        "geometry_valid": int(output.geometry_valid[index]),
        "geometry_condition_number": float(output.geometry_condition_number[index]),
        "geometry_residual_rms": float(output.geometry_residual_rms[index]),
        "geometry_support_count": int(output.geometry_support_count[index]),
        "fallback_used": int(output.fallback_used[index]),
        "wheel_scale": float(output.wheel_scale[index, 0]),
        "visual_residual_vx_mps": float(output.residual[index, 0]),
        "visual_residual_vy_mps": float(output.residual[index, 1]),
        "visual_residual_gate": float(output.residual_gate[index, 0]),
        "kinematic_residual_vx_mps": float(output.kinematic_residual[index, 0]),
        "kinematic_residual_vy_mps": float(output.kinematic_residual[index, 1]),
        "kinematic_correction_vx_mps": float(output.kinematic_applied_correction[index, 0]),
        "kinematic_correction_vy_mps": float(output.kinematic_applied_correction[index, 1]),
        "kinematic_gate_vx": float(output.kinematic_gate[index, 0]),
        "kinematic_gate_vy": float(output.kinematic_gate[index, 1]),
        "kinematic_log_variance_vx": float(output.kinematic_log_variance[index, 0]),
        "kinematic_log_variance_vy": float(output.kinematic_log_variance[index, 1]),
        "kinematic_coverage": float(output.kinematic_coverage[index, 0]),
        "kinematic_available": int(output.kinematic_available[index]),
        "yaw_rate_radps": float(raw_kinematics[index, -1, 0]),
        "wheel_relative_std": float(raw_kinematics[index, -1, -1]),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen V2 checkpoints on stage1_test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=["geometry_only", "visual", "visual_kinematic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--split_manifest")
    parser.add_argument("--frame_cache_root")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The nuScenes holdout evaluator is server-only and requires CUDA.")

    cfg = load_config(args.config)
    for key, value in (
        ("dataroot", args.dataroot),
        ("can_bus_root", args.can_bus_root),
        ("flow_root", args.flow_root),
        ("split_manifest_path", args.split_manifest),
        ("frame_cache_root", args.frame_cache_root),
    ):
        if value:
            cfg["data"][key] = value
    cfg["data"]["geometry_cache_mode"] = "off"
    cfg["data"]["geometry_cache_root"] = None
    if args.mode:
        cfg["model"]["mode"] = args.mode
    cfg["model"]["kinematic_dropout"] = 0.0
    cfg["model"]["wheel_scale_jitter"] = 0.0
    dataset = build_dataset(cfg, "stage1_test")
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 8)),
        pin_memory=True,
    )
    model = build_model(cfg)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    elif cfg["model"]["mode"] != "geometry_only":
        raise ValueError("A checkpoint is required for visual and visual_kinematic evaluation.")
    device = torch.device("cuda")
    model.eval().to(device)

    rows_by_variant: dict[str, list[dict]] = {variant: [] for variant in args.variants}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        baseline = forward_batch(model, batch)
        for variant in args.variants:
            output, prediction = ablation_prediction(model, batch, baseline, variant)
            for index in range(int(prediction.shape[0])):
                rows_by_variant[variant].append(
                    _row(raw_batch, batch, output, prediction, index, variant, args.seed)
                )

    stats_path = cfg["data"].get("kinematic_stats_path")
    statistics = json.loads(Path(stats_path).read_text(encoding="utf-8")) if stats_path else {}
    high_lateral_threshold = float(statistics.get("high_lateral_threshold_mps", 0.5))
    metrics = {
        "partition": "stage1_test",
        "split_manifest_path": cfg["data"].get("split_manifest_path"),
        "mode": cfg["model"]["mode"],
        "seed": args.seed,
        "high_lateral_threshold_mps": high_lateral_threshold,
        "variants": {
            variant: metrics_from_rows(rows, high_lateral_threshold)
            for variant, rows in rows_by_variant.items()
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for variant in args.variants for row in rows_by_variant[variant]]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    safe_metrics = json_safe(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(safe_metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(safe_metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
