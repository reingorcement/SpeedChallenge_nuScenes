from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HOST_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_PROJECT_ROOT))

from experiments.nuscenes_version_comparison.bootstrap import SOURCE_PROJECT_ROOT  # noqa: E402

PROJECT_ROOT = SOURCE_PROJECT_ROOT

from dataset.nuscenes_chassis import FAIR_CHASSIS_INDEX  # noqa: E402
from experiments.nuscenes_version_comparison import ALGORITHMS  # noqa: E402
from experiments.nuscenes_version_comparison.adapters.dataset_adapter import (  # noqa: E402
    build_dataset_for_algorithm,
)
from experiments.nuscenes_version_comparison.adapters.legacy_adapter import (  # noqa: E402
    BaseVelocityAdapter,
    build_legacy_adapter,
)
from experiments.nuscenes_version_comparison.config import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    load_resolved_config,
    resolve_project_path,
    write_yaml,
)
from experiments.nuscenes_version_comparison.perturbations import (  # noqa: E402
    PerturbationContext,
    apply_perturbation,
)
from utils.seed import set_seed  # noqa: E402


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _path_overrides(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "dataroot": args.dataroot,
        "can_bus_root": args.can_bus_root,
        "flow_root": args.flow_root,
        "frame_cache_root": args.frame_cache_root,
        "geometry_cache_root": args.geometry_cache_root,
        "split_manifest_path": args.split_manifest,
        "kinematic_stats_path": args.kinematic_stats,
    }


def _build_adapter(config: dict[str, Any], checkpoint: str | None):
    algorithm = str(config["_comparison"]["name"])
    family = str(config["_comparison"]["family"])
    if algorithm == "base_velocity":
        if checkpoint:
            raise ValueError("base_velocity must not receive a checkpoint.")
        return BaseVelocityAdapter()
    if family == "legacy":
        if not checkpoint:
            raise ValueError(f"{algorithm} requires --checkpoint.")
        adapter = build_legacy_adapter(config, load_pretrained=False, project_root=PROJECT_ROOT)
    else:
        from experiments.nuscenes_version_comparison.adapters.v2_adapter import build_v2_adapter

        adapter = build_v2_adapter(config)
        if algorithm != "v2_geometry" and not checkpoint:
            raise ValueError(f"{algorithm} requires --checkpoint.")
    if checkpoint:
        adapter.load_checkpoint(checkpoint)
    return adapter


def _current_legacy_chassis(batch: dict[str, Any]) -> torch.Tensor:
    chassis = batch["chassis"]
    indexes = batch.get("chassis_current_index")
    if indexes is None:
        return chassis[:, -1]
    indexes = indexes.to(chassis.device, dtype=torch.long).reshape(-1)
    batch_indexes = torch.arange(chassis.shape[0], device=chassis.device)
    return chassis[batch_indexes, indexes]


def _scenario_signals(batch: dict[str, Any], family: str) -> dict[str, torch.Tensor]:
    if family == "v2":
        raw = batch["kinematic_values_raw"][:, -1]
        return {
            "yaw_rate_radps": raw[:, 0],
            "steering_feedback": raw[:, 2],
            "longitudinal_accel_mps2": raw[:, 3],
            "lateral_accel_mps2": raw[:, 4],
            "throttle": raw[:, 5],
            "brake": raw[:, 6],
            "wheel_relative_std": raw[:, 11],
        }
    current = _current_legacy_chassis(batch)
    wheel_mean = current[:, FAIR_CHASSIS_INDEX["wheel_speed_mean"]].abs().clamp_min(0.5)
    return {
        "yaw_rate_radps": current[:, FAIR_CHASSIS_INDEX["yaw_rate"]],
        "steering_feedback": current[:, FAIR_CHASSIS_INDEX["steering_feedback"]],
        "longitudinal_accel_mps2": current[:, FAIR_CHASSIS_INDEX["longitudinal_accel"]],
        "lateral_accel_mps2": current[:, FAIR_CHASSIS_INDEX["transversal_accel"]],
        "throttle": current[:, FAIR_CHASSIS_INDEX["throttle"]],
        "brake": current[:, FAIR_CHASSIS_INDEX["brake"]],
        "wheel_relative_std": current[:, FAIR_CHASSIS_INDEX["wheel_speed_std"]] / wheel_mean,
    }


def _diagnostic_value(diagnostics: dict[str, Any], key: str, index: int, axis: int = 0):
    value = diagnostics.get(key)
    if not isinstance(value, torch.Tensor):
        return ""
    item = value[index]
    if item.ndim:
        item = item.reshape(-1)[axis]
    return float(item)


def _gate_columns(diagnostics: dict[str, Any], index: int) -> dict[str, float | str]:
    values: dict[str, float | str] = {
        "gate_vision": "",
        "gate_kinematic": "",
        "gate_dynamics": "",
    }
    gate = diagnostics.get("gate_weights")
    names = diagnostics.get("gate_names")
    if isinstance(gate, torch.Tensor) and names:
        for gate_index, name in enumerate(names):
            values[f"gate_{name}"] = float(gate[index, gate_index])
    return values


def _rows(
    raw_batch: dict[str, Any],
    clean_batch: dict[str, Any],
    prediction: torch.Tensor,
    diagnostics: dict[str, Any],
    *,
    algorithm: str,
    family: str,
    reference_commit: str,
    runtime_commit: str,
    seed: int,
    variant: str,
    partition: str,
    domain: str,
) -> list[dict[str, Any]]:
    signals = _scenario_signals(clean_batch, family)
    target = clean_batch["velocity"]
    base = clean_batch["base_velocity"]
    output = []
    for index in range(prediction.shape[0]):
        error = prediction[index] - target[index]
        row = {
            "algorithm": algorithm,
            "family": family,
            "reference_commit": reference_commit,
            "runtime_commit": runtime_commit,
            "seed": seed,
            "variant": variant,
            "partition": partition,
            "domain": domain,
            "scene_name": raw_batch["scene_name"][index],
            "center_token": raw_batch["center_token"][index],
            "center_timestamp_us": int(clean_batch["center_timestamp"][index]),
            "target_vx_mps": float(target[index, 0]),
            "target_vy_mps": float(target[index, 1]),
            "pred_vx_mps": float(prediction[index, 0]),
            "pred_vy_mps": float(prediction[index, 1]),
            "base_vx_mps": float(base[index, 0]),
            "base_vy_mps": float(base[index, 1]),
            "error_vx_mps": float(error[0]),
            "error_vy_mps": float(error[1]),
            "endpoint_error_mps": float(torch.linalg.vector_norm(error)),
            **{name: float(value[index]) for name, value in signals.items()},
            **_gate_columns(diagnostics, index),
            "visual_vx_mps": _diagnostic_value(diagnostics, "visual_velocity", index, 0),
            "visual_vy_mps": _diagnostic_value(diagnostics, "visual_velocity", index, 1),
            "geometry_vx_mps": _diagnostic_value(diagnostics, "geometry_velocity", index, 0),
            "geometry_vy_mps": _diagnostic_value(diagnostics, "geometry_velocity", index, 1),
            "geometry_valid": _diagnostic_value(diagnostics, "geometry_valid", index),
            "geometry_condition_number": _diagnostic_value(
                diagnostics, "geometry_condition_number", index
            ),
            "fallback_used": _diagnostic_value(diagnostics, "fallback_used", index),
            "wheel_scale": _diagnostic_value(diagnostics, "wheel_scale", index),
            "visual_residual_gate": _diagnostic_value(
                diagnostics, "visual_residual_gate", index
            ),
            "kinematic_gate_vx": _diagnostic_value(diagnostics, "kinematic_gate", index, 0),
            "kinematic_gate_vy": _diagnostic_value(diagnostics, "kinematic_gate", index, 1),
            "kinematic_correction_vx_mps": _diagnostic_value(
                diagnostics, "kinematic_correction", index, 0
            ),
            "kinematic_correction_vy_mps": _diagnostic_value(
                diagnostics, "kinematic_correction", index, 1
            ),
        }
        output.append(row)
    return output


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot compute metrics from no rows.")
    error_x = torch.tensor([float(row["error_vx_mps"]) for row in rows])
    error_y = torch.tensor([float(row["error_vy_mps"]) for row in rows])
    endpoint = torch.tensor([float(row["endpoint_error_mps"]) for row in rows])
    target_speed = torch.tensor(
        [math.hypot(float(row["target_vx_mps"]), float(row["target_vy_mps"])) for row in rows]
    )
    predicted_speed = torch.tensor(
        [math.hypot(float(row["pred_vx_mps"]), float(row["pred_vy_mps"])) for row in rows]
    )
    return {
        "sample_count": float(len(rows)),
        "scene_count": float(len({str(row["scene_name"]) for row in rows})),
        "mae_vx_mps": float(error_x.abs().mean()),
        "rmse_vx_mps": float(error_x.square().mean().sqrt()),
        "mae_vy_mps": float(error_y.abs().mean()),
        "rmse_vy_mps": float(error_y.square().mean().sqrt()),
        "endpoint_error_mps": float(endpoint.mean()),
        "endpoint_p90_mps": float(torch.quantile(endpoint, 0.90)),
        "endpoint_p95_mps": float(torch.quantile(endpoint, 0.95)),
        "endpoint_p99_mps": float(torch.quantile(endpoint, 0.99)),
        "speed_magnitude_mae_mps": float((predicted_speed - target_speed).abs().mean()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen comparison checkpoint.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    parser.add_argument("--partition", choices=["train_core", "dev", "stage1_test"], default="stage1_test")
    parser.add_argument("--domain", choices=["all", "boston", "singapore"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dataroot")
    parser.add_argument("--can-bus-root")
    parser.add_argument("--flow-root")
    parser.add_argument("--frame-cache-root")
    parser.add_argument("--geometry-cache-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--kinematic-stats")
    args = parser.parse_args()

    config, protocol, _ = load_resolved_config(
        args.algorithm,
        protocol_path=args.protocol,
        seed=args.seed,
        path_overrides=_path_overrides(args),
    )
    family = str(config["_comparison"]["family"])
    perturbation_family = "v2" if family == "v2" else "legacy"
    if family == "v2":
        config["data"]["geometry_cache_mode"] = "off"
        config["data"]["geometry_cache_root"] = None
        config["model"]["kinematic_dropout"] = 0.0
        config["model"]["wheel_scale_jitter"] = 0.0
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    variants = args.variants or list(protocol["stress"]["variants"])
    if args.include_calibration:
        variants.extend(protocol["stress"]["v2_calibration_variants"])
    variants = list(dict.fromkeys(variants))
    if family != "v2" and any(name in protocol["stress"]["v2_calibration_variants"] for name in variants):
        raise ValueError("Calibration variants are V2-only diagnostics.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Formal nuScenes evaluation requires CUDA; use --allow-cpu only for smoke checks.")
    set_seed(args.seed)
    dataset = build_dataset_for_algorithm(config, args.partition, domain=args.domain)
    workers = int(config["train"].get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(config["train"].get("prefetch_factor", 2)) if workers else None,
    )
    adapter = _build_adapter(config, args.checkpoint).eval().to(device)
    stats_path = config["data"].get("kinematic_stats_path")
    if stats_path:
        stats_path = resolve_project_path(stats_path)
    context = PerturbationContext.from_config(protocol, stats_path)
    rows_by_variant = {variant: [] for variant in variants}
    runtime_commit = _git_commit()
    for step, raw_batch in enumerate(tqdm(loader, desc=f"evaluate {args.algorithm}")):
        if args.max_batches and step >= args.max_batches:
            break
        clean_batch = move_batch(raw_batch, device)
        for variant in variants:
            perturbed = apply_perturbation(clean_batch, variant, perturbation_family, context)
            if not torch.equal(perturbed["velocity"], clean_batch["velocity"]):
                raise RuntimeError(f"Perturbation {variant!r} changed supervision labels.")
            prediction, diagnostics = adapter(perturbed)
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError(f"Non-finite prediction for variant {variant!r}.")
            rows_by_variant[variant].extend(
                _rows(
                    raw_batch,
                    clean_batch,
                    prediction,
                    diagnostics,
                    algorithm=args.algorithm,
                    family=family,
                    reference_commit=str(config["_comparison"]["reference_commit"]),
                    runtime_commit=runtime_commit,
                    seed=args.seed,
                    variant=variant,
                    partition=args.partition,
                    domain=args.domain,
                )
            )
    metrics = {variant: metrics_from_rows(rows) for variant, rows in rows_by_variant.items()}
    output_root = resolve_project_path(protocol["paths"]["output_root"])
    output_dir = (
        output_root
        / args.run_id
        / "evaluation"
        / args.partition
        / args.domain
        / args.algorithm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for variant in variants for row in rows_by_variant[variant]]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "algorithm": args.algorithm,
        "reference_commit": config["_comparison"]["reference_commit"],
        "runtime_commit": runtime_commit,
        "seed": args.seed,
        "partition": args.partition,
        "domain": args.domain,
        "variants": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_yaml(output_dir / "resolved_config.yaml", config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
