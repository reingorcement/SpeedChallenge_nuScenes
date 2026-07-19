from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


HOST_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_PROJECT_ROOT))

from experiments.nuscenes_version_comparison.bootstrap import SOURCE_PROJECT_ROOT  # noqa: E402

PROJECT_ROOT = SOURCE_PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset  # noqa: E402
from experiments.nuscenes_version_comparison.adapters.dataset_adapter import (  # noqa: E402
    build_v2_dataset,
    dataset_identities,
    select_partition_scenes,
)
from experiments.nuscenes_version_comparison.adapters.legacy_adapter import (  # noqa: E402
    BaseVelocityAdapter,
    build_legacy_adapter,
)
from experiments.nuscenes_version_comparison.config import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    load_yaml,
    resolve_project_path,
    write_yaml,
)
from experiments.nuscenes_version_comparison.evaluate_comparison import (  # noqa: E402
    _rows,
    metrics_from_rows,
    move_batch,
)
from experiments.nuscenes_version_comparison.perturbations import (  # noqa: E402
    PerturbationContext,
    V2_CALIBRATION_VARIANTS,
    apply_perturbation,
)
from utils.seed import set_seed  # noqa: E402


DEFAULT_VARIANTS = (
    "clean",
    "can_missing",
    "yaw_bias_pos",
    "wheel_scale_low",
    "camera_missing",
)
ALGORITHM_METADATA = {
    "base_velocity": {
        "family": "reference",
        "label": "Base [vx, 0]",
        "reference_commit": "n/a",
    },
    "legacy_kd": {
        "family": "legacy",
        "label": "Legacy KD-only",
        "reference_commit": "7de2aad",
    },
    "legacy_vkd": {
        "family": "legacy",
        "label": "Legacy VKD gated",
        "reference_commit": "7de2aad",
    },
    "v2_visual": {
        "family": "v2",
        "label": "V2 visual",
        "reference_commit": "9e0dc23",
    },
    "v2_visual_kinematic": {
        "family": "v2",
        "label": "V2 visual-kinematic",
        "reference_commit": "9e0dc23",
    },
}


def _torch_load(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a saved config.")
    return checkpoint


def _metadata(algorithm: str) -> dict[str, Any]:
    base = dict(ALGORITHM_METADATA[algorithm])
    return {
        "name": algorithm,
        "family": base["family"],
        "label": base["label"],
        "reference_commit": base["reference_commit"],
    }


def _apply_common_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    protocol: dict[str, Any],
    *,
    family: str,
) -> None:
    data = config.setdefault("data", {})
    train = config.setdefault("train", {})
    paths = protocol["paths"]
    dataroot = args.dataroot or os.environ.get("NUSCENES_ROOT") or paths["dataroot"]
    data["dataroot"] = dataroot
    data["version"] = protocol["data"]["version"]
    data["can_bus_root"] = (
        args.can_bus_root
        or os.environ.get("NUSCENES_CAN_ROOT")
        or str(Path(dataroot) / "can_bus")
    )
    data["split_manifest_path"] = args.split_manifest or os.environ.get("V2_SPLIT_MANIFEST") or paths["split_manifest_path"]
    data["kinematic_stats_path"] = args.kinematic_stats or os.environ.get("V2_KINEMATIC_STATS") or paths["kinematic_stats_path"]
    data["allow_pose_fallback"] = False
    data["chassis_mode"] = "fair_chassis"
    data["online_mode"] = True
    data["base_velocity_source"] = "zero_lateral"
    train["batch_size"] = int(args.batch_size)
    train["num_workers"] = int(args.num_workers)
    if family == "legacy":
        data["frame_cache_root"] = (
            args.legacy_frame_cache_root or os.environ.get("LEGACY_FRAME_ROOT") or None
        )
    elif family == "v2":
        data["flow_root"] = args.flow_root or os.environ.get("V2_FLOW_ROOT") or paths["flow_root"]
        data["frame_cache_root"] = args.v2_frame_cache_root or os.environ.get("V2_FRAME_ROOT") or paths["frame_cache_root"]
        data["geometry_cache_root"] = args.geometry_cache_root or os.environ.get("V2_GEOMETRY_ROOT") or paths.get("geometry_cache_root")
        if args.disable_v2_geometry_cache:
            data["geometry_cache_mode"] = "off"
            data["geometry_cache_root"] = None
        config.setdefault("model", {})["kinematic_dropout"] = 0.0
        config.setdefault("model", {})["wheel_scale_jitter"] = 0.0


def _config_from_checkpoint(
    algorithm: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(_torch_load(checkpoint_path)["config"])
    config["_comparison"] = _metadata(algorithm)
    _apply_common_overrides(
        config,
        args,
        protocol,
        family=str(config["_comparison"]["family"]),
    )
    return config


def _official_split(partition: str) -> str:
    if partition in {"train_core", "dev"}:
        return "train"
    if partition == "stage1_test":
        return "val"
    raise PermissionError(f"Partition {partition!r} is unavailable; final_system_test remains sealed.")


def _build_legacy_dataset_native(
    config: dict[str, Any],
    partition: str,
    *,
    domain: str,
) -> NuScenesChassisFusionDataset:
    data = config["data"]
    scene_names = select_partition_scenes(
        data["split_manifest_path"],
        partition,
        version=str(data["version"]),
        domain=domain,
    )
    return NuScenesChassisFusionDataset(
        dataroot=data["dataroot"],
        version=data["version"],
        split=_official_split(partition),
        img_size=(int(data["img_height"]), int(data["img_width"])),
        seq_len=int(data["seq_len"]),
        cam_channel=data["cam_channel"],
        can_bus_root=data["can_bus_root"],
        allow_pose_fallback=False,
        scene_names=scene_names,
        image_cache_size=int(data.get("image_cache_size", 0)),
        chassis_mode="fair_chassis",
        chassis_window_sec=float(data.get("chassis_window_sec", 1.0)),
        chassis_hz=float(data.get("chassis_hz", 20.0)),
        online_mode=bool(data.get("online_mode", False)),
        image_mode=str(data.get("image_mode", "gray")),
        flow_mode=str(data.get("flow_mode", "none")),
        flow_root=data.get("flow_root"),
        geometry_mode=str(data.get("geometry_mode", "none")),
        base_velocity_source="zero_lateral",
        frame_cache_root=data.get("frame_cache_root"),
        max_scenes=data.get("max_scenes"),
    )


def _dataset_for_config(config: dict[str, Any], partition: str, domain: str):
    family = str(config["_comparison"]["family"])
    if family == "v2":
        return build_v2_dataset(config, partition, domain=domain)
    return _build_legacy_dataset_native(config, partition, domain=domain)


def _identity_key(identity: tuple[str, str, int]) -> tuple[str, str]:
    return identity[0], identity[1]


def _common_subset_indices(
    datasets: dict[str, Any],
    *,
    max_samples: int,
) -> tuple[dict[str, Subset], list[tuple[str, str]]]:
    identity_lists = {name: dataset_identities(dataset) for name, dataset in datasets.items()}
    key_sets = {
        name: {_identity_key(identity) for identity in identities}
        for name, identities in identity_lists.items()
    }
    common_keys = set.intersection(*key_sets.values())
    if not common_keys:
        raise ValueError("No common scene/token keys exist across selected algorithms.")
    reference_name = "legacy_kd" if "legacy_kd" in identity_lists else sorted(identity_lists)[0]
    ordered_keys = [
        _identity_key(identity)
        for identity in identity_lists[reference_name]
        if _identity_key(identity) in common_keys
    ]
    if max_samples > 0 and len(ordered_keys) > max_samples:
        by_scene: dict[str, list[tuple[str, str]]] = {}
        for key in ordered_keys:
            by_scene.setdefault(key[0], []).append(key)
        selected_keys: list[tuple[str, str]] = []
        while len(selected_keys) < max_samples:
            before = len(selected_keys)
            for scene_name in sorted(by_scene):
                if by_scene[scene_name]:
                    selected_keys.append(by_scene[scene_name].pop(0))
                    if len(selected_keys) >= max_samples:
                        break
            if len(selected_keys) == before:
                break
        ordered_keys = selected_keys
    selected = set(ordered_keys)
    subsets: dict[str, Subset] = {}
    for name, dataset in datasets.items():
        index_by_key = {
            _identity_key(identity): index
            for index, identity in enumerate(identity_lists[name])
        }
        subsets[name] = Subset(dataset, [index_by_key[key] for key in ordered_keys if key in selected])
    return subsets, ordered_keys


def _build_adapter(algorithm: str, config: dict[str, Any], checkpoint_path: Path | None):
    family = str(config["_comparison"]["family"])
    if algorithm == "base_velocity":
        return BaseVelocityAdapter()
    if family == "legacy":
        adapter = build_legacy_adapter(config, load_pretrained=False, project_root=PROJECT_ROOT)
    else:
        from experiments.nuscenes_version_comparison.adapters.v2_adapter import build_v2_adapter

        adapter = build_v2_adapter(config)
    if checkpoint_path is None:
        raise ValueError(f"{algorithm} requires a checkpoint.")
    adapter.load_checkpoint(checkpoint_path)
    return adapter


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _evaluate_algorithm(
    algorithm: str,
    config: dict[str, Any],
    dataset,
    checkpoint_path: Path | None,
    variants: tuple[str, ...],
    context: PerturbationContext,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    family = str(config["_comparison"]["family"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("Quick evaluation requires CUDA unless --allow-cpu is set.")
    workers = int(config["train"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(config["train"].get("prefetch_factor", 2)) if workers else None,
    )
    adapter = _build_adapter(algorithm, config, checkpoint_path).eval().to(device)
    perturbation_family = "v2" if family == "v2" else "legacy"
    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for raw_batch in tqdm(loader, desc=f"quick eval {algorithm}"):
        clean_batch = move_batch(raw_batch, device)
        for variant in variants:
            if family != "v2" and variant in V2_CALIBRATION_VARIANTS:
                continue
            perturbed = apply_perturbation(clean_batch, variant, perturbation_family, context)
            if not torch.equal(perturbed["velocity"], clean_batch["velocity"]):
                raise RuntimeError(f"Perturbation {variant!r} changed labels.")
            prediction, diagnostics = adapter(perturbed)
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError(f"Non-finite prediction for {algorithm} variant={variant}.")
            rows_by_variant.setdefault(variant, []).extend(
                _rows(
                    raw_batch,
                    clean_batch,
                    prediction,
                    diagnostics,
                    algorithm=algorithm,
                    family=family,
                    reference_commit=str(config["_comparison"]["reference_commit"]),
                    runtime_commit="quick_existing_checkpoint",
                    seed=int(args.seed),
                    variant=variant,
                    partition=args.partition,
                    domain=args.domain,
                )
            )
    all_rows = [row for variant in variants for row in rows_by_variant.get(variant, [])]
    _write_rows(output_dir / algorithm / "predictions.csv", all_rows)
    metrics = {
        variant: metrics_from_rows(rows)
        for variant, rows in rows_by_variant.items()
        if rows
    }
    (output_dir / algorithm / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_yaml(output_dir / algorithm / "resolved_config.yaml", config)
    return all_rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _clean_rows(rows: list[dict[str, Any]], algorithm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["algorithm"] == algorithm and row["variant"] == "clean"]


def _rows_for(rows: list[dict[str, Any]], algorithm: str, variant: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["algorithm"] == algorithm and row["variant"] == variant]


def _paired_by_key(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_map = {(row["scene_name"], row["center_token"]): row for row in left}
    right_map = {(row["scene_name"], row["center_token"]): row for row in right}
    keys = sorted(set(left_map) & set(right_map))
    return [(left_map[key], right_map[key]) for key in keys]


def _bootstrap_scene_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_scene: dict[str, list[float]] = {}
    for left, right in pairs:
        delta = float(left["endpoint_error_mps"]) - float(right["endpoint_error_mps"])
        per_scene.setdefault(str(left["scene_name"]), []).append(delta)
    scene_values = [statistics.fmean(values) for _, values in sorted(per_scene.items())]
    if not scene_values:
        return {"scene_count": 0, "mean_delta_mps": math.nan}
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choices(scene_values, k=len(scene_values))) for _ in range(samples)]
    draws.sort()
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]
    return {
        "scene_count": len(scene_values),
        "sample_count": len(pairs),
        "mean_delta_mps": statistics.fmean(scene_values),
        "ci95_low_mps": low,
        "ci95_high_mps": high,
        "conclusion": "clear_improvement" if high < 0.0 else "trend_only",
    }


def _thresholds(clean_rows: list[dict[str, Any]]) -> dict[str, float]:
    def q(field: str, absolute: bool = False, positive: bool = False) -> float:
        values = [float(row[field]) for row in clean_rows]
        if absolute:
            values = [abs(value) for value in values]
        if positive:
            values = [value for value in values if value > 0.0]
        if not values:
            return math.inf
        values.sort()
        index = min(int(0.9 * (len(values) - 1)), len(values) - 1)
        return values[index]

    return {
        "turning_yaw_rate_radps": 0.1,
        "high_lateral_velocity_mps": q("target_vy_mps", absolute=True),
        "wheel_inconsistent_relative_std": q("wheel_relative_std"),
        "hard_accel_mps2": q("longitudinal_accel_mps2", absolute=True),
        "high_lateral_accel_mps2": q("lateral_accel_mps2", absolute=True),
        "high_steering_feedback": q("steering_feedback", absolute=True),
        "high_throttle": q("throttle", positive=True),
        "braking": q("brake", positive=True),
    }


def _scenario(row: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    speed = math.hypot(float(row["target_vx_mps"]), float(row["target_vy_mps"]))
    names = ["all"]
    names.append("low_speed" if speed < 2.0 else "mid_speed" if speed < 10.0 else "high_speed")
    if abs(float(row["yaw_rate_radps"])) >= thresholds["turning_yaw_rate_radps"]:
        names.append("turning")
    if abs(float(row["target_vy_mps"])) >= thresholds["high_lateral_velocity_mps"]:
        names.append("high_lateral")
    if float(row["wheel_relative_std"]) >= thresholds["wheel_inconsistent_relative_std"]:
        names.append("wheel_inconsistent")
    if abs(float(row["longitudinal_accel_mps2"])) >= thresholds["hard_accel_mps2"]:
        names.append("hard_accel")
    if abs(float(row["lateral_accel_mps2"])) >= thresholds["high_lateral_accel_mps2"]:
        names.append("high_lateral_accel")
    if abs(float(row["steering_feedback"])) >= thresholds["high_steering_feedback"]:
        names.append("high_steering")
    if float(row["throttle"]) >= thresholds["high_throttle"]:
        names.append("high_throttle")
    if float(row["brake"]) >= thresholds["braking"]:
        names.append("braking")
    return names


def _scenario_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] != "clean":
            continue
        for name in _scenario(row, thresholds):
            grouped.setdefault((row["algorithm"], name), []).append(row)
    return [
        {"algorithm": algorithm, "scenario": scenario, **metrics_from_rows(group)}
        for (algorithm, scenario), group in sorted(grouped.items())
    ]


def _make_report(
    all_rows: list[dict[str, Any]],
    algorithms: list[str],
    variants: tuple[str, ...],
    args: argparse.Namespace,
    common_keys: list[tuple[str, str]],
    skipped: dict[str, str],
) -> dict[str, Any]:
    clean_metrics = {
        algorithm: metrics_from_rows(_clean_rows(all_rows, algorithm))
        for algorithm in algorithms
        if _clean_rows(all_rows, algorithm)
    }
    comparisons: dict[str, Any] = {}
    candidates = []
    if "v2_visual_kinematic" in algorithms:
        candidates.extend([("v2_visual_kinematic", "legacy_vkd"), ("v2_visual_kinematic", "legacy_kd")])
    if "v2_visual" in algorithms:
        candidates.extend([("v2_visual", "legacy_vkd"), ("v2_visual", "legacy_kd")])
    candidates.append(("legacy_vkd", "legacy_kd"))
    for left, right in candidates:
        if left not in algorithms or right not in algorithms:
            continue
        for variant in variants:
            pairs = _paired_by_key(_rows_for(all_rows, left, variant), _rows_for(all_rows, right, variant))
            if not pairs:
                continue
            key = f"{left}_minus_{right}:{variant}"
            result = _bootstrap_scene_delta(
                pairs,
                samples=int(args.bootstrap_samples),
                seed=int(args.seed),
            )
            left_clean = metrics_from_rows(_rows_for(all_rows, left, "clean"))
            right_clean = metrics_from_rows(_rows_for(all_rows, right, "clean"))
            left_variant = metrics_from_rows(_rows_for(all_rows, left, variant))
            right_variant = metrics_from_rows(_rows_for(all_rows, right, variant))
            result["left_endpoint_mps"] = left_variant["endpoint_error_mps"]
            result["right_endpoint_mps"] = right_variant["endpoint_error_mps"]
            result["left_degradation_vs_clean_mps"] = (
                left_variant["endpoint_error_mps"] - left_clean["endpoint_error_mps"]
            )
            result["right_degradation_vs_clean_mps"] = (
                right_variant["endpoint_error_mps"] - right_clean["endpoint_error_mps"]
            )
            comparisons[key] = result
    reference_clean = _clean_rows(all_rows, "legacy_kd") or [row for row in all_rows if row["variant"] == "clean"]
    thresholds = _thresholds(reference_clean)
    return {
        "mode": "quick_existing_checkpoint_evaluation",
        "run_id": args.run_id,
        "partition": args.partition,
        "domain": args.domain,
        "seed": int(args.seed),
        "selected_common_sample_count": len(common_keys),
        "algorithms": algorithms,
        "skipped_algorithms": skipped,
        "variants": list(variants),
        "clean_metrics": clean_metrics,
        "comparisons": comparisons,
        "scenario_metrics": _scenario_metrics(all_rows, thresholds),
        "interpretation_boundary": {
            "verified": (
                "Existing-checkpoint open-loop accuracy on aligned nuScenes samples, "
                "synthetic sensor perturbation trends, and high-excitation subgroup trends."
            ),
            "trend_only": (
                "This is a fast screening test, not a retrained fair protocol with repeated seeds. "
                "Checkpoint-native architectures are used, while evaluation inputs are forced causal."
            ),
            "not_verified": (
                "True physical parameter mismatch, cross-vehicle dynamics generalization, "
                "and closed-loop estimator-controller coupling."
            ),
            "current_v2": (
                "Use v2_visual_kinematic only when a real checkpoint is supplied; "
                "v2_visual has no kinematic or dynamics branch."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quick Existing-Checkpoint nuScenes Evaluation",
        "",
        "This report reuses existing checkpoints and does not retrain models.",
        "",
        "## Clean Accuracy",
        "",
        "| Algorithm | Samples | Scenes | Endpoint EE | MAE vx | MAE vy | P95 EE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for algorithm, metrics in report["clean_metrics"].items():
        lines.append(
            f"| {algorithm} | {int(metrics['sample_count'])} | {int(metrics['scene_count'])} | "
            f"{metrics['endpoint_error_mps']:.5f} | {metrics['mae_vx_mps']:.5f} | "
            f"{metrics['mae_vy_mps']:.5f} | {metrics['endpoint_p95_mps']:.5f} |"
        )
    lines.extend(["", "## Paired Deltas", ""])
    for name, result in report["comparisons"].items():
        if not name.endswith(":clean"):
            continue
        lines.append(
            f"- `{name}`: delta {result['mean_delta_mps']:.5f} m/s, "
            f"95% CI [{result['ci95_low_mps']:.5f}, {result['ci95_high_mps']:.5f}], "
            f"{result['conclusion']}."
        )
    lines.extend(["", "## Stress Summary", ""])
    for name, result in report["comparisons"].items():
        if name.endswith(":clean"):
            continue
        lines.append(
            f"- `{name}`: absolute delta {result['mean_delta_mps']:.5f} m/s; "
            f"degradation {result['left_degradation_vs_clean_mps']:.5f} vs "
            f"{result['right_degradation_vs_clean_mps']:.5f} m/s."
        )
    if report["skipped_algorithms"]:
        lines.extend(["", "## Skipped", ""])
        for algorithm, reason in report["skipped_algorithms"].items():
            lines.append(f"- `{algorithm}`: {reason}")
    boundary = report["interpretation_boundary"]
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- Verified: {boundary['verified']}",
            f"- Trend-only: {boundary['trend_only']}",
            f"- Not verified: {boundary['not_verified']}",
            f"- Current V2 note: {boundary['current_v2']}",
            "",
        ]
    )
    return "\n".join(lines)


def _checkpoint_arg(args: argparse.Namespace, algorithm: str) -> Path | None:
    value = getattr(args, f"{algorithm}_checkpoint")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickly evaluate existing KD/VKD/V2 checkpoints without retraining."
    )
    parser.add_argument("--run-id", default="quick_existing")
    parser.add_argument("--profile", choices=["legacy", "full"], default="legacy")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    parser.add_argument("--partition", choices=["dev", "stage1_test"], default="stage1_test")
    parser.add_argument("--domain", choices=["all", "boston", "singapore"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-v2-geometry-cache", action="store_true")
    parser.add_argument("--dataroot")
    parser.add_argument("--can-bus-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--kinematic-stats")
    parser.add_argument("--flow-root")
    parser.add_argument("--geometry-cache-root")
    parser.add_argument("--legacy-frame-cache-root")
    parser.add_argument("--v2-frame-cache-root")
    parser.add_argument("--legacy-kd-checkpoint", default="checkpoints/velocity_xy_kd_only/best.pth")
    parser.add_argument("--legacy-vkd-checkpoint", default="checkpoints/velocity_xy_vkd_gated/best.pth")
    parser.add_argument("--v2-visual-checkpoint", default="checkpoints/velocity_xy_v2_visual/best.pth")
    parser.add_argument("--v2-visual-kinematic-checkpoint", default="checkpoints/velocity_xy_v2_visual_kinematic/best.pth")
    args = parser.parse_args()

    set_seed(int(args.seed))
    protocol = load_yaml(args.protocol)
    variants = tuple(dict.fromkeys(args.variants + (list(V2_CALIBRATION_VARIANTS) if args.include_calibration else [])))
    output_root = resolve_project_path(protocol["paths"]["output_root"]) / args.run_id / "quick_existing_eval"
    output_root.mkdir(parents=True, exist_ok=True)

    configs: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, Path | None] = {}
    skipped: dict[str, str] = {}
    algorithms = ["legacy_kd", "legacy_vkd"]
    if args.profile == "full":
        algorithms.extend(["v2_visual", "v2_visual_kinematic"])
    for algorithm in algorithms:
        checkpoint = _checkpoint_arg(args, algorithm)
        if checkpoint is None or not checkpoint.is_file():
            skipped[algorithm] = f"checkpoint not found: {checkpoint}"
            continue
        try:
            configs[algorithm] = _config_from_checkpoint(algorithm, checkpoint, args, protocol)
            checkpoints[algorithm] = checkpoint
        except Exception as exc:
            skipped[algorithm] = f"config load failed: {exc}"
    if "legacy_kd" not in configs:
        raise RuntimeError("legacy_kd checkpoint is required for the quick baseline.")
    base_config = deepcopy(configs["legacy_kd"])
    base_config["_comparison"] = _metadata("base_velocity")
    configs = {"base_velocity": base_config, **configs}
    checkpoints["base_velocity"] = None

    datasets = {
        algorithm: _dataset_for_config(config, args.partition, args.domain)
        for algorithm, config in configs.items()
    }
    subsets, common_keys = _common_subset_indices(datasets, max_samples=int(args.max_samples))
    active_algorithms = list(configs)
    stats_path = configs.get("v2_visual", base_config)["data"].get("kinematic_stats_path")
    context = PerturbationContext.from_config(protocol, stats_path)

    all_rows: list[dict[str, Any]] = []
    for algorithm in active_algorithms:
        rows = _evaluate_algorithm(
            algorithm,
            configs[algorithm],
            subsets[algorithm],
            checkpoints[algorithm],
            variants,
            context,
            args,
            output_root,
        )
        all_rows.extend(rows)

    _write_rows(output_root / "all_predictions.csv", all_rows)
    report = _make_report(all_rows, active_algorithms, variants, args, common_keys, skipped)
    (output_root / "quick_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "quick_report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Quick evaluation complete: {output_root / 'quick_report.md'}")


if __name__ == "__main__":
    main()
