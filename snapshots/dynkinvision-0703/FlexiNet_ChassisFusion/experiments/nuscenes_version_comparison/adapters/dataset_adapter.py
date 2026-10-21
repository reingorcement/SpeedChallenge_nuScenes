from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
from typing import Any, Iterable

from dataset.nuscenes_chassis import NuScenesChassisFusionDataset
from experiments.nuscenes_version_comparison.splits import (
    load_split_manifest,
    manifest_checksum,
    partition_scene_names,
    validate_split_manifest,
)


ALLOWED_PARTITIONS = frozenset({"train_core", "dev", "stage1_test"})


def select_partition_scenes(
    manifest_path: str | Path,
    partition: str,
    *,
    version: str,
    domain: str = "all",
) -> list[str]:
    if partition not in ALLOWED_PARTITIONS:
        raise PermissionError(
            f"Partition {partition!r} is unavailable; final_system_test remains sealed."
        )
    names = partition_scene_names(
        manifest_path,
        partition,
        expected_version=version,
    )
    if domain == "all":
        return names
    if domain not in {"boston", "singapore"}:
        raise ValueError("domain must be 'all', 'boston', or 'singapore'.")
    manifest = load_split_manifest(manifest_path)
    metadata = manifest.get("scene_metadata", {})
    selected = [
        name
        for name in names
        if str(metadata.get(name, {}).get("location", "")).lower().startswith(domain)
    ]
    if not selected:
        raise ValueError(f"No {domain!r} scenes exist in partition {partition!r}.")
    return selected


def write_training_domain_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    version: str,
    domain: str,
) -> Path:
    if domain == "all":
        return Path(source_path)
    manifest = deepcopy(load_split_manifest(source_path))
    for partition in ("train_core", "dev"):
        manifest["partitions"][partition] = select_partition_scenes(
            source_path, partition, version=version, domain=domain
        )
    manifest["notes"] = (
        f"Comparison training manifest filtered to {domain}; final_system_test remains sealed."
    )
    manifest["checksum_sha256"] = manifest_checksum(manifest)
    validate_split_manifest(manifest)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _official_split(partition: str) -> str:
    if partition in {"train_core", "dev"}:
        return "train"
    if partition == "stage1_test":
        return "val"
    raise PermissionError(f"Partition {partition!r} is not available to this experiment.")


def _build_base_dataset(
    config: dict[str, Any],
    partition: str,
    scene_names: list[str],
    *,
    v2: bool,
) -> NuScenesChassisFusionDataset:
    data = config["data"]
    height_key = "visual_height" if v2 else "img_height"
    width_key = "visual_width" if v2 else "img_width"
    dataset_kwargs = dict(
        dataroot=data["dataroot"],
        version=data["version"],
        split=_official_split(partition),
        img_size=(int(data[height_key]), int(data[width_key])),
        seq_len=int(data["seq_len"]),
        cam_channel=data["cam_channel"],
        can_bus_root=data["can_bus_root"],
        allow_pose_fallback=False,
        scene_names=scene_names,
        image_cache_size=int(data.get("image_cache_size", 0)),
        chassis_mode="fair_chassis",
        chassis_window_sec=float(data.get("chassis_window_sec", 1.0)),
        chassis_hz=float(data.get("chassis_hz", 20.0)),
        online_mode=True,
        image_mode="rgb",
        flow_mode="none",
        geometry_mode="none",
        frame_cache_root=data.get("frame_cache_root"),
        max_scenes=data.get("max_scenes"),
    )
    if v2:
        dataset_kwargs["retain_nusc_metadata"] = True
    return NuScenesChassisFusionDataset(**dataset_kwargs)


def build_legacy_dataset(
    config: dict[str, Any], partition: str, *, domain: str = "all"
) -> NuScenesChassisFusionDataset:
    data = config["data"]
    manifest_path = data["split_manifest_path"]
    scene_names = select_partition_scenes(
        manifest_path, partition, version=str(data["version"]), domain=domain
    )
    return _build_base_dataset(config, partition, scene_names, v2=False)


def build_v2_dataset(
    config: dict[str, Any], partition: str, *, domain: str = "all"
):
    from dataset.v2_schema import V2NuScenesDatasetAdapter
    from train_v2 import build_geometry_cache_contract

    data = config["data"]
    scene_names = select_partition_scenes(
        data["split_manifest_path"],
        partition,
        version=str(data["version"]),
        domain=domain,
    )
    base = _build_base_dataset(config, partition, scene_names, v2=True)
    augmentation = data.get("photometric_augmentation") if partition == "train_core" else None
    return V2NuScenesDatasetAdapter(
        base,
        flow_root=data["flow_root"],
        camera_height_m=float(data["camera_height_m"]),
        road_mask_top_fraction=float(data.get("road_mask_top_fraction", 0.45)),
        geometry_size=(int(data["geometry_height"]), int(data["geometry_width"])),
        kinematic_stats_path=data.get("kinematic_stats_path"),
        field_max_age_sec=data.get("field_max_age_sec"),
        photometric_augmentation=augmentation,
        geometry_cache_root=data.get("geometry_cache_root"),
        geometry_cache_mode=str(data.get("geometry_cache_mode", "off")),
        geometry_cache_contract=build_geometry_cache_contract(config),
        flow_backend=data.get("flow_backend"),
    )


def build_dataset_for_algorithm(
    config: dict[str, Any], partition: str, *, domain: str = "all"
):
    family = str(config["_comparison"]["family"])
    if family == "v2":
        return build_v2_dataset(config, partition, domain=domain)
    return build_legacy_dataset(config, partition, domain=domain)


def _samples(dataset) -> Iterable[dict[str, Any]]:
    base = getattr(dataset, "base_dataset", dataset)
    return getattr(base, "samples")


def dataset_identities(dataset) -> list[tuple[str, str, int]]:
    return [
        (
            str(sample["scene_name"]),
            str(sample["center_token"]),
            int(sample["center_timestamp"]),
        )
        for sample in _samples(dataset)
    ]


def assert_dataset_alignment(left, right) -> None:
    left_ids = dataset_identities(left)
    right_ids = dataset_identities(right)
    if left_ids == right_ids:
        return
    left_set = set(left_ids)
    right_set = set(right_ids)
    raise ValueError(
        "Dataset identities do not align: "
        f"left_only={sorted(left_set - right_set)[:3]}, "
        f"right_only={sorted(right_set - left_set)[:3]}."
    )
