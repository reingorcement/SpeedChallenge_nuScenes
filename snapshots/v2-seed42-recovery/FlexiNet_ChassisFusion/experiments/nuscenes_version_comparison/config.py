from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import SOURCE_PROJECT_ROOT


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_PROJECT_ROOT
ARTIFACT_PROJECT_ROOT = SOURCE_PROJECT_ROOT
DEFAULT_PROTOCOL_PATH = EXPERIMENT_ROOT / "configs" / "common_protocol.yaml"

ALGORITHM_CONFIGS = {
    "base_velocity": "base_velocity.yaml",
    "legacy_kd": "legacy_kd.yaml",
    "legacy_vkd": "legacy_vkd.yaml",
    "v2_geometry": "v2_geometry.yaml",
    "v2_visual": "v2_visual.yaml",
    "v2_kinematic_pretrain": "v2_kinematic_pretrain.yaml",
    "v2_visual_kinematic": "v2_visual_kinematic.yaml",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ARTIFACT_PROJECT_ROOT / path


def algorithm_config_path(name: str) -> Path:
    try:
        filename = ALGORITHM_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"Unknown comparison algorithm {name!r}.") from error
    return EXPERIMENT_ROOT / "configs" / filename


def load_resolved_config(
    algorithm: str,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    seed: int | None = None,
    path_overrides: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_yaml(protocol_path)
    specification = load_yaml(algorithm_config_path(algorithm))
    metadata = specification.get("algorithm", {})
    if metadata.get("name") != algorithm:
        raise ValueError(f"Algorithm config identity mismatch for {algorithm!r}.")
    base_path = resolve_project_path(metadata["base_config"])
    config = deep_merge(load_yaml(base_path), specification.get("overrides", {}))

    family = str(metadata["family"])
    paths = deepcopy(protocol["paths"])
    for key, value in (path_overrides or {}).items():
        if value:
            paths[key] = value
    common_data = protocol["data"]
    config.setdefault("data", {})
    config["data"].update(
        {
            "dataroot": paths["dataroot"],
            "version": common_data["version"],
            "seq_len": int(common_data["seq_len"]),
            "cam_channel": common_data["cam_channel"],
            "can_bus_root": paths["can_bus_root"],
            "frame_cache_root": paths["frame_cache_root"],
            "split_manifest_path": paths["split_manifest_path"],
            "kinematic_stats_path": paths["kinematic_stats_path"],
            "chassis_window_sec": float(common_data["chassis_window_sec"]),
            "chassis_hz": float(common_data["chassis_hz"]),
            "online_mode": True,
            "max_scenes": common_data.get("max_scenes"),
        }
    )
    if family in {"legacy", "reference"}:
        config["data"].update(
            {
                "img_height": int(common_data["visual_height"]),
                "img_width": int(common_data["visual_width"]),
                "image_mode": "rgb",
                "chassis_mode": "fair_chassis",
                "allow_pose_fallback": False,
                "flow_mode": "none",
                "geometry_mode": "none",
                "base_velocity_source": "zero_lateral",
            }
        )
    if family == "v2":
        config["data"].update(
            {
                "visual_height": int(common_data["visual_height"]),
                "visual_width": int(common_data["visual_width"]),
                "geometry_height": int(common_data["geometry_height"]),
                "geometry_width": int(common_data["geometry_width"]),
                "flow_root": paths["flow_root"],
                "geometry_cache_root": paths.get("geometry_cache_root"),
                "split_manifest_path": paths["split_manifest_path"],
                "kinematic_stats_path": paths["kinematic_stats_path"],
            }
        )
    chosen_seed = int(seed if seed is not None else protocol["protocol"]["seeds"][0])
    config.setdefault("train", {})["seed"] = chosen_seed
    config["_comparison"] = {
        **metadata,
        "runtime_seed": chosen_seed,
        "protocol_path": str(Path(protocol_path)),
        "reference_commit": metadata["source_commit"],
    }
    validate_resolved_config(config, protocol)
    return config, protocol, specification


def validate_resolved_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    data = config["data"]
    if not bool(data.get("online_mode")):
        raise ValueError("Comparison inputs must be causal (online_mode=true).")
    if str(data.get("chassis_mode", "fair_chassis")) != "fair_chassis":
        raise ValueError("pose_assisted and other chassis modes are forbidden.")
    if int(data["seq_len"]) != int(protocol["data"]["seq_len"]):
        raise ValueError("All learned algorithms must use the protocol sequence length.")


def write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
