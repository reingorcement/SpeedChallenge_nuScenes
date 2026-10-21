from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.physics_schema import (
    FAIR_PHYSICS_FEATURE_COUNT,
    PHYSICS_EXTRA_FEATURES,
    PHYSICS_INPUT_DIM,
    PHYSICS_INPUT_INDEX,
)


FAIR_CHASSIS_FEATURES = [
    "vehicle_speed",
    "odom_speed",
    "yaw_rate",
    "throttle",
    "brake",
    "steering",
    "steering_feedback",
    "longitudinal_accel",
    "transversal_accel",
    "fl_wheel_speed",
    "fr_wheel_speed",
    "rl_wheel_speed",
    "rr_wheel_speed",
    "wheel_speed_mean",
    "wheel_speed_std",
    "base_vx",
]
POSE_ASSISTED_EXTRA_FEATURES = [
    "pose_speed",
    "pose_vel_x",
    "pose_vel_y",
    "pose_accel_x",
    "pose_accel_y",
    "pose_rotation_rate_z",
]
FAIR_CHASSIS_DIM = len(FAIR_CHASSIS_FEATURES)
POSE_ASSISTED_CHASSIS_DIM = FAIR_CHASSIS_DIM + len(POSE_ASSISTED_EXTRA_FEATURES)
FAIR_CHASSIS_INDEX = {name: idx for idx, name in enumerate(FAIR_CHASSIS_FEATURES)}
KINEMATIC_FEATURE_NAMES = [
    "vehicle_speed",
    "odom_speed",
    "yaw_rate",
    "steering",
    "steering_feedback",
    "longitudinal_accel",
    "transversal_accel",
    "fl_wheel_speed",
    "fr_wheel_speed",
    "rl_wheel_speed",
    "rr_wheel_speed",
    "wheel_speed_mean",
    "wheel_speed_std",
    "base_vx",
]
DYNAMICS_PROXY_FEATURE_NAMES = [
    "base_vx",
    "base_vy",
    "yaw_rate",
    "steering",
    "steering_feedback",
    "longitudinal_accel",
    "transversal_accel",
    "wheel_speed_std",
]


def _has_finite_field(record: Optional[dict], key: str) -> bool:
    if record is None or key not in record:
        return False
    try:
        return math.isfinite(float(record[key]))
    except (TypeError, ValueError):
        return False


def _finite_field(record: Optional[dict], key: str, scale: float = 1.0) -> float:
    if not _has_finite_field(record, key):
        return 0.0
    assert record is not None
    return float(record[key]) / scale


def _current_sequence_index(length: int, online_mode: bool) -> int:
    if length <= 0:
        raise ValueError("Sequence length must be positive.")
    return length - 1 if online_mode else length // 2


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _yaw_from_quaternion(rotation: Sequence[float]) -> float:
    if len(rotation) != 4:
        raise ValueError("nuScenes ego_pose rotation must be a quaternion [w, x, y, z].")
    w, x, y, z = [float(v) for v in rotation]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_global_velocity_to_ego(vx_global: float, vy_global: float, rotation: Sequence[float]) -> tuple[float, float]:
    yaw = _yaw_from_quaternion(rotation)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_ego = cos_yaw * vx_global + sin_yaw * vy_global
    vy_ego = -sin_yaw * vx_global + cos_yaw * vy_global
    return vx_ego, vy_ego


class PrecomputedVelocityLookup:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, tuple[list[int], list[tuple[float, float]]]] = {}

    def lookup(self, scene_name: str, timestamp_us: int) -> tuple[float, float]:
        timestamps, velocities = self._load_scene(scene_name)
        index = bisect_right(timestamps, timestamp_us) - 1
        if index < 0:
            raise KeyError(f"No precomputed EOVM velocity at or before {timestamp_us} for {scene_name}.")
        return velocities[index]

    def _load_scene(self, scene_name: str) -> tuple[list[int], list[tuple[float, float]]]:
        if scene_name in self._cache:
            return self._cache[scene_name]
        path = self.root / f"{scene_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Precomputed EOVM velocity CSV not found: {path}")
        timestamps: list[int] = []
        velocities: list[tuple[float, float]] = []
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestamps.append(int(row["timestamp_us"]))
                velocities.append((float(row["vx"]), float(row["vy"])))
        self._cache[scene_name] = (timestamps, velocities)
        return self._cache[scene_name]


class PrecomputedPhysicsExtraLookup:
    """Causal lookup for optional full-vehicle signals stored per scene."""

    def __init__(self, root: str | Path, max_age_us: int = 500_000) -> None:
        self.root = Path(root)
        if max_age_us <= 0:
            raise ValueError("max_age_us must be positive.")
        self.max_age_us = int(max_age_us)
        self._cache: dict[str, tuple[list[int], list[dict[str, float]]]] = {}

    def lookup(self, scene_name: str, timestamp_us: int) -> tuple[list[float], list[bool]]:
        timestamps, rows = self._load_scene(scene_name)
        index = bisect_right(timestamps, timestamp_us) - 1
        if index < 0 or timestamp_us - timestamps[index] > self.max_age_us:
            return [0.0] * len(PHYSICS_EXTRA_FEATURES), [False] * len(PHYSICS_EXTRA_FEATURES)
        row = rows[index]
        values = [float(row.get(name, 0.0)) for name in PHYSICS_EXTRA_FEATURES]
        valid = [name in row for name in PHYSICS_EXTRA_FEATURES]
        return values, valid

    def _load_scene(self, scene_name: str) -> tuple[list[int], list[dict[str, float]]]:
        if scene_name in self._cache:
            return self._cache[scene_name]
        path = self.root / f"{scene_name}.csv"
        if not path.is_file():
            self._cache[scene_name] = ([], [])
            return self._cache[scene_name]
        timestamps: list[int] = []
        rows: list[dict[str, float]] = []
        with path.open("r", newline="", encoding="utf-8") as file:
            for raw_row in csv.DictReader(file):
                if not raw_row.get("timestamp_us"):
                    continue
                timestamps.append(int(raw_row["timestamp_us"]))
                rows.append(
                    {
                        name: float(raw_row[name])
                        for name in PHYSICS_EXTRA_FEATURES
                        if _has_finite_field(raw_row, name)
                    }
                )
        order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        self._cache[scene_name] = (
            [timestamps[index] for index in order],
            [rows[index] for index in order],
        )
        return self._cache[scene_name]


class NuScenesCanBusAdapter:
    def __init__(self, can_bus_root: str | Path, max_age_us: int = 250_000) -> None:
        self.can_bus_root = Path(can_bus_root)
        if not self.can_bus_root.exists():
            raise FileNotFoundError(f"CAN bus root not found: {self.can_bus_root}")
        if max_age_us <= 0:
            raise ValueError("max_age_us must be positive.")
        self.max_age_us = int(max_age_us)
        self._message_cache: dict[tuple[str, str], List[dict]] = {}
        self._timestamp_cache: dict[tuple[str, str], List[int]] = {}

    def available_scene_names(self) -> List[str]:
        return sorted({p.name[:10] for p in self.can_bus_root.glob("scene-*_*.json")})

    def load_messages(self, scene_name: str, message_name: str) -> List[dict]:
        cache_key = (scene_name, message_name)
        if cache_key in self._message_cache:
            return self._message_cache[cache_key]

        path = self.can_bus_root / f"{scene_name}_{message_name}.json"
        if not path.is_file():
            self._message_cache[cache_key] = []
            return []
        with path.open("r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            self._message_cache[cache_key] = []
            self._timestamp_cache[cache_key] = []
            return []
        messages = sorted(messages, key=lambda r: int(r.get("utime", 0)))
        self._message_cache[cache_key] = messages
        self._timestamp_cache[cache_key] = [int(r.get("utime", 0)) for r in messages]
        return messages

    def _select_latest_before(self, scene_name: str, message_name: str, timestamp_us: int) -> Optional[dict]:
        record, _ = self.select_latest_before_with_age(scene_name, message_name, timestamp_us)
        return record

    def select_latest_before_with_age(
        self,
        scene_name: str,
        message_name: str,
        timestamp_us: int,
    ) -> tuple[Optional[dict], float]:
        """Return the latest causal record and its true age in seconds."""

        records = self.load_messages(scene_name, message_name)
        timestamps = self._timestamp_cache.get((scene_name, message_name), [])
        idx = bisect_right(timestamps, timestamp_us) - 1
        if idx < 0:
            return None, self.max_age_us / 1_000_000.0
        age_us = timestamp_us - timestamps[idx]
        if age_us < 0 or age_us > self.max_age_us:
            return None, self.max_age_us / 1_000_000.0
        return records[idx], age_us / 1_000_000.0

    def _feature_values(
        self,
        scene_name: str,
        timestamp_us: int,
        chassis_mode: str = "fair_chassis",
    ) -> Tuple[List[float], float]:
        pose = self._select_latest_before(scene_name, "pose", timestamp_us)
        monitor = self._select_latest_before(scene_name, "vehicle_monitor", timestamp_us)
        wheel = self._select_latest_before(scene_name, "zoe_veh_info", timestamp_us)
        steer = self._select_latest_before(scene_name, "steeranglefeedback", timestamp_us)

        vehicle_speed = 0.0
        odom_speed = 0.0
        pose_speed = 0.0
        yaw_rate = 0.0
        throttle = 0.0
        brake = 0.0
        steering = 0.0
        steering_feedback = 0.0
        longitudinal_accel = 0.0
        transversal_accel = 0.0
        wheel_speeds = [0.0, 0.0, 0.0, 0.0]
        wheel_speed_mean = 0.0
        wheel_speed_std = 0.0
        accel_x = 0.0
        accel_y = 0.0
        rotation_rate_z = 0.0
        pose_vel_x = 0.0
        pose_vel_y = 0.0
        base_vx = 0.0

        if monitor is not None:
            vehicle_speed = _finite_field(monitor, "vehicle_speed", 3.6)
            yaw_rate = math.radians(_finite_field(monitor, "yaw_rate"))
            throttle = _finite_field(monitor, "throttle", 1000.0)
            brake = _finite_field(monitor, "brake", 126.0)
            steering = _finite_field(monitor, "steering", 780.0)
            base_vx = vehicle_speed

        if wheel is not None:
            wheel_fields = ["FL_wheel_speed", "FR_wheel_speed", "RL_wheel_speed", "RR_wheel_speed"]
            raw_wheels = [
                _finite_field(wheel, field)
                for field in wheel_fields
            ]
            wheel_speeds = [value / 1000.0 for value in raw_wheels]
            valid_wheel_speeds = [
                value
                for value, field in zip(wheel_speeds, wheel_fields)
                if _has_finite_field(wheel, field)
            ]
            if valid_wheel_speeds:
                wheel_speed_mean = sum(valid_wheel_speeds) / len(valid_wheel_speeds)
                variance = sum((value - wheel_speed_mean) ** 2 for value in valid_wheel_speeds) / len(
                    valid_wheel_speeds
                )
                wheel_speed_std = math.sqrt(variance)
            odom_speed = _finite_field(wheel, "odom_speed", 3.6)
            longitudinal_accel = _finite_field(wheel, "longitudinal_accel")
            transversal_accel = _finite_field(wheel, "transversal_accel")
            if base_vx == 0.0:
                base_vx = odom_speed
            if base_vx == 0.0:
                base_vx = wheel_speed_mean

        if steer is not None:
            steering_feedback = _finite_field(steer, "value", 7.7)

        if pose is not None:
            vel = pose.get("vel", [0.0, 0.0, 0.0])
            accel = pose.get("accel", [0.0, 0.0, 0.0])
            rotation_rate = pose.get("rotation_rate", [0.0, 0.0, 0.0])
            if len(vel) >= 2:
                pose_vel_x = float(vel[0])
                pose_vel_y = float(vel[1])
                pose_speed = math.sqrt(pose_vel_x**2 + pose_vel_y**2)
            if len(accel) >= 2:
                accel_x = float(accel[0])
                accel_y = float(accel[1])
            if len(rotation_rate) >= 3:
                rotation_rate_z = float(rotation_rate[2])
            if chassis_mode == "pose_assisted" and base_vx == 0.0:
                base_vx = pose_speed

        fair_values = [
            vehicle_speed,
            odom_speed,
            yaw_rate,
            throttle,
            brake,
            steering,
            steering_feedback,
            longitudinal_accel,
            transversal_accel,
            *wheel_speeds,
            wheel_speed_mean,
            wheel_speed_std,
            base_vx,
        ]
        if len(fair_values) != FAIR_CHASSIS_DIM:
            raise RuntimeError(f"fair_chassis feature length mismatch: {len(fair_values)}")

        if chassis_mode == "fair_chassis":
            return fair_values, base_vx
        if chassis_mode == "pose_assisted":
            pose_values = [
                pose_speed,
                pose_vel_x,
                pose_vel_y,
                accel_x,
                accel_y,
                rotation_rate_z,
            ]
            return fair_values + pose_values, base_vx
        raise ValueError("chassis_mode must be 'fair_chassis' or 'pose_assisted'.")

    def extract_feature_vector(
        self,
        scene_name: str,
        timestamp_us: int,
        chassis_mode: str = "fair_chassis",
    ) -> Tuple[torch.Tensor, float]:
        values, base_vx = self._feature_values(scene_name, timestamp_us, chassis_mode=chassis_mode)
        return torch.tensor(values, dtype=torch.float32), base_vx

    def extract_feature_sequence(
        self,
        scene_name: str,
        center_timestamp_us: int,
        chassis_mode: str = "fair_chassis",
        window_sec: float = 0.0,
        hz: float = 1.0,
        online_mode: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        timestamps = self.sample_timestamps(
            center_timestamp_us=center_timestamp_us,
            window_sec=window_sec,
            hz=hz,
            online_mode=online_mode,
        )

        rows = [
            self._feature_values(scene_name, timestamp_us, chassis_mode=chassis_mode)[0]
            for timestamp_us in timestamps
        ]
        _, base_vx = self._feature_values(scene_name, center_timestamp_us, chassis_mode=chassis_mode)
        return torch.tensor(rows, dtype=torch.float32), base_vx

    @staticmethod
    def sample_timestamps(
        center_timestamp_us: int,
        window_sec: float,
        hz: float,
        online_mode: bool,
    ) -> list[int]:
        if window_sec <= 0.0:
            return [center_timestamp_us]
        sample_count = max(int(round(window_sec * hz)) + 1, 1)
        window_us = int(round(window_sec * 1_000_000))
        if online_mode:
            start_us = center_timestamp_us - window_us
            end_us = center_timestamp_us
        else:
            half_window = window_us // 2
            start_us = center_timestamp_us - half_window
            end_us = center_timestamp_us + (window_us - half_window)
        if sample_count == 1:
            return [center_timestamp_us]
        return [
            int(round(start_us + (end_us - start_us) * index / (sample_count - 1)))
            for index in range(sample_count)
        ]

    def extract_physics_sequence(
        self,
        scene_name: str,
        center_timestamp_us: int,
        window_sec: float,
        hz: float,
        online_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        timestamps = self.sample_timestamps(center_timestamp_us, window_sec, hz, online_mode)
        inputs = torch.zeros(len(timestamps), PHYSICS_INPUT_DIM, dtype=torch.float32)
        valid = torch.zeros(len(timestamps), PHYSICS_INPUT_DIM, dtype=torch.bool)
        for row_index, timestamp_us in enumerate(timestamps):
            values, _ = self._feature_values(scene_name, timestamp_us, chassis_mode="fair_chassis")
            monitor = self._select_latest_before(scene_name, "vehicle_monitor", timestamp_us)
            wheel = self._select_latest_before(scene_name, "zoe_veh_info", timestamp_us)
            steer = self._select_latest_before(scene_name, "steeranglefeedback", timestamp_us)
            inputs[row_index, :FAIR_PHYSICS_FEATURE_COUNT] = torch.tensor(
                [
                    values[FAIR_CHASSIS_INDEX["base_vx"]],
                    values[FAIR_CHASSIS_INDEX["yaw_rate"]],
                    values[FAIR_CHASSIS_INDEX["steering"]],
                    values[FAIR_CHASSIS_INDEX["steering_feedback"]],
                    values[FAIR_CHASSIS_INDEX["longitudinal_accel"]],
                    values[FAIR_CHASSIS_INDEX["transversal_accel"]],
                    values[FAIR_CHASSIS_INDEX["fl_wheel_speed"]],
                    values[FAIR_CHASSIS_INDEX["fr_wheel_speed"]],
                    values[FAIR_CHASSIS_INDEX["rl_wheel_speed"]],
                    values[FAIR_CHASSIS_INDEX["rr_wheel_speed"]],
                ],
                dtype=torch.float32,
            )
            wheel_fields = ["FL_wheel_speed", "FR_wheel_speed", "RL_wheel_speed", "RR_wheel_speed"]
            base_valid = _has_finite_field(monitor, "vehicle_speed") or _has_finite_field(wheel, "odom_speed")
            base_valid = base_valid or any(_has_finite_field(wheel, field) for field in wheel_fields)
            valid[row_index, PHYSICS_INPUT_INDEX["base_vx"]] = base_valid
            valid[row_index, PHYSICS_INPUT_INDEX["yaw_rate"]] = _has_finite_field(monitor, "yaw_rate")
            valid[row_index, PHYSICS_INPUT_INDEX["steering_command"]] = _has_finite_field(monitor, "steering")
            valid[row_index, PHYSICS_INPUT_INDEX["steering_feedback"]] = _has_finite_field(steer, "value")
            valid[row_index, PHYSICS_INPUT_INDEX["longitudinal_accel"]] = _has_finite_field(
                wheel, "longitudinal_accel"
            )
            valid[row_index, PHYSICS_INPUT_INDEX["lateral_accel"]] = _has_finite_field(
                wheel, "transversal_accel"
            )
            for offset, field in enumerate(wheel_fields):
                valid[row_index, PHYSICS_INPUT_INDEX["wheel_speed_fl"] + offset] = _has_finite_field(
                    wheel, field
                )
            inputs[row_index, ~valid[row_index]] = 0.0
        dt = torch.empty(len(timestamps), dtype=torch.float32)
        if len(timestamps) == 1:
            dt[0] = 0.0
        else:
            dt[0] = 0.0
            dt[1:] = torch.tensor(
                [(timestamps[index] - timestamps[index - 1]) / 1_000_000.0 for index in range(1, len(timestamps))]
            )
        return inputs, valid, dt, timestamps


class NuScenesChassisFusionDataset(Dataset):
    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        split: str = "train",
        img_size: Tuple[int, int] = (64, 64),
        seq_len: int = 13,
        cam_channel: str = "CAM_FRONT",
        can_bus_root: Optional[str | Path] = None,
        allow_pose_fallback: bool = False,
        scene_indices: Optional[Sequence[int]] = None,
        scene_names: Optional[Sequence[str]] = None,
        image_cache_size: int = 0,
        chassis_mode: str = "fair_chassis",
        chassis_window_sec: float = 0.0,
        chassis_hz: float = 1.0,
        online_mode: bool = False,
        image_mode: str = "gray",
        flow_mode: str = "none",
        flow_root: Optional[str | Path] = None,
        geometry_mode: str = "basic",
        base_velocity_source: str = "zero_lateral",
        eovm_base_root: Optional[str | Path] = None,
        physics_extra_root: Optional[str | Path] = None,
        physics_units_audited: bool = False,
        can_max_age_sec: float = 0.25,
        physics_extra_max_age_sec: float = 0.5,
        frame_cache_root: Optional[str | Path] = None,
        max_scenes: Optional[int] = None,
        retain_nusc_metadata: bool = False,
    ) -> None:
        from nuscenes.nuscenes import NuScenes

        self.version = version
        self.split = split
        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        self.seq_len = seq_len
        self.cam_channel = cam_channel
        self.allow_pose_fallback = allow_pose_fallback
        if chassis_mode not in {"fair_chassis", "pose_assisted"}:
            raise ValueError("chassis_mode must be 'fair_chassis' or 'pose_assisted'.")
        if image_mode not in {"gray", "rgb"}:
            raise ValueError("image_mode must be 'gray' or 'rgb'.")
        if flow_mode not in {"none", "precomputed"}:
            raise ValueError("flow_mode must be 'none' or 'precomputed'.")
        if geometry_mode not in {"none", "basic"}:
            raise ValueError("geometry_mode must be 'none' or 'basic'.")
        if base_velocity_source not in {"zero_lateral", "eovm_ekf"}:
            raise ValueError("base_velocity_source must be 'zero_lateral' or 'eovm_ekf'.")
        self.chassis_mode = chassis_mode
        self.chassis_window_sec = float(chassis_window_sec)
        self.chassis_hz = float(chassis_hz)
        self.online_mode = bool(online_mode)
        self.image_mode = image_mode
        self.flow_mode = flow_mode
        self.flow_root = Path(flow_root) if flow_root else None
        self.geometry_mode = geometry_mode
        self.base_velocity_source = base_velocity_source
        self.physics_units_audited = bool(physics_units_audited)
        self.physics_extra_lookup = (
            PrecomputedPhysicsExtraLookup(
                physics_extra_root,
                max_age_us=int(float(physics_extra_max_age_sec) * 1_000_000),
            )
            if physics_extra_root
            else None
        )
        self.eovm_lookup = PrecomputedVelocityLookup(eovm_base_root) if eovm_base_root else None
        if base_velocity_source == "eovm_ekf" and self.eovm_lookup is None:
            raise ValueError("eovm_base_root is required when base_velocity_source='eovm_ekf'.")
        self.can_bus_adapter = (
            NuScenesCanBusAdapter(
                can_bus_root,
                max_age_us=int(float(can_max_age_sec) * 1_000_000),
            )
            if can_bus_root
            else None
        )
        self.image_cache_size = max(int(image_cache_size), 0)
        self._image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.frame_cache_root = Path(frame_cache_root) if frame_cache_root else None
        self.retain_nusc_metadata = bool(retain_nusc_metadata)
        self._frame_token_to_loc: dict[str, tuple[str, int]] = {}
        self._frame_scene_arrays: dict[str, np.ndarray] = {}
        self._label_cache: Optional[dict[str, list[float]]] = None
        self.img_size = tuple(img_size)
        self.samples: List[dict] = []
        self.scene_names: List[str] = []

        if scene_indices is not None and scene_names is not None:
            raise ValueError("scene_indices and scene_names are mutually exclusive.")
        if scene_names is not None:
            requested_scene_names = set(scene_names)
            official_scene_names = self._scene_names_for_split(version, split)
            outside_split = requested_scene_names - official_scene_names
            if outside_split:
                raise ValueError(
                    f"Requested scenes are outside official {split!r}: {sorted(outside_split)[:3]}"
                )
            available = {scene["name"]: scene for scene in self.nusc.scene}
            missing = requested_scene_names - set(available)
            if missing:
                raise ValueError(f"Requested scenes are absent from nuScenes metadata: {sorted(missing)[:3]}")
            scenes = [available[name] for name in sorted(requested_scene_names)]
        elif scene_indices is None:
            split_scene_names = self._scene_names_for_split(version, split)
            scenes = [scene for scene in self.nusc.scene if scene["name"] in split_scene_names]
        else:
            scenes = []
            for idx in scene_indices:
                if idx >= len(self.nusc.scene):
                    break
                scenes.append(self.nusc.scene[idx])

        if max_scenes is not None and int(max_scenes) > 0:
            scenes = scenes[: int(max_scenes)]

        expected_scene_names = {scene["name"] for scene in scenes}

        if not scenes:
            raise ValueError(f"No scenes found for version={version!r}, split={split!r}.")

        available_can_scene_names = (
            set(self.can_bus_adapter.available_scene_names())
            if self.can_bus_adapter is not None
            else set()
        )
        skipped_missing_can = 0
        skipped_missing_camera = 0

        for scene in scenes:
            scene_name = scene["name"]
            if self.can_bus_adapter is not None and scene_name not in available_can_scene_names:
                skipped_missing_can += 1
                continue

            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            if cam_channel not in first_sample["data"]:
                skipped_missing_camera += 1
                continue

            current_cam_token = first_sample["data"][cam_channel]
            scene_tokens: List[str] = []
            while current_cam_token:
                scene_tokens.append(current_cam_token)
                cam_data = self.nusc.get("sample_data", current_cam_token)
                current_cam_token = cam_data.get("next", "")

            if len(scene_tokens) >= seq_len:
                self.scene_names.append(scene_name)

            for i in range(len(scene_tokens) - seq_len + 1):
                tokens = scene_tokens[i : i + seq_len]
                center_idx = _current_sequence_index(seq_len, self.online_mode)
                center_token = tokens[center_idx]
                center_sd = self.nusc.get("sample_data", center_token)
                if not center_sd.get("prev", "") or not center_sd.get("next", ""):
                    continue
                self.samples.append(
                    {
                        "scene_name": scene_name,
                        "tokens": tokens,
                        "center_token": center_token,
                        "center_timestamp": int(center_sd["timestamp"]),
                    }
                )

        if scene_names is not None:
            missing_built_scenes = expected_scene_names - set(self.scene_names)
            if missing_built_scenes:
                raise RuntimeError(
                    "Requested manifest scenes were silently unavailable: "
                    f"{sorted(missing_built_scenes)[:3]}. Regenerate the manifest after eligibility audit."
                )

        if not self.samples:
            raise ValueError(
                "No samples were built for "
                f"version={version!r}, split={split!r}, cam_channel={cam_channel!r}, seq_len={seq_len}. "
                f"Skipped scenes without CAN files: {skipped_missing_can}. "
                f"Skipped scenes without the camera channel: {skipped_missing_camera}. "
                "Check dataroot, can_bus_root, split, and camera channel."
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=1)
                if image_mode == "gray"
                else transforms.Lambda(_convert_image_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
                if image_mode == "gray"
                else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if self.frame_cache_root is not None:
            self._load_frame_cache_manifest()

    def _load_frame_cache_manifest(self) -> None:
        assert self.frame_cache_root is not None
        manifest_path = self.frame_cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"frame_cache_root is set but manifest.json was not found: {manifest_path}. "
                "Run scripts/precompute_frames.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        mismatches = []
        if int(manifest.get("img_height", -1)) != int(self.img_size[0]):
            mismatches.append(f"img_height={manifest.get('img_height')} != {self.img_size[0]}")
        if int(manifest.get("img_width", -1)) != int(self.img_size[1]):
            mismatches.append(f"img_width={manifest.get('img_width')} != {self.img_size[1]}")
        if str(manifest.get("image_mode")) != self.image_mode:
            mismatches.append(f"image_mode={manifest.get('image_mode')} != {self.image_mode}")
        if str(manifest.get("version")) != self.version:
            mismatches.append(f"version={manifest.get('version')} != {self.version}")
        if str(manifest.get("cam_channel")) != self.cam_channel:
            mismatches.append(f"cam_channel={manifest.get('cam_channel')} != {self.cam_channel}")
        if mismatches:
            raise ValueError(
                "Frame cache manifest does not match dataset config: "
                + "; ".join(mismatches)
                + f". Regenerate the cache at {self.frame_cache_root} with matching settings."
            )

        for scene_name, tokens in manifest.get("scenes", {}).items():
            for idx, token in enumerate(tokens):
                self._frame_token_to_loc[token] = (scene_name, idx)

        labels_path = self.frame_cache_root / "labels.json"
        if labels_path.is_file():
            with labels_path.open("r", encoding="utf-8") as f:
                self._label_cache = json.load(f)

        missing = [
            sample["tokens"][i]
            for sample in self.samples
            for i in range(len(sample["tokens"]))
            if sample["tokens"][i] not in self._frame_token_to_loc
        ]
        if missing:
            raise ValueError(
                f"Frame cache is missing {len(missing)} tokens required by split={self.split!r} "
                f"(e.g. {missing[0]}). Regenerate the cache covering this split."
            )

        # Release the heavy NuScenes metadata object so DataLoader workers do not each
        # copy it on fork. Only safe when __getitem__ no longer needs it: frames come
        # from the cache, labels come from labels.json, and chassis comes from the CAN
        # adapter (the pose-fallback path still requires NuScenes).
        if (
            self._label_cache is not None
            and self.can_bus_adapter is not None
            and not bool(getattr(self, "retain_nusc_metadata", False))
        ):
            missing_labels = [s["center_token"] for s in self.samples if s["center_token"] not in self._label_cache]
            if not missing_labels:
                self.nusc = None  # type: ignore[assignment]

    def _frame_cache_enabled(self) -> bool:
        return bool(getattr(self, "_frame_token_to_loc", None))

    def _get_frame_scene_array(self, scene_name: str) -> np.ndarray:
        array = self._frame_scene_arrays.get(scene_name)
        if array is None:
            assert self.frame_cache_root is not None
            array = np.load(self.frame_cache_root / f"{scene_name}.npy", mmap_mode="r")
            self._frame_scene_arrays[scene_name] = array
        return array

    def _load_frame_from_cache(self, token: str) -> torch.Tensor:
        scene_name, idx = self._frame_token_to_loc[token]
        raw = np.asarray(self._get_frame_scene_array(scene_name)[idx])
        tensor = torch.from_numpy(raw.copy()).float().div_(255.0)
        if self.image_mode == "gray":
            tensor = tensor.unsqueeze(0)
            tensor.sub_(0.5).div_(0.5)
        else:
            tensor = tensor.permute(2, 0, 1).contiguous()
            tensor.sub_(0.5).div_(0.5)
        return tensor

    def _load_frame_tensor(self, token: str) -> torch.Tensor:
        if self._frame_cache_enabled():
            return self._load_frame_from_cache(token)

        if self.image_cache_size > 0 and token in self._image_cache:
            frame = self._image_cache.pop(token)
            self._image_cache[token] = frame
            return frame

        img_path = Path(self.nusc.get_sample_data_path(token))
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with Image.open(img_path) as img:
            frame = self.transform(img).float()

        if self.image_cache_size > 0:
            self._image_cache[token] = frame
            if len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return frame

    @staticmethod
    def _scene_names_for_split(version: str, split: str) -> set[str]:
        from nuscenes.utils.splits import create_splits_scenes

        splits = create_splits_scenes()
        split = split.lower()

        if version == "v1.0-mini":
            if split == "train":
                return set(splits["mini_train"])
            if split == "val":
                return set(splits["mini_val"])
            if split == "trainval":
                return set(splits["mini_train"]) | set(splits["mini_val"])
            raise ValueError("v1.0-mini supports split='train', 'val', or 'trainval'.")

        if version == "v1.0-trainval":
            if split in {"train", "val"}:
                return set(splits[split])
            if split == "trainval":
                return set(splits["train"]) | set(splits["val"])
            raise ValueError("v1.0-trainval supports split='train', 'val', or 'trainval'.")

        if version == "v1.0-test":
            if split == "test":
                return set(splits["test"])
            raise ValueError("v1.0-test supports only split='test'.")

        raise ValueError(f"Unsupported nuScenes version: {version!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_velocity_label(self, center_token: str) -> torch.Tensor:
        label_cache = getattr(self, "_label_cache", None)
        if label_cache is not None:
            cached = label_cache.get(center_token)
            if cached is not None:
                return torch.tensor(cached, dtype=torch.float32)
            if self.nusc is None:
                raise KeyError(
                    f"Velocity label for {center_token} missing from cache and NuScenes was released. "
                    "Regenerate the frame cache with labels."
                )
        center_sd = self.nusc.get("sample_data", center_token)
        prev_token = center_sd.get("prev", "")
        next_token = center_sd.get("next", "")
        if not prev_token or not next_token:
            raise ValueError(f"Cannot compute centered velocity label for boundary frame: {center_token}")
        prev_sd = self.nusc.get("sample_data", prev_token)
        next_sd = self.nusc.get("sample_data", next_token)
        prev_pose = self.nusc.get("ego_pose", prev_sd["ego_pose_token"])
        center_pose = self.nusc.get("ego_pose", center_sd["ego_pose_token"])
        next_pose = self.nusc.get("ego_pose", next_sd["ego_pose_token"])
        x_prev, y_prev = prev_pose["translation"][0], prev_pose["translation"][1]
        x_next, y_next = next_pose["translation"][0], next_pose["translation"][1]
        dt = (next_pose["timestamp"] - prev_pose["timestamp"]) / 1e6
        if dt <= 0.0:
            raise ValueError(f"Non-positive dt for centered velocity label: {center_token}")
        vx_global = (x_next - x_prev) / dt
        vy_global = (y_next - y_prev) / dt
        vx_ego, vy_ego = _rotate_global_velocity_to_ego(vx_global, vy_global, center_pose["rotation"])
        return torch.tensor([vx_ego, vy_ego], dtype=torch.float32)

    def _compute_speed_label(self, center_token: str) -> float:
        velocity = self._compute_velocity_label(center_token)
        return float(torch.linalg.vector_norm(velocity).item())

    def _compute_pose_fallback(self, center_token: str) -> Tuple[torch.Tensor, float]:
        center_sd = self.nusc.get("sample_data", center_token)
        prev_token = center_sd.get("prev", "")
        if not prev_token:
            return torch.zeros(1, FAIR_CHASSIS_DIM, dtype=torch.float32), 0.0
        prev_sd = self.nusc.get("sample_data", prev_token)
        pose1 = self.nusc.get("ego_pose", prev_sd["ego_pose_token"])
        pose2 = self.nusc.get("ego_pose", center_sd["ego_pose_token"])
        x1, y1 = pose1["translation"][0], pose1["translation"][1]
        x2, y2 = pose2["translation"][0], pose2["translation"][1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dt = (pose2["timestamp"] - pose1["timestamp"]) / 1e6
        speed = distance / dt if dt > 0 else 0.0
        feature = torch.zeros(FAIR_CHASSIS_DIM, dtype=torch.float32)
        feature[0] = speed
        feature[1] = speed
        feature[-1] = speed
        return feature.unsqueeze(0), speed

    def _compute_geometry_prior(self, chassis_sequence: torch.Tensor) -> torch.Tensor:
        if self.geometry_mode == "none":
            return torch.empty(0, dtype=torch.float32)
        center_idx = _current_sequence_index(
            chassis_sequence.shape[0],
            bool(getattr(self, "online_mode", False)),
        )
        center_vec = chassis_sequence[center_idx]
        return torch.tensor(
            [
                float(center_vec[2].item()),  # yaw_rate
                float(center_vec[5].item()),  # steering
                float(center_vec[6].item()),  # steering_feedback
                float(center_vec[FAIR_CHASSIS_FEATURES.index("base_vx")].item()),
            ],
            dtype=torch.float32,
        )

    def _load_flow_sequence(self, tokens: Sequence[str]) -> torch.Tensor:
        if self.flow_mode == "none":
            return torch.empty(0, dtype=torch.float32)
        if self.flow_root is None:
            raise FileNotFoundError("flow_root is required when flow_mode='precomputed'.")

        flows = []
        for token in tokens[:-1]:
            pt_path = self.flow_root / f"{token}.pt"
            npy_path = self.flow_root / f"{token}.npy"
            if pt_path.is_file():
                flow = torch.load(pt_path, map_location="cpu").float()
            elif npy_path.is_file():
                flow = torch.from_numpy(np.load(npy_path)).float()
            else:
                raise FileNotFoundError(f"Precomputed flow not found for token {token}: {pt_path} or {npy_path}")
            if flow.ndim == 3 and flow.shape[-1] == 2:
                flow = flow.permute(2, 0, 1)
            if flow.shape[0] != 2:
                raise ValueError(f"Optical flow for token {token} must have shape [2,H,W] or [H,W,2].")
            flows.append(flow)
        return torch.stack(flows) if flows else torch.empty(0, dtype=torch.float32)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        frames = [self._load_frame_tensor(token) for token in tokens]
        stacked_frames = torch.stack(frames)
        velocity_label = self._compute_velocity_label(sample["center_token"])
        speed_label = float(torch.linalg.vector_norm(velocity_label).item())

        if self.can_bus_adapter is not None:
            kin_seq, base_vx = self.can_bus_adapter.extract_feature_sequence(
                sample["scene_name"],
                sample["center_timestamp"],
                chassis_mode=self.chassis_mode,
                window_sec=self.chassis_window_sec,
                hz=self.chassis_hz,
                online_mode=self.online_mode,
            )
            physics_inputs, physics_valid_mask, physics_dt, physics_timestamps = (
                self.can_bus_adapter.extract_physics_sequence(
                    sample["scene_name"],
                    sample["center_timestamp"],
                    window_sec=self.chassis_window_sec,
                    hz=self.chassis_hz,
                    online_mode=self.online_mode,
                )
            )
        elif self.allow_pose_fallback:
            kin_seq, base_vx = self._compute_pose_fallback(sample["center_token"])
            physics_inputs = torch.zeros(kin_seq.shape[0], PHYSICS_INPUT_DIM, dtype=torch.float32)
            physics_valid_mask = torch.zeros_like(physics_inputs, dtype=torch.bool)
            physics_inputs[:, PHYSICS_INPUT_INDEX["base_vx"]] = base_vx
            physics_valid_mask[:, PHYSICS_INPUT_INDEX["base_vx"]] = True
            physics_dt = torch.zeros(kin_seq.shape[0], dtype=torch.float32)
            physics_timestamps = [sample["center_timestamp"]] * kin_seq.shape[0]
        else:
            raise FileNotFoundError(
                "CAN bus root is not configured. Provide --can_bus_root or enable --allow_pose_fallback."
            )
        if self.base_velocity_source == "eovm_ekf":
            assert self.eovm_lookup is not None
            base_velocity_values = self.eovm_lookup.lookup(sample["scene_name"], sample["center_timestamp"])
            base_velocity = torch.tensor(base_velocity_values, dtype=torch.float32)
        else:
            base_velocity = torch.tensor([base_vx, 0.0], dtype=torch.float32)

        if self.physics_extra_lookup is not None:
            for row_index, timestamp_us in enumerate(physics_timestamps):
                extra_values, extra_valid = self.physics_extra_lookup.lookup(sample["scene_name"], timestamp_us)
                physics_inputs[row_index, FAIR_PHYSICS_FEATURE_COUNT:] = torch.tensor(extra_values)
                physics_valid_mask[row_index, FAIR_PHYSICS_FEATURE_COUNT:] = torch.tensor(extra_valid)

        return {
            "images": stacked_frames,
            "chassis": kin_seq,
            "base_velocity": base_velocity,
            "velocity": velocity_label,
            "flow": self._load_flow_sequence(tokens),
            "geometry": self._compute_geometry_prior(kin_seq),
            "physics_inputs": physics_inputs,
            "physics_valid_mask": physics_valid_mask,
            "physics_dt": physics_dt,
            "chassis_current_index": torch.tensor(
                _current_sequence_index(kin_seq.shape[0], self.online_mode),
                dtype=torch.long,
            ),
            "base_speed": torch.tensor([base_vx], dtype=torch.float32),
            "speed": torch.tensor([speed_label], dtype=torch.float32),
            "scene_name": sample["scene_name"],
            "center_token": sample["center_token"],
            "center_timestamp": torch.tensor(sample["center_timestamp"], dtype=torch.long),
        }
