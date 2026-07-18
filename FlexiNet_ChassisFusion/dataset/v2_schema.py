from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from model.v2.rotation_flow import scale_intrinsics


KINEMATIC_VALUE_FEATURES = (
    "yaw_rate",
    "steering_command",
    "steering_feedback",
    "longitudinal_accel",
    "lateral_accel",
    "throttle",
    "brake",
    "wheel_residual_fl",
    "wheel_residual_fr",
    "wheel_residual_rl",
    "wheel_residual_rr",
    "wheel_speed_relative_std",
)
KINEMATIC_VALUE_DIM = len(KINEMATIC_VALUE_FEATURES)
KINEMATIC_INPUT_DIM = KINEMATIC_VALUE_DIM * 3
DEFAULT_FIELD_MAX_AGE_SEC = (0.10, 0.15, 0.15, 0.10, 0.10, 0.25, 0.25) + (0.10,) * 5


def _field_limits_tensor(limits: float | Sequence[float]) -> torch.Tensor:
    if isinstance(limits, (int, float)):
        values = [float(limits)] * KINEMATIC_VALUE_DIM
    else:
        values = [float(value) for value in limits]
    if len(values) != KINEMATIC_VALUE_DIM or min(values) <= 0.0:
        raise ValueError(f"field age limits must contain {KINEMATIC_VALUE_DIM} positive values.")
    return torch.tensor(values, dtype=torch.float32)


def _finite_value(record: dict | None, key: str, scale: float = 1.0) -> tuple[float, bool]:
    if record is None or key not in record:
        return 0.0, False
    try:
        value = float(record[key]) / scale
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0, False
    return (value, True) if math.isfinite(value) else (0.0, False)


def build_kinematic_step(
    monitor: dict | None,
    monitor_age_sec: float,
    wheel: dict | None,
    wheel_age_sec: float,
    steer: dict | None,
    steer_age_sec: float,
    max_age_sec: float | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deployable values, masks, ages, and separate absolute wheel signals."""

    field_limits = _field_limits_tensor(max_age_sec)
    values = torch.zeros(KINEMATIC_VALUE_DIM, dtype=torch.float32)
    valid = torch.zeros(KINEMATIC_VALUE_DIM, dtype=torch.bool)
    ages = field_limits.clone()

    field_specs = (
        (0, monitor, "yaw_rate", 180.0 / math.pi, monitor_age_sec),
        (1, monitor, "steering", 780.0, monitor_age_sec),
        (2, steer, "value", 7.7, steer_age_sec),
        (3, wheel, "longitudinal_accel", 1.0, wheel_age_sec),
        (4, wheel, "transversal_accel", 1.0, wheel_age_sec),
        (5, monitor, "throttle", 1000.0, monitor_age_sec),
        (6, monitor, "brake", 126.0, monitor_age_sec),
    )
    for index, record, key, scale, age_sec in field_specs:
        value, field_valid = _finite_value(record, key, scale)
        if field_valid and 0.0 <= float(age_sec) <= float(field_limits[index]):
            values[index] = value
            valid[index] = True
            ages[index] = float(age_sec)

    wheel_fields = ("FL_wheel_speed", "FR_wheel_speed", "RL_wheel_speed", "RR_wheel_speed")
    wheel_values = torch.zeros(4, dtype=torch.float32)
    wheel_valid = torch.zeros(4, dtype=torch.bool)
    wheel_ages = field_limits[7:11].clone()
    for index, field in enumerate(wheel_fields):
        value, field_valid = _finite_value(wheel, field, 1000.0)
        limit = float(field_limits[7 + index])
        if field_valid and 0.0 <= float(wheel_age_sec) <= limit:
            wheel_values[index] = value
            wheel_valid[index] = True
            wheel_ages[index] = float(wheel_age_sec)

    valid_count = int(wheel_valid.sum().item())
    if valid_count >= 2:
        wheel_mean = wheel_values[wheel_valid].mean()
        denominator = wheel_mean.abs().clamp_min(0.5)
        residuals = (wheel_values - wheel_mean) / denominator
        values[7:11] = torch.where(wheel_valid, residuals, torch.zeros_like(residuals))
        valid[7:11] = wheel_valid
        ages[7:11] = wheel_ages
        values[11] = wheel_values[wheel_valid].std(unbiased=False) / denominator
        valid[11] = True
        ages[11] = wheel_ages[wheel_valid].max()

    normalized_ages = (ages / field_limits.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    return values, valid, normalized_ages, wheel_values, wheel_valid, wheel_ages


def apply_sequence_photometric_augmentation(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    gamma: float,
) -> torch.Tensor:
    """Apply one set of photometric factors to every frame in a normalized sequence."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [T,3,H,W].")
    unit_images = (images + 1.0) * 0.5
    augmented = []
    for frame in unit_images:
        frame = TF.adjust_brightness(frame, brightness)
        frame = TF.adjust_contrast(frame, contrast)
        frame = TF.adjust_saturation(frame, saturation)
        frame = TF.adjust_hue(frame, hue)
        frame = TF.adjust_gamma(frame.clamp(0.0, 1.0), gamma)
        augmented.append(frame.clamp(0.0, 1.0))
    return torch.stack(augmented) * 2.0 - 1.0


def normalize_kinematic_values(
    values: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if values.shape != valid.shape or values.shape[-1] != KINEMATIC_VALUE_DIM:
        raise ValueError("values and valid must match the V2 kinematic schema.")
    normalized = (values - mean) / std.clamp_min(1.0e-3)
    return torch.where(valid, normalized, torch.zeros_like(normalized)).clamp(-5.0, 5.0)


def quaternion_to_rotation_matrix(rotation: list[float]) -> torch.Tensor:
    if len(rotation) != 4:
        raise ValueError("rotation must be [w,x,y,z].")
    w, x, y, z = rotation
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("rotation quaternion has zero norm.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


class V2NuScenesDatasetAdapter(Dataset):
    """Add causal geometry inputs to an online NuScenesChassisFusionDataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        flow_root: str | Path,
        camera_height_m: float,
        road_mask_top_fraction: float = 0.45,
        geometry_size: tuple[int, int] | None = (288, 512),
        kinematic_stats_path: str | Path | None = None,
        field_max_age_sec: Sequence[float] | None = None,
        photometric_augmentation: dict[str, Any] | None = None,
    ) -> None:
        if not bool(getattr(base_dataset, "online_mode", False)):
            raise ValueError("V2 requires base_dataset.online_mode=True to prevent future-frame leakage.")
        if int(getattr(base_dataset, "seq_len", 0)) < 2:
            raise ValueError("V2 requires at least two causal image frames.")
        if float(camera_height_m) <= 0.0:
            raise ValueError("camera_height_m must be a positive calibrated or nominal value.")
        self.base_dataset = base_dataset
        if getattr(base_dataset, "nusc", None) is None:
            raise ValueError(
                "V2 needs nuScenes calibration metadata during __getitem__; do not use a frame cache mode that releases nusc."
            )
        self.flow_root = Path(flow_root)
        self.camera_height_m = float(camera_height_m)
        if not 0.0 <= road_mask_top_fraction < 1.0:
            raise ValueError("road_mask_top_fraction must be in [0,1).")
        self.road_mask_top_fraction = float(road_mask_top_fraction)
        if geometry_size is not None and (len(geometry_size) != 2 or min(geometry_size) <= 0):
            raise ValueError("geometry_size must be a positive (height,width) pair.")
        self.geometry_size = tuple(int(value) for value in geometry_size) if geometry_size else None
        self.photometric_augmentation = dict(photometric_augmentation or {})
        self._kinematic_mean = torch.zeros(KINEMATIC_VALUE_DIM, dtype=torch.float32)
        self._kinematic_std = torch.ones(KINEMATIC_VALUE_DIM, dtype=torch.float32)
        self._scenario_weights: dict[str, float] = {}
        statistics = None
        if kinematic_stats_path:
            statistics = json.loads(Path(kinematic_stats_path).read_text(encoding="utf-8"))
            if statistics.get("source") != "nuScenes train split only":
                raise ValueError("Kinematic statistics must be fitted from the nuScenes train split only.")
            if tuple(statistics.get("features", ())) != KINEMATIC_VALUE_FEATURES:
                raise ValueError("Kinematic statistics feature order does not match the V2 schema.")
            self._kinematic_mean = torch.tensor(statistics["mean"], dtype=torch.float32)
            self._kinematic_std = torch.tensor(statistics["std"], dtype=torch.float32).clamp_min(1.0e-3)
            if self._kinematic_mean.shape != (KINEMATIC_VALUE_DIM,) or self._kinematic_std.shape != (
                KINEMATIC_VALUE_DIM,
            ):
                raise ValueError("Kinematic mean/std must contain exactly 12 values.")
            self._scenario_weights = {
                str(token): float(weight)
                for token, weight in statistics.get("sample_weights", {}).items()
            }
        if field_max_age_sec is not None:
            self._field_max_age_sec = _field_limits_tensor(field_max_age_sec)
        elif statistics is not None:
            self._field_max_age_sec = _field_limits_tensor(statistics["field_max_age_sec"])
        else:
            self._field_max_age_sec = _field_limits_tensor(DEFAULT_FIELD_MAX_AGE_SEC)
        self._metadata = self._build_metadata()
        if (
            bool(getattr(base_dataset, "_frame_cache_enabled", lambda: False)())
            and getattr(base_dataset, "_label_cache", None) is not None
            and getattr(base_dataset, "can_bus_adapter", None) is not None
        ):
            base_dataset.nusc = None
        if not self.flow_root.is_dir():
            raise FileNotFoundError(f"V2 flow root not found: {self.flow_root}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {
            "base_dataset",
            "flow_root",
            "camera_height_m",
            "road_mask_top_fraction",
            "geometry_size",
            "photometric_augmentation",
            "_kinematic_mean",
            "_kinematic_std",
            "_scenario_weights",
            "_field_max_age_sec",
            "_metadata",
        }:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        if self.photometric_augmentation and bool(self.photometric_augmentation.get("enabled", False)):
            probability = float(self.photometric_augmentation.get("probability", 0.5))
            if float(torch.rand(())) < probability:
                brightness = float(self.photometric_augmentation.get("brightness", 0.1))
                contrast = float(self.photometric_augmentation.get("contrast", 0.1))
                saturation = float(self.photometric_augmentation.get("saturation", 0.05))
                hue = float(self.photometric_augmentation.get("hue", 0.02))
                gamma = self.photometric_augmentation.get("gamma", [0.9, 1.1])
                sample = lambda bound: 1.0 + (2.0 * float(torch.rand(())) - 1.0) * float(bound)
                hue_value = (2.0 * float(torch.rand(())) - 1.0) * hue
                gamma_value = float(gamma[0]) + float(torch.rand(())) * (float(gamma[1]) - float(gamma[0]))
                item["images"] = apply_sequence_photometric_augmentation(
                    item["images"],
                    sample(brightness),
                    sample(contrast),
                    sample(saturation),
                    hue_value,
                    gamma_value,
                )
        sample = self.base_dataset.samples[index]
        source_token, target_token = sample["tokens"][-2:]
        flow, confidence, source_size = self._load_flow(source_token)
        geometry_size = self.geometry_size or tuple(int(value) for value in flow.shape[-2:])
        geometry_h, geometry_w = geometry_size
        flow, confidence = self._resize_flow(flow, confidence, geometry_size)
        road_mask = torch.zeros(1, geometry_h, geometry_w, dtype=torch.bool)
        road_mask[:, int(round(geometry_h * self.road_mask_top_fraction)) :] = True

        metadata = self._metadata[index]
        intrinsics = scale_intrinsics(
            metadata["intrinsics"], metadata["original_size"], geometry_size
        )
        rotation_ego_from_camera = metadata["rotation_ego_from_camera"]
        rotation_camera_from_ego = rotation_ego_from_camera.T
        plane_normal_camera = metadata["plane_normal_camera"]

        (
            kinematic_values,
            kinematic_valid,
            kinematic_age,
            kinematic_dt,
            wheel_values,
            wheel_valid,
            wheel_age,
        ) = self._build_kinematic_sequence(sample["scene_name"], sample["center_timestamp"])
        kinematic_values_raw = kinematic_values.clone()
        normalized_values = normalize_kinematic_values(
            kinematic_values,
            kinematic_valid,
            self._kinematic_mean,
            self._kinematic_std,
        )
        yaw_rate = kinematic_values_raw[-1, 0]
        yaw_valid = kinematic_valid[-1, 0]
        angular_rate_ego = torch.stack((yaw_rate.new_zeros(()), yaw_rate.new_zeros(()), yaw_rate))
        angular_rate_camera = rotation_camera_from_ego @ angular_rate_ego

        item.update(
            {
                "flow_v2": flow,
                "flow_confidence": confidence,
                "road_mask": road_mask,
                "flow_source_size": torch.tensor(source_size, dtype=torch.long),
                "intrinsics": intrinsics,
                "frame_dt": torch.tensor(
                    metadata["frame_dt"],
                    dtype=torch.float32,
                ),
                "angular_rate_camera": angular_rate_camera,
                "angular_rate_valid": torch.tensor([False, False, bool(yaw_valid)], dtype=torch.bool),
                "angular_rate_age_sec": kinematic_age[-1, 0] * self._field_max_age_sec[0],
                "plane_normal_camera": plane_normal_camera,
                "plane_offset": torch.tensor(self.camera_height_m, dtype=torch.float32),
                "rotation_ego_from_camera": rotation_ego_from_camera,
                "kinematic_values": normalized_values,
                "kinematic_values_raw": kinematic_values_raw,
                "kinematic_valid": kinematic_valid,
                "kinematic_age": kinematic_age,
                "kinematic_dt": kinematic_dt,
                "wheel_speed": wheel_values[-1],
                "wheel_speed_valid": wheel_valid[-1],
                "wheel_speed_age_sec": wheel_age[-1],
                "output_reference": "ego_reference",
            }
        )
        return item

    def scenario_weights(self) -> torch.Tensor:
        if not self._scenario_weights:
            raise ValueError("Scenario sampling requires sample_weights in kinematic statistics.")
        weights = []
        for sample in self.base_dataset.samples:
            token = str(sample["center_token"])
            if token not in self._scenario_weights:
                raise ValueError(f"Missing scenario weight for training token {token}.")
            weights.append(self._scenario_weights[token])
        return torch.tensor(weights, dtype=torch.double)

    def _build_kinematic_sequence(
        self,
        scene_name: str,
        center_timestamp_us: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        adapter = self.base_dataset.can_bus_adapter
        if adapter is None:
            raise ValueError("V2 weak kinematics requires the nuScenes CAN bus adapter.")
        timestamps = adapter.sample_timestamps(
            center_timestamp_us=center_timestamp_us,
            window_sec=self.base_dataset.chassis_window_sec,
            hz=self.base_dataset.chassis_hz,
            online_mode=True,
        )
        rows = []
        masks = []
        ages = []
        wheel_rows = []
        wheel_masks = []
        wheel_ages = []
        for timestamp_us in timestamps:
            monitor, monitor_age = adapter.select_latest_before_with_age(
                scene_name, "vehicle_monitor", timestamp_us
            )
            wheel, wheel_age = adapter.select_latest_before_with_age(
                scene_name, "zoe_veh_info", timestamp_us
            )
            steer, steer_age = adapter.select_latest_before_with_age(
                scene_name, "steeranglefeedback", timestamp_us
            )
            row = build_kinematic_step(
                monitor,
                monitor_age,
                wheel,
                wheel_age,
                steer,
                steer_age,
                self._field_max_age_sec,
            )
            values, valid, normalized_age, wheels, wheels_valid, wheels_age = row
            rows.append(values)
            masks.append(valid)
            ages.append(normalized_age)
            wheel_rows.append(wheels)
            wheel_masks.append(wheels_valid)
            wheel_ages.append(wheels_age)
        dt = torch.zeros(len(timestamps), dtype=torch.float32)
        if len(timestamps) > 1:
            dt[1:] = torch.tensor(
                [
                    (timestamps[index] - timestamps[index - 1]) / 1_000_000.0
                    for index in range(1, len(timestamps))
                ],
                dtype=torch.float32,
            )
        return (
            torch.stack(rows),
            torch.stack(masks),
            torch.stack(ages),
            dt,
            torch.stack(wheel_rows),
            torch.stack(wheel_masks),
            torch.stack(wheel_ages),
        )

    def _build_metadata(self) -> list[dict[str, Any]]:
        nusc = self.base_dataset.nusc
        metadata = []
        upward_ego = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        for sample in self.base_dataset.samples:
            source_token, target_token = sample["tokens"][-2:]
            source_sd = nusc.get("sample_data", source_token)
            target_sd = nusc.get("sample_data", target_token)
            calibrated = nusc.get("calibrated_sensor", target_sd["calibrated_sensor_token"])
            rotation_ego_from_camera = quaternion_to_rotation_matrix(calibrated["rotation"])
            metadata.append(
                {
                    "intrinsics": torch.tensor(calibrated["camera_intrinsic"], dtype=torch.float32),
                    "original_size": (int(target_sd["height"]), int(target_sd["width"])),
                    "rotation_ego_from_camera": rotation_ego_from_camera,
                    "plane_normal_camera": rotation_ego_from_camera.T @ upward_ego,
                    "frame_dt": (
                        int(target_sd["timestamp"]) - int(source_sd["timestamp"])
                    ) / 1_000_000.0,
                }
            )
        return metadata

    def _load_flow(self, token: str) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        pt_path = self.flow_root / f"{token}.pt"
        npy_path = self.flow_root / f"{token}.npy"
        if pt_path.is_file():
            package = torch.load(pt_path, map_location="cpu")
            if not isinstance(package, dict) or "flow" not in package:
                raise ValueError(f"V2 flow package must be a dict with a flow tensor: {pt_path}")
            flow = torch.as_tensor(package["flow"], dtype=torch.float32)
            if flow.ndim == 3 and flow.shape[-1] == 2:
                flow = flow.permute(2, 0, 1)
            confidence = torch.as_tensor(
                package.get("confidence", torch.ones(1, *flow.shape[-2:])), dtype=torch.float32
            )
            source_size = tuple(int(value) for value in package.get("target_size", flow.shape[-2:]))
        elif npy_path.is_file():
            array = np.load(npy_path)
            flow = torch.from_numpy(array).float()
            if flow.ndim == 3 and flow.shape[-1] == 2:
                flow = flow.permute(2, 0, 1)
            confidence = torch.ones(1, *flow.shape[-2:], dtype=torch.float32)
            source_size = tuple(int(value) for value in flow.shape[-2:])
        else:
            raise FileNotFoundError(f"Missing V2 flow for {token}: {pt_path} or {npy_path}")
        if flow.ndim == 3 and flow.shape[-1] == 2:
            flow = flow.permute(2, 0, 1)
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(0)
        if flow.shape[0] != 2 or confidence.shape[0] != 1:
            raise ValueError("V2 flow/confidence must have shapes [2,H,W] and [1,H,W].")
        return flow, confidence, source_size

    @staticmethod
    def _resize_flow(
        flow: torch.Tensor,
        confidence: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_h, source_w = flow.shape[-2:]
        target_h, target_w = target_size
        if (source_h, source_w) == target_size:
            return flow, confidence
        resized = F.interpolate(flow[None], target_size, mode="bilinear", align_corners=False)[0]
        resized[0] *= float(target_w) / float(source_w)
        resized[1] *= float(target_h) / float(source_h)
        confidence = F.interpolate(confidence[None], target_size, mode="bilinear", align_corners=False)[0]
        return resized, confidence.clamp(0.0, 1.0)
