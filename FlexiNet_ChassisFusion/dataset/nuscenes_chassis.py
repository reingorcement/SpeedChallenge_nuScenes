from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NuScenesCanBusAdapter:
    def __init__(self, can_bus_root: str | Path) -> None:
        self.can_bus_root = Path(can_bus_root)
        if not self.can_bus_root.exists():
            raise FileNotFoundError(f"CAN bus root not found: {self.can_bus_root}")
        self._message_cache: dict[tuple[str, str], List[dict]] = {}

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
            return []
        self._message_cache[cache_key] = messages
        return messages

    @staticmethod
    def _select_latest_before(records: Sequence[dict], timestamp_us: int) -> Optional[dict]:
        candidates = [r for r in records if r.get("utime", 0) <= timestamp_us]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["utime"])

    def extract_feature_vector(self, scene_name: str, timestamp_us: int) -> Tuple[torch.Tensor, float]:
        pose = self._select_latest_before(self.load_messages(scene_name, "pose"), timestamp_us)
        monitor = self._select_latest_before(self.load_messages(scene_name, "vehicle_monitor"), timestamp_us)
        wheel = self._select_latest_before(self.load_messages(scene_name, "zoe_veh_info"), timestamp_us)
        steer = self._select_latest_before(self.load_messages(scene_name, "steeranglefeedback"), timestamp_us)

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
        wheel_speed_mean = 0.0
        wheel_speed_std = 0.0
        accel_x = 0.0
        accel_y = 0.0
        rotation_rate_z = 0.0
        base_speed = 0.0

        if monitor is not None:
            vehicle_speed = float(monitor.get("vehicle_speed", 0.0)) / 3.6
            yaw_rate = math.radians(float(monitor.get("yaw_rate", 0.0)))
            throttle = float(monitor.get("throttle", 0.0)) / 1000.0
            brake = float(monitor.get("brake", 0.0)) / 126.0
            steering = float(monitor.get("steering", 0.0)) / 780.0
            base_speed = vehicle_speed

        if wheel is not None:
            wheels = [
                float(wheel.get("FL_wheel_speed", 0.0)),
                float(wheel.get("FR_wheel_speed", 0.0)),
                float(wheel.get("RL_wheel_speed", 0.0)),
                float(wheel.get("RR_wheel_speed", 0.0)),
            ]
            wheel_speed_mean = sum(wheels) / len(wheels) / 1000.0
            wheel_speed_std = float(torch.tensor(wheels).std(unbiased=False).item()) / 1000.0
            odom_speed = float(wheel.get("odom_speed", 0.0)) / 3.6
            longitudinal_accel = float(wheel.get("longitudinal_accel", 0.0))
            transversal_accel = float(wheel.get("transversal_accel", 0.0))
            if base_speed == 0.0:
                base_speed = odom_speed

        if steer is not None:
            steering_feedback = float(steer.get("value", 0.0)) / 7.7

        if pose is not None:
            vel = pose.get("vel", [0.0, 0.0, 0.0])
            accel = pose.get("accel", [0.0, 0.0, 0.0])
            rotation_rate = pose.get("rotation_rate", [0.0, 0.0, 0.0])
            if len(vel) >= 2:
                pose_speed = math.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2)
            if len(accel) >= 2:
                accel_x = float(accel[0])
                accel_y = float(accel[1])
            if len(rotation_rate) >= 3:
                rotation_rate_z = float(rotation_rate[2])
            if base_speed == 0.0:
                base_speed = pose_speed

        feature_vec = torch.tensor(
            [
                vehicle_speed,
                odom_speed,
                pose_speed,
                yaw_rate,
                throttle,
                brake,
                steering,
                steering_feedback,
                longitudinal_accel,
                transversal_accel,
                wheel_speed_mean,
                wheel_speed_std,
                accel_x,
                accel_y,
                rotation_rate_z,
                base_speed,
            ],
            dtype=torch.float32,
        )
        return feature_vec, base_speed


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
    ) -> None:
        from nuscenes.nuscenes import NuScenes

        self.version = version
        self.split = split
        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        self.seq_len = seq_len
        self.cam_channel = cam_channel
        self.allow_pose_fallback = allow_pose_fallback
        self.can_bus_adapter = NuScenesCanBusAdapter(can_bus_root) if can_bus_root else None
        self.samples: List[dict] = []
        self.scene_names: List[str] = []

        if scene_indices is None:
            split_scene_names = self._scene_names_for_split(version, split)
            scenes = [scene for scene in self.nusc.scene if scene["name"] in split_scene_names]
        else:
            scenes = []
            for idx in scene_indices:
                if idx >= len(self.nusc.scene):
                    break
                scenes.append(self.nusc.scene[idx])

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
                center_idx = seq_len // 2
                center_token = tokens[center_idx]
                center_sd = self.nusc.get("sample_data", center_token)
                self.samples.append(
                    {
                        "scene_name": scene_name,
                        "tokens": tokens,
                        "center_token": center_token,
                        "center_timestamp": int(center_sd["timestamp"]),
                    }
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
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

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

    def _compute_speed_label(self, center_token: str) -> float:
        center_sd = self.nusc.get("sample_data", center_token)
        next_token = center_sd.get("next", "")
        if not next_token:
            return 0.0
        next_sd = self.nusc.get("sample_data", next_token)
        pose1 = self.nusc.get("ego_pose", center_sd["ego_pose_token"])
        pose2 = self.nusc.get("ego_pose", next_sd["ego_pose_token"])
        x1, y1 = pose1["translation"][0], pose1["translation"][1]
        x2, y2 = pose2["translation"][0], pose2["translation"][1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dt = (pose2["timestamp"] - pose1["timestamp"]) / 1e6
        return distance / dt if dt > 0 else 0.0

    def _compute_pose_fallback(self, center_token: str) -> torch.Tensor:
        center_sd = self.nusc.get("sample_data", center_token)
        prev_token = center_sd.get("prev", "")
        if not prev_token:
            return torch.zeros(16, dtype=torch.float32)
        prev_sd = self.nusc.get("sample_data", prev_token)
        pose1 = self.nusc.get("ego_pose", prev_sd["ego_pose_token"])
        pose2 = self.nusc.get("ego_pose", center_sd["ego_pose_token"])
        x1, y1 = pose1["translation"][0], pose1["translation"][1]
        x2, y2 = pose2["translation"][0], pose2["translation"][1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dt = (pose2["timestamp"] - pose1["timestamp"]) / 1e6
        speed = distance / dt if dt > 0 else 0.0
        return torch.tensor(
            [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, speed, speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        frames = []
        for token in tokens:
            img_path = Path(self.nusc.get_sample_data_path(token))
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            with Image.open(img_path) as img:
                frames.append(self.transform(img.convert("L")))

        stacked_frames = torch.stack(frames).float()
        speed_label = self._compute_speed_label(sample["center_token"])

        if self.can_bus_adapter is not None:
            kin_vec, base_speed = self.can_bus_adapter.extract_feature_vector(sample["scene_name"], sample["center_timestamp"])
        elif self.allow_pose_fallback:
            kin_vec = self._compute_pose_fallback(sample["center_token"])
            base_speed = float(kin_vec[0].item())
        else:
            raise FileNotFoundError(
                "CAN bus root is not configured. Provide --can_bus_root or enable --allow_pose_fallback."
            )

        return {
            "images": stacked_frames,
            "chassis": kin_vec,
            "base_speed": torch.tensor([base_speed], dtype=torch.float32),
            "speed": torch.tensor([speed_label], dtype=torch.float32),
            "scene_name": sample["scene_name"],
            "center_token": sample["center_token"],
            "center_timestamp": torch.tensor(sample["center_timestamp"], dtype=torch.long),
        }
