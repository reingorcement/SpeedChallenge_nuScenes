from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NuScenesVisionEvalDataset(Dataset):
    """nuScenes pure-vision dataset aligned with the chassis-fusion validation samples."""

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        split: str = "val",
        img_size: Tuple[int, int] = (64, 64),
        seq_len: int = 13,
        cam_channel: str = "CAM_FRONT",
        can_bus_root: Optional[str | Path] = None,
        filter_can_bus_scenes: bool = True,
        scene_indices: Optional[Sequence[int]] = None,
        image_cache_size: int = 0,
    ) -> None:
        from nuscenes.nuscenes import NuScenes

        self.version = version
        self.split = split
        self.dataroot = Path(dataroot)
        self.seq_len = int(seq_len)
        self.cam_channel = cam_channel
        self.image_cache_size = max(int(image_cache_size), 0)
        self._image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)
        self.samples: list[dict] = []
        self.scene_names: list[str] = []

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

        available_can_scene_names: set[str] = set()
        if filter_can_bus_scenes:
            if can_bus_root is None:
                raise ValueError("can_bus_root is required when filter_can_bus_scenes=True.")
            can_bus_path = Path(can_bus_root)
            if not can_bus_path.exists():
                raise FileNotFoundError(f"CAN bus root not found: {can_bus_path}")
            available_can_scene_names = sorted({p.name[:10] for p in can_bus_path.glob("scene-*_*.json")})
            available_can_scene_names = set(available_can_scene_names)

        skipped_missing_can = 0
        skipped_missing_camera = 0
        for scene in scenes:
            scene_name = scene["name"]
            if filter_can_bus_scenes and scene_name not in available_can_scene_names:
                skipped_missing_can += 1
                continue

            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            if cam_channel not in first_sample["data"]:
                skipped_missing_camera += 1
                continue

            current_cam_token = first_sample["data"][cam_channel]
            scene_tokens: list[str] = []
            while current_cam_token:
                scene_tokens.append(current_cam_token)
                cam_data = self.nusc.get("sample_data", current_cam_token)
                current_cam_token = cam_data.get("next", "")

            if len(scene_tokens) >= self.seq_len:
                self.scene_names.append(scene_name)

            for i in range(len(scene_tokens) - self.seq_len + 1):
                tokens = scene_tokens[i : i + self.seq_len]
                center_idx = self.seq_len // 2
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

    def _load_frame_tensor(self, token: str) -> torch.Tensor:
        if self.image_cache_size > 0 and token in self._image_cache:
            frame = self._image_cache.pop(token)
            self._image_cache[token] = frame
            return frame

        img_path = Path(self.nusc.get_sample_data_path(token))
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with Image.open(img_path) as img:
            frame = self.transform(img.convert("L")).float()

        if self.image_cache_size > 0:
            self._image_cache[token] = frame
            if len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return frame

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

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = [self._load_frame_tensor(token) for token in sample["tokens"]]
        stacked_frames = torch.stack(frames)
        speed_label = self._compute_speed_label(sample["center_token"])

        return {
            "images": stacked_frames,
            "speed": torch.tensor([speed_label], dtype=torch.float32),
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "scene_name": sample["scene_name"],
            "center_token": sample["center_token"],
            "center_timestamp": torch.tensor(sample["center_timestamp"], dtype=torch.long),
        }
