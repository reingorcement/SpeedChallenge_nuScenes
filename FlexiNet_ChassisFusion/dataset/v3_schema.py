from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.v2_schema import (
    apply_sequence_photometric_augmentation,
    quaternion_to_rotation_matrix,
)
from model.v2.rotation_flow import scale_intrinsics


FORBIDDEN_VISUAL_KEYS = frozenset(
    {
        "angular_rate_camera",
        "base_speed",
        "base_velocity",
        "chassis",
        "geometry",
        "kinematic_age",
        "kinematic_valid",
        "kinematic_values",
        "physics_inputs",
        "wheel_speed",
    }
)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    trace = torch.diagonal(rotation).sum()
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if float(angle.abs()) < 1.0e-5:
        return 0.5 * vee
    return angle * vee / (2.0 * torch.sin(angle).clamp_min(1.0e-8))


class V3NuScenesVisualDataset(Dataset):
    """Strict visual-only adapter; pose records are targets/metadata, never inputs."""

    def __init__(
        self,
        base_dataset: Dataset,
        flow_root: str | Path,
        camera_height_m: float,
        pair_count: int = 3,
        geometry_size: tuple[int, int] = (288, 512),
        flow_backend: str | None = "raft_large",
        photometric_augmentation: dict[str, Any] | None = None,
        load_flows: bool = True,
    ) -> None:
        if not bool(getattr(base_dataset, "online_mode", False)):
            raise ValueError("V3 requires a causal base dataset with online_mode=True.")
        if getattr(base_dataset, "can_bus_adapter", None) is not None:
            raise ValueError(
                "V3 base dataset must be constructed without a CAN bus adapter."
            )
        if int(getattr(base_dataset, "seq_len", 0)) != 5:
            raise ValueError("V3 has a fixed causal seq_len=5 data contract.")
        if int(pair_count) != 3:
            raise ValueError(
                "V3 datasets always emit three flow pairs; ablate pairs in the model."
            )
        self.base_dataset = base_dataset
        self.flow_root = Path(flow_root)
        self.camera_height_m = float(camera_height_m)
        self.pair_count = int(pair_count)
        self.geometry_size = tuple(int(value) for value in geometry_size)
        self.flow_backend = flow_backend
        self.photometric_augmentation = photometric_augmentation
        self.load_flows = bool(load_flows)
        if self.load_flows and not self.flow_root.is_dir():
            raise FileNotFoundError(f"V3 flow root not found: {self.flow_root}")
        self._metadata = self._build_metadata() if self.load_flows else None
        if (
            bool(getattr(base_dataset, "_frame_cache_enabled", lambda: False)())
            and getattr(base_dataset, "_label_cache", None) is not None
        ):
            base_dataset.nusc = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {
            "base_dataset",
            "flow_root",
            "camera_height_m",
            "pair_count",
            "geometry_size",
            "flow_backend",
            "photometric_augmentation",
            "load_flows",
            "_metadata",
        }:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset.samples[index]
        if sample["tokens"][-1] != sample["center_token"]:
            raise ValueError("V3 samples must end at the causal center token.")
        images = torch.stack(
            [self.base_dataset._load_frame_tensor(token) for token in sample["tokens"]]
        )
        if images.ndim != 4 or images.shape[:2] != (5, 3):
            raise ValueError("V3 images must have shape [5,3,H,W].")
        images = self._augment(images)
        visual = {"images": images}
        target = {
            "velocity": self.base_dataset._compute_velocity_label(
                sample["center_token"]
            ),
        }
        if self.load_flows:
            if self._metadata is None:
                raise RuntimeError("V3 flow metadata was not initialized.")
            metadata = self._metadata[index]
            flows = []
            confidences = []
            for source_token, target_token in metadata["pair_tokens"]:
                flow, confidence = self._load_flow(source_token, target_token)
                flow, confidence = self._resize_flow(
                    flow, confidence, self.geometry_size
                )
                flows.append(flow)
                confidences.append(confidence)
            visual.update(
                {
                    "flows": torch.stack(flows),
                    "flow_confidence": torch.stack(confidences),
                    "pair_dt": metadata["pair_dt"],
                    "intrinsics": scale_intrinsics(
                        metadata["intrinsics"],
                        metadata["original_size"],
                        self.geometry_size,
                    ),
                    "plane_normal_camera": metadata["plane_normal_camera"],
                    "plane_offset": torch.tensor(
                        self.camera_height_m, dtype=torch.float32
                    ),
                    "rotation_ego_from_camera": metadata[
                        "rotation_ego_from_camera"
                    ],
                    "translation_ego_from_camera": metadata[
                        "translation_ego_from_camera"
                    ],
                }
            )
            target["angular_velocity_camera"] = metadata[
                "angular_velocity_camera"
            ]
        item = {
            "visual": visual,
            "target": target,
            "scene_name": sample["scene_name"],
            "center_token": sample["center_token"],
            "center_timestamp": torch.tensor(
                sample["center_timestamp"], dtype=torch.long
            ),
        }
        forbidden = FORBIDDEN_VISUAL_KEYS & (
            item.keys() | visual.keys() | target.keys()
        )
        if forbidden:
            raise RuntimeError(f"V3 emitted forbidden fields: {sorted(forbidden)}")
        return item

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.photometric_augmentation
        if not cfg or not bool(cfg.get("enabled", False)):
            return images
        if float(torch.rand(())) >= float(cfg.get("probability", 0.5)):
            return images

        def sample(bound: float) -> float:
            return 1.0 + (2.0 * float(torch.rand(())) - 1.0) * float(bound)

        hue = (2.0 * float(torch.rand(())) - 1.0) * float(cfg.get("hue", 0.02))
        gamma_range = cfg.get("gamma", [0.9, 1.1])
        gamma = float(gamma_range[0]) + float(torch.rand(())) * (
            float(gamma_range[1]) - float(gamma_range[0])
        )
        return apply_sequence_photometric_augmentation(
            images,
            sample(cfg.get("brightness", 0.1)),
            sample(cfg.get("contrast", 0.1)),
            sample(cfg.get("saturation", 0.05)),
            hue,
            gamma,
        )

    def _build_metadata(self) -> list[dict[str, Any]]:
        nusc = self.base_dataset.nusc
        if nusc is None:
            raise ValueError(
                "V3 metadata must be built before releasing nuScenes metadata."
            )
        upward_ego = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        result = []
        for sample in self.base_dataset.samples:
            tokens = sample["tokens"][-(self.pair_count + 1) :]
            pair_tokens = list(zip(tokens[:-1], tokens[1:]))
            pair_dt = []
            angular_velocity = []
            target_sd = nusc.get("sample_data", tokens[-1])
            calibrated = nusc.get(
                "calibrated_sensor", target_sd["calibrated_sensor_token"]
            )
            rotation_ego_from_camera = quaternion_to_rotation_matrix(
                calibrated["rotation"]
            )
            for source_token, target_token in pair_tokens:
                source_sd = nusc.get("sample_data", source_token)
                pair_target_sd = nusc.get("sample_data", target_token)
                dt = (
                    int(pair_target_sd["timestamp"]) - int(source_sd["timestamp"])
                ) / 1_000_000.0
                if dt <= 0.0:
                    raise ValueError(
                        f"Non-positive V3 frame interval for {source_token}."
                    )
                source_pose = nusc.get("ego_pose", source_sd["ego_pose_token"])
                target_pose = nusc.get("ego_pose", pair_target_sd["ego_pose_token"])
                source_world_from_camera = (
                    quaternion_to_rotation_matrix(source_pose["rotation"])
                    @ rotation_ego_from_camera
                )
                target_world_from_camera = (
                    quaternion_to_rotation_matrix(target_pose["rotation"])
                    @ rotation_ego_from_camera
                )
                relative_camera_rotation = (
                    source_world_from_camera.T @ target_world_from_camera
                )
                pair_dt.append(dt)
                angular_velocity.append(so3_log(relative_camera_rotation) / dt)
            result.append(
                {
                    "pair_tokens": pair_tokens,
                    "pair_dt": torch.tensor(pair_dt, dtype=torch.float32),
                    "angular_velocity_camera": torch.stack(angular_velocity),
                    "intrinsics": torch.tensor(
                        calibrated["camera_intrinsic"], dtype=torch.float32
                    ),
                    "original_size": (
                        int(target_sd["height"]),
                        int(target_sd["width"]),
                    ),
                    "rotation_ego_from_camera": rotation_ego_from_camera,
                    "translation_ego_from_camera": torch.tensor(
                        calibrated["translation"], dtype=torch.float32
                    ),
                    "plane_normal_camera": rotation_ego_from_camera.T @ upward_ego,
                }
            )
        return result

    def _load_flow(
        self, source_token: str, target_token: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.flow_root / f"{source_token}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing V3 flow package: {path}")
        package = torch.load(path, map_location="cpu")
        if not isinstance(package, dict) or "flow" not in package:
            raise ValueError(f"V3 flow package must contain a flow tensor: {path}")
        if "confidence" not in package:
            raise ValueError(f"V3 flow package must contain FB confidence: {path}")
        if (
            package.get("source_token") != source_token
            or package.get("target_token") != target_token
        ):
            raise ValueError(f"V3 flow token mismatch in {path}.")
        if self.flow_backend and package.get("backend") != self.flow_backend:
            raise ValueError(
                f"V3 flow backend mismatch in {path}: "
                f"{package.get('backend')!r} != {self.flow_backend!r}"
            )
        flow = torch.as_tensor(package["flow"], dtype=torch.float32)
        confidence = torch.as_tensor(package["confidence"], dtype=torch.float32)
        if flow.ndim == 3 and flow.shape[-1] == 2:
            flow = flow.permute(2, 0, 1)
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(0)
        if flow.ndim != 3 or flow.shape[0] != 2:
            raise ValueError(f"V3 flow must have shape [2,H,W]: {path}")
        package_size = package.get("target_size")
        if package_size is not None and tuple(
            int(value) for value in package_size
        ) != tuple(flow.shape[-2:]):
            raise ValueError(f"V3 flow size metadata mismatch in {path}.")
        if confidence.shape != (1, *flow.shape[-2:]):
            raise ValueError(f"V3 confidence must have shape [1,H,W]: {path}")
        return flow, confidence

    @staticmethod
    def _resize_flow(
        flow: torch.Tensor,
        confidence: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_h, source_w = flow.shape[-2:]
        target_h, target_w = target_size
        if (source_h, source_w) == target_size:
            return flow, confidence.clamp(0.0, 1.0)
        resized = F.interpolate(
            flow[None], target_size, mode="bilinear", align_corners=False
        )[0]
        resized[0] *= float(target_w) / float(source_w)
        resized[1] *= float(target_h) / float(source_h)
        resized_confidence = F.interpolate(
            confidence[None], target_size, mode="bilinear", align_corners=False
        )[0]
        return resized, resized_confidence.clamp(0.0, 1.0)
