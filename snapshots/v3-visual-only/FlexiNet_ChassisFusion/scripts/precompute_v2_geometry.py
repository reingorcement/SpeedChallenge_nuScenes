from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_geometry_cache import GEOMETRY_CACHE_VERSION, geometry_cache_fingerprint
from model.v2 import TimedSignal, V2NuScenesVelocityModel
from train_v2 import build_dataset, build_geometry_cache_contract, load_config


class GeometryInputView(Dataset):
    def __init__(self, adapter, indices: list[int]) -> None:
        self.adapter = adapter
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        index = self.indices[item]
        sample = self.adapter.base_dataset.samples[index]
        geometry = self.adapter.online_geometry_inputs(index)
        values, valid, ages, *_ = self.adapter._build_kinematic_sequence(
            sample["scene_name"], sample["center_timestamp"]
        )
        yaw_rate = values[-1, 0]
        rotation_ego_from_camera = geometry["rotation_ego_from_camera"]
        angular_rate_ego = torch.stack(
            (yaw_rate.new_zeros(()), yaw_rate.new_zeros(()), yaw_rate)
        )
        geometry.update(
            {
                "angular_rate_camera": rotation_ego_from_camera.T @ angular_rate_ego,
                "angular_rate_valid": torch.tensor(
                    [False, False, bool(valid[-1, 0])], dtype=torch.bool
                ),
                "angular_rate_age_sec": ages[-1, 0] * self.adapter._field_max_age_sec[0],
                "scene_name": sample["scene_name"],
                "center_token": sample["center_token"],
            }
        )
        return geometry


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_scene(root: Path, scene_name: str, records: dict[str, list], tokens: list[str]) -> None:
    package = {
        "velocity": torch.stack(records["velocity"]).float(),
        "covariance": torch.stack(records["covariance"]).float(),
        "condition_number": torch.stack(records["condition_number"]).float(),
        "residual_rms": torch.stack(records["residual_rms"]).float(),
        "valid": torch.stack(records["valid"]).bool(),
        "support_count": torch.stack(records["support_count"]).long(),
    }
    temporary = root / f"{scene_name}.pt.tmp"
    torch.save(package, temporary)
    temporary.replace(root / f"{scene_name}.pt")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute deterministic V2 road geometry.")
    parser.add_argument("--config", default="configs/velocity_xy_v2_visual.yaml")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--partitions", nargs="+", default=["train_core", "dev", "stage1_test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--frame_cache_root")
    parser.add_argument("--split_manifest")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V2 geometry precomputation requires CUDA.")

    cfg = load_config(args.config)
    for key, value in (
        ("dataroot", args.dataroot),
        ("can_bus_root", args.can_bus_root),
        ("flow_root", args.flow_root),
        ("frame_cache_root", args.frame_cache_root),
        ("split_manifest_path", args.split_manifest),
    ):
        if value:
            cfg["data"][key] = value
    cfg["data"]["geometry_cache_mode"] = "off"
    cfg["data"]["geometry_cache_root"] = None
    cfg["data"]["photometric_augmentation"] = {"enabled": False}
    contract = build_geometry_cache_contract(cfg)
    fingerprint = geometry_cache_fingerprint(contract)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "version": GEOMETRY_CACHE_VERSION,
        "fingerprint": fingerprint,
        "contract": contract,
        "scenes": {},
    }
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint or existing.get("contract") != contract:
            raise ValueError("Existing geometry cache has a different contract; use a new root.")
        manifest = existing

    device = torch.device("cuda")
    model = V2NuScenesVelocityModel(
        mode="geometry_only",
        sensor_max_age_sec=float(cfg["model"].get("sensor_max_age_sec", 0.25)),
        output_reference=str(cfg["model"].get("output_reference", "ego_reference")),
    ).to(device).eval()

    for partition in args.partitions:
        dataset = build_dataset(cfg, partition)
        existing_scenes = {
            scene
            for scene in manifest["scenes"]
            if (output_root / f"{scene}.pt").is_file()
        }
        indices = [
            index
            for index, sample in enumerate(dataset.base_dataset.samples)
            if sample["scene_name"] not in existing_scenes
        ]
        if not indices:
            print(f"Geometry cache partition {partition}: all scenes already complete.")
            continue
        expected_counts = Counter(
            dataset.base_dataset.samples[index]["scene_name"] for index in indices
        )
        loader = DataLoader(
            GeometryInputView(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        records: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        tokens: dict[str, list[str]] = defaultdict(list)
        for raw_batch in tqdm(loader, desc=f"Geometry cache {partition}"):
            batch_size = len(raw_batch["scene_name"])
            gyro = TimedSignal(
                value=raw_batch["angular_rate_camera"].to(device, non_blocking=True),
                valid=raw_batch["angular_rate_valid"].to(device, non_blocking=True),
                age_sec=raw_batch["angular_rate_age_sec"].to(device, non_blocking=True),
            )
            output = model(
                images=torch.zeros(batch_size, 1, device=device),
                flow=raw_batch["flow_v2"].to(device, non_blocking=True),
                flow_confidence=raw_batch["flow_confidence"].to(device, non_blocking=True),
                intrinsics=raw_batch["intrinsics"].to(device, non_blocking=True),
                frame_dt=raw_batch["frame_dt"].to(device, non_blocking=True),
                angular_rate_camera=gyro,
                plane_normal_camera=raw_batch["plane_normal_camera"].to(device, non_blocking=True),
                plane_offset=raw_batch["plane_offset"].to(device, non_blocking=True),
                rotation_ego_from_camera=raw_batch["rotation_ego_from_camera"].to(
                    device, non_blocking=True
                ),
                road_mask=raw_batch["road_mask"].to(device, non_blocking=True),
            )
            fields = {
                "velocity": output.geometry_velocity,
                "covariance": output.geometry_covariance,
                "condition_number": output.geometry_condition_number,
                "residual_rms": output.geometry_residual_rms,
                "valid": output.geometry_valid,
                "support_count": output.geometry_support_count,
            }
            for index, scene_name in enumerate(raw_batch["scene_name"]):
                tokens[scene_name].append(raw_batch["center_token"][index])
                for field, tensor in fields.items():
                    records[scene_name][field].append(tensor[index].detach().cpu())
                if len(tokens[scene_name]) == expected_counts[scene_name]:
                    _write_scene(output_root, scene_name, records.pop(scene_name), tokens[scene_name])
                    manifest["scenes"][scene_name] = tokens.pop(scene_name)
                    _atomic_json(manifest_path, manifest)
        if records:
            raise RuntimeError(f"Incomplete scene buffers after {partition}: {sorted(records)}")
    print(
        json.dumps(
            {
                "scene_count": len(manifest["scenes"]),
                "sample_count": sum(len(tokens) for tokens in manifest["scenes"].values()),
                "fingerprint": fingerprint,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
