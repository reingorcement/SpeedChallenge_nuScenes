from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_NUSCENES_ROOT = "/media/ps/AC78E69278E65B1A/zy/nuscenes"


def _scene_names_for_split(version: str, split: str) -> set[str] | None:
    """Return the set of scene names for a split, or ``None`` to mean 'all scenes'."""
    from nuscenes.utils.splits import create_splits_scenes

    split = split.lower()
    if split in {"all", "trainval"}:
        return None
    splits = create_splits_scenes()
    alias = {"train": "train", "val": "val", "test": "test"}
    if version == "v1.0-mini":
        alias = {"train": "mini_train", "val": "mini_val"}
    if split in alias:
        return set(splits[alias[split]])
    if split in splits:
        return set(splits[split])
    raise ValueError(f"Unsupported split {split!r} for version {version!r}.")


def _build_transform(height: int, width: int, image_mode: str):
    from torchvision import transforms

    steps = [transforms.Resize((height, width))]
    if image_mode == "gray":
        steps.append(transforms.Grayscale(num_output_channels=1))
    else:
        steps.append(transforms.Lambda(lambda img: img.convert("RGB")))
    return transforms.Compose(steps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute resized nuScenes camera frames (and velocity labels) to disk."
    )
    parser.add_argument("--dataroot", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="trainval", help="train/val/test/trainval/all")
    parser.add_argument("--cam_channel", default="CAM_FRONT")
    parser.add_argument("--output_root", default=None, help="Defaults to <dataroot>/precomputed_frames/<cam>_<h>x<w>_<mode>")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--image_mode", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--no_labels", action="store_true", help="Skip precomputing velocity labels")
    parser.add_argument("--overwrite", action="store_true", help="Recompute scenes whose .npy already exists")
    parser.add_argument("--limit_scenes", type=int, default=0, help="Optional max number of scenes (debug)")
    args = parser.parse_args()

    try:
        from PIL import Image
        from nuscenes.nuscenes import NuScenes
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script requires pillow, torchvision and nuscenes-devkit on the server.") from exc

    from dataset.nuscenes_chassis import _rotate_global_velocity_to_ego

    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(args.dataroot) / "precomputed_frames" / f"{args.cam_channel}_{args.height}x{args.width}_{args.image_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    scene_filter = _scene_names_for_split(args.version, args.split)
    transform = _build_transform(args.height, args.width, args.image_mode)

    scenes = [
        scene
        for scene in nusc.scene
        if scene_filter is None or scene["name"] in scene_filter
    ]
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]

    # Incrementally merge with an existing cache in the same directory so multiple
    # runs (e.g. one per split, or resuming an interrupted run) accumulate instead
    # of overwriting each other's manifest/labels.
    manifest_path = output_root / "manifest.json"
    labels_path = output_root / "labels.json"
    existing_scenes: dict[str, list[str]] = {}
    existing_labels: dict[str, list[float]] = {}
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open("r", encoding="utf-8") as f:
            prior = json.load(f)
        prior_cfg = (
            int(prior.get("img_height", -1)),
            int(prior.get("img_width", -1)),
            str(prior.get("image_mode")),
            str(prior.get("version")),
            str(prior.get("cam_channel")),
        )
        this_cfg = (args.height, args.width, args.image_mode, args.version, args.cam_channel)
        if prior_cfg != this_cfg:
            raise ValueError(
                f"Existing cache at {output_root} was built with {prior_cfg} but this run uses "
                f"{this_cfg}. Use a different --output_root or pass --overwrite."
            )
        existing_scenes = prior.get("scenes", {})
        if labels_path.is_file():
            with labels_path.open("r", encoding="utf-8") as f:
                existing_labels = json.load(f)

    manifest_scenes: dict[str, list[str]] = dict(existing_scenes)
    labels: dict[str, list[float]] = dict(existing_labels)
    saved_scenes = 0
    total_frames = 0

    for scene in tqdm(scenes, desc="Precompute frames", unit="scene"):
        scene_name = scene["name"]
        scene_path = output_root / f"{scene_name}.npy"
        if not args.overwrite and scene_path.is_file() and scene_name in existing_scenes:
            total_frames += len(existing_scenes[scene_name])
            continue

        first_sample = nusc.get("sample", scene["first_sample_token"])
        if args.cam_channel not in first_sample["data"]:
            continue

        tokens: list[str] = []
        token = first_sample["data"][args.cam_channel]
        while token:
            tokens.append(token)
            token = nusc.get("sample_data", token).get("next", "")
        if not tokens:
            continue
        manifest_scenes[scene_name] = tokens

        if not args.no_labels:
            for tok in tokens:
                sd = nusc.get("sample_data", tok)
                prev_token = sd.get("prev", "")
                next_token = sd.get("next", "")
                if not prev_token or not next_token:
                    continue
                prev_sd = nusc.get("sample_data", prev_token)
                next_sd = nusc.get("sample_data", next_token)
                prev_pose = nusc.get("ego_pose", prev_sd["ego_pose_token"])
                center_pose = nusc.get("ego_pose", sd["ego_pose_token"])
                next_pose = nusc.get("ego_pose", next_sd["ego_pose_token"])
                dt = (next_pose["timestamp"] - prev_pose["timestamp"]) / 1e6
                if dt <= 0.0:
                    continue
                vx_global = (next_pose["translation"][0] - prev_pose["translation"][0]) / dt
                vy_global = (next_pose["translation"][1] - prev_pose["translation"][1]) / dt
                vx_ego, vy_ego = _rotate_global_velocity_to_ego(vx_global, vy_global, center_pose["rotation"])
                labels[tok] = [float(vx_ego), float(vy_ego)]

        frames = []
        for tok in tokens:
            img_path = Path(nusc.get_sample_data_path(tok))
            with Image.open(img_path) as img:
                processed = transform(img)
                frames.append(np.asarray(processed, dtype=np.uint8))
        stacked = np.stack(frames, axis=0)
        np.save(scene_path, stacked)
        saved_scenes += 1
        total_frames += stacked.shape[0]

    manifest = {
        "version": args.version,
        "cam_channel": args.cam_channel,
        "img_height": args.height,
        "img_width": args.width,
        "image_mode": args.image_mode,
        "num_scenes": len(manifest_scenes),
        "num_frames": total_frames,
        "scenes": manifest_scenes,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    if not args.no_labels:
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(labels, f)

    print(
        f"Saved {saved_scenes} scene arrays ({total_frames} frames) to {output_root}. "
        f"Labels: {'skipped' if args.no_labels else len(labels)}."
    )


if __name__ == "__main__":
    main()
