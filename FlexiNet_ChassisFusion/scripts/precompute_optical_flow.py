from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(size)
        return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute adjacent-frame optical flow for nuScenes camera frames.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--cam_channel", default="CAM_FRONT")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()

    try:
        import cv2
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script requires opencv-python and nuscenes-devkit on the server.") from exc

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    splits = create_splits_scenes()
    scene_names = set(splits[args.split]) if args.split in splits else {scene["name"] for scene in nusc.scene}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    saved = 0
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        sample = nusc.get("sample", scene["first_sample_token"])
        if args.cam_channel not in sample["data"]:
            continue
        token = sample["data"][args.cam_channel]
        while token:
            sd = nusc.get("sample_data", token)
            next_token = sd.get("next", "")
            if not next_token:
                break
            next_sd = nusc.get("sample_data", next_token)
            frame = _load_gray(Path(nusc.get_sample_data_path(token)), size)
            next_frame = _load_gray(Path(nusc.get_sample_data_path(next_token)), size)
            flow = cv2.calcOpticalFlowFarneback(
                frame,
                next_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            np.save(output_root / f"{token}.npy", flow.astype(np.float32))
            saved += 1
            token = next_token

    print(f"Saved {saved} optical-flow files to {output_root}")


if __name__ == "__main__":
    main()
