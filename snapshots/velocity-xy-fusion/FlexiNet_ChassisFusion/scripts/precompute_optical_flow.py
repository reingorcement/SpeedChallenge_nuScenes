from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_NUSCENES_ROOT = "/media/ps/AC78E69278E65B1A/zy/nuscenes"
DEFAULT_FLOW_ROOT = f"{DEFAULT_NUSCENES_ROOT}/precomputed_flow/CAM_FRONT"


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("L").resize(size)
        return np.asarray(image)


def _make_cuda_farneback(cv2):
    if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() <= 0:
        return None
    if not hasattr(cv2, "cuda_FarnebackOpticalFlow"):
        return None
    return cv2.cuda_FarnebackOpticalFlow.create(
        3,      # numLevels
        0.5,    # pyrScale
        False,  # fastPyramids
        15,     # winSize
        3,      # numIters
        5,      # polyN
        1.2,    # polySigma
        0,      # flags
    )


def _calc_flow_cpu(cv2, frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
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


def _calc_flow_cuda(cv2, farneback, frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    frame_gpu = cv2.cuda_GpuMat()
    next_frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame)
    next_frame_gpu.upload(next_frame)
    flow_gpu = farneback.calc(frame_gpu, next_frame_gpu, None)
    return flow_gpu.download()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute adjacent-frame optical flow for nuScenes camera frames.")
    parser.add_argument("--dataroot", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--cam_channel", default="CAM_FRONT")
    parser.add_argument("--output_root", default=DEFAULT_FLOW_ROOT)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Recompute files that already exist")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of adjacent pairs to process")
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

    cuda_farneback = None
    if args.device in {"auto", "cuda"}:
        cuda_farneback = _make_cuda_farneback(cv2)
        if cuda_farneback is None and args.device == "cuda":
            raise RuntimeError("CUDA Farneback optical flow is unavailable in this OpenCV build.")

    device_name = "cuda" if cuda_farneback is not None else "cpu"
    print(f"Optical-flow backend: {device_name}")
    if device_name == "cuda":
        print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

    pairs: list[tuple[str, str]] = []
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
            out_path = output_root / f"{token}.npy"
            if args.overwrite or not out_path.is_file():
                pairs.append((token, next_token))
                if args.limit > 0 and len(pairs) >= args.limit:
                    break
            token = next_token
        if args.limit > 0 and len(pairs) >= args.limit:
            break

    saved = 0
    for token, next_token in tqdm(pairs, desc="Precompute flow", unit="pair"):
        frame = _load_gray(Path(nusc.get_sample_data_path(token)), size)
        next_frame = _load_gray(Path(nusc.get_sample_data_path(next_token)), size)
        if cuda_farneback is not None:
            flow = _calc_flow_cuda(cv2, cuda_farneback, frame, next_frame)
        else:
            flow = _calc_flow_cpu(cv2, frame, next_frame)
        np.save(output_root / f"{token}.npy", flow.astype(np.float32))
        saved += 1

    skipped = "unknown" if args.overwrite else "existing files skipped"
    print(f"Saved {saved} optical-flow files to {output_root} ({skipped})")


if __name__ == "__main__":
    main()
