from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.v2_splits import partition_scene_names  # noqa: E402


def target_size(source_size: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    source_h, source_w = source_size
    if width <= 0 and height <= 0:
        raise ValueError("Set --width or --height.")
    if width <= 0:
        width = round(height * source_w / source_h / 8) * 8
    if height <= 0:
        height = round(width * source_h / source_w / 8) * 8
    source_ratio = source_w / source_h
    target_ratio = width / height
    if abs(target_ratio / source_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Requested {width}x{height} changes aspect ratio from {source_ratio:.4f} to {target_ratio:.4f}."
        )
    if width % 8 or height % 8:
        raise ValueError("RAFT target dimensions must be divisible by 8.")
    return height, width


def load_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return TF.resize(tensor, list(size), antialias=True)


def warp_backward_flow(backward: torch.Tensor, forward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = forward.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=forward.device, dtype=forward.dtype),
        torch.arange(width, device=forward.device, dtype=forward.dtype),
        indexing="ij",
    )
    target_x = x[None] + forward[:, 0]
    target_y = y[None] + forward[:, 1]
    grid_x = 2.0 * target_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * target_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(backward, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
    return warped, valid[:, None]


def consistency_confidence(forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
    backward_warped, valid = warp_backward_flow(backward, forward)
    error = torch.linalg.vector_norm(forward + backward_warped, dim=1, keepdim=True)
    scale = 0.5 + 0.01 * torch.linalg.vector_norm(forward, dim=1, keepdim=True)
    return (torch.exp(-error / scale) * valid).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute aspect-ratio-preserving RAFT flow and confidence.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--split_manifest")
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["train_core", "dev", "stage1_test"],
    )
    parser.add_argument("--cam_channel", default="CAM_FRONT")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--model", choices=["raft_small", "raft_large"], default="raft_large")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("V2 RAFT preprocessing is server-only and requires CUDA.")
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if args.model == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    transforms = weights.transforms()
    device = torch.device("cuda")
    model.eval().to(device)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    if args.split_manifest:
        scene_names = set()
        for partition in args.partitions:
            scene_names.update(
                partition_scene_names(
                    args.split_manifest,
                    partition,
                    expected_version=args.version,
                )
            )
    else:
        split_scenes = create_splits_scenes()
        scene_names = set(split_scenes.get(args.split, [scene["name"] for scene in nusc.scene]))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = []
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        sample = nusc.get("sample", scene["first_sample_token"])
        token = sample["data"].get(args.cam_channel, "")
        while token:
            source = nusc.get("sample_data", token)
            next_token = source.get("next", "")
            if not next_token:
                break
            output = output_root / f"{token}.pt"
            if args.overwrite or not output.is_file():
                pairs.append((token, next_token, output))
            if args.limit and len(pairs) >= args.limit:
                break
            token = next_token
        if args.limit and len(pairs) >= args.limit:
            break

    for source_token, target_token, output in tqdm(pairs, desc="V2 RAFT flow", unit="pair"):
        source_sd = nusc.get("sample_data", source_token)
        original_size = (int(source_sd["height"]), int(source_sd["width"]))
        size = target_size(original_size, args.width, args.height)
        source = load_rgb(Path(nusc.get_sample_data_path(source_token)), size)[None].to(device)
        target = load_rgb(Path(nusc.get_sample_data_path(target_token)), size)[None].to(device)
        source_input, target_input = transforms(source, target)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            forward = model(source_input, target_input)[-1].float()
            backward = model(target_input, source_input)[-1].float()
            confidence = consistency_confidence(forward, backward)
        torch.save(
            {
                "flow": forward[0].cpu().half(),
                "confidence": confidence[0].cpu().half(),
                "source_size": original_size,
                "target_size": size,
                "backend": args.model,
                "source_token": source_token,
                "target_token": target_token,
            },
            output,
        )
    print(f"Saved {len(pairs)} V2 flow packages to {output_root}")


if __name__ == "__main__":
    main()
