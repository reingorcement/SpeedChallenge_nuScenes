from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_v2 import build_dataset, build_model, forward_batch, load_config, move_batch


FIELDS = (
    "geometry_velocity",
    "geometry_covariance",
    "geometry_condition_number",
    "geometry_residual_rms",
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cached V2 geometry against online solves.")
    parser.add_argument("--config", default="configs/velocity_xy_v2_visual.yaml")
    parser.add_argument("--geometry_cache_root", required=True)
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/v2_geometry_cache_verification.json")
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--frame_cache_root")
    parser.add_argument("--split_manifest")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V2 geometry cache verification requires CUDA.")

    base_cfg = load_config(args.config)
    for key, value in (
        ("dataroot", args.dataroot),
        ("can_bus_root", args.can_bus_root),
        ("flow_root", args.flow_root),
        ("frame_cache_root", args.frame_cache_root),
        ("split_manifest_path", args.split_manifest),
    ):
        if value:
            base_cfg["data"][key] = value
    base_cfg["data"]["photometric_augmentation"] = {"enabled": False}
    base_cfg["model"]["mode"] = "geometry_only"

    online_cfg = copy.deepcopy(base_cfg)
    online_cfg["data"]["geometry_cache_mode"] = "off"
    online_cfg["data"]["geometry_cache_root"] = None
    cached_cfg = copy.deepcopy(base_cfg)
    cached_cfg["data"]["geometry_cache_mode"] = "required"
    cached_cfg["data"]["geometry_cache_root"] = args.geometry_cache_root
    online_dataset = build_dataset(online_cfg, args.partition)
    cached_dataset = build_dataset(cached_cfg, args.partition)
    if len(online_dataset) != len(cached_dataset):
        raise RuntimeError("Online and cached datasets have different lengths.")
    sample_count = min(args.samples, len(online_dataset))
    indices = sorted(random.Random(args.seed).sample(range(len(online_dataset)), sample_count))
    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
    }
    online_loader = DataLoader(Subset(online_dataset, indices), **loader_args)
    cached_loader = DataLoader(Subset(cached_dataset, indices), **loader_args)
    device = torch.device("cuda")
    model = build_model(online_cfg).to(device).eval()
    maxima = {field: 0.0 for field in FIELDS}

    for online_raw, cached_raw in zip(online_loader, cached_loader):
        if "flow_v2" in cached_raw or "cached_geometry" not in cached_raw:
            raise RuntimeError("Cached training batch still contains online geometry inputs.")
        online = move_batch(online_raw, device)
        cached = move_batch(cached_raw, device)
        online_output = forward_batch(model, online)
        cached_output = forward_batch(model, cached)
        for field in FIELDS:
            online_value = getattr(online_output, field)
            cached_value = getattr(cached_output, field)
            torch.testing.assert_close(
                cached_value, online_value, rtol=1.0e-5, atol=1.0e-5, equal_nan=True
            )
            finite = torch.isfinite(online_value) & torch.isfinite(cached_value)
            if bool(finite.any()):
                maxima[field] = max(
                    maxima[field],
                    float((cached_value[finite] - online_value[finite]).abs().max()),
                )
        if not torch.equal(cached_output.geometry_valid, online_output.geometry_valid):
            raise AssertionError("Cached geometry validity differs from online geometry.")
        if not torch.equal(
            cached_output.geometry_support_count, online_output.geometry_support_count
        ):
            raise AssertionError("Cached geometry support count differs from online geometry.")

    report = {
        "status": "passed",
        "partition": args.partition,
        "sample_count": sample_count,
        "rtol": 1.0e-5,
        "atol": 1.0e-5,
        "max_abs_difference": maxima,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
