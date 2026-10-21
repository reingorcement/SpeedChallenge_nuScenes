from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.v2 import V2VelocityLoss
from train_v2 import (
    _configure_trainable_modules,
    _load_initial_weights,
    _parameter_groups,
    _set_epoch_training_mode,
    build_dataset,
    build_model,
    forward_batch,
    load_config,
    move_batch,
)
from utils.seed import set_seed


def _profile(
    cfg: dict,
    workers: int,
    prefetch_factor: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, float | int]:
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda")
    dataset = build_dataset(cfg, "train_core")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    model = build_model(cfg)
    _load_initial_weights(model, cfg, "visual")
    _configure_trainable_modules(model, "visual")
    _set_epoch_training_mode(model, "visual", epoch=0, freeze_backbone_epochs=5)
    model.to(device)
    criterion = V2VelocityLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(_parameter_groups(model, cfg, "visual"), weight_decay=0.0)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=float(cfg["train"].get("amp_init_scale", 65536.0)),
        growth_interval=int(cfg["train"].get("amp_growth_interval", 10000)),
    )
    iterator = iter(loader)
    load_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    total_seconds = 0.0
    total_batches = warmup_batches + measured_batches
    for index in range(total_batches):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        raw_batch = next(iterator)
        load_end = time.perf_counter()
        transfer_start = time.perf_counter()
        batch = move_batch(raw_batch, device)
        torch.cuda.synchronize()
        transfer_end = time.perf_counter()
        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True):
            output = forward_batch(model, batch)
            loss = criterion(output, batch["velocity"]).total
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        compute_end = time.perf_counter()
        total_end = time.perf_counter()
        if index >= warmup_batches:
            load_seconds += load_end - load_start
            transfer_seconds += transfer_end - transfer_start
            compute_seconds += compute_end - compute_start
            total_seconds += total_end - total_start
    batch_size = int(cfg["train"]["batch_size"])
    return {
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "measured_batches": measured_batches,
        "batch_size": batch_size,
        "load_seconds_per_batch": load_seconds / measured_batches,
        "transfer_seconds_per_batch": transfer_seconds / measured_batches,
        "compute_seconds_per_batch": compute_seconds / measured_batches,
        "total_seconds_per_batch": total_seconds / measured_batches,
        "samples_per_second": measured_batches * batch_size / total_seconds,
    }


def select_profile_result(
    baseline: dict,
    cached_results: list[dict],
    min_speedup: float,
    prefetch_factor: int,
) -> dict:
    passed = [result for result in cached_results if result.get("status") == "passed"]
    if not passed:
        raise RuntimeError("Every cached DataLoader candidate failed.")
    selected = max(passed, key=lambda result: float(result["samples_per_second"]))
    speedup = float(selected["samples_per_second"]) / float(baseline["samples_per_second"])
    return {
        "baseline": baseline,
        "cached_candidates": cached_results,
        "selected_num_workers": int(selected["num_workers"]),
        "selected_prefetch_factor": prefetch_factor,
        "speedup": speedup,
        "minimum_required_speedup": min_speedup,
        "accepted": speedup >= min_speedup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile online and cached V2 training paths.")
    parser.add_argument("--config", default="configs/velocity_xy_v2_visual.yaml")
    parser.add_argument("--geometry_cache_root", required=True)
    parser.add_argument("--frame_cache_root", required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--warmup_batches", type=int, default=20)
    parser.add_argument("--measured_batches", type=int, default=200)
    parser.add_argument("--min_speedup", type=float, default=1.2)
    parser.add_argument("--output", default="logs/v2_throughput_profile.json")
    parser.add_argument("--dataroot")
    parser.add_argument("--can_bus_root")
    parser.add_argument("--flow_root")
    parser.add_argument("--split_manifest")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V2 throughput profiling requires CUDA.")

    cfg = load_config(args.config)
    for key, value in (
        ("dataroot", args.dataroot),
        ("can_bus_root", args.can_bus_root),
        ("flow_root", args.flow_root),
        ("split_manifest_path", args.split_manifest),
    ):
        if value:
            cfg["data"][key] = value
    cfg["model"]["mode"] = "visual"
    cfg["data"]["photometric_augmentation"] = {"enabled": True, "probability": 0.5}

    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["data"]["frame_cache_root"] = None
    baseline_cfg["data"]["geometry_cache_root"] = None
    baseline_cfg["data"]["geometry_cache_mode"] = "off"
    baseline = _profile(
        baseline_cfg,
        workers=8,
        prefetch_factor=2,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
    )
    gc.collect()
    torch.cuda.empty_cache()

    cached_results = []
    for workers in args.workers:
        cached_cfg = copy.deepcopy(cfg)
        cached_cfg["data"]["frame_cache_root"] = args.frame_cache_root
        cached_cfg["data"]["geometry_cache_root"] = args.geometry_cache_root
        cached_cfg["data"]["geometry_cache_mode"] = "required"
        try:
            result = _profile(
                cached_cfg,
                workers=workers,
                prefetch_factor=args.prefetch_factor,
                warmup_batches=args.warmup_batches,
                measured_batches=args.measured_batches,
            )
            result["status"] = "passed"
        except (RuntimeError, OSError) as error:
            result = {"num_workers": workers, "status": "failed", "error": str(error)}
        cached_results.append(result)
        gc.collect()
        torch.cuda.empty_cache()
    report = select_profile_result(
        baseline, cached_results, args.min_speedup, args.prefetch_factor
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise RuntimeError(
            f"Cached throughput speedup {report['speedup']:.3f}x is below "
            f"{args.min_speedup:.3f}x."
        )


if __name__ == "__main__":
    main()
