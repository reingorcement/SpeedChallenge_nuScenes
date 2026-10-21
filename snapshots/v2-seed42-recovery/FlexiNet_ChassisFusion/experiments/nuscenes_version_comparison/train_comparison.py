from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HOST_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_PROJECT_ROOT))

from experiments.nuscenes_version_comparison.bootstrap import SOURCE_PROJECT_ROOT  # noqa: E402

PROJECT_ROOT = SOURCE_PROJECT_ROOT

from experiments.nuscenes_version_comparison.adapters.dataset_adapter import (  # noqa: E402
    build_legacy_dataset,
    write_training_domain_manifest,
)
from experiments.nuscenes_version_comparison.adapters.legacy_adapter import (  # noqa: E402
    build_legacy_adapter,
)
from experiments.nuscenes_version_comparison.config import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    load_resolved_config,
    resolve_project_path,
    write_yaml,
)
from scripts.path_sanity_check import build_optimizer  # noqa: E402
from train import _loss_from_batch, build_criterion  # noqa: E402
from train_v2 import move_batch  # noqa: E402
from utils.seed import set_seed  # noqa: E402


TRAINABLE_ALGORITHMS = (
    "legacy_kd",
    "legacy_vkd",
    "v2_visual",
    "v2_kinematic_pretrain",
    "v2_visual_kinematic",
)


def _path_overrides(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "dataroot": args.dataroot,
        "can_bus_root": args.can_bus_root,
        "flow_root": args.flow_root,
        "frame_cache_root": args.frame_cache_root,
        "geometry_cache_root": args.geometry_cache_root,
        "split_manifest_path": args.split_manifest,
        "kinematic_stats_path": args.kinematic_stats,
    }


def _run_paths(protocol: dict[str, Any], run_id: str, algorithm: str, seed: int):
    output_root = resolve_project_path(protocol["paths"]["output_root"])
    checkpoint_root = resolve_project_path(protocol["paths"]["checkpoint_root"])
    output_dir = output_root / run_id / "train" / algorithm / f"seed_{seed}"
    checkpoint_dir = checkpoint_root / run_id / algorithm / f"seed_{seed}"
    return output_dir, checkpoint_dir


def _criterion(config: dict[str, Any]) -> torch.nn.Module:
    train = config["train"]
    return build_criterion(
        str(train["loss"]),
        sigma_vx=float(train.get("sigma_vx", 1.0)),
        sigma_vy=float(train.get("sigma_vy", 1.0)),
        speed_weight=float(train.get("speed_loss_weight", 0.0)),
        beta_loss_weight=float(train.get("beta_loss_weight", 0.0)),
        prior_loss_weight=float(train.get("prior_loss_weight", 0.0)),
        smoothness_loss_weight=float(train.get("smoothness_loss_weight", 0.0)),
        consistency_loss_weight=float(train.get("consistency_loss_weight", 0.0)),
        physics_prior_loss_weight=float(train.get("physics_prior_loss_weight", 0.0)),
        physics_correction_loss_weight=float(train.get("physics_correction_loss_weight", 0.0)),
        beta_vx_eps=float(train.get("beta_vx_eps", 0.5)),
    )


def _loader(dataset, config: dict[str, Any], device: torch.device, shuffle: bool) -> DataLoader:
    workers = int(config["train"].get("num_workers", 8))
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(config["train"].get("prefetch_factor", 2)) if workers else None,
    )


def _prediction_rows(raw_batch: dict[str, Any], prediction: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    target = raw_batch["velocity"]
    for index in range(prediction.shape[0]):
        error = prediction[index].cpu() - target[index].cpu()
        rows.append(
            {
                "scene_name": raw_batch["scene_name"][index],
                "center_token": raw_batch["center_token"][index],
                "target_vx_mps": float(target[index, 0]),
                "target_vy_mps": float(target[index, 1]),
                "pred_vx_mps": float(prediction[index, 0]),
                "pred_vy_mps": float(prediction[index, 1]),
                "endpoint_error_mps": float(torch.linalg.vector_norm(error)),
            }
        )
    return rows


@torch.inference_mode()
def _evaluate_legacy(adapter, loader, criterion, device, max_batches: int = 0):
    adapter.eval()
    rows: list[dict[str, Any]] = []
    weighted_loss = 0.0
    count = 0
    for step, raw_batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        prediction, _ = adapter(batch)
        loss = _loss_from_batch(
            criterion,
            prediction,
            batch["velocity"],
            batch["base_velocity"],
            batch,
            device,
            adapter.model,
        )
        batch_size = int(prediction.shape[0])
        weighted_loss += float(loss) * batch_size
        count += batch_size
        rows.extend(_prediction_rows(raw_batch, prediction.detach().cpu()))
    if not rows:
        raise RuntimeError("Legacy validation produced no samples.")
    endpoints = torch.tensor([row["endpoint_error_mps"] for row in rows])
    error_vx = torch.tensor([row["pred_vx_mps"] - row["target_vx_mps"] for row in rows])
    error_vy = torch.tensor([row["pred_vy_mps"] - row["target_vy_mps"] for row in rows])
    return {
        "loss": weighted_loss / max(count, 1),
        "sample_count": count,
        "selection_metric_mps": float(endpoints.mean()),
        "mae_vx_mps": float(error_vx.abs().mean()),
        "rmse_vx_mps": float(error_vx.square().mean().sqrt()),
        "mae_vy_mps": float(error_vy.abs().mean()),
        "rmse_vy_mps": float(error_vy.square().mean().sqrt()),
        "endpoint_error_mps": float(endpoints.mean()),
    }, rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_legacy(
    config: dict[str, Any],
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    domain: str,
    smoke: bool,
) -> None:
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not smoke:
        raise RuntimeError("Formal nuScenes comparison training requires CUDA.")
    train_set = build_legacy_dataset(config, "train_core", domain=domain)
    dev_set = build_legacy_dataset(config, "dev", domain=domain)
    train_loader = _loader(train_set, config, device, shuffle=True)
    dev_loader = _loader(dev_set, config, device, shuffle=False)
    adapter = build_legacy_adapter(config, load_pretrained=True, project_root=PROJECT_ROOT).to(device)
    criterion = _criterion(config).to(device)
    optimizer = build_optimizer(config, adapter.model)
    epochs = 1 if smoke else int(config["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    use_amp = bool(config["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_limit = 2 if smoke else int(config["train"].get("limit_train_batches", 0))
    eval_limit = 2 if smoke else int(config["train"].get("limit_eval_batches", 0))
    grad_clip = float(config["train"].get("grad_clip", 2.0))
    patience = int(config["train"].get("early_stopping_patience", 4))
    min_delta = float(config["train"].get("early_stopping_min_delta", 1.0e-4))
    history = []
    best_score = math.inf
    no_improve = 0

    for epoch in range(epochs):
        adapter.train()
        running_loss = 0.0
        batch_count = 0
        bar = tqdm(train_loader, desc=f"{config['_comparison']['name']} {epoch + 1}/{epochs}")
        for step, raw_batch in enumerate(bar):
            if train_limit and step >= train_limit:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction, _ = adapter(batch)
                loss = _loss_from_batch(
                    criterion,
                    prediction,
                    batch["velocity"],
                    batch["base_velocity"],
                    batch,
                    device,
                    adapter.model,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite legacy loss at epoch={epoch + 1}, batch={step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            batch_count += 1
        scheduler.step()
        metrics, rows = _evaluate_legacy(adapter, dev_loader, criterion, device, eval_limit)
        metrics.update(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(batch_count, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history.append(metrics)
        score = float(metrics["selection_metric_mps"])
        improved = score < best_score - min_delta
        if improved:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
        checkpoint = {
            "model": adapter.model.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
            "config": config,
            "comparison": config["_comparison"],
        }
        torch.save(checkpoint, checkpoint_dir / "last.pth")
        if improved:
            torch.save(checkpoint, checkpoint_dir / "best.pth")
            (output_dir / "best_eval").mkdir(exist_ok=True)
            (output_dir / "best_eval" / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            _write_rows(output_dir / "best_eval" / "predictions.csv", rows)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if patience and no_improve >= patience:
            break
    if not (checkpoint_dir / "best.pth").is_file():
        raise RuntimeError("Legacy training produced no best checkpoint.")


def _train_v2(
    config: dict[str, Any],
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    smoke: bool,
) -> None:
    resolved_path = output_dir / "input_config.yaml"
    write_yaml(resolved_path, config)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_v2.py"),
        "--config",
        str(resolved_path),
        "--save_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(output_dir),
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one algorithm under the comparison protocol.")
    parser.add_argument("--algorithm", choices=TRAINABLE_ALGORITHMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", choices=["all", "boston"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--visual-baseline-metrics")
    parser.add_argument("--dataroot")
    parser.add_argument("--can-bus-root")
    parser.add_argument("--flow-root")
    parser.add_argument("--frame-cache-root")
    parser.add_argument("--geometry-cache-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--kinematic-stats")
    args = parser.parse_args()

    config, protocol, _ = load_resolved_config(
        args.algorithm,
        protocol_path=args.protocol,
        seed=args.seed,
        path_overrides=_path_overrides(args),
    )
    output_dir, checkpoint_dir = _run_paths(protocol, args.run_id, args.algorithm, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
        if "effective_batch_size" in config["train"]:
            config["train"]["effective_batch_size"] = max(
                int(config["train"]["effective_batch_size"]), args.batch_size
            )
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.max_epochs:
        config["train"]["epochs"] = args.max_epochs
    if args.init_checkpoint:
        config["model"]["init_checkpoint"] = args.init_checkpoint
    if args.visual_baseline_metrics:
        config["train"]["visual_baseline_metrics"] = args.visual_baseline_metrics
    if args.domain != "all" and config["_comparison"]["family"] == "v2":
        filtered = output_dir / f"split_manifest_train_{args.domain}.json"
        config["data"]["split_manifest_path"] = str(
            write_training_domain_manifest(
                config["data"]["split_manifest_path"],
                filtered,
                version=str(config["data"]["version"]),
                domain=args.domain,
            )
        )
        # The geometry-cache contract hashes training statistics. Domain-only
        # normalization therefore requires online geometry unless a matching
        # domain cache is generated explicitly.
        config["data"]["geometry_cache_mode"] = "off"
        config["data"]["geometry_cache_root"] = None
    write_yaml(output_dir / "comparison_config.yaml", config)
    if config["_comparison"]["family"] == "legacy":
        _train_legacy(config, output_dir, checkpoint_dir, domain=args.domain, smoke=args.smoke)
    else:
        _train_v2(config, output_dir, checkpoint_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
