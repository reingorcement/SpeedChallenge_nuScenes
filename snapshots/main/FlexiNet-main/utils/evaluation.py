from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class VisionEvaluationResult:
    loss: float
    preds: list[float]
    targets: list[float]
    sample_indices: list[int]
    scene_names: list[str]
    center_tokens: list[str]
    center_timestamps: list[int]


def mean_absolute_error(preds: Sequence[float], targets: Sequence[float]) -> float:
    if not preds:
        return 0.0
    return sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)


def root_mean_squared_error(preds: Sequence[float], targets: Sequence[float]) -> float:
    if not preds:
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds))


def r2_score(preds: Sequence[float], targets: Sequence[float]) -> float:
    if len(preds) < 2:
        return 0.0
    target_mean = sum(targets) / len(targets)
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, targets))
    ss_tot = sum((t - target_mean) ** 2 for t in targets)
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def build_criterion(name: str) -> nn.Module:
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name in {"smoothl1", "huber"}:
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss: {name}")


def compute_metrics(preds: Sequence[float], targets: Sequence[float], loss: float | None = None) -> dict[str, float]:
    metrics = {
        "mae_mps": mean_absolute_error(preds, targets),
        "rmse_mps": root_mean_squared_error(preds, targets),
        "r2": r2_score(preds, targets),
    }
    metrics["mae_kmh"] = metrics["mae_mps"] * 3.6
    metrics["rmse_kmh"] = metrics["rmse_mps"] * 3.6
    if loss is not None:
        metrics["loss"] = loss
    return metrics


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> VisionEvaluationResult:
    model.eval()
    losses: list[float] = []
    preds: list[float] = []
    targets: list[float] = []
    sample_indices: list[int] = []
    scene_names: list[str] = []
    center_tokens: list[str] = []
    center_timestamps: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["images"].to(device).float()
            speed = batch["speed"].to(device).float()
            pred = model(images)
            loss = criterion(pred, speed)

            losses.append(loss.item())
            preds.extend(pred.squeeze(1).detach().cpu().tolist())
            targets.extend(speed.squeeze(1).detach().cpu().tolist())

            sample_indices.extend([int(v) for v in batch["sample_index"].detach().cpu().tolist()])
            scene_names.extend(list(batch["scene_name"]))
            center_tokens.extend(list(batch["center_token"]))
            center_timestamps.extend([int(v) for v in batch["center_timestamp"].detach().cpu().tolist()])

    return VisionEvaluationResult(
        loss=sum(losses) / max(len(losses), 1),
        preds=preds,
        targets=targets,
        sample_indices=sample_indices,
        scene_names=scene_names,
        center_tokens=center_tokens,
        center_timestamps=center_timestamps,
    )


def save_metrics_json(metrics: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions_csv(result: VisionEvaluationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "scene_name",
                "center_token",
                "center_timestamp",
                "target_speed_mps",
                "pred_speed_mps",
                "error_mps",
                "abs_error_mps",
            ],
        )
        writer.writeheader()
        for i, (pred, target) in enumerate(zip(result.preds, result.targets)):
            writer.writerow(
                {
                    "sample_index": _safe_get(result.sample_indices, i, i),
                    "scene_name": _safe_get(result.scene_names, i, ""),
                    "center_token": _safe_get(result.center_tokens, i, ""),
                    "center_timestamp": _safe_get(result.center_timestamps, i, ""),
                    "target_speed_mps": target,
                    "pred_speed_mps": pred,
                    "error_mps": pred - target,
                    "abs_error_mps": abs(pred - target),
                }
            )


def write_evaluation_artifacts(result: VisionEvaluationResult, output_dir: str | Path, prefix: str = "val") -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(result.preds, result.targets, loss=result.loss)
    save_metrics_json(metrics, output_dir / f"{prefix}_metrics.json")
    save_predictions_csv(result, output_dir / f"{prefix}_predictions.csv")
    plot_prediction_diagnostics(result, output_dir, prefix=prefix)
    return metrics


def plot_prediction_diagnostics(result: VisionEvaluationResult, output_dir: str | Path, prefix: str = "val") -> None:
    plt = _load_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = result.preds
    targets = result.targets
    errors = [p - t for p, t in zip(preds, targets)]
    abs_errors = [abs(e) for e in errors]

    _plot_pred_vs_target(plt, output_dir / f"{prefix}_pred_vs_target.png", preds, targets)
    _plot_scatter(
        plt,
        output_dir / f"{prefix}_residual_vs_speed.png",
        targets,
        errors,
        xlabel="Target speed (m/s)",
        ylabel="Prediction error (m/s)",
        title="Pure Vision Residual vs Target Speed",
        hline=0.0,
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_abs_error_histogram.png",
        abs_errors,
        xlabel="Absolute error (m/s)",
        title="Pure Vision Absolute Error Distribution",
    )


def copy_small_results_for_git(source_dir: str | Path, results_dir: str | Path, include_predictions: bool = False) -> None:
    source_dir = Path(source_dir)
    results_dir = Path(results_dir)
    if not source_dir.exists():
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".json", ".png", ".svg"}
    allowed_names = {"training_history.csv"}
    if include_predictions:
        allowed_names.add("val_predictions.csv")

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in allowed_suffixes and path.name not in allowed_names:
            continue
        target_path = results_dir / path.relative_to(source_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)


def _plot_pred_vs_target(plt, path: Path, preds: Sequence[float], targets: Sequence[float]) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, s=10, alpha=0.45, label="pure_vision")
    values = list(targets) + list(preds)
    min_v = min(values)
    max_v = max(values)
    plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.2, label="ideal")
    plt.xlabel("Target speed (m/s)")
    plt.ylabel("Predicted speed (m/s)")
    plt.title("Pure Vision Prediction vs Target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_scatter(
    plt,
    path: Path,
    x_values: Sequence[float],
    y_values: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    hline: float | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, s=10, alpha=0.45)
    if hline is not None:
        plt.axhline(hline, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_histogram(plt, path: Path, values: Sequence[float], xlabel: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, alpha=0.82)
    plt.xlabel(xlabel)
    plt.ylabel("Sample count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _safe_get(values: Sequence[object], index: int, default: object) -> object:
    return values[index] if index < len(values) else default


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for evaluation plots. Install it with: pip install matplotlib") from exc
    return plt
