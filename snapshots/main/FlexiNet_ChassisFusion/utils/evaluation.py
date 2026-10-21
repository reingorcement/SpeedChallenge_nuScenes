from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class EvaluationResult:
    loss: float
    preds: list[float]
    targets: list[float]
    base_speeds: list[float]
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


def compute_metrics(
    preds: Sequence[float],
    targets: Sequence[float],
    base_speeds: Sequence[float] | None = None,
    loss: float | None = None,
) -> dict[str, float]:
    metrics = {
        "mae_mps": mean_absolute_error(preds, targets),
        "rmse_mps": root_mean_squared_error(preds, targets),
        "r2": r2_score(preds, targets),
    }
    metrics["mae_kmh"] = metrics["mae_mps"] * 3.6
    metrics["rmse_kmh"] = metrics["rmse_mps"] * 3.6
    if loss is not None:
        metrics["loss"] = loss

    if base_speeds is not None and len(base_speeds) == len(targets):
        base_mae = mean_absolute_error(base_speeds, targets)
        base_rmse = root_mean_squared_error(base_speeds, targets)
        metrics["base_mae_mps"] = base_mae
        metrics["base_rmse_mps"] = base_rmse
        metrics["base_mae_kmh"] = base_mae * 3.6
        metrics["base_rmse_kmh"] = base_rmse * 3.6
        metrics["mae_gain_vs_base_mps"] = base_mae - metrics["mae_mps"]
        metrics["rmse_gain_vs_base_mps"] = base_rmse - metrics["rmse_mps"]
    return metrics


def save_metrics_json(metrics: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions_csv(result: EvaluationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_name",
                "center_token",
                "center_timestamp",
                "target_speed_mps",
                "pred_speed_mps",
                "base_speed_mps",
                "error_mps",
                "abs_error_mps",
                "base_error_mps",
                "base_abs_error_mps",
                "delta_v_mps",
            ],
        )
        writer.writeheader()
        for i, (pred, target, base_speed) in enumerate(zip(result.preds, result.targets, result.base_speeds)):
            writer.writerow(
                {
                    "scene_name": _safe_get(result.scene_names, i, ""),
                    "center_token": _safe_get(result.center_tokens, i, ""),
                    "center_timestamp": _safe_get(result.center_timestamps, i, ""),
                    "target_speed_mps": target,
                    "pred_speed_mps": pred,
                    "base_speed_mps": base_speed,
                    "error_mps": pred - target,
                    "abs_error_mps": abs(pred - target),
                    "base_error_mps": base_speed - target,
                    "base_abs_error_mps": abs(base_speed - target),
                    "delta_v_mps": pred - base_speed,
                }
            )


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_json(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_training_history(history: list[dict[str, float]], output_dir: str | Path) -> None:
    if not history:
        return
    plt = _load_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    _plot_lines(
        plt,
        output_dir / "train_val_loss_curve.png",
        epochs,
        {
            "train_loss": [row["train_loss"] for row in history],
            "val_loss": [row["val_loss"] for row in history],
        },
        ylabel="Loss",
        title="Training and Validation Loss",
    )
    _plot_lines(
        plt,
        output_dir / "val_mae_rmse_curve.png",
        epochs,
        {
            "val_mae_mps": [row["val_mae_mps"] for row in history],
            "val_rmse_mps": [row["val_rmse_mps"] for row in history],
            "base_mae_mps": [row["base_mae_mps"] for row in history],
        },
        ylabel="Speed Error (m/s)",
        title="Validation Error",
    )


def write_evaluation_artifacts(result: EvaluationResult, output_dir: str | Path, prefix: str = "val") -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(result.preds, result.targets, result.base_speeds, loss=result.loss)
    save_metrics_json(metrics, output_dir / f"{prefix}_metrics.json")
    save_predictions_csv(result, output_dir / f"{prefix}_predictions.csv")
    plot_prediction_diagnostics(result, output_dir, prefix=prefix)
    return metrics


def plot_prediction_diagnostics(result: EvaluationResult, output_dir: str | Path, prefix: str = "val") -> None:
    plt = _load_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = result.preds
    targets = result.targets
    base_speeds = result.base_speeds
    errors = [p - t for p, t in zip(preds, targets)]
    abs_errors = [abs(e) for e in errors]
    delta_v = [p - b for p, b in zip(preds, base_speeds)]

    _plot_pred_vs_target(plt, output_dir / f"{prefix}_pred_vs_target.png", preds, targets, base_speeds)
    _plot_scatter(
        plt,
        output_dir / f"{prefix}_residual_vs_speed.png",
        targets,
        errors,
        xlabel="Target speed (m/s)",
        ylabel="Prediction error (m/s)",
        title="Residual vs Target Speed",
        hline=0.0,
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_abs_error_histogram.png",
        abs_errors,
        xlabel="Absolute error (m/s)",
        title="Absolute Error Distribution",
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_delta_v_histogram.png",
        delta_v,
        xlabel="Predicted correction delta_v (m/s)",
        title="Residual Correction Distribution",
    )
    _plot_speed_bin_mae(plt, output_dir / f"{prefix}_speed_bin_mae.png", preds, targets, base_speeds)


def copy_small_results_for_git(source_dir: str | Path, results_dir: str | Path, include_predictions: bool = False) -> None:
    source_dir = Path(source_dir)
    results_dir = Path(results_dir)
    if not source_dir.exists():
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".json", ".png"}
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


def _plot_lines(plt, path: Path, x_values: Sequence[int], series: dict[str, Sequence[float]], ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.plot(x_values, values, marker="o", linewidth=1.8, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_pred_vs_target(plt, path: Path, preds: Sequence[float], targets: Sequence[float], base_speeds: Sequence[float]) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, base_speeds, s=10, alpha=0.35, label="base_speed")
    plt.scatter(targets, preds, s=10, alpha=0.45, label="model")
    min_v = min(list(targets) + list(preds) + list(base_speeds))
    max_v = max(list(targets) + list(preds) + list(base_speeds))
    plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.2, label="ideal")
    plt.xlabel("Target speed (m/s)")
    plt.ylabel("Predicted speed (m/s)")
    plt.title("Prediction vs Target")
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


def _plot_speed_bin_mae(
    plt,
    path: Path,
    preds: Sequence[float],
    targets: Sequence[float],
    base_speeds: Sequence[float],
) -> None:
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, float("inf")]
    labels = ["0-2", "2-5", "5-10", "10-15", "15+"]
    model_values = []
    base_values = []
    for low, high in zip(bins[:-1], bins[1:]):
        indexes = [i for i, target in enumerate(targets) if low <= target < high]
        if not indexes:
            model_values.append(0.0)
            base_values.append(0.0)
            continue
        model_values.append(sum(abs(preds[i] - targets[i]) for i in indexes) / len(indexes))
        base_values.append(sum(abs(base_speeds[i] - targets[i]) for i in indexes) / len(indexes))

    x_positions = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar([x - width / 2 for x in x_positions], base_values, width=width, label="base_speed")
    plt.bar([x + width / 2 for x in x_positions], model_values, width=width, label="model")
    plt.xticks(x_positions, labels)
    plt.xlabel("Target speed bin (m/s)")
    plt.ylabel("MAE (m/s)")
    plt.title("MAE by Speed Range")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
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
        raise RuntimeError(
            "matplotlib is required for evaluation plots. Install it in the training environment, "
            "for example: pip install matplotlib"
        ) from exc
    return plt
