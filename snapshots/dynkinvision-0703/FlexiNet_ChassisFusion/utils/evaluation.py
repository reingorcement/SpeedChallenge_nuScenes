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
    preds: list[list[float]]
    targets: list[list[float]]
    base_velocities: list[list[float]]
    scene_names: list[str]
    center_tokens: list[str]
    center_timestamps: list[int]
    yaw_rates: list[float] | None = None
    wheel_speed_stds: list[float] | None = None
    longitudinal_accels: list[float] | None = None
    transversal_accels: list[float] | None = None
    gate_weights: list[list[float]] | None = None
    gate_names: list[str] | None = None


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


def _as_velocity_list(values: Sequence[float] | Sequence[Sequence[float]]) -> list[list[float]]:
    if not values:
        return []
    first = values[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        return [[float(v[0]), float(v[1])] for v in values]  # type: ignore[index]
    return [[float(v), 0.0] for v in values]  # type: ignore[arg-type]


def _axis(values: Sequence[Sequence[float]], index: int) -> list[float]:
    return [float(v[index]) for v in values]


def _speed_magnitudes(values: Sequence[Sequence[float]]) -> list[float]:
    return [math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2) for v in values]


def _endpoint_errors(preds: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> list[float]:
    return [
        math.sqrt((float(pred[0]) - float(target[0])) ** 2 + (float(pred[1]) - float(target[1])) ** 2)
        for pred, target in zip(preds, targets)
    ]


def compute_metrics(
    preds: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    base_velocities: Sequence[float] | Sequence[Sequence[float]] | None = None,
    loss: float | None = None,
    yaw_rates: Sequence[float] | None = None,
    wheel_speed_stds: Sequence[float] | None = None,
    longitudinal_accels: Sequence[float] | None = None,
    transversal_accels: Sequence[float] | None = None,
) -> dict[str, float]:
    pred_vecs = _as_velocity_list(preds)
    target_vecs = _as_velocity_list(targets)
    pred_vx = _axis(pred_vecs, 0)
    pred_vy = _axis(pred_vecs, 1)
    target_vx = _axis(target_vecs, 0)
    target_vy = _axis(target_vecs, 1)
    pred_speed = _speed_magnitudes(pred_vecs)
    target_speed = _speed_magnitudes(target_vecs)
    endpoint_errors = _endpoint_errors(pred_vecs, target_vecs)
    metrics = {
        "mae_vx_mps": mean_absolute_error(pred_vx, target_vx),
        "rmse_vx_mps": root_mean_squared_error(pred_vx, target_vx),
        "mae_vy_mps": mean_absolute_error(pred_vy, target_vy),
        "rmse_vy_mps": root_mean_squared_error(pred_vy, target_vy),
        "vector_endpoint_error_mps": sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else 0.0,
        "speed_magnitude_mae_mps": mean_absolute_error(pred_speed, target_speed),
        "speed_magnitude_rmse_mps": root_mean_squared_error(pred_speed, target_speed),
        "r2_vx": r2_score(pred_vx, target_vx),
        "r2_vy": r2_score(pred_vy, target_vy),
    }
    metrics["selection_metric_mps"] = metrics["mae_vx_mps"] + metrics["mae_vy_mps"]
    metrics["mae_vx_kmh"] = metrics["mae_vx_mps"] * 3.6
    metrics["mae_vy_kmh"] = metrics["mae_vy_mps"] * 3.6
    if loss is not None:
        metrics["loss"] = loss

    if base_velocities is not None and len(base_velocities) == len(target_vecs):
        base_vecs = _as_velocity_list(base_velocities)
        base_vx = _axis(base_vecs, 0)
        base_vy = _axis(base_vecs, 1)
        base_speed = _speed_magnitudes(base_vecs)
        metrics["base_mae_vx_mps"] = mean_absolute_error(base_vx, target_vx)
        metrics["base_rmse_vx_mps"] = root_mean_squared_error(base_vx, target_vx)
        metrics["base_mae_vy_mps"] = mean_absolute_error(base_vy, target_vy)
        metrics["base_rmse_vy_mps"] = root_mean_squared_error(base_vy, target_vy)
        metrics["base_speed_magnitude_mae_mps"] = mean_absolute_error(base_speed, target_speed)
        metrics["gain_vs_base_vx_mps"] = metrics["base_mae_vx_mps"] - metrics["mae_vx_mps"]
        metrics["gain_vs_base_vy_mps"] = metrics["base_mae_vy_mps"] - metrics["mae_vy_mps"]
        metrics["gain_vs_base_vector_mps"] = (
            sum(_endpoint_errors(base_vecs, target_vecs)) / len(base_vecs)
            if base_vecs
            else 0.0
        ) - metrics["vector_endpoint_error_mps"]

    metrics.update(
        _compute_group_metrics(
            pred_vecs,
            target_vecs,
            yaw_rates=yaw_rates,
            wheel_speed_stds=wheel_speed_stds,
            longitudinal_accels=longitudinal_accels,
            transversal_accels=transversal_accels,
        )
    )
    return metrics


def _compute_group_metrics(
    preds: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    yaw_rates: Sequence[float] | None = None,
    wheel_speed_stds: Sequence[float] | None = None,
    longitudinal_accels: Sequence[float] | None = None,
    transversal_accels: Sequence[float] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    speeds = _speed_magnitudes(targets)
    groups = {
        "low_speed": [i for i, speed in enumerate(speeds) if speed < 2.0],
        "mid_speed": [i for i, speed in enumerate(speeds) if 2.0 <= speed < 10.0],
        "high_speed": [i for i, speed in enumerate(speeds) if speed >= 10.0],
    }
    abs_vy = [abs(v[1]) for v in targets]
    if abs_vy:
        sorted_abs_vy = sorted(abs_vy)
        threshold = sorted_abs_vy[int(0.9 * (len(sorted_abs_vy) - 1))]
        groups["high_lateral"] = [i for i, value in enumerate(abs_vy) if value >= threshold]
        lateral_threshold = sorted_abs_vy[int(0.5 * (len(sorted_abs_vy) - 1))]
        groups["low_speed_lateral"] = [
            i for i, value in enumerate(abs_vy) if speeds[i] < 2.0 and value >= lateral_threshold
        ]
    if yaw_rates is not None and len(yaw_rates) == len(targets):
        groups["turning"] = [i for i, yaw_rate in enumerate(yaw_rates) if abs(yaw_rate) >= 0.1]
        groups["straight"] = [i for i, yaw_rate in enumerate(yaw_rates) if abs(yaw_rate) < 0.1]
    if wheel_speed_stds is not None and len(wheel_speed_stds) == len(targets):
        groups["wheel_inconsistent"] = _top_fraction_indexes(wheel_speed_stds, fraction=0.1, use_abs=False)
    if longitudinal_accels is not None and len(longitudinal_accels) == len(targets):
        groups["hard_accel"] = _top_fraction_indexes(longitudinal_accels, fraction=0.1, use_abs=True)
    if transversal_accels is not None and len(transversal_accels) == len(targets):
        groups["high_lat_accel"] = _top_fraction_indexes(transversal_accels, fraction=0.1, use_abs=True)

    for name, indexes in groups.items():
        if not indexes:
            continue
        pred_subset = [preds[i] for i in indexes]
        target_subset = [targets[i] for i in indexes]
        errors = _endpoint_errors(pred_subset, target_subset)
        metrics[f"{name}_sample_count"] = len(indexes)
        metrics[f"{name}_mae_vx_mps"] = mean_absolute_error(_axis(pred_subset, 0), _axis(target_subset, 0))
        metrics[f"{name}_vector_endpoint_error_mps"] = sum(errors) / len(errors)
        metrics[f"{name}_mae_vy_mps"] = mean_absolute_error(_axis(pred_subset, 1), _axis(target_subset, 1))
    return metrics


def _top_fraction_indexes(values: Sequence[float], fraction: float, use_abs: bool) -> list[int]:
    if not values:
        return []
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return list(range(len(values)))
    scored = [abs(float(value)) if use_abs else float(value) for value in values]
    sample_count = max(1, math.ceil(len(scored) * fraction))
    ranked = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
    return sorted(ranked[:sample_count])


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
                "target_vx_mps",
                "target_vy_mps",
                "pred_vx_mps",
                "pred_vy_mps",
                "base_vx_mps",
                "base_vy_mps",
                "error_vx_mps",
                "error_vy_mps",
                "endpoint_error_mps",
                "base_endpoint_error_mps",
                "delta_vx_mps",
                "delta_vy_mps",
                "yaw_rate_radps",
                "longitudinal_accel_mps2",
                "transversal_accel_mps2",
                "wheel_speed_std_mps",
                "gate_vision",
                "gate_kinematic",
                "gate_dynamics",
            ],
        )
        writer.writeheader()
        for i, (pred, target, base_velocity) in enumerate(zip(result.preds, result.targets, result.base_velocities)):
            endpoint_error = math.sqrt((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)
            base_endpoint_error = math.sqrt(
                (base_velocity[0] - target[0]) ** 2 + (base_velocity[1] - target[1]) ** 2
            )
            gate_values = _gate_values_for_row(result, i)
            writer.writerow(
                {
                    "scene_name": _safe_get(result.scene_names, i, ""),
                    "center_token": _safe_get(result.center_tokens, i, ""),
                    "center_timestamp": _safe_get(result.center_timestamps, i, ""),
                    "target_vx_mps": target[0],
                    "target_vy_mps": target[1],
                    "pred_vx_mps": pred[0],
                    "pred_vy_mps": pred[1],
                    "base_vx_mps": base_velocity[0],
                    "base_vy_mps": base_velocity[1],
                    "error_vx_mps": pred[0] - target[0],
                    "error_vy_mps": pred[1] - target[1],
                    "endpoint_error_mps": endpoint_error,
                    "base_endpoint_error_mps": base_endpoint_error,
                    "delta_vx_mps": pred[0] - base_velocity[0],
                    "delta_vy_mps": pred[1] - base_velocity[1],
                    "yaw_rate_radps": _safe_optional_get(result.yaw_rates, i, ""),
                    "longitudinal_accel_mps2": _safe_optional_get(result.longitudinal_accels, i, ""),
                    "transversal_accel_mps2": _safe_optional_get(result.transversal_accels, i, ""),
                    "wheel_speed_std_mps": _safe_optional_get(result.wheel_speed_stds, i, ""),
                    "gate_vision": gate_values.get("vision", ""),
                    "gate_kinematic": gate_values.get("kinematic", ""),
                    "gate_dynamics": gate_values.get("dynamics", ""),
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
            "val_mae_vx_mps": [row["val_mae_vx_mps"] for row in history],
            "val_mae_vy_mps": [row["val_mae_vy_mps"] for row in history],
            "val_vector_endpoint_error_mps": [row["val_vector_endpoint_error_mps"] for row in history],
            "base_mae_vx_mps": [row["base_mae_vx_mps"] for row in history],
        },
        ylabel="Speed Error (m/s)",
        title="Validation Velocity Error",
    )


def write_evaluation_artifacts(result: EvaluationResult, output_dir: str | Path, prefix: str = "val") -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(
        result.preds,
        result.targets,
        result.base_velocities,
        loss=result.loss,
        yaw_rates=result.yaw_rates,
        wheel_speed_stds=result.wheel_speed_stds,
        longitudinal_accels=result.longitudinal_accels,
        transversal_accels=result.transversal_accels,
    )
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
    base_velocities = result.base_velocities
    endpoint_errors = _endpoint_errors(preds, targets)
    delta_vx = [p[0] - b[0] for p, b in zip(preds, base_velocities)]
    delta_vy = [p[1] - b[1] for p, b in zip(preds, base_velocities)]

    _plot_pred_vs_target(
        plt,
        output_dir / f"{prefix}_speed_pred_vs_target.png",
        _speed_magnitudes(preds),
        _speed_magnitudes(targets),
        _speed_magnitudes(base_velocities),
    )
    _plot_scatter(
        plt,
        output_dir / f"{prefix}_endpoint_error_vs_speed.png",
        _speed_magnitudes(targets),
        endpoint_errors,
        xlabel="Target speed (m/s)",
        ylabel="Endpoint error (m/s)",
        title="Vector Endpoint Error vs Target Speed",
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_endpoint_error_histogram.png",
        endpoint_errors,
        xlabel="Endpoint error (m/s)",
        title="Vector Endpoint Error Distribution",
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_delta_vx_histogram.png",
        delta_vx,
        xlabel="Predicted correction delta_vx (m/s)",
        title="Longitudinal Residual Correction Distribution",
    )
    _plot_histogram(
        plt,
        output_dir / f"{prefix}_delta_vy_histogram.png",
        delta_vy,
        xlabel="Predicted correction delta_vy (m/s)",
        title="Lateral Residual Correction Distribution",
    )
    _plot_speed_bin_mae(plt, output_dir / f"{prefix}_speed_bin_endpoint_error.png", preds, targets, base_velocities)


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
    plt.scatter(targets, base_speeds, s=10, alpha=0.35, label="base_velocity magnitude")
    plt.scatter(targets, preds, s=10, alpha=0.45, label="model magnitude")
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
    preds: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    base_velocities: Sequence[Sequence[float]],
) -> None:
    bins = [0.0, 2.0, 5.0, 10.0, 15.0, float("inf")]
    labels = ["0-2", "2-5", "5-10", "10-15", "15+"]
    model_values = []
    base_values = []
    target_speeds = _speed_magnitudes(targets)
    for low, high in zip(bins[:-1], bins[1:]):
        indexes = [i for i, target_speed in enumerate(target_speeds) if low <= target_speed < high]
        if not indexes:
            model_values.append(0.0)
            base_values.append(0.0)
            continue
        model_values.append(
            sum(math.sqrt((preds[i][0] - targets[i][0]) ** 2 + (preds[i][1] - targets[i][1]) ** 2) for i in indexes)
            / len(indexes)
        )
        base_values.append(
            sum(
                math.sqrt(
                    (base_velocities[i][0] - targets[i][0]) ** 2
                    + (base_velocities[i][1] - targets[i][1]) ** 2
                )
                for i in indexes
            )
            / len(indexes)
        )

    x_positions = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar([x - width / 2 for x in x_positions], base_values, width=width, label="base_velocity")
    plt.bar([x + width / 2 for x in x_positions], model_values, width=width, label="model")
    plt.xticks(x_positions, labels)
    plt.xlabel("Target speed bin (m/s)")
    plt.ylabel("Endpoint error (m/s)")
    plt.title("Vector Endpoint Error by Speed Range")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _safe_get(values: Sequence[object], index: int, default: object) -> object:
    return values[index] if index < len(values) else default


def _safe_optional_get(values: Sequence[object] | None, index: int, default: object) -> object:
    if values is None:
        return default
    return values[index] if index < len(values) else default


def _gate_values_for_row(result: EvaluationResult, index: int) -> dict[str, float]:
    if not result.gate_weights or not result.gate_names or index >= len(result.gate_weights):
        return {}
    return {
        name: float(weight)
        for name, weight in zip(result.gate_names, result.gate_weights[index])
    }


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
