"""Offline post-processing of evaluation artifacts.

Inputs are always the files written by :func:`utils.evaluation.write_evaluation_artifacts`
(``<prefix>_predictions.csv`` and ``<prefix>_metrics.json``). The functions below are pure
helpers so they can be unit-tested without GPUs or real datasets.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


_PREDICTION_FLOAT_FIELDS = (
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
)


def load_predictions(path: str | Path) -> list[dict]:
    """Load ``*_predictions.csv`` into a list of dicts with floats parsed."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for key in _PREDICTION_FLOAT_FIELDS:
                if key in row and row[key] != "":
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = math.nan
            row["target_speed_mps"] = math.sqrt(
                float(row.get("target_vx_mps", 0.0)) ** 2
                + float(row.get("target_vy_mps", 0.0)) ** 2
            )
            rows.append(row)
    return rows


def bad_cases(rows: Sequence[dict], top_k: int = 50, by: str = "endpoint_error_mps") -> list[dict]:
    """Return the rows with the largest error first (default: endpoint error)."""
    valid = [row for row in rows if isinstance(row.get(by), float) and not math.isnan(row[by])]
    return sorted(valid, key=lambda row: row[by], reverse=True)[: max(int(top_k), 0)]


@dataclass
class PerSceneRow:
    scene_name: str
    sample_count: int
    mae_vx_mps: float
    mae_vy_mps: float
    endpoint_error_mps: float
    base_endpoint_error_mps: float
    gain_endpoint_error_mps: float


def per_scene_metrics(rows: Sequence[dict]) -> list[PerSceneRow]:
    """Aggregate predictions per scene and report MAE / endpoint error / gain."""
    bucket: dict[str, list[dict]] = {}
    for row in rows:
        scene = str(row.get("scene_name", ""))
        bucket.setdefault(scene, []).append(row)

    results: list[PerSceneRow] = []
    for scene, items in bucket.items():
        n = len(items)
        if n == 0:
            continue
        mae_vx = sum(abs(item["error_vx_mps"]) for item in items) / n
        mae_vy = sum(abs(item["error_vy_mps"]) for item in items) / n
        ep = sum(item["endpoint_error_mps"] for item in items) / n
        base_ep = sum(item["base_endpoint_error_mps"] for item in items) / n
        results.append(
            PerSceneRow(
                scene_name=scene,
                sample_count=n,
                mae_vx_mps=mae_vx,
                mae_vy_mps=mae_vy,
                endpoint_error_mps=ep,
                base_endpoint_error_mps=base_ep,
                gain_endpoint_error_mps=base_ep - ep,
            )
        )
    results.sort(key=lambda row: row.endpoint_error_mps, reverse=True)
    return results


@dataclass
class SpeedBinRow:
    bin_label: str
    low_mps: float
    high_mps: float
    sample_count: int
    mae_vx_mps: float
    mae_vy_mps: float
    endpoint_error_mps: float
    base_endpoint_error_mps: float
    gain_endpoint_error_mps: float


@dataclass
class ScenarioRow:
    scenario: str
    sample_count: int
    mae_vx_mps: float
    mae_vy_mps: float
    endpoint_error_mps: float
    base_endpoint_error_mps: float
    gain_endpoint_error_mps: float


_DEFAULT_SPEED_BINS = (
    ("0-2", 0.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15+", 15.0, float("inf")),
)


def per_speed_bin_metrics(
    rows: Sequence[dict],
    bins: Sequence[tuple[str, float, float]] = _DEFAULT_SPEED_BINS,
) -> list[SpeedBinRow]:
    out: list[SpeedBinRow] = []
    for label, low, high in bins:
        items = [
            row
            for row in rows
            if isinstance(row.get("target_speed_mps"), float)
            and low <= row["target_speed_mps"] < high
        ]
        n = len(items)
        if n == 0:
            out.append(SpeedBinRow(label, low, high, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        mae_vx = sum(abs(item["error_vx_mps"]) for item in items) / n
        mae_vy = sum(abs(item["error_vy_mps"]) for item in items) / n
        ep = sum(item["endpoint_error_mps"] for item in items) / n
        base_ep = sum(item["base_endpoint_error_mps"] for item in items) / n
        out.append(
            SpeedBinRow(
                bin_label=label,
                low_mps=low,
                high_mps=high,
                sample_count=n,
                mae_vx_mps=mae_vx,
                mae_vy_mps=mae_vy,
                endpoint_error_mps=ep,
                base_endpoint_error_mps=base_ep,
                gain_endpoint_error_mps=base_ep - ep,
            )
        )
    return out


def per_scenario_metrics(rows: Sequence[dict]) -> list[ScenarioRow]:
    groups = _scenario_groups(rows)
    out: list[ScenarioRow] = []
    for scenario, indexes in groups.items():
        items = [rows[i] for i in indexes]
        n = len(items)
        if n == 0:
            continue
        mae_vx = sum(abs(item["error_vx_mps"]) for item in items) / n
        mae_vy = sum(abs(item["error_vy_mps"]) for item in items) / n
        ep = sum(item["endpoint_error_mps"] for item in items) / n
        base_ep = sum(item["base_endpoint_error_mps"] for item in items) / n
        out.append(
            ScenarioRow(
                scenario=scenario,
                sample_count=n,
                mae_vx_mps=mae_vx,
                mae_vy_mps=mae_vy,
                endpoint_error_mps=ep,
                base_endpoint_error_mps=base_ep,
                gain_endpoint_error_mps=base_ep - ep,
            )
        )
    out.sort(key=lambda row: row.endpoint_error_mps, reverse=True)
    return out


def _scenario_groups(rows: Sequence[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    groups["turning"] = [
        i for i, row in enumerate(rows) if abs(float(row.get("yaw_rate_radps", 0.0))) >= 0.1
    ]
    abs_vy = [abs(float(row.get("target_vy_mps", 0.0))) for row in rows]
    if abs_vy:
        high_lateral_threshold = _quantile_threshold(abs_vy, 0.9)
        low_lateral_threshold = _quantile_threshold(abs_vy, 0.5)
        groups["high_lateral"] = [i for i, value in enumerate(abs_vy) if value >= high_lateral_threshold]
        groups["low_speed_lateral"] = [
            i
            for i, value in enumerate(abs_vy)
            if float(rows[i].get("target_speed_mps", 0.0)) < 2.0 and value >= low_lateral_threshold
        ]
    groups["wheel_inconsistent"] = _top_row_indexes(rows, "wheel_speed_std_mps", quantile=0.9, use_abs=False)
    groups["hard_accel"] = _top_row_indexes(rows, "longitudinal_accel_mps2", quantile=0.9, use_abs=True)
    groups["high_lat_accel"] = _top_row_indexes(rows, "transversal_accel_mps2", quantile=0.9, use_abs=True)
    return groups


def _top_row_indexes(rows: Sequence[dict], key: str, quantile: float, use_abs: bool) -> list[int]:
    values = []
    indexes = []
    for i, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, float) and not math.isnan(value):
            values.append(abs(value) if use_abs else value)
            indexes.append(i)
    if not values:
        return []
    threshold = _quantile_threshold(values, quantile)
    return [index for index, value in zip(indexes, values) if value >= threshold]


def _quantile_threshold(values: Sequence[float], quantile: float) -> float:
    sorted_values = sorted(values)
    return sorted_values[int(quantile * (len(sorted_values) - 1))]


def _write_dicts_csv(rows: Iterable[dict], path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_float(value: float, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f"{value}"
        return f"{value:.{digits}f}"
    return str(value)


def _format_metric_table(metrics: dict) -> list[str]:
    keys = [
        "loss",
        "mae_vx_mps",
        "mae_vy_mps",
        "rmse_vx_mps",
        "rmse_vy_mps",
        "vector_endpoint_error_mps",
        "speed_magnitude_mae_mps",
        "r2_vx",
        "r2_vy",
        "base_mae_vx_mps",
        "base_mae_vy_mps",
        "gain_vs_base_vx_mps",
        "gain_vs_base_vy_mps",
        "gain_vs_base_vector_mps",
        "low_speed_vector_endpoint_error_mps",
        "mid_speed_vector_endpoint_error_mps",
        "high_speed_vector_endpoint_error_mps",
        "turning_vector_endpoint_error_mps",
        "straight_vector_endpoint_error_mps",
        "high_lateral_vector_endpoint_error_mps",
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key in keys:
        if key in metrics:
            lines.append(f"| `{key}` | {_format_float(metrics[key])} |")
    return lines


def _format_per_bin_table(rows: Sequence[SpeedBinRow]) -> list[str]:
    lines = [
        "| Speed bin (m/s) | Samples | Endpoint err | Base endpoint err | Gain | MAE vx | MAE vy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.bin_label,
                    str(row.sample_count),
                    _format_float(row.endpoint_error_mps),
                    _format_float(row.base_endpoint_error_mps),
                    _format_float(row.gain_endpoint_error_mps),
                    _format_float(row.mae_vx_mps),
                    _format_float(row.mae_vy_mps),
                ]
            )
            + " |"
        )
    return lines


def _format_scenario_table(rows: Sequence[ScenarioRow]) -> list[str]:
    lines = [
        "| Scenario | Samples | Endpoint err | Base endpoint err | Gain | MAE vx | MAE vy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.scenario,
                    str(row.sample_count),
                    _format_float(row.endpoint_error_mps),
                    _format_float(row.base_endpoint_error_mps),
                    _format_float(row.gain_endpoint_error_mps),
                    _format_float(row.mae_vx_mps),
                    _format_float(row.mae_vy_mps),
                ]
            )
            + " |"
        )
    return lines


def _format_worst_scene_table(rows: Sequence[PerSceneRow], top_k: int = 10) -> list[str]:
    lines = [
        "| Rank | Scene | Samples | Endpoint err | Base endpoint err | Gain |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows[: max(top_k, 0)], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row.scene_name,
                    str(row.sample_count),
                    _format_float(row.endpoint_error_mps),
                    _format_float(row.base_endpoint_error_mps),
                    _format_float(row.gain_endpoint_error_mps),
                ]
            )
            + " |"
        )
    return lines


def _format_bad_case_table(rows: Sequence[dict], top_k: int = 10) -> list[str]:
    lines = [
        "| Rank | Scene | Endpoint err | Target speed | Pred (vx, vy) | Target (vx, vy) |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(rows[: max(top_k, 0)], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.get("scene_name", "")),
                    _format_float(row.get("endpoint_error_mps", 0.0)),
                    _format_float(row.get("target_speed_mps", 0.0)),
                    f"({_format_float(row.get('pred_vx_mps', 0.0), 3)}, {_format_float(row.get('pred_vy_mps', 0.0), 3)})",
                    f"({_format_float(row.get('target_vx_mps', 0.0), 3)}, {_format_float(row.get('target_vy_mps', 0.0), 3)})",
                ]
            )
            + " |"
        )
    return lines


def write_analysis_bundle(
    eval_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "val",
    bad_case_top_k: int = 50,
    report_top_k: int = 10,
) -> dict[str, Path]:
    """Read existing eval artifacts under ``eval_dir`` and write the analysis bundle.

    The output directory defaults to ``eval_dir / "analysis"`` so the bundle sits
    next to the original ``*_metrics.json`` / ``*_predictions.csv`` without
    polluting them.
    """

    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir) if output_dir is not None else eval_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = eval_dir / f"{prefix}_predictions.csv"
    metrics_path = eval_dir / f"{prefix}_metrics.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = load_predictions(predictions_path)

    worst_samples = bad_cases(rows, top_k=bad_case_top_k)
    scene_rows = per_scene_metrics(rows)
    bin_rows = per_speed_bin_metrics(rows)
    scenario_rows = per_scenario_metrics(rows)

    bad_case_path = output_dir / f"{prefix}_bad_cases_top{bad_case_top_k}.csv"
    _write_dicts_csv(
        worst_samples,
        bad_case_path,
        fieldnames=[
            "scene_name",
            "center_token",
            "center_timestamp",
            "target_speed_mps",
            "endpoint_error_mps",
            "base_endpoint_error_mps",
            "pred_vx_mps",
            "pred_vy_mps",
            "target_vx_mps",
            "target_vy_mps",
            "base_vx_mps",
            "base_vy_mps",
            "delta_vx_mps",
            "delta_vy_mps",
        ],
    )

    scene_path = output_dir / f"{prefix}_per_scene_metrics.csv"
    _write_dicts_csv(
        [asdict(row) for row in scene_rows],
        scene_path,
        fieldnames=[
            "scene_name",
            "sample_count",
            "mae_vx_mps",
            "mae_vy_mps",
            "endpoint_error_mps",
            "base_endpoint_error_mps",
            "gain_endpoint_error_mps",
        ],
    )

    bin_path = output_dir / f"{prefix}_per_speed_bin_metrics.csv"
    _write_dicts_csv(
        [asdict(row) for row in bin_rows],
        bin_path,
        fieldnames=[
            "bin_label",
            "low_mps",
            "high_mps",
            "sample_count",
            "mae_vx_mps",
            "mae_vy_mps",
            "endpoint_error_mps",
            "base_endpoint_error_mps",
            "gain_endpoint_error_mps",
        ],
    )

    scenario_path = output_dir / f"{prefix}_per_scenario_metrics.csv"
    _write_dicts_csv(
        [asdict(row) for row in scenario_rows],
        scenario_path,
        fieldnames=[
            "scenario",
            "sample_count",
            "mae_vx_mps",
            "mae_vy_mps",
            "endpoint_error_mps",
            "base_endpoint_error_mps",
            "gain_endpoint_error_mps",
        ],
    )

    report_path = output_dir / f"{prefix}_report.md"
    summary_lines = [
        f"# Evaluation report ({prefix})",
        "",
        f"- Total samples: **{len(rows)}**",
        f"- Source eval dir: `{eval_dir}`",
        "",
        "## Overall metrics",
        *_format_metric_table(metrics),
        "",
        "## Performance by speed bin",
        *_format_per_bin_table(bin_rows),
        "",
        "## Scenario Metrics",
        *_format_scenario_table(scenario_rows),
        "",
        f"## Worst {report_top_k} scenes (by mean endpoint error)",
        *_format_worst_scene_table(scene_rows, top_k=report_top_k),
        "",
        f"## Worst {report_top_k} samples (by endpoint error)",
        *_format_bad_case_table(worst_samples, top_k=report_top_k),
        "",
        "## Files",
        f"- `{bad_case_path.name}` — top-{bad_case_top_k} bad cases",
        f"- `{scene_path.name}` — per-scene aggregated metrics",
        f"- `{bin_path.name}` — per-speed-bin aggregated metrics",
        f"- `{scenario_path.name}` — per-scenario aggregated metrics",
        "",
    ]
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "bad_cases": bad_case_path,
        "per_scene": scene_path,
        "per_speed_bin": bin_path,
        "per_scenario": scenario_path,
        "report": report_path,
    }
