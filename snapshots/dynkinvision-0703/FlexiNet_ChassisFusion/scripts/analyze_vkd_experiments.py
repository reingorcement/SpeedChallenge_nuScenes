"""Analyze the 0704 VKD velocity-xy experiment bundle.

The script is offline-only: it reads existing ``best_eval`` artifacts from the
four trained experiments and writes comparison tables, plots, and a Markdown
report. It does not run inference or modify training outputs.

Example:
    python scripts/analyze_vkd_experiments.py \
        --root . \
        --output_dir analysis_reports/vkd_0704
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


METHODS: tuple[tuple[str, str], ...] = (
    ("vkd_gated", "Vision + Kinematics + Dynamics gated"),
    ("kd_only", "Kinematics + Dynamics only"),
    ("vision_only", "Vision only"),
    ("vision_kinematics", "Vision + Kinematics"),
)

PRIMARY_METRICS: tuple[str, ...] = (
    "vector_endpoint_error_mps",
    "mae_vx_mps",
    "mae_vy_mps",
    "rmse_vx_mps",
    "rmse_vy_mps",
    "speed_magnitude_mae_mps",
    "r2_vx",
    "r2_vy",
    "gain_vs_base_vector_mps",
    "gain_vs_base_vx_mps",
    "gain_vs_base_vy_mps",
)

SCENARIO_ORDER: tuple[str, ...] = (
    "all",
    "low_speed",
    "mid_speed",
    "high_speed",
    "turning",
    "straight",
    "high_lateral",
    "low_speed_lateral",
    "wheel_inconsistent",
    "hard_accel",
    "high_lat_accel",
)

FLOAT_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class MethodArtifacts:
    name: str
    label: str
    eval_dir: Path
    metrics_path: Path
    predictions_path: Path
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_float(value: str) -> float:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for key in FLOAT_FIELDS:
                if key in row:
                    row[key] = _parse_float(str(row[key]))
            vx = float(row.get("target_vx_mps", 0.0))
            vy = float(row.get("target_vy_mps", 0.0))
            row["target_speed_mps"] = math.sqrt(vx * vx + vy * vy)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _maybe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any, digits: int = 5) -> str:
    val = _maybe_float(value)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def _pct(delta: float, baseline: float) -> float:
    return delta / baseline * 100.0 if baseline else math.nan


def _quantile_threshold(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    sorted_values = np.sort(values)
    index = int(q * (len(sorted_values) - 1))
    return float(sorted_values[index])


def _top_fraction_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    count = max(1, int(math.ceil(values.size * fraction)))
    order = np.argsort(values)[::-1]
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _column(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_maybe_float(row.get(key)) for row in rows], dtype=np.float64)


def _valid_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def _valid_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else math.nan


def _valid_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else math.nan


def _load_method(root: Path, name: str, label: str) -> MethodArtifacts:
    run_eval_dir = root / "runs" / "FlexiNet_ChassisFusion_velocity_xy" / name / "best_eval"
    result_eval_dir = root / "results_git" / "FlexiNet_ChassisFusion_velocity_xy" / name / "best_eval"

    predictions_path = run_eval_dir / "val_predictions.csv"
    metrics_path = result_eval_dir / "val_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_eval_dir / "val_metrics.json"
    if not predictions_path.exists():
        predictions_path = result_eval_dir / "val_predictions.csv"

    missing = [str(path) for path in (metrics_path, predictions_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for {name}: {missing}")

    return MethodArtifacts(
        name=name,
        label=label,
        eval_dir=predictions_path.parent,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        metrics=_read_json(metrics_path),
        rows=_read_predictions(predictions_path),
    )


def _validate_alignment(methods: Sequence[MethodArtifacts]) -> dict[str, Any]:
    ref = methods[0]
    ref_tokens = [row["center_token"] for row in ref.rows]
    ref_vx = _column(ref.rows, "target_vx_mps")
    ref_vy = _column(ref.rows, "target_vy_mps")
    max_target_diff = 0.0

    for method in methods[1:]:
        tokens = [row["center_token"] for row in method.rows]
        if tokens != ref_tokens:
            raise RuntimeError(f"center_token alignment mismatch: {method.name} vs {ref.name}")
        diff_vx = np.nanmax(np.abs(_column(method.rows, "target_vx_mps") - ref_vx))
        diff_vy = np.nanmax(np.abs(_column(method.rows, "target_vy_mps") - ref_vy))
        max_target_diff = max(max_target_diff, float(diff_vx), float(diff_vy))

    return {
        "reference_method": ref.name,
        "sample_count": len(ref.rows),
        "max_target_diff": max_target_diff,
    }


def _scenario_masks(ref_rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    target_speed = _column(ref_rows, "target_speed_mps")
    target_vy_abs = np.abs(_column(ref_rows, "target_vy_mps"))
    yaw_rate_abs = np.abs(_column(ref_rows, "yaw_rate_radps"))
    wheel_std = _column(ref_rows, "wheel_speed_std_mps")
    long_accel_abs = np.abs(_column(ref_rows, "longitudinal_accel_mps2"))
    lat_accel_abs = np.abs(_column(ref_rows, "transversal_accel_mps2"))

    high_lateral_threshold = _quantile_threshold(target_vy_abs[np.isfinite(target_vy_abs)], 0.90)
    low_lateral_threshold = _quantile_threshold(target_vy_abs[np.isfinite(target_vy_abs)], 0.50)

    masks = {
        "all": np.ones(len(ref_rows), dtype=bool),
        "low_speed": target_speed < 2.0,
        "mid_speed": (target_speed >= 2.0) & (target_speed < 10.0),
        "high_speed": target_speed >= 10.0,
        "turning": yaw_rate_abs >= 0.1,
        "straight": yaw_rate_abs < 0.1,
        "high_lateral": target_vy_abs >= high_lateral_threshold,
        "low_speed_lateral": (target_speed < 2.0) & (target_vy_abs >= low_lateral_threshold),
        "wheel_inconsistent": _top_fraction_mask(wheel_std, fraction=0.10),
        "hard_accel": _top_fraction_mask(long_accel_abs, fraction=0.10),
        "high_lat_accel": _top_fraction_mask(lat_accel_abs, fraction=0.10),
    }
    return {key: np.asarray(mask, dtype=bool) for key, mask in masks.items()}


def _compute_subset_metrics(rows: Sequence[dict[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    error_vx = _column(rows, "error_vx_mps")[mask]
    error_vy = _column(rows, "error_vy_mps")[mask]
    endpoint = _column(rows, "endpoint_error_mps")[mask]
    base_endpoint = _column(rows, "base_endpoint_error_mps")[mask]

    return {
        "sample_count": int(mask.sum()),
        "mae_vx_mps": _valid_mean(np.abs(error_vx)),
        "mae_vy_mps": _valid_mean(np.abs(error_vy)),
        "rmse_vx_mps": float(np.sqrt(_valid_mean(error_vx * error_vx))),
        "rmse_vy_mps": float(np.sqrt(_valid_mean(error_vy * error_vy))),
        "endpoint_error_mps": _valid_mean(endpoint),
        "endpoint_error_p50_mps": _valid_percentile(endpoint, 50),
        "endpoint_error_p90_mps": _valid_percentile(endpoint, 90),
        "endpoint_error_p95_mps": _valid_percentile(endpoint, 95),
        "base_endpoint_error_mps": _valid_mean(base_endpoint),
        "gain_vs_base_endpoint_mps": _valid_mean(base_endpoint) - _valid_mean(endpoint),
    }


def _build_overall_rows(methods: Sequence[MethodArtifacts]) -> list[dict[str, Any]]:
    vkd_metrics = methods[0].metrics
    rows: list[dict[str, Any]] = []
    for method in methods:
        record: dict[str, Any] = {
            "method": method.name,
            "label": method.label,
            "sample_count": len(method.rows),
        }
        for key in PRIMARY_METRICS:
            record[key] = method.metrics.get(key, math.nan)

        for key in ("vector_endpoint_error_mps", "mae_vx_mps", "mae_vy_mps"):
            current = _maybe_float(method.metrics.get(key))
            vkd = _maybe_float(vkd_metrics.get(key))
            delta = current - vkd
            record[f"delta_vs_vkd_{key}"] = delta
            record[f"delta_vs_vkd_{key}_pct"] = _pct(delta, current)
        rows.append(record)
    return rows


def _build_scenario_rows(
    methods: Sequence[MethodArtifacts],
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vkd_by_scenario = {
        scenario: _compute_subset_metrics(methods[0].rows, mask)
        for scenario, mask in masks.items()
    }
    for scenario in SCENARIO_ORDER:
        mask = masks[scenario]
        vkd_endpoint = vkd_by_scenario[scenario]["endpoint_error_mps"]
        for method in methods:
            metrics = _compute_subset_metrics(method.rows, mask)
            endpoint = metrics["endpoint_error_mps"]
            record = {
                "scenario": scenario,
                "method": method.name,
                "label": method.label,
                **metrics,
                "delta_endpoint_vs_vkd_mps": endpoint - vkd_endpoint,
                "delta_endpoint_vs_vkd_pct": _pct(endpoint - vkd_endpoint, endpoint),
            }
            rows.append(record)
    return rows


def _build_error_percentile_rows(methods: Sequence[MethodArtifacts]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        endpoint = _column(method.rows, "endpoint_error_mps")
        rows.append({
            "method": method.name,
            "p50_mps": _valid_percentile(endpoint, 50),
            "p75_mps": _valid_percentile(endpoint, 75),
            "p90_mps": _valid_percentile(endpoint, 90),
            "p95_mps": _valid_percentile(endpoint, 95),
            "p99_mps": _valid_percentile(endpoint, 99),
            "max_mps": float(np.nanmax(endpoint)),
        })
    return rows


def _build_gate_rows(
    methods: Sequence[MethodArtifacts],
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        gate_vision = _column(method.rows, "gate_vision")
        gate_kinematic = _column(method.rows, "gate_kinematic")
        gate_dynamics = _column(method.rows, "gate_dynamics")
        if not (
            np.isfinite(gate_vision).any()
            or np.isfinite(gate_kinematic).any()
            or np.isfinite(gate_dynamics).any()
        ):
            continue

        endpoint = _column(method.rows, "endpoint_error_mps")
        for scenario in SCENARIO_ORDER:
            mask = masks[scenario]
            record = {
                "method": method.name,
                "scenario": scenario,
                "sample_count": int(mask.sum()),
                "gate_vision_mean": _valid_mean(gate_vision[mask]),
                "gate_vision_std": _valid_std(gate_vision[mask]),
                "gate_kinematic_mean": _valid_mean(gate_kinematic[mask]),
                "gate_kinematic_std": _valid_std(gate_kinematic[mask]),
                "gate_dynamics_mean": _valid_mean(gate_dynamics[mask]),
                "gate_dynamics_std": _valid_std(gate_dynamics[mask]),
                "endpoint_error_mps": _valid_mean(endpoint[mask]),
            }
            rows.append(record)
    return rows


def _build_bad_case_overlap_rows(
    methods: Sequence[MethodArtifacts],
    top_k: int,
) -> list[dict[str, Any]]:
    endpoint_by_method: dict[str, dict[str, float]] = {}
    top_tokens_by_method: dict[str, set[str]] = {}
    for method in methods:
        token_to_error = {
            str(row["center_token"]): _maybe_float(row.get("endpoint_error_mps"))
            for row in method.rows
        }
        endpoint_by_method[method.name] = token_to_error
        top_tokens = sorted(token_to_error, key=lambda token: token_to_error[token], reverse=True)[:top_k]
        top_tokens_by_method[method.name] = set(top_tokens)

    candidate_tokens = set().union(*top_tokens_by_method.values())
    rows: list[dict[str, Any]] = []
    for token in candidate_tokens:
        record: dict[str, Any] = {
            "center_token": token,
            "top_bad_case_method_count": sum(token in top_tokens for top_tokens in top_tokens_by_method.values()),
        }
        for method in methods:
            record[f"{method.name}_endpoint_error_mps"] = endpoint_by_method[method.name].get(token, math.nan)
            record[f"{method.name}_is_top{top_k}"] = int(token in top_tokens_by_method[method.name])
        rows.append(record)

    rows.sort(
        key=lambda row: (
            int(row["top_bad_case_method_count"]),
            _maybe_float(row.get("vkd_gated_endpoint_error_mps")),
        ),
        reverse=True,
    )
    return rows


def _format_md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _plot_outputs(
    output_dir: Path,
    methods: Sequence[MethodArtifacts],
    overall_rows: Sequence[dict[str, Any]],
    scenario_rows: Sequence[dict[str, Any]],
    gate_rows: Sequence[dict[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    method_names = [method.name for method in methods]
    x = np.arange(len(method_names))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.25
    for offset, key, label in (
        (-width, "vector_endpoint_error_mps", "Vector EE"),
        (0.0, "mae_vx_mps", "MAE vx"),
        (width, "mae_vy_mps", "MAE vy"),
    ):
        values = [_maybe_float(row[key]) for row in overall_rows]
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel("Error (m/s)")
    ax.set_title("Overall Velocity Error")
    ax.legend()
    fig.tight_layout()
    out = figure_dir / "overall_metrics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    written.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.35
    vx = [_maybe_float(row["mae_vx_mps"]) for row in overall_rows]
    vy = [_maybe_float(row["mae_vy_mps"]) for row in overall_rows]
    ax.bar(x - width / 2, vx, width=width, label="MAE vx")
    ax.bar(x + width / 2, vy, width=width, label="MAE vy")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel("MAE (m/s)")
    ax.set_title("Longitudinal vs Lateral Error")
    ax.legend()
    fig.tight_layout()
    out = figure_dir / "vx_vy_mae.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    written.append(out)

    key_scenarios = [
        "all",
        "low_speed",
        "mid_speed",
        "high_speed",
        "turning",
        "high_lateral",
        "wheel_inconsistent",
        "hard_accel",
        "high_lat_accel",
    ]
    scenario_lookup = {
        (row["scenario"], row["method"]): _maybe_float(row["endpoint_error_mps"])
        for row in scenario_rows
    }
    matrix = np.asarray(
        [[scenario_lookup[(scenario, method)] for method in method_names] for scenario in key_scenarios],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(key_scenarios)))
    ax.set_yticklabels(key_scenarios)
    ax.set_title("Endpoint Error by Scenario")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Endpoint error (m/s)")
    fig.tight_layout()
    out = figure_dir / "scenario_endpoint_error.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    written.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for method in methods:
        endpoint = _column(method.rows, "endpoint_error_mps")
        endpoint = endpoint[np.isfinite(endpoint)]
        ax.hist(endpoint, bins=80, alpha=0.35, density=True, label=method.name)
    ax.set_xlabel("Endpoint error (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Endpoint Error Distribution")
    ax.legend()
    fig.tight_layout()
    out = figure_dir / "error_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    written.append(out)

    vkd_gate_rows = [row for row in gate_rows if row["method"] == "vkd_gated"]
    if vkd_gate_rows:
        gate_lookup = {row["scenario"]: row for row in vkd_gate_rows}
        gate_scenarios = [scenario for scenario in key_scenarios if scenario in gate_lookup]
        x_gate = np.arange(len(gate_scenarios))
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        width = 0.25
        for offset, key, label in (
            (-width, "gate_vision_mean", "vision"),
            (0.0, "gate_kinematic_mean", "kinematic"),
            (width, "gate_dynamics_mean", "dynamics"),
        ):
            ax.bar(x_gate + offset, [_maybe_float(gate_lookup[s][key]) for s in gate_scenarios], width, label=label)
        ax.set_xticks(x_gate)
        ax.set_xticklabels(gate_scenarios, rotation=25, ha="right")
        ax.set_ylabel("Mean gate weight")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("VKD Gate Weights by Scenario")
        ax.legend()
        fig.tight_layout()
        out = figure_dir / "gate_by_scenario.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out)

    return written


def _write_report(
    output_dir: Path,
    methods: Sequence[MethodArtifacts],
    alignment: dict[str, Any],
    overall_rows: Sequence[dict[str, Any]],
    scenario_rows: Sequence[dict[str, Any]],
    gate_rows: Sequence[dict[str, Any]],
    figure_paths: Sequence[Path],
) -> Path:
    by_method = {row["method"]: row for row in overall_rows}
    ranked = sorted(overall_rows, key=lambda row: _maybe_float(row["vector_endpoint_error_mps"]))
    best = ranked[0]
    vkd = by_method["vkd_gated"]
    kd = by_method["kd_only"]
    vk = by_method["vision_kinematics"]
    vision = by_method["vision_only"]

    def gain_line(other: dict[str, Any], metric: str) -> str:
        vkd_value = _maybe_float(vkd[metric])
        other_value = _maybe_float(other[metric])
        delta = other_value - vkd_value
        return f"{delta:.5f} m/s ({_pct(delta, other_value):.2f}% lower than {other['method']})"

    primary_table = _format_md_table(
        ["method", "vector EE", "MAE vx", "MAE vy", "R2 vx", "R2 vy", "gain vs base EE"],
        [
            [
                row["method"],
                _fmt(row["vector_endpoint_error_mps"]),
                _fmt(row["mae_vx_mps"]),
                _fmt(row["mae_vy_mps"]),
                _fmt(row["r2_vx"]),
                _fmt(row["r2_vy"]),
                _fmt(row["gain_vs_base_vector_mps"]),
            ]
            for row in overall_rows
        ],
    )

    key_scenarios = ("turning", "high_lateral", "wheel_inconsistent", "hard_accel", "high_lat_accel")
    scenario_lookup = {
        (row["scenario"], row["method"]): row
        for row in scenario_rows
    }
    scenario_table = _format_md_table(
        ["scenario", "n", "vkd EE", "kd EE", "vision+k EE", "vision EE", "vkd vs vision+k"],
        [
            [
                scenario,
                scenario_lookup[(scenario, "vkd_gated")]["sample_count"],
                _fmt(scenario_lookup[(scenario, "vkd_gated")]["endpoint_error_mps"]),
                _fmt(scenario_lookup[(scenario, "kd_only")]["endpoint_error_mps"]),
                _fmt(scenario_lookup[(scenario, "vision_kinematics")]["endpoint_error_mps"]),
                _fmt(scenario_lookup[(scenario, "vision_only")]["endpoint_error_mps"]),
                _fmt(
                    _maybe_float(scenario_lookup[(scenario, "vision_kinematics")]["endpoint_error_mps"])
                    - _maybe_float(scenario_lookup[(scenario, "vkd_gated")]["endpoint_error_mps"])
                ),
            ]
            for scenario in key_scenarios
        ],
    )

    vkd_gate_rows = [row for row in gate_rows if row["method"] == "vkd_gated" and row["scenario"] in key_scenarios]
    gate_table = _format_md_table(
        ["scenario", "gate vision", "gate kinematic", "gate dynamics"],
        [
            [
                row["scenario"],
                _fmt(row["gate_vision_mean"], digits=4),
                _fmt(row["gate_kinematic_mean"], digits=4),
                _fmt(row["gate_dynamics_mean"], digits=4),
            ]
            for row in vkd_gate_rows
        ],
    )

    if figure_paths:
        figure_list = "\n".join(
            f"- `{path.relative_to(output_dir).as_posix()}`" for path in figure_paths
        )
    else:
        figure_list = "- Figure generation was skipped or unavailable in this environment."

    report = f"""# VKD 0704 Experiment Analysis

## Scope

This report compares the four completed velocity-xy experiments from the 0704 run:

{chr(10).join(f"- `{method.name}`: {method.label}" for method in methods)}

Inputs were read from each method's `best_eval` artifacts. Alignment check:

- Reference method: `{alignment['reference_method']}`
- Samples: `{alignment['sample_count']}`
- Max target difference across aligned methods: `{alignment['max_target_diff']:.6g}`

This is a single-seed descriptive comparison. It should not be worded as a statistically significant result.

## Main Result

Best method by `vector_endpoint_error_mps`: `{best['method']}`.

`vkd_gated` improves vector endpoint error over:

- `kd_only` by {gain_line(kd, 'vector_endpoint_error_mps')}
- `vision_kinematics` by {gain_line(vk, 'vector_endpoint_error_mps')}
- `vision_only` by {gain_line(vision, 'vector_endpoint_error_mps')}

{primary_table}

## Scenario Breakdown

These scenarios are computed from the aligned validation samples using the same masks for all methods.

{scenario_table}

## Gate Analysis

`vkd_gated` gate weights show how the confidence-gated fusion distributes trust across branches.

{gate_table}

## Generated Figures

{figure_list}

## Output Files

- `comparison_summary.csv`: main metric comparison.
- `scenario_comparison.csv`: per-scenario comparison with common masks.
- `gate_summary.csv`: gate weights by scenario for methods that expose gate columns.
- `error_percentiles.csv`: endpoint-error tail behavior.
- `bad_case_overlap.csv`: overlap of top bad cases across methods.

## Interpretation Guardrails

- Use `vector_endpoint_error_mps` as the primary ranking metric.
- Use `mae_vx_mps` and `mae_vy_mps` to explain whether gains come from longitudinal or lateral velocity.
- Use `turning`, `high_lateral`, `wheel_inconsistent`, `hard_accel`, and `high_lat_accel` for mechanism-oriented claims.
- Do not claim significance without repeated seeds.
"""

    path = output_dir / "REPORT_VKD_0704.md"
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="FlexiNet_ChassisFusion project root")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_reports/vkd_0704"),
        help="Directory for analysis outputs",
    )
    parser.add_argument("--bad_case_top_k", type=int, default=50)
    parser.add_argument("--no_plots", action="store_true", help="Skip matplotlib figure generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [_load_method(root, name, label) for name, label in METHODS]
    alignment = _validate_alignment(methods)
    masks = _scenario_masks(methods[0].rows)

    overall_rows = _build_overall_rows(methods)
    scenario_rows = _build_scenario_rows(methods, masks)
    gate_rows = _build_gate_rows(methods, masks)
    percentile_rows = _build_error_percentile_rows(methods)
    bad_case_rows = _build_bad_case_overlap_rows(methods, top_k=int(args.bad_case_top_k))

    overall_fields = [
        "method",
        "label",
        "sample_count",
        *PRIMARY_METRICS,
        "delta_vs_vkd_vector_endpoint_error_mps",
        "delta_vs_vkd_vector_endpoint_error_mps_pct",
        "delta_vs_vkd_mae_vx_mps",
        "delta_vs_vkd_mae_vx_mps_pct",
        "delta_vs_vkd_mae_vy_mps",
        "delta_vs_vkd_mae_vy_mps_pct",
    ]
    scenario_fields = [
        "scenario",
        "method",
        "label",
        "sample_count",
        "mae_vx_mps",
        "mae_vy_mps",
        "rmse_vx_mps",
        "rmse_vy_mps",
        "endpoint_error_mps",
        "endpoint_error_p50_mps",
        "endpoint_error_p90_mps",
        "endpoint_error_p95_mps",
        "base_endpoint_error_mps",
        "gain_vs_base_endpoint_mps",
        "delta_endpoint_vs_vkd_mps",
        "delta_endpoint_vs_vkd_pct",
    ]
    gate_fields = [
        "method",
        "scenario",
        "sample_count",
        "gate_vision_mean",
        "gate_vision_std",
        "gate_kinematic_mean",
        "gate_kinematic_std",
        "gate_dynamics_mean",
        "gate_dynamics_std",
        "endpoint_error_mps",
    ]
    percentile_fields = ["method", "p50_mps", "p75_mps", "p90_mps", "p95_mps", "p99_mps", "max_mps"]
    bad_case_fields = [
        "center_token",
        "top_bad_case_method_count",
        *[
            field
            for method, _label in METHODS
            for field in (f"{method}_endpoint_error_mps", f"{method}_is_top{args.bad_case_top_k}")
        ],
    ]

    _write_csv(output_dir / "comparison_summary.csv", overall_rows, overall_fields)
    _write_csv(output_dir / "scenario_comparison.csv", scenario_rows, scenario_fields)
    _write_csv(output_dir / "gate_summary.csv", gate_rows, gate_fields)
    _write_csv(output_dir / "error_percentiles.csv", percentile_rows, percentile_fields)
    _write_csv(output_dir / "bad_case_overlap.csv", bad_case_rows, bad_case_fields)

    summary_payload = {
        "methods": [
            {
                "name": method.name,
                "label": method.label,
                "metrics_path": str(method.metrics_path),
                "predictions_path": str(method.predictions_path),
            }
            for method in methods
        ],
        "alignment": alignment,
        "primary_metrics": overall_rows,
        "scenario_order": list(SCENARIO_ORDER),
        "single_seed_warning": True,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figure_paths: list[Path] = []
    if not args.no_plots:
        try:
            figure_paths = _plot_outputs(output_dir, methods, overall_rows, scenario_rows, gate_rows)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("matplotlib is not installed; skipped figure generation. Re-run with matplotlib or --no_plots.")

    report_path = _write_report(output_dir, methods, alignment, overall_rows, scenario_rows, gate_rows, figure_paths)

    print(f"Wrote analysis outputs to: {output_dir}")
    print(f"Wrote report: {report_path}")
    print("Primary ranking by vector_endpoint_error_mps:")
    for row in sorted(overall_rows, key=lambda r: _maybe_float(r["vector_endpoint_error_mps"])):
        print(f"  {row['method']}: {row['vector_endpoint_error_mps']:.6f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
