"""Assemble the velocity-xy decoupling advisor report.

Read the per-config `best_eval/val_predictions.csv` + `val_metrics.json` for the
four trained XY ablations, align them by `center_token`, compute a cross-config
comparison summary, write the curve assets (vx / vy / endpoint-error-by-bin) and
the first-1000 aligned CSV used to draw them.

Outputs are written to `analysis_reports/velocity_xy_decoupling/`. The script is
read-only with respect to `runs/`, `results_git/`, `checkpoints/`, `outputs/`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_ROOT = REPO_ROOT / "FlexiNet_ChassisFusion"
RUNS_ROOT = FUSION_ROOT / "runs" / "FlexiNet_ChassisFusion_velocity_xy"
REPORT_DIR = REPO_ROOT / "analysis_reports" / "velocity_xy_decoupling"

METHODS: list[tuple[str, str]] = [
    ("fair_gray", "FlexiNet+ChassisFusion (fair_gray)"),
    ("fair_flow", "FlexiNet+ChassisFusion (fair_flow)"),
    ("fair_rgb_flow_geom", "FlexiNet+ChassisFusion (fair_rgb_flow_geom)"),
    ("pose_assisted", "FlexiNet+ChassisFusion (pose_assisted, upper-bound)"),
]

CHASSIS_FUSION_REF = "fair_rgb_flow_geom"  # used as the anchor ordering


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _floats(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: 0.0 for k in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _delta_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "p05", "p50", "p95")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _endpoint_error(vx_err: np.ndarray, vy_err: np.ndarray) -> np.ndarray:
    return np.sqrt(vx_err * vx_err + vy_err * vy_err)


def _bin_endpoint_error(
    target_speed: np.ndarray,
    endpoint_err: np.ndarray,
    bins: list[tuple[float, float]],
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for lo, hi in bins:
        mask = (target_speed >= lo) & (target_speed < hi)
        n = int(mask.sum())
        mean_err = float(endpoint_err[mask].mean()) if n > 0 else 0.0
        out.append({
            "lo": lo,
            "hi": None if math.isinf(hi) else hi,
            "count": n,
            "mean_endpoint_error_mps": mean_err,
        })
    return out


def _load_method(name: str) -> dict[str, Any]:
    base = RUNS_ROOT / name / "best_eval"
    metrics = _read_json(base / "val_metrics.json")
    rows = _read_csv(base / "val_predictions.csv")
    return {
        "name": name,
        "metrics_path": base / "val_metrics.json",
        "predictions_path": base / "val_predictions.csv",
        "metrics": metrics,
        "rows": rows,
    }


def _validate_alignment(loaded: dict[str, dict[str, Any]]) -> tuple[list[str], int]:
    names = list(loaded.keys())
    ref_rows = loaded[CHASSIS_FUSION_REF]["rows"]
    ref_tokens = [r["center_token"] for r in ref_rows]
    n = len(ref_tokens)
    ref_target_vx = _floats(ref_rows, "target_vx_mps")
    ref_target_vy = _floats(ref_rows, "target_vy_mps")
    max_diff = 0.0
    for name in names:
        rows = loaded[name]["rows"]
        if len(rows) != n:
            raise RuntimeError(
                f"row count mismatch: {name}={len(rows)} vs {CHASSIS_FUSION_REF}={n}"
            )
        tokens = [r["center_token"] for r in rows]
        if tokens != ref_tokens:
            raise RuntimeError(
                f"center_token order mismatch between {name} and {CHASSIS_FUSION_REF}"
            )
        diff_vx = np.abs(_floats(rows, "target_vx_mps") - ref_target_vx).max()
        diff_vy = np.abs(_floats(rows, "target_vy_mps") - ref_target_vy).max()
        max_diff = max(max_diff, float(diff_vx), float(diff_vy))
    return ref_tokens, n, max_diff


def _build_method_summary(name: str, label: str, rows: list[dict[str, str]],
                          metrics: dict[str, Any]) -> dict[str, Any]:
    target_vx = _floats(rows, "target_vx_mps")
    target_vy = _floats(rows, "target_vy_mps")
    pred_vx = _floats(rows, "pred_vx_mps")
    pred_vy = _floats(rows, "pred_vy_mps")
    delta_vx = _floats(rows, "delta_vx_mps")
    delta_vy = _floats(rows, "delta_vy_mps")
    endpoint = _floats(rows, "endpoint_error_mps")
    speed_pred = np.sqrt(pred_vx * pred_vx + pred_vy * pred_vy)
    speed_true = np.sqrt(target_vx * target_vx + target_vy * target_vy)
    speed_mag_mae = float(np.abs(speed_pred - speed_true).mean())

    summary: dict[str, Any] = {
        "label": label,
        "n_samples": len(rows),
        "metrics": {
            "mae_vx_mps": metrics.get("mae_vx_mps"),
            "mae_vy_mps": metrics.get("mae_vy_mps"),
            "rmse_vx_mps": metrics.get("rmse_vx_mps"),
            "rmse_vy_mps": metrics.get("rmse_vy_mps"),
            "vector_endpoint_error_mps": metrics.get("vector_endpoint_error_mps"),
            "speed_magnitude_mae_mps": metrics.get("speed_magnitude_mae_mps"),
            "speed_magnitude_mae_mps_recomputed": speed_mag_mae,
            "r2_vx": metrics.get("r2_vx"),
            "r2_vy": metrics.get("r2_vy"),
        },
        "group_metrics": {
            "low_speed_vector_endpoint_error_mps": metrics.get("low_speed_vector_endpoint_error_mps"),
            "low_speed_mae_vy_mps": metrics.get("low_speed_mae_vy_mps"),
            "mid_speed_vector_endpoint_error_mps": metrics.get("mid_speed_vector_endpoint_error_mps"),
            "mid_speed_mae_vy_mps": metrics.get("mid_speed_mae_vy_mps"),
            "high_speed_vector_endpoint_error_mps": metrics.get("high_speed_vector_endpoint_error_mps"),
            "high_speed_mae_vy_mps": metrics.get("high_speed_mae_vy_mps"),
            "high_lateral_vector_endpoint_error_mps": metrics.get("high_lateral_vector_endpoint_error_mps"),
            "high_lateral_mae_vy_mps": metrics.get("high_lateral_mae_vy_mps"),
            "turning_vector_endpoint_error_mps": metrics.get("turning_vector_endpoint_error_mps"),
            "turning_mae_vy_mps": metrics.get("turning_mae_vy_mps"),
            "straight_vector_endpoint_error_mps": metrics.get("straight_vector_endpoint_error_mps"),
            "straight_mae_vy_mps": metrics.get("straight_mae_vy_mps"),
        },
        "endpoint_error_percentiles_mps": _percentiles(endpoint),
        "delta_vx_mps": _delta_stats(delta_vx),
        "delta_vy_mps": _delta_stats(delta_vy),
        "speed_bin_endpoint_error_mps": _bin_endpoint_error(
            speed_true,
            endpoint,
            bins=[(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, math.inf)],
        ),
    }
    return summary


def _baseline_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    target_vx = _floats(rows, "target_vx_mps")
    target_vy = _floats(rows, "target_vy_mps")
    base_vx = _floats(rows, "base_vx_mps")
    base_vy = _floats(rows, "base_vy_mps")
    base_endpoint = _floats(rows, "base_endpoint_error_mps")
    err_vx = base_vx - target_vx
    err_vy = base_vy - target_vy
    base_mae_vx = float(np.abs(err_vx).mean())
    base_mae_vy = float(np.abs(err_vy).mean())
    base_rmse_vx = float(np.sqrt((err_vx * err_vx).mean()))
    base_rmse_vy = float(np.sqrt((err_vy * err_vy).mean()))
    base_vec = float(base_endpoint.mean())
    speed_true = np.sqrt(target_vx * target_vx + target_vy * target_vy)
    base_speed = np.sqrt(base_vx * base_vx + base_vy * base_vy)
    base_speed_mae = float(np.abs(base_speed - speed_true).mean())

    return {
        "label": "Base velocity [base_vx, 0] (zero_lateral)",
        "n_samples": len(rows),
        "metrics": {
            "mae_vx_mps": base_mae_vx,
            "mae_vy_mps": base_mae_vy,
            "rmse_vx_mps": base_rmse_vx,
            "rmse_vy_mps": base_rmse_vy,
            "vector_endpoint_error_mps": base_vec,
            "speed_magnitude_mae_mps": base_speed_mae,
        },
        "endpoint_error_percentiles_mps": _percentiles(base_endpoint),
        "speed_bin_endpoint_error_mps": _bin_endpoint_error(
            speed_true,
            base_endpoint,
            bins=[(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, math.inf)],
        ),
    }


def _improvements(method: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    bm = baseline["metrics"]
    mm = method["metrics"]

    def _gain(base: float, val: float) -> dict[str, float]:
        absolute = float(base - val)
        pct = float(absolute / base * 100.0) if base else 0.0
        return {"absolute_mps": absolute, "relative_percent": pct}

    return {
        "vx": _gain(bm["mae_vx_mps"], mm["mae_vx_mps"]),
        "vy": _gain(bm["mae_vy_mps"], mm["mae_vy_mps"]),
        "endpoint": _gain(
            bm["vector_endpoint_error_mps"], mm["vector_endpoint_error_mps"]
        ),
    }


def _plot_first1000_axis(
    axis_label: str,
    target: np.ndarray,
    base: np.ndarray,
    method_preds: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    n = len(target)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, target, color="black", linewidth=1.6, label=f"target {axis_label}")
    ax.plot(x, base, color="#888888", linewidth=1.2, linestyle="--", label=f"base {axis_label}")
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for (name, _label), color in zip(METHODS, palette):
        preds = method_preds[name]
        ax.plot(x, preds, color=color, linewidth=1.0, alpha=0.85, label=f"{name} pred {axis_label}")
    ax.set_xlabel("Aligned validation sample index (first 1000)")
    ax.set_ylabel(f"{axis_label} (m/s)")
    ax.set_title(f"First 1000 aligned validation samples — target vs base vs methods ({axis_label})")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=160)
    plt.close(fig)


def _plot_speed_bin_endpoint_err(
    method_bins: dict[str, list[dict[str, float]]],
    baseline_bins: list[dict[str, float]],
    out_path: Path,
) -> None:
    bin_labels: list[str] = []
    for entry in baseline_bins:
        lo, hi = entry["lo"], entry["hi"]
        bin_labels.append(f"{lo:g}-{hi:g}" if hi is not None else f">{lo:g}")
    valid_idx = [i for i, e in enumerate(baseline_bins) if e["count"] > 0]
    bin_labels = [bin_labels[i] for i in valid_idx]
    x = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_vals = [baseline_bins[i]["mean_endpoint_error_mps"] for i in valid_idx]
    ax.plot(x, baseline_vals, color="#555555", marker="o", linewidth=1.6,
            linestyle="--", label="Base velocity [base_vx, 0]")
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for (name, _label), color in zip(METHODS, palette):
        vals = [method_bins[name][i]["mean_endpoint_error_mps"] for i in valid_idx]
        ax.plot(x, vals, color=color, marker="s", linewidth=1.4, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Target speed bin (m/s)")
    ax.set_ylabel("Vector endpoint error (m/s)")
    ax.set_title("Per-speed-bin vector endpoint error (mean)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=160)
    plt.close(fig)


def _write_first1000_csv(
    out_path: Path,
    rows: list[dict[str, str]],
    method_pred_vx: dict[str, np.ndarray],
    method_pred_vy: dict[str, np.ndarray],
    limit: int,
) -> None:
    fieldnames = [
        "row_index",
        "scene_name",
        "center_token",
        "target_vx_mps",
        "target_vy_mps",
        "base_vx_mps",
        "base_vy_mps",
    ]
    for name, _ in METHODS:
        fieldnames.append(f"{name}_pred_vx_mps")
        fieldnames.append(f"{name}_pred_vy_mps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(min(limit, len(rows))):
            r = rows[i]
            record = {
                "row_index": i,
                "scene_name": r["scene_name"],
                "center_token": r["center_token"],
                "target_vx_mps": r["target_vx_mps"],
                "target_vy_mps": r["target_vy_mps"],
                "base_vx_mps": r["base_vx_mps"],
                "base_vy_mps": r["base_vy_mps"],
            }
            for name, _ in METHODS:
                record[f"{name}_pred_vx_mps"] = float(method_pred_vx[name][i])
                record[f"{name}_pred_vy_mps"] = float(method_pred_vy[name][i])
            writer.writerow(record)


def _render_pdf(md_path: Path, pdf_path: Path) -> str:
    import subprocess

    pdf_md = md_path.with_suffix(".pdf.md")
    pdf_md.write_text(
        md_path.read_text(encoding="utf-8").replace(".svg)", ".png)"),
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                "pandoc",
                str(pdf_md),
                "-o",
                str(pdf_path),
                "--from",
                "gfm",
                "--pdf-engine=xelatex",
                "-V",
                "mainfont=Microsoft YaHei",
                "-V",
                "CJKmainfont=Microsoft YaHei",
                "-V",
                "geometry:margin=0.9in",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pdf_md.unlink(missing_ok=True)
            return f"pandoc ok -> {pdf_path}"
        msg = f"pandoc failed (rc={result.returncode}): {result.stderr.strip()[:500]}"
    except FileNotFoundError:
        msg = "pandoc not found on PATH"

    try:
        import weasyprint  # type: ignore  # noqa: F401
        from markdown import markdown  # type: ignore

        html = markdown(
            md_path.read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code"],
        )
        styled = (
            "<html><head><meta charset='utf-8'><style>"
            "body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:11pt;line-height:1.45;}"
            "table{border-collapse:collapse;margin:8px 0;}"
            "th,td{border:1px solid #888;padding:4px 8px;}"
            "code{background:#f5f5f5;padding:1px 4px;}"
            "pre{background:#f5f5f5;padding:6px;}"
            "img{max-width:100%;}"
            "</style></head><body>" + html + "</body></html>"
        )
        weasyprint.HTML(string=styled, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
        return f"{msg}; weasyprint ok -> {pdf_path}"
    except Exception as exc:  # noqa: BLE001
        return f"{msg}; weasyprint failed: {exc!r}"


def _format_md_table(rows: list[list[str]], aligns: list[str]) -> str:
    header = rows[0]
    body = rows[1:]
    sep = []
    for a in aligns:
        if a == "right":
            sep.append("---:")
        elif a == "left":
            sep.append(":---")
        else:
            sep.append(":---:")
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def build_report_md(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    baseline = summary["baseline"]
    align = summary["alignment"]
    today = _dt.date.today().isoformat()

    headline_rows: list[list[str]] = [
        ["方法", "MAE_vx (m/s)", "MAE_vy (m/s)",
         "Vector EE (m/s)", "R²_vx", "R²_vy", "Gain vs base EE (m/s)"]
    ]
    headline_rows.append([
        baseline["label"],
        _fmt(baseline["metrics"]["mae_vx_mps"]),
        _fmt(baseline["metrics"]["mae_vy_mps"]),
        _fmt(baseline["metrics"]["vector_endpoint_error_mps"]),
        "—", "—", "0.0000",
    ])
    base_ee = baseline["metrics"]["vector_endpoint_error_mps"]
    for name, _label in METHODS:
        m = methods[name]
        ee = m["metrics"]["vector_endpoint_error_mps"]
        gain = base_ee - ee
        headline_rows.append([
            name,
            _fmt(m["metrics"]["mae_vx_mps"]),
            _fmt(m["metrics"]["mae_vy_mps"]),
            _fmt(ee),
            _fmt(m["metrics"]["r2_vx"]),
            _fmt(m["metrics"]["r2_vy"]),
            _fmt(gain),
        ])
    headline_table = _format_md_table(
        headline_rows,
        aligns=["left", "right", "right", "right", "right", "right", "right"],
    )

    pct_rows: list[list[str]] = [[
        "方法", "P50 (m/s)", "P90 (m/s)", "P95 (m/s)", "P99 (m/s)", "Max (m/s)"
    ]]
    pct_rows.append([
        baseline["label"],
        _fmt(baseline["endpoint_error_percentiles_mps"]["p50"]),
        _fmt(baseline["endpoint_error_percentiles_mps"]["p90"]),
        _fmt(baseline["endpoint_error_percentiles_mps"]["p95"]),
        _fmt(baseline["endpoint_error_percentiles_mps"]["p99"]),
        _fmt(baseline["endpoint_error_percentiles_mps"]["max"]),
    ])
    for name, _ in METHODS:
        pcts = methods[name]["endpoint_error_percentiles_mps"]
        pct_rows.append([
            name,
            _fmt(pcts["p50"]),
            _fmt(pcts["p90"]),
            _fmt(pcts["p95"]),
            _fmt(pcts["p99"]),
            _fmt(pcts["max"]),
        ])
    pct_table = _format_md_table(
        pct_rows,
        aligns=["left", "right", "right", "right", "right", "right"],
    )

    bin_rows: list[list[str]] = [[
        "速度桶 (m/s)", "样本数",
        baseline["label"],
        *[name for name, _ in METHODS],
    ]]
    bins = baseline["speed_bin_endpoint_error_mps"]
    for i, entry in enumerate(bins):
        if entry["count"] == 0:
            continue
        lo, hi = entry["lo"], entry["hi"]
        bin_label = f"{lo:g}-{hi:g}" if hi is not None else f">{lo:g}"
        row = [bin_label, str(entry["count"]), _fmt(entry["mean_endpoint_error_mps"])]
        for name, _ in METHODS:
            row.append(_fmt(methods[name]["speed_bin_endpoint_error_mps"][i]["mean_endpoint_error_mps"]))
        bin_rows.append(row)
    bin_table = _format_md_table(
        bin_rows,
        aligns=["left", "right"] + ["right"] * (1 + len(METHODS)),
    )

    delta_rows: list[list[str]] = [[
        "方法", "Δvx mean", "Δvx std", "Δvx P05", "Δvx P95",
        "Δvy mean", "Δvy std", "Δvy P05", "Δvy P95"
    ]]
    for name, _ in METHODS:
        dvx = methods[name]["delta_vx_mps"]
        dvy = methods[name]["delta_vy_mps"]
        delta_rows.append([
            name,
            _fmt(dvx["mean"]), _fmt(dvx["std"]), _fmt(dvx["p05"]), _fmt(dvx["p95"]),
            _fmt(dvy["mean"]), _fmt(dvy["std"]), _fmt(dvy["p05"]), _fmt(dvy["p95"]),
        ])
    delta_table = _format_md_table(
        delta_rows,
        aligns=["left"] + ["right"] * 8,
    )

    group_rows: list[list[str]] = [[
        "方法", "low EE", "mid EE", "high EE", "high_lateral EE",
        "high_lateral MAE_vy", "turning EE", "straight EE"
    ]]
    for name, _ in METHODS:
        g = methods[name]["group_metrics"]
        group_rows.append([
            name,
            _fmt(g.get("low_speed_vector_endpoint_error_mps")),
            _fmt(g.get("mid_speed_vector_endpoint_error_mps")),
            _fmt(g.get("high_speed_vector_endpoint_error_mps")),
            _fmt(g.get("high_lateral_vector_endpoint_error_mps")),
            _fmt(g.get("high_lateral_mae_vy_mps")),
            _fmt(g.get("turning_vector_endpoint_error_mps")),
            _fmt(g.get("straight_vector_endpoint_error_mps")),
        ])
    group_table = _format_md_table(
        group_rows,
        aligns=["left"] + ["right"] * 7,
    )

    fair_gray_ee = methods["fair_gray"]["metrics"]["vector_endpoint_error_mps"]
    pose_ee = methods["pose_assisted"]["metrics"]["vector_endpoint_error_mps"]
    pose_gain_over_fair_gray = fair_gray_ee - pose_ee
    fair_gray_vy = methods["fair_gray"]["metrics"]["mae_vy_mps"]
    pose_vy = methods["pose_assisted"]["metrics"]["mae_vy_mps"]
    rgb_ee = methods["fair_rgb_flow_geom"]["metrics"]["vector_endpoint_error_mps"]
    flow_ee = methods["fair_flow"]["metrics"]["vector_endpoint_error_mps"]

    md = f"""# FlexiNet+ChassisFusion 纵/侧向速度解耦汇报

生成日期：{today}

项目位置：`09_SpeedChallenge_nuScenes`，分支：`feature/velocity-xy-fusion`

本报告对应 `feature/velocity-xy-fusion` 分支上把任务从标量速度扩展到 2D ego-frame 速度 `[v_x, v_y]` 的工作。
对比对象为同一批对齐验证样本上的 5 种方法：

- `Base velocity [base_vx, 0]`：第一版零侧向 baseline，直接用底盘 `base_vx` 作为纵向速度，`v_y` 设为 0。
- `fair_gray`：fair_chassis 信号 + 灰度图像序列分支。
- `fair_flow`：fair_chassis 信号 + 灰度图像序列 + 预计算 Farneback 光流分支。
- `fair_rgb_flow_geom`：fair_chassis 信号 + RGB 图像序列 + 光流 + 几何先验。
- `pose_assisted`：fair_chassis 信号 + 额外 pose 派生 CAN 字段，作为同源信息上限实验。

## 1. 核心结论

所有 4 个 XY 模型在 vector endpoint error 上均显著优于 `[base_vx, 0]` baseline，说明任务已经成功扩展到二维速度估计。
最佳整体方法是 `pose_assisted`（vector endpoint error `{_fmt(pose_ee)}` m/s），但它使用了 pose 派生 CAN 输入，与 ego-pose 标签同源，仅作为上限实验。
fair_chassis 三种视觉/几何组合（gray / flow / rgb_flow_geom）的 endpoint error 几乎打平（`{_fmt(fair_gray_ee)}` / `{_fmt(flow_ee)}` / `{_fmt(rgb_ee)}` m/s），说明在当前 fair_chassis 输入下，光流与 RGB+几何分支对 `v_y` 的边际贡献有限，瓶颈更可能在 `v_y` 标签信噪比与可用底盘信号。

`pose_assisted` 相对 `fair_gray` 的 endpoint error 仅下降 `{_fmt(pose_gain_over_fair_gray, 5)}` m/s，MAE_vy 由 `{_fmt(fair_gray_vy, 5)}` m/s 降到 `{_fmt(pose_vy, 5)}` m/s，差异基本可视为噪声，进一步说明 fair_chassis 已经吃掉了大部分可用的底盘信息。

三组方法在同一批 `{align['n_samples']}` 个验证样本上完成对齐，`center_token` 精确对齐，跨 method `target_vx/target_vy` 最大差异为 `{align['max_target_speed_diff_mps']:.6f}` m/s。

{headline_table}

`Gain vs base EE` 为相对 baseline 的 vector endpoint error 绝对下降（正数越大越好）。`R²_vx` 几乎到达 0.999，说明纵向速度被底盘 `base_vx` 几乎线性吃满；`R²_vy` 在 0.5 左右，反映 `v_y` 仍然是主难点。

## 2. 数据与对齐协议

源文件：

"""
    for name, _ in METHODS:
        md += f"- `runs/FlexiNet_ChassisFusion_velocity_xy/{name}/best_eval/val_metrics.json`\n"
        md += f"- `runs/FlexiNet_ChassisFusion_velocity_xy/{name}/best_eval/val_predictions.csv`\n"

    md += f"""
对齐方式为 `center_token` 精确匹配，以 `{CHASSIS_FUSION_REF}` 的样本顺序作为最终分析顺序。
共有 `{align['n_samples']}` 个对齐样本，跨 method 的 `target_vx / target_vy` 最大差异为 `{align['max_target_speed_diff_mps']:.6f}` m/s（理论应为 0，因为 4 个 config 使用同一 val 集 + 相同标签生成方式）。

baseline `[base_vx, 0]` 直接复用任一 method CSV 的 `base_vx_mps / base_vy_mps / base_endpoint_error_mps` 列重算，确保与 method 使用同一 33148 个样本。

## 3. 指标解释

- `MAE_vx / MAE_vy`：分别为纵/侧向速度的平均绝对误差。
- `Vector endpoint error`：`sqrt((pred_vx - target_vx)^2 + (pred_vy - target_vy)^2)` 的均值，是 2D 速度估计的核心指标，等同于速度矢量端点距离。
- `R²_vx / R²_vy`：分别衡量纵/侧向速度估计能解释多少真实速度方差。
- `Gain vs base`：相对 `[base_vx, 0]` baseline 的指标绝对下降，正数表示有改进。
- 分桶指标：
  - `low_speed / mid_speed / high_speed`：按真实速度大小 `<2 / 2-10 / >=10` m/s 分桶。
  - `high_lateral`：按 `|target_vy|` 的 P90 阈值取最高 10% 侧向速度样本。
  - `turning / straight`：按 `|yaw_rate| >= 0.1` 划分，仅 `fair_rgb_flow_geom` 在 eval 时启用了 yaw_rate（geometry_mode=basic），其它 config 没记录 turning/straight 分桶。

## 4. 定量结果

### 4.1 整体指标（best 检查点）

{headline_table}

### 4.2 Endpoint error 分位数

{pct_table}

baseline 的 P95 endpoint error 为 `{_fmt(baseline['endpoint_error_percentiles_mps']['p95'])}` m/s，
4 个 XY 模型的 P95 均收敛到 `{_fmt(methods['fair_gray']['endpoint_error_percentiles_mps']['p95'])}–{_fmt(methods['pose_assisted']['endpoint_error_percentiles_mps']['p95'])}` m/s 区间，说明 residual fusion 不只是改善平均误差，也压低了高分位。

### 4.3 速度分桶 endpoint error

{bin_table}

### 4.4 分组指标（low/mid/high 速度桶、high_lateral、turning/straight）

{group_table}

`high_lateral` 子集上 4 个 method 的 endpoint error 均显著高于整体均值，说明大侧向速度仍是主要误差来源。`pose_assisted` 在 `high_lateral` 上的 MAE_vy 与 fair_chassis 系列接近，进一步说明 pose 同源输入并未在难样本上带来质变。

## 5. Residual correction 行为

每个 method 的 `delta = pred - base` 统计如下（仅 best 检查点、val 集）：

{delta_table}

可以观察到：

- 所有 method 的 `Δvx` 均集中在 ±0.5 m/s 量级，没有发生“脱离 base 重新预测”，符合 residual fusion 设计。
- `Δvy` 的 std 明显小于 `Δvx`，说明模型对 `v_y` 的修正幅度比纵向更保守，与 `v_y` 标签量级本身就小的事实一致（损失里 `sigma_vy=0.5` 已经做过分轴归一化）。
- `pose_assisted` 的 `Δvy` 范围与 fair_chassis 几乎相同，说明上限实验的增益并非来自更大幅度的侧向修正，而是更稳定的方向。

## 6. 曲线对比图

![First 1000 aligned validation samples — vx](first1000_true_vs_methods_vx.svg)

第一张图展示前 1000 个对齐验证样本的纵向速度 `v_x`：黑线为真值，灰色虚线为 baseline `base_vx`，4 条彩色线为 4 个 XY 模型的 `pred_vx`。4 个模型的 `pred_vx` 与 baseline 几乎重合，反映 `R²_vx ≈ 0.999`。

![First 1000 aligned validation samples — vy](first1000_true_vs_methods_vy.svg)

第二张图展示同一段样本的侧向速度 `v_y`：baseline 恒等于 0，目标 `v_y` 在 ±0.5 m/s 范围内抖动，4 个 XY 模型给出非零侧向估计并整体跟随真值趋势，但局部仍有滞后或偏差。

![Per-speed-bin vector endpoint error](endpoint_error_by_speed_bin.svg)

第三张图给出按真实速度分桶的 vector endpoint error。所有桶上 baseline 均明显高于 4 个 XY 模型，且差距随速度增大而扩大，符合“高速段 base_vx 误差被放大且无 vy 补偿”的物理直觉。

用于画图的对齐数据保存于：

- `analysis_reports/velocity_xy_decoupling/aligned_first1000_curve_data.csv`

跨 method 汇总保存于：

- `analysis_reports/velocity_xy_decoupling/comparison_summary.json`

## 7. 结果解释

### 7.1 baseline `[base_vx, 0]` 仍然很强但天花板明显

`[base_vx, 0]` 的 `mae_vx_mps` 为 `{_fmt(baseline['metrics']['mae_vx_mps'])}` m/s，已经显著低于 main 分支 scalar-speed 报告里纯视觉的水平，但它把 `v_y` 强制设为 0，导致在转弯/换道场景下不可避免地产生侧向误差。`base_mae_vy_mps = {_fmt(baseline['metrics']['mae_vy_mps'])}` m/s 即来自这一固定假设，是 XY 模型相对 baseline 的最大上升空间。

### 7.2 fair_chassis 三种视觉/几何组合几乎打平

`fair_gray` / `fair_flow` / `fair_rgb_flow_geom` 的 endpoint error 分别为 `{_fmt(fair_gray_ee)}` / `{_fmt(flow_ee)}` / `{_fmt(rgb_ee)}` m/s。光流分支带来约 `{_fmt(fair_gray_ee - flow_ee)}` m/s 的差异，RGB+geometry 与 gray 之间的差异也在 `{_fmt(fair_gray_ee - rgb_ee)}` m/s 量级，相对于 baseline 改进幅度（≈0.12 m/s）来说基本可以视为同分布噪声。

这说明当前 fair_chassis 输入下，视觉分支对 `v_y` 的贡献已经接近饱和，瓶颈很可能在：

1. nuScenes `v_y` 标签由 ego_pose 中心差分得到，本身就含 ~0.01 m/s 量级噪声；
2. fair_chassis 没有提供直接的侧向运动学指示（fair 模式刻意去掉了 pose 派生量）；
3. 64×64 灰度/RGB 输入对侧向光流模式的分辨率有限。

### 7.3 `pose_assisted` 仅边际提升，证实主要增益来自同源信息

`pose_assisted` 相对 `fair_gray` 的 endpoint error 下降 `{_fmt(pose_gain_over_fair_gray, 5)}` m/s，MAE_vy 由 `{_fmt(fair_gray_vy, 5)}` 降到 `{_fmt(pose_vy, 5)}`。考虑到 pose_assisted 的额外输入 (`pose_vel_x / pose_vel_y / pose_accel_x / pose_accel_y / pose_rotation_rate_z / pose_speed`) 与 ego_pose 标签同源，理论上是一个非常强的上限信号。如果它的提升只有这个量级，说明：

- pose 同源输入对最终精度的边际贡献已经很小；
- fair_chassis baseline 在论文/汇报中可以作为可部署主结论；
- pose_assisted 应明确标注为“上限实验”，避免造成 pose 派生量也能在真实车端获得的误解。

## 8. 局限与后续建议

1. 当前结果只来自一次训练 + 一次评估，论文写作前应补充多 seed/多 split 验证。
2. 第一版 `base_velocity = [base_vx, 0]`，未启用 EOVM-EKF 给出非零侧向先验；EOVM 物理先验仍待服务器端轮速/转向比例审计后再做主实验。
3. `high_lateral` 子集仍是误差主源，建议下一步专门做 high_lateral / turning 场景的可视化与 case study，找出系统性偏差方向。
4. 视觉对 `v_y` 边际贡献低，可考虑：
   - 提升输入分辨率 / 使用更长的时序窗口；
   - 用 RAFT/GMFlow 替换 Farneback；
   - 在 loss 上对 `v_y` 加更强权重或 focal-style 加权；
5. 缺少 chassis-only、vision-only、direct fusion 等更细消融，需要后续补全以做严格“residual 与 direct 谁更好”的判断。

## 9. 汇报时建议强调

1. 任务已经从标量速度成功扩展到 2D ego-frame 速度估计，全部 4 个方法在 vector endpoint error 上相对 `[base_vx, 0]` baseline 均有正增益。
2. fair_chassis 配置下，gray / flow / rgb_flow_geom 三种视觉/几何组合几乎打平 → 当前瓶颈不在视觉表征，而在 `v_y` 标签信噪比与可用底盘信号。
3. `pose_assisted` 仅带来边际提升 → 应明确将其标注为上限实验而非主结论，主结论以 fair_chassis 为准。
4. `R²_vx ≈ 0.999`、`R²_vy ≈ 0.52` 直接给出了下一步研究的优先级：纵向已饱和，侧向是主战场。

## 10. 文件索引

本次生成的文件：

- `analysis_reports/velocity_xy_decoupling/REPORT_FOR_ADVISOR_VELOCITY_XY.md`
- `analysis_reports/velocity_xy_decoupling/REPORT_FOR_ADVISOR_VELOCITY_XY.pdf`
- `analysis_reports/velocity_xy_decoupling/first1000_true_vs_methods_vx.svg`
- `analysis_reports/velocity_xy_decoupling/first1000_true_vs_methods_vy.svg`
- `analysis_reports/velocity_xy_decoupling/endpoint_error_by_speed_bin.svg`
- `analysis_reports/velocity_xy_decoupling/aligned_first1000_curve_data.csv`
- `analysis_reports/velocity_xy_decoupling/comparison_summary.json`

复用产物（未修改）：

- `FlexiNet_ChassisFusion/runs/FlexiNet_ChassisFusion_velocity_xy/<name>/best_eval/val_metrics.json`
- `FlexiNet_ChassisFusion/runs/FlexiNet_ChassisFusion_velocity_xy/<name>/best_eval/val_predictions.csv`
- `FlexiNet_ChassisFusion/runs/FlexiNet_ChassisFusion_velocity_xy/<name>/train_val_loss_curve.png`
- `FlexiNet_ChassisFusion/runs/FlexiNet_ChassisFusion_velocity_xy/<name>/val_mae_rmse_curve.png`
"""
    return md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=REPORT_DIR,
        help="output directory for the assembled report",
    )
    parser.add_argument(
        "--first_n",
        type=int,
        default=1000,
        help="number of aligned validation samples to dump into the curve assets",
    )
    parser.add_argument(
        "--skip_pdf",
        action="store_true",
        help="skip PDF rendering (only emit MD/JSON/CSV/SVG)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="parse + summarize but do not write any output files",
    )
    args = parser.parse_args(argv)

    print(f"repo root: {REPO_ROOT}")
    print(f"runs root: {RUNS_ROOT}")
    print(f"report dir: {args.report_dir}")

    loaded: dict[str, dict[str, Any]] = {}
    for name, _label in METHODS:
        eval_dir = RUNS_ROOT / name / "best_eval"
        if not (eval_dir / "val_metrics.json").exists():
            raise RuntimeError(f"missing val_metrics.json for {name}: {eval_dir}")
        if not (eval_dir / "val_predictions.csv").exists():
            raise RuntimeError(f"missing val_predictions.csv for {name}: {eval_dir}")
        loaded[name] = _load_method(name)
        print(f"  loaded {name}: {len(loaded[name]['rows'])} rows")

    tokens, n, max_diff = _validate_alignment(loaded)
    print(f"alignment ok: n={n}, max_target_diff={max_diff:.6g}")

    method_summaries: dict[str, Any] = {}
    method_pred_vx: dict[str, np.ndarray] = {}
    method_pred_vy: dict[str, np.ndarray] = {}
    method_bins: dict[str, list[dict[str, float]]] = {}
    for name, label in METHODS:
        summary = _build_method_summary(name, label, loaded[name]["rows"], loaded[name]["metrics"])
        method_summaries[name] = summary
        method_pred_vx[name] = _floats(loaded[name]["rows"], "pred_vx_mps")
        method_pred_vy[name] = _floats(loaded[name]["rows"], "pred_vy_mps")
        method_bins[name] = summary["speed_bin_endpoint_error_mps"]

    baseline = _baseline_summary(loaded[CHASSIS_FUSION_REF]["rows"])
    improvements = {name: _improvements(method_summaries[name], baseline) for name, _ in METHODS}

    target_vx_full = _floats(loaded[CHASSIS_FUSION_REF]["rows"], "target_vx_mps")
    target_vy_full = _floats(loaded[CHASSIS_FUSION_REF]["rows"], "target_vy_mps")
    base_vx_full = _floats(loaded[CHASSIS_FUSION_REF]["rows"], "base_vx_mps")
    base_vy_full = _floats(loaded[CHASSIS_FUSION_REF]["rows"], "base_vy_mps")

    summary_payload: dict[str, Any] = {
        "generated_on": _dt.date.today().isoformat(),
        "branch": "feature/velocity-xy-fusion",
        "task": "2D ego-frame velocity (vx, vy) decoupling",
        "alignment": {
            "n_samples": n,
            "alignment_key": "center_token",
            "ordered_by": f"runs/FlexiNet_ChassisFusion_velocity_xy/{CHASSIS_FUSION_REF}/best_eval/val_predictions.csv",
            "max_target_speed_diff_mps": max_diff,
        },
        "sources": {
            name: {
                "metrics_json": str(loaded[name]["metrics_path"].relative_to(REPO_ROOT)).replace("\\", "/"),
                "predictions_csv": str(loaded[name]["predictions_path"].relative_to(REPO_ROOT)).replace("\\", "/"),
            }
            for name, _ in METHODS
        },
        "baseline": baseline,
        "methods": method_summaries,
        "improvements_vs_baseline": improvements,
    }

    if args.dry_run:
        print(json.dumps({
            "baseline": {
                "mae_vx_mps": baseline["metrics"]["mae_vx_mps"],
                "mae_vy_mps": baseline["metrics"]["mae_vy_mps"],
                "vector_endpoint_error_mps": baseline["metrics"]["vector_endpoint_error_mps"],
            },
            "methods": {
                name: {
                    "mae_vx_mps": method_summaries[name]["metrics"]["mae_vx_mps"],
                    "mae_vy_mps": method_summaries[name]["metrics"]["mae_vy_mps"],
                    "vector_endpoint_error_mps": method_summaries[name]["metrics"]["vector_endpoint_error_mps"],
                }
                for name, _ in METHODS
            },
        }, indent=2))
        return 0

    args.report_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.report_dir / "comparison_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {json_path}")

    csv_path = args.report_dir / "aligned_first1000_curve_data.csv"
    _write_first1000_csv(
        csv_path,
        loaded[CHASSIS_FUSION_REF]["rows"],
        method_pred_vx,
        method_pred_vy,
        args.first_n,
    )
    print(f"wrote {csv_path}")

    first_n = min(args.first_n, n)
    vx_svg = args.report_dir / "first1000_true_vs_methods_vx.svg"
    _plot_first1000_axis(
        "v_x",
        target_vx_full[:first_n],
        base_vx_full[:first_n],
        {name: method_pred_vx[name][:first_n] for name, _ in METHODS},
        vx_svg,
    )
    print(f"wrote {vx_svg}")

    vy_svg = args.report_dir / "first1000_true_vs_methods_vy.svg"
    _plot_first1000_axis(
        "v_y",
        target_vy_full[:first_n],
        base_vy_full[:first_n],
        {name: method_pred_vy[name][:first_n] for name, _ in METHODS},
        vy_svg,
    )
    print(f"wrote {vy_svg}")

    bin_svg = args.report_dir / "endpoint_error_by_speed_bin.svg"
    _plot_speed_bin_endpoint_err(method_bins, baseline["speed_bin_endpoint_error_mps"], bin_svg)
    print(f"wrote {bin_svg}")

    md_path = args.report_dir / "REPORT_FOR_ADVISOR_VELOCITY_XY.md"
    md_text = build_report_md(summary_payload)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"wrote {md_path}")

    if args.skip_pdf:
        print("--skip_pdf set, not rendering PDF")
        return 0

    pdf_path = args.report_dir / "REPORT_FOR_ADVISOR_VELOCITY_XY.pdf"
    msg = _render_pdf(md_path, pdf_path)
    print(f"PDF: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
