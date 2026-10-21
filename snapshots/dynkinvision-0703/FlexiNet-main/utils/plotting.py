from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def plot_true_vs_estimated_from_csv(
    predictions_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "val",
    first_n: int = 1000,
    max_full_points: int = 5000,
) -> None:
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_prediction_rows(predictions_csv)
    if not rows:
        return

    sample_indices = [int(row["sample_index"]) for row in rows]
    targets = [float(row["target_speed_mps"]) for row in rows]
    preds = [float(row["pred_speed_mps"]) for row in rows]

    full_step = max(len(rows) // max_full_points, 1)
    full_idx = list(range(0, len(rows), full_step))
    _plot_series_svg(
        path=output_dir / f"{prefix}_true_vs_estimated_comparison.svg",
        title="Pure Vision Speed Comparison: Full Validation Set",
        subtitle=f"Source: {predictions_csv.as_posix()}, n={len(rows)}",
        x_values=[sample_indices[i] for i in full_idx],
        series={
            "Ground truth": [targets[i] for i in full_idx],
            "Estimated": [preds[i] for i in full_idx],
        },
        width=1500,
        height=900,
    )

    n = min(int(first_n), len(rows))
    _plot_series_svg(
        path=output_dir / f"{prefix}_true_vs_estimated_zoom_first{n}.svg",
        title=f"Pure Vision Speed Comparison: First {n} Validation Samples",
        subtitle="Ground truth and estimated speed",
        x_values=list(range(n)),
        series={
            "Ground truth": targets[:n],
            "Estimated": preds[:n],
        },
        width=1300,
        height=520,
    )


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_series_svg(
    path: Path,
    title: str,
    subtitle: str,
    x_values: Sequence[int],
    series: dict[str, Sequence[float]],
    width: int,
    height: int,
) -> None:
    colors = {
        "Ground truth": "#1f77b4",
        "Estimated": "#d62728",
    }
    left = 80
    right = 60
    top = 100
    bottom = 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_y = [value for values in series.values() for value in values]
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = max((y_max - y_min) * 0.08, 0.1)
    y_min -= y_pad
    y_max += y_pad

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial, Helvetica, sans-serif;} .legend{font-size:13px;}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-size="22" font-weight="700" fill="#222222">{_escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="68" text-anchor="middle" font-size="14" fill="#555555">{_escape(subtitle)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#222222" stroke-width="1"/>',
    ]

    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = sy(y)
        svg.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#d9dde3" stroke-width="1"/>')
        svg.append(f'<text x="{left - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="12" fill="#222222">{y:.1f}</text>')

    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        px = sx(x)
        svg.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#d9dde3" stroke-width="1"/>')
        svg.append(f'<text x="{px:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="12" fill="#222222">{int(round(x))}</text>')

    svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 35}" text-anchor="middle" font-size="14" fill="#222222">Validation sample index</text>')
    svg.append(
        f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" fill="#222222" '
        f'transform="rotate(-90 28 {top + plot_h / 2:.1f})">Speed (m/s)</text>'
    )

    for label, values in series.items():
        points = " ".join(f"L {sx(x):.2f} {sy(y):.2f}" for x, y in zip(x_values, values))
        if points.startswith("L "):
            points = "M " + points[2:]
        svg.append(
            f'<path d="{points}" fill="none" stroke="{colors.get(label, "#333333")}" '
            f'stroke-width="2.0" opacity="0.95"/>'
        )

    legend_x = width - right - 165
    legend_y = top + 5
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="165" height="58" fill="white" stroke="#cccccc" opacity="0.95"/>')
    for i, label in enumerate(series.keys()):
        y = legend_y + 18 + i * 24
        svg.append(f'<line x1="{legend_x + 12}" y1="{y}" x2="{legend_x + 40}" y2="{y}" stroke="{colors.get(label, "#333333")}" stroke-width="3"/>')
        svg.append(f'<text class="legend" x="{legend_x + 48}" y="{y + 4}" fill="#222222">{_escape(label)}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
