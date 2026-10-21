"""Offline evaluation only. This program is never imported by inference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roll_velocity.io import write_json


def join_labels(predictions, truth, tolerance=1e-6):
    required = {"time", "vx", "vy"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Label CSV requires {sorted(required)} (SI units)")
    if truth.time.duplicated().any() or not np.isfinite(truth[list(required)].to_numpy(float)).all():
        raise ValueError("Label timestamps must be unique and labels finite")
    if predictions.duplicated(["method", "image_id"]).any():
        raise ValueError("Duplicate method/image_id predictions")
    truth = truth.rename(columns={"time": "truth_time", "vx": "truth_vx", "vy": "truth_vy"})
    selected = ["truth_time", "truth_vx", "truth_vy"] + (["roll"] if "roll" in truth else [])
    joined = pd.merge_asof(predictions.sort_values("time_s"), truth[selected].sort_values("truth_time"),
                           left_on="time_s", right_on="truth_time", direction="nearest", tolerance=tolerance)
    if joined.truth_time.isna().any():
        raise ValueError("Missing timestamp-matched labels; no interpolation or time-shift search is performed")
    joined["valid"] = joined.valid.astype(str).str.lower().isin(["true", "1"])
    joined["eligible"] = joined.reason.ne("warmup")
    finite = np.isfinite(joined[["vx_mps", "vy_mps"]].to_numpy(float)).all(1)
    if (joined.valid & ~finite).any():
        raise ValueError("A valid prediction contains non-finite speeds")
    joined["active"] = (np.degrees(joined["roll"]).abs() >= 1) & (joined.truth_vy.abs() >= .2) if "roll" in joined else False
    joined["quiet"] = (np.degrees(joined["roll"]).abs() < 1) & (joined.truth_vy.abs() < .05) if "roll" in joined else False
    joined["common_valid"] = joined.groupby("image_id").valid.transform("all")
    counts = joined.groupby("image_id").method.nunique()
    if not (counts == joined.method.nunique()).all():
        raise ValueError("All methods must contain the same image_id set, including invalid rows")
    return joined


def metrics(joined):
    result = []
    for method, part in joined.groupby("method"):
        for subset in ("all", "active", "quiet"):
            selected = part.eligible & (True if subset == "all" else part[subset])
            for population in ("own_valid", "common_valid"):
                valid = selected & part.valid & (True if population == "own_valid" else part.common_valid)
                for axis in ("vx", "vy"):
                    error = part.loc[valid, axis + "_mps"] - part.loc[valid, "truth_" + axis]
                    result.append(dict(method=method, subset=subset, population=population, axis=axis,
                                       eligible=int(selected.sum()), valid=int(valid.sum()),
                                       coverage=float(valid.sum() / selected.sum()) if selected.any() else np.nan,
                                       mae=error.abs().mean(), rmse=np.sqrt((error ** 2).mean()), bias=error.mean(),
                                       p95=error.abs().quantile(.95) if len(error) else np.nan))
    return pd.DataFrame(result)


def evaluate_run(prediction_file, truth_file, output, plot=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    joined = join_labels(pd.read_csv(prediction_file), pd.read_csv(truth_file))
    joined.to_csv(output / "per_frame.csv", index=False)
    summary = metrics(joined)
    summary.to_csv(output / "metrics.csv", index=False)
    zero = joined.drop_duplicates("image_id").query("eligible")
    baseline = []
    for subset in ("all", "active", "quiet"):
        part = zero if subset == "all" else zero[zero[subset]]
        baseline.append(dict(subset=subset, count=len(part), zero_vy_mae=part.truth_vy.abs().mean()))
    pd.DataFrame(baseline).to_csv(output / "zero_lateral_baseline.csv", index=False)
    write_json(output / "evaluation.json", dict(predictions=str(prediction_file), labels=str(truth_file),
               alignment="timestamp match <=1e-6 s, output frame time; no lag fitting",
               active="abs(roll in degrees)>=1 AND abs(truth_vy)>=0.2 m/s",
               quiet="abs(roll in degrees)<1 AND abs(truth_vy)<0.05 m/s",
               has_roll_labels="roll" in joined, warmup_excluded=True, invalid_filled_with_zero=False))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, axis in zip(axes, ("vx", "vy")):
            ax.plot(zero.time_s, zero["truth_" + axis], color="black", label="Ground truth", linewidth=1.8)
            for method, part in joined.groupby("method", sort=False):
                part = part.sort_values("time_s")
                # Keep invalid rows as NaN so the line never bridges a failed interval.
                values = part[axis + "_mps"].where(part.valid & part.eligible)
                ax.plot(part.time_s, values, label=method, linewidth=1)
            ax.set_ylabel(axis + " (m/s)")
            ax.grid(alpha=.25)
            ax.legend(ncol=3)
        axes[-1].set_xlabel("Output frame time (s)")
        fig.tight_layout()
        fig.savefig(output / "velocity.png", dpi=160)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = evaluate_run(args.predictions, args.truth, args.output, not args.no_plot)
    print(result.query("subset == 'all' and population == 'own_valid'").to_string(index=False))


if __name__ == "__main__":
    main()
