"""Output-frame evaluation with explicit target contracts and failure coverage."""
from pathlib import Path

import numpy as np
import pandas as pd

from .io import artifact_path, sha256, write_json
from .training import load_labels


def join_labels(predictions, truth, spec, tolerance=1e-6):
    required = {"method", "image_id", "time_s", "valid", "reason", "vx_mps", "vy_mps", "target_version", "reference_point"}
    if not required.issubset(predictions.columns):
        raise ValueError("Missing prediction fields")
    if predictions.duplicated(["method", "image_id"]).any() or predictions.duplicated(["method", "time_s"]).any():
        raise ValueError("Duplicate prediction IDs/timestamps")
    if truth.time.duplicated().any() or not np.isfinite(truth[["time", "vx", "vy"]].to_numpy(float)).all():
        raise ValueError("Invalid labels")
    if not predictions.target_version.eq(spec["version"]).all() or not predictions.reference_point.eq(spec["reference_point"]).all():
        raise ValueError("Prediction/label coordinate contract mismatch; evaluate legacy baselines separately")
    counts = predictions.groupby("image_id").method.nunique()
    if not counts.eq(predictions.method.nunique()).all():
        raise ValueError("All methods must include the same frames, including failures")
    if not predictions.groupby("image_id").time_s.nunique().eq(1).all():
        raise ValueError("Methods disagree on frame timestamps")
    labels = truth.rename(columns={"time": "truth_time", "vx": "truth_vx", "vy": "truth_vy"})
    selected = ["truth_time", "truth_vx", "truth_vy"] + (["roll"] if "roll" in labels else [])
    joined = pd.merge_asof(predictions.sort_values("time_s"), labels[selected].sort_values("truth_time"),
                           left_on="time_s", right_on="truth_time", direction="nearest", tolerance=tolerance)
    joined["eligible"] = joined.reason.ne("warmup")
    # Centered pose labels can omit the last sequence frame; callers must slice
    # predictions explicitly instead of silently discarding unmatched eligible rows.
    if (joined.eligible & joined.truth_time.isna()).any():
        raise ValueError("Missing labels at output frame times; no lag fitting or implicit row dropping")
    joined["valid"] = joined.valid.astype(str).str.lower().isin(["true", "1"])
    finite = np.isfinite(joined[["vx_mps", "vy_mps"]].to_numpy(float)).all(1)
    if (joined.valid & ~finite).any():
        raise ValueError("Valid prediction contains non-finite speeds")
    joined["common_valid"] = joined.groupby("image_id").valid.transform("all")
    joined["active"] = (np.degrees(joined.roll).abs() >= 1) & (joined.truth_vy.abs() >= .2) if "roll" in joined else False
    joined["quiet"] = (np.degrees(joined.roll).abs() < 1) & (joined.truth_vy.abs() < .05) if "roll" in joined else False
    return joined


def metrics(joined):
    rows = []
    for method, part in joined.groupby("method"):
        for subset in ["all", "active", "quiet"]:
            selected = part.eligible & (True if subset == "all" else part[subset])
            for population in ["own_valid", "common_valid"]:
                valid = selected & part.valid & (True if population == "own_valid" else part.common_valid)
                for axis in ["vx", "vy"]:
                    error = part.loc[valid, axis + "_mps"] - part.loc[valid, "truth_" + axis]
                    rows.append(dict(method=method, subset=subset, population=population, axis=axis,
                                     eligible=int(selected.sum()), valid=int(valid.sum()),
                                     coverage=float(valid.sum() / selected.sum()) if selected.any() else np.nan,
                                     mae=error.abs().mean(), rmse=np.sqrt((error ** 2).mean()), bias=error.mean(),
                                     p95=error.abs().quantile(.95) if len(error) else np.nan))
    return pd.DataFrame(rows)


def macro_metrics(per_case):
    """Coverage includes every eligible case, even if it has zero valid outputs."""
    result = []
    for key, group in per_case.groupby(["method", "subset", "population", "axis"]):
        eligible = group[group.eligible > 0]
        good = eligible[eligible.valid > 0]
        result.append(dict(zip(["method", "subset", "population", "axis"], key)) |
                      dict(cases=len(eligible), cases_with_estimates=len(good), failed_cases=int((eligible.valid == 0).sum()),
                           eligible=int(eligible.eligible.sum()), valid=int(eligible.valid.sum()),
                           coverage=float(eligible.valid.sum() / eligible.eligible.sum()) if len(eligible) else np.nan,
                           min_coverage=eligible.coverage.min(), mae=good.mae.mean(), rmse=good.rmse.mean(), worst_case_p95=good.p95.max()))
    return pd.DataFrame(result)


def evaluate(predictions, labels, output, methods=None):
    output = artifact_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be empty")
    frame = pd.read_csv(predictions)
    if methods is not None:
        if set(methods) - set(frame.method):
            raise ValueError("Requested method missing from predictions")
        frame = frame[frame.method.isin(methods)]
    truth, spec = load_labels(labels)
    joined = join_labels(frame, truth, spec)
    report = metrics(joined)
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "per_frame.csv", index=False)
    report.to_csv(output / "metrics.csv", index=False)
    write_json(output / "evaluation.json", dict(predictions_sha256=sha256(predictions), labels_sha256=sha256(labels),
               label_meta_sha256=sha256(Path(labels).with_suffix(".meta.json")), target_spec=spec,
               alignment="output_frame, timestamp tolerance 1e-6 seconds; no optimized lag", methods=sorted(frame.method.unique())))
    return report
