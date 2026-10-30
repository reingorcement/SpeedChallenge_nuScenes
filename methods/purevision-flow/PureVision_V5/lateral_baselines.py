"""Fit one constant from train labels, then evaluate zero/constant/model on dev."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from purevision_v5.common import code_version, now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import validate_lateral_rule
from purevision_v5.selection import label_rows


def error_metrics(prediction, target):
    error = np.asarray(prediction, float) - np.asarray(target, float)
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Lateral baseline requires finite nonempty scalar predictions")
    return dict(count=len(error), mae_vy=float(np.abs(error).mean()), rmse_vy=float(np.sqrt(np.square(error).mean())),
                bias_vy=float(error.mean()), p95_abs_vy=float(np.quantile(np.abs(error), .95)))


def summarize_lateral(prediction, target, scenes, rule):
    prediction, target, scenes = np.asarray(prediction), np.asarray(target), np.asarray(scenes)
    if prediction.shape != target.shape or target.ndim != 1 or scenes.shape != target.shape:
        raise ValueError("Lateral prediction/target/scene shapes differ")
    selected = np.abs(target) >= rule["nonzero_threshold_m_s"]
    subset = dict(count=int(selected.sum()), scene_count=len(set(scenes[selected])),
                  threshold_m_s=rule["nonzero_threshold_m_s"],
                  adequate=bool(selected.sum() >= rule["min_subset_windows"]
                                and len(set(scenes[selected])) >= rule["min_subset_scenes"]))
    if selected.any():
        subset["metrics"] = error_metrics(prediction[selected], target[selected])
    return dict(overall=error_metrics(prediction, target), nonzero_vy=subset,
                per_scene={str(s): error_metrics(prediction[scenes == s], target[scenes == s]) for s in sorted(set(scenes))})


def baseline_report(manifest_path, selection_path=None, rule=None, predictions=()):
    rule = validate_lateral_rule(rule)
    train_rows, selection = label_rows(manifest_path, "train_core", selection_path)
    dev_rows, _ = label_rows(manifest_path, "dev", selection_path)
    # Fitting is completed solely from train_core before any dev statistic is used.
    constant = float(np.median([r["target_vy"] for r in train_rows]))
    target = np.asarray([r["target_vy"] for r in dev_rows])
    scenes = np.asarray([r["scene"] for r in dev_rows])
    result = dict(schema="v5_train_median_lateral_baseline_v1", generated_at=now(),
                  scope="resolution_pilot" if selection else "full_train_dev", dataset_sha256=sha256(manifest_path),
                  scene_selection=selection, effective_lateral_rule=rule,
                  fit=dict(partition="train_core", count=len(train_rows), statistic="median_of_float32_window_target_vy",
                           constant_vy_m_s=constant, training_targets_identity=object_hash(train_rows)),
                  evaluation=dict(partition="dev", count=len(dev_rows), targets_identity=object_hash(dev_rows)),
                  zero_vy=summarize_lateral(np.zeros_like(target), target, scenes, rule),
                  train_median_vy=summarize_lateral(np.full_like(target, constant), target, scenes, rule), models={})
    for path in predictions:
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        identity = lambda r: (r["scene"], r["token"], float(r["time_s"]))
        by_key = {identity(r): r for r in rows}
        if len(rows) != len(by_key) or set(by_key) != {identity(r) for r in dev_rows}:
            raise ValueError("Model predictions do not cover the exact selected dev windows")
        ordered = [by_key[identity(r)] for r in dev_rows]
        if any(float(a[k]) != b[k] for a, b in zip(ordered, dev_rows) for k in ("target_vx", "target_vy")):
            raise ValueError("Model prediction targets differ from verified dev labels")
        result["models"][str(Path(path).resolve())] = dict(predictions_sha256=sha256(path),
            **summarize_lateral([float(r["vy"]) for r in ordered], target, scenes, rule))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--config", help="Use the experiment's declared lateral_rule")
    parser.add_argument("--predictions", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Baseline output must be new; old evidence is immutable")
    rule = read_json(args.config).get("lateral_rule") if args.config else None
    result = baseline_report(args.manifest, args.selection, rule, args.predictions)
    result["code"] = code_version()
    write_json(args.output, result)
    print({k: result[k] for k in ("scope", "fit", "evaluation")})


if __name__ == "__main__":
    main()
