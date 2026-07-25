from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load_predictions(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "center_token" not in rows[0]:
        raise ValueError(f"Prediction CSV has no center_token rows: {path}")
    indexed = {row["center_token"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(
            f"Prediction CSV contains duplicate center_token values: {path}"
        )
    return indexed


def _value(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return float(value)
    raise ValueError(f"Prediction row is missing all supported fields: {names}")


def _errors(row: dict[str, str]) -> tuple[float, float, float]:
    target_vx = _value(row, "target_vx_mps", "target_vx")
    target_vy = _value(row, "target_vy_mps", "target_vy")
    pred_vx = _value(row, "pred_vx_mps", "predicted_vx")
    pred_vy = _value(row, "pred_vy_mps", "predicted_vy")
    error_vx = abs(pred_vx - target_vx)
    error_vy = abs(pred_vy - target_vy)
    return error_vx, error_vy, math.hypot(pred_vx - target_vx, pred_vy - target_vy)


def paired_scene_deltas(
    candidate_path: str | Path, baseline_path: str | Path
) -> dict[str, tuple[float, float, float]]:
    candidate = load_predictions(candidate_path)
    baseline = load_predictions(baseline_path)
    if set(candidate) != set(baseline):
        missing_candidate = set(baseline) - set(candidate)
        missing_baseline = set(candidate) - set(baseline)
        raise ValueError(
            "Prediction token sets differ: "
            f"candidate missing={len(missing_candidate)}, baseline missing={len(missing_baseline)}."
        )
    by_scene: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for token, candidate_row in candidate.items():
        baseline_row = baseline[token]
        if candidate_row["scene_name"] != baseline_row["scene_name"]:
            raise ValueError(f"Scene mismatch for center_token={token}.")
        candidate_target = (
            _value(candidate_row, "target_vx_mps", "target_vx"),
            _value(candidate_row, "target_vy_mps", "target_vy"),
        )
        baseline_target = (
            _value(baseline_row, "target_vx_mps", "target_vx"),
            _value(baseline_row, "target_vy_mps", "target_vy"),
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(candidate_target, baseline_target)
        ):
            raise ValueError(f"Target mismatch for center_token={token}.")
        candidate_error = _errors(candidate_row)
        baseline_error = _errors(baseline_row)
        by_scene[candidate_row["scene_name"]].append(
            tuple(c - b for c, b in zip(candidate_error, baseline_error))
        )
    return {
        scene: tuple(
            sum(values[axis] for values in rows) / len(rows) for axis in range(3)
        )
        for scene, rows in by_scene.items()
    }


def bootstrap(
    seed_pairs: list[dict[str, tuple[float, float, float]]],
    samples: int,
    random_seed: int,
) -> dict[str, object]:
    scene_sets = [set(pair) for pair in seed_pairs]
    if any(scene_set != scene_sets[0] for scene_set in scene_sets[1:]):
        raise ValueError(
            "All random-seed prediction pairs must contain the same scenes."
        )
    scenes = sorted(scene_sets[0])
    if not scenes:
        raise ValueError("No paired scenes were found.")
    names = ("mae_vx_mps", "mae_vy_mps", "epe_mps")
    per_seed = []
    for pair in seed_pairs:
        per_seed.append(
            {
                name: sum(pair[scene][axis] for scene in scenes) / len(scenes)
                for axis, name in enumerate(names)
            }
        )
    generator = random.Random(random_seed)
    draws = [[] for _ in names]
    for _ in range(samples):
        selected = generator.choices(scenes, k=len(scenes))
        for axis in range(len(names)):
            value = sum(
                pair[scene][axis] for pair in seed_pairs for scene in selected
            ) / (len(seed_pairs) * len(selected))
            draws[axis].append(value)
    metrics = {}
    for name, values in zip(names, draws):
        values.sort()
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        estimate = sum(row[name] for row in per_seed) / len(per_seed)
        metrics[name] = {
            "candidate_minus_baseline": estimate,
            "ci95": [low, high],
            "significant_improvement": high < 0.0,
        }
    return {
        "scene_count": len(scenes),
        "seed_count": len(seed_pairs),
        "bootstrap_samples": samples,
        "per_seed": per_seed,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline scene-paired bootstrap for V3 versus token-aligned baselines."
    )
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.candidate) != len(args.baseline):
        raise ValueError(
            "--candidate and --baseline must provide the same number of seeds."
        )
    if args.samples < 100:
        raise ValueError("At least 100 bootstrap samples are required.")
    pairs = [
        paired_scene_deltas(candidate, baseline)
        for candidate, baseline in zip(args.candidate, args.baseline)
    ]
    result = bootstrap(pairs, args.samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
