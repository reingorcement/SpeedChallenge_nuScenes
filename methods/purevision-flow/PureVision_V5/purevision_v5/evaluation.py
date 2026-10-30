"""Component metrics, per-scene outputs, and a predeclared lateral acceptance rule."""
from __future__ import annotations

import numpy as np


DEFAULT_LATERAL_RULE = dict(max_relative_mae_increase=.05, max_absolute_mae_increase_m_s=.01,
                            nonzero_threshold_m_s=.2, min_subset_windows=100, min_subset_scenes=5)


def validate_lateral_rule(rule=None):
    """One source for legacy defaults and explicitly configured future evaluations."""
    value = dict(DEFAULT_LATERAL_RULE if rule is None else rule)
    if set(value) != set(DEFAULT_LATERAL_RULE):
        raise ValueError("Lateral rule has missing or unexpected fields")
    for key in ("max_relative_mae_increase", "max_absolute_mae_increase_m_s", "nonzero_threshold_m_s"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not np.isfinite(value[key]) or value[key] < 0:
            raise ValueError("Lateral thresholds must be finite nonnegative numbers")
    for key in ("min_subset_windows", "min_subset_scenes"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 1:
            raise ValueError("Lateral subset minima must be positive integers")
    return value

def metrics(prediction, target):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Predictions and labels must be [N,2]")
    if not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Cannot silently omit missing/nonfinite predictions")
    error = prediction - target
    result = dict(count=len(error), epe=float(np.linalg.norm(error, axis=1).mean()))
    for i, axis in enumerate(("vx", "vy")):
        result.update({f"mae_{axis}": float(np.abs(error[:, i]).mean()),
                       f"rmse_{axis}": float(np.sqrt(np.square(error[:, i]).mean())),
                       f"bias_{axis}": float(error[:, i].mean()),
                       f"p95_abs_{axis}": float(np.quantile(np.abs(error[:, i]), .95))})
    result["zero_vy_mae"] = float(np.abs(target[:, 1]).mean())
    return result


def summarize(prediction, target, scenes, lateral_rule=None):
    rule = validate_lateral_rule(lateral_rule)
    prediction, target, scenes = np.asarray(prediction), np.asarray(target), np.asarray(scenes)
    if scenes.shape != (len(target),):
        raise ValueError("One scene identity is required per prediction")
    result = dict(overall=metrics(prediction, target), scene_count=len(set(scenes)),
                  per_scene={str(s): metrics(prediction[scenes == s], target[scenes == s]) for s in sorted(set(scenes))},
                  effective_lateral_rule=rule)
    selected = np.abs(target[:, 1]) >= rule["nonzero_threshold_m_s"]
    result["nonzero_vy"] = dict(threshold_m_s=rule["nonzero_threshold_m_s"], count=int(selected.sum()),
                                scene_count=len(set(scenes[selected])))
    if selected.any():
        result["nonzero_vy"]["metrics"] = metrics(prediction[selected], target[selected])
    result["nonzero_vy"]["adequate"] = bool(selected.sum() >= rule["min_subset_windows"]
                                            and len(set(scenes[selected])) >= rule["min_subset_scenes"])
    return result


def lateral_acceptance(candidate, baseline, lateral_rule=None):
    rule = validate_lateral_rule(lateral_rule)
    for value in (candidate, baseline):
        if value.get("effective_lateral_rule", DEFAULT_LATERAL_RULE) != rule:
            raise ValueError("Lateral summary and acceptance rules differ")
    checks = {}
    for name in ("overall", "nonzero_vy"):
        if name == "nonzero_vy":
            if not baseline[name]["adequate"] or not candidate[name]["adequate"]:
                checks[name] = dict(status="insufficient_data")
                continue
            a, b = candidate[name]["metrics"], baseline[name]["metrics"]
        else:
            a, b = candidate[name], baseline[name]
        tolerance = max(rule["max_absolute_mae_increase_m_s"], rule["max_relative_mae_increase"] * b["mae_vy"])
        checks[name] = dict(delta_mae_vy=a["mae_vy"] - b["mae_vy"], tolerance_m_s=tolerance,
                            status="pass" if a["mae_vy"] <= b["mae_vy"] + tolerance else "reject")
    return dict(checks=checks, acceptable=all(c["status"] == "pass" for c in checks.values()),
                effective_lateral_rule=rule,
                scope="exploratory single-seed PLANAR_PROXY; EPE gain alone does not establish improvement")
