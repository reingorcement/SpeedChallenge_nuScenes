"""Component metrics, per-scene outputs, and a predeclared lateral acceptance rule."""
from __future__ import annotations

import numpy as np


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


def summarize(prediction, target, scenes):
    prediction, target, scenes = np.asarray(prediction), np.asarray(target), np.asarray(scenes)
    result = dict(overall=metrics(prediction, target), scene_count=len(set(scenes)),
                  per_scene={str(s): metrics(prediction[scenes == s], target[scenes == s]) for s in sorted(set(scenes))})
    selected = np.abs(target[:, 1]) >= .2
    result["nonzero_vy"] = dict(threshold_m_s=.2, count=int(selected.sum()),
                                scene_count=len(set(scenes[selected])))
    if selected.any():
        result["nonzero_vy"]["metrics"] = metrics(prediction[selected], target[selected])
    result["nonzero_vy"]["adequate"] = bool(selected.sum() >= 100 and len(set(scenes[selected])) >= 5)
    return result


def lateral_acceptance(candidate, baseline):
    checks = {}
    for name in ("overall", "nonzero_vy"):
        if name == "nonzero_vy":
            if not baseline[name]["adequate"] or not candidate[name]["adequate"]:
                checks[name] = dict(status="insufficient_data")
                continue
            a, b = candidate[name]["metrics"], baseline[name]["metrics"]
        else:
            a, b = candidate[name], baseline[name]
        tolerance = max(.01, .05 * b["mae_vy"])
        checks[name] = dict(delta_mae_vy=a["mae_vy"] - b["mae_vy"], tolerance_m_s=tolerance,
                            status="pass" if a["mae_vy"] <= b["mae_vy"] + tolerance else "reject")
    return dict(checks=checks, acceptable=all(c["status"] == "pass" for c in checks.values()),
                scope="exploratory single-seed PLANAR_PROXY; EPE gain alone does not establish improvement")
