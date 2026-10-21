from __future__ import annotations

import math
from typing import Iterable, Sequence


def mae(preds: Sequence[float], targets: Sequence[float]) -> float:
    n = max(len(preds), 1)
    return sum(abs(p - t) for p, t in zip(preds, targets)) / n


def rmse(preds: Sequence[float], targets: Sequence[float]) -> float:
    n = max(len(preds), 1)
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / n)

