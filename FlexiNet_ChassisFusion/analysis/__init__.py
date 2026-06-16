"""Post-training testing and analysis utilities for FlexiNet ChassisFusion.

This package keeps every new analysis-only piece of code separate from the
existing training/test entry points (``train.py`` / ``test.py``) and from the
shared metric/plot helpers in ``utils/evaluation.py``. The core flow is:

1. ``inference.run_inference_from_checkpoint`` reloads a checkpoint and runs
   the existing :func:`train.run_eval` pipeline on the chosen split.
2. ``postprocess`` consumes the resulting ``*_predictions.csv`` and produces
   bad-case lists, per-scene tables, per-speed-bin tables and a Markdown
   summary report.
3. The CLI scripts (``evaluate_checkpoint.py`` and ``compare_checkpoints.py``)
   stitch the two layers together so a single command turns a checkpoint into a
   full analysis bundle.
"""

from .inference import run_inference_from_checkpoint
from .postprocess import (
    PerSceneRow,
    SpeedBinRow,
    load_predictions,
    bad_cases,
    per_scene_metrics,
    per_speed_bin_metrics,
    write_analysis_bundle,
)

__all__ = [
    "run_inference_from_checkpoint",
    "PerSceneRow",
    "SpeedBinRow",
    "load_predictions",
    "bad_cases",
    "per_scene_metrics",
    "per_speed_bin_metrics",
    "write_analysis_bundle",
]
