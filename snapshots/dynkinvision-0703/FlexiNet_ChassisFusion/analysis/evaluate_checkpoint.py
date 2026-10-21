"""End-to-end evaluation + analysis for a single checkpoint.

Typical use::

    python analysis/evaluate_checkpoint.py \
        --config configs/velocity_xy_fair_gray.yaml \
        --checkpoint checkpoints/velocity_xy_fair_gray/best.pth \
        --output_dir outputs/velocity_xy_fair_gray/eval_best

The script will:

1. Reload the checkpoint and run :func:`train.run_eval` on the chosen split
   (``val`` by default).
2. Write the standard metric/CSV/PNG artifacts via
   :func:`utils.evaluation.write_evaluation_artifacts` (same files as ``test.py``).
3. Generate the analysis bundle (bad-case CSV, per-scene CSV, per-speed-bin CSV,
   ``*_report.md``) under ``<output_dir>/analysis/``.

If you have already evaluated and just want to (re)run the analysis bundle,
pass ``--skip_inference`` and point ``--output_dir`` to the existing eval folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config_from_yaml  # noqa: E402
from utils.evaluation import copy_small_results_for_git, write_evaluation_artifacts  # noqa: E402

from analysis.inference import run_inference_from_checkpoint  # noqa: E402
from analysis.postprocess import write_analysis_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and emit the analysis bundle.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (val/test/...)")
    parser.add_argument("--output_dir", type=str, default="outputs/FlexiNet_ChassisFusion/eval")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_git/FlexiNet_ChassisFusion/test_eval",
        help="Where to mirror small JSON/PNG results for git tracking",
    )
    parser.add_argument("--prefix", type=str, default="val", help="Filename prefix used by write_evaluation_artifacts")
    parser.add_argument("--allow_pose_fallback", action="store_true")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Stop after N batches (0 = full eval)")
    parser.add_argument("--bad_case_top_k", type=int, default=50)
    parser.add_argument("--report_top_k", type=int, default=10)
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference and only (re)run the analysis bundle on existing artifacts in --output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        cfg = load_config_from_yaml(Path(args.config))
        artifacts = run_inference_from_checkpoint(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.split,
            allow_pose_fallback=args.allow_pose_fallback,
            max_eval_batches=int(args.max_eval_batches),
        )
        print(
            f"Eval scenes: {artifacts.scene_count} | Eval samples: {artifacts.dataset_size} | "
            f"Split: {artifacts.split} | Checkpoint: {artifacts.checkpoint_path}"
        )
        metrics = write_evaluation_artifacts(artifacts.result, output_dir, prefix=args.prefix)
        copy_small_results_for_git(output_dir, args.results_dir, include_predictions=False)
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"MAE vx: {metrics['mae_vx_mps']:.4f} m/s")
        print(f"MAE vy: {metrics['mae_vy_mps']:.4f} m/s")
        print(f"Vector endpoint error: {metrics['vector_endpoint_error_mps']:.4f} m/s")
        print(f"Base MAE vx: {metrics.get('base_mae_vx_mps', 0.0):.4f} m/s")
        print(f"Base MAE vy: {metrics.get('base_mae_vy_mps', 0.0):.4f} m/s")
        print(f"Saved evaluation artifacts to: {output_dir}")
    else:
        print(f"Skipping inference; reading existing artifacts from {output_dir}")

    bundle = write_analysis_bundle(
        output_dir,
        prefix=args.prefix,
        bad_case_top_k=int(args.bad_case_top_k),
        report_top_k=int(args.report_top_k),
    )
    copy_small_results_for_git(output_dir, args.results_dir, include_predictions=False)
    print("Analysis bundle:")
    for label, path in bundle.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
