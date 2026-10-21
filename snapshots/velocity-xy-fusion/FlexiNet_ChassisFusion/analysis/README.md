# `analysis/` — Post-training testing & result analysis

This folder is the structured replacement for the workflow described in
`POST_TRAINING_TEST_AND_ANALYSIS.md`. It does **not** retrain anything; it only
reloads a checkpoint, runs validation/test inference, and turns the resulting
predictions into machine-readable tables and a Markdown report.

The model now predicts a 2D ego-frame velocity `[vx, vy]`, so the analysis is
written around `vector_endpoint_error_mps`, `mae_vx_mps`, `mae_vy_mps` and the
`base_velocity` baseline. The legacy md still describes the older scalar-speed
view; treat that document as historical context.

## Layout

| File | Purpose |
| --- | --- |
| `inference.py` | Reusable `run_inference_from_checkpoint` (used by both CLIs). Mirrors `test.py` but exposes the pieces as functions. |
| `postprocess.py` | Pure-Python helpers that consume `*_predictions.csv` + `*_metrics.json` and emit bad-case / per-scene / per-bin tables and `*_report.md`. |
| `evaluate_checkpoint.py` | Single-checkpoint CLI. Runs eval (or just the analysis if `--skip_inference`). |
| `compare_checkpoints.py` | Two-checkpoint CLI. Compares any two eval folders (default: `best.pth` vs `last.pth`). |

The standard files written by `utils/evaluation.py` are reused as-is:
`val_metrics.json`, `val_predictions.csv`, and the diagnostic PNGs (pred vs
target, endpoint-error vs speed, histograms, per-bin endpoint error).

## Outputs

For each evaluation directory `<output_dir>/`, an `analysis/` subdirectory is
created containing:

```
<output_dir>/
  val_metrics.json                       # already produced by write_evaluation_artifacts
  val_predictions.csv                    # ditto
  val_*.png                              # ditto
  analysis/
    val_bad_cases_top50.csv              # samples sorted by endpoint error desc
    val_per_scene_metrics.csv            # MAE_vx, MAE_vy, endpoint err, gain per scene
    val_per_speed_bin_metrics.csv        # same metrics binned by target speed
    val_report.md                        # Markdown summary you can open directly
```

`compare_checkpoints.py` additionally writes `comparison.{md,csv}` to its
`--report_dir`.

## Single checkpoint

```bash
# Most common: evaluate best.pth on the val split for an experiment
python analysis/evaluate_checkpoint.py \
  --config configs/velocity_xy_fair_gray.yaml \
  --checkpoint checkpoints/velocity_xy_fair_gray/best.pth \
  --output_dir outputs/velocity_xy_fair_gray/eval_best \
  --results_dir results_git/FlexiNet_ChassisFusion_velocity_xy/fair_gray/eval_best
```

If you have already evaluated and only want the analysis bundle (e.g. you
changed the bad-case top-K or want the Markdown report regenerated):

```bash
python analysis/evaluate_checkpoint.py \
  --skip_inference \
  --output_dir outputs/velocity_xy_fair_gray/eval_best
```

## `best` vs `last` comparison

```bash
python analysis/compare_checkpoints.py \
  --config configs/velocity_xy_fair_gray.yaml \
  --checkpoint_a checkpoints/velocity_xy_fair_gray/best.pth \
  --checkpoint_b checkpoints/velocity_xy_fair_gray/last.pth \
  --output_dir_a outputs/velocity_xy_fair_gray/eval_best \
  --output_dir_b outputs/velocity_xy_fair_gray/eval_last \
  --report_dir   outputs/velocity_xy_fair_gray/eval_compare
```

Open `outputs/velocity_xy_fair_gray/eval_compare/comparison.md` for the
side-by-side metric table with deltas and preferred direction.

## Recommended reading order

1. `<output_dir>/val_metrics.json` — overall scores. The headline numbers are
   `vector_endpoint_error_mps`, `mae_vx_mps`, `mae_vy_mps`, and the
   `gain_vs_base_*` set. Higher gain ⇒ the model adds value over `base_velocity`.
2. `<output_dir>/analysis/val_per_speed_bin_metrics.csv` — does the model help
   in every speed regime, or only some?
3. `<output_dir>/analysis/val_per_scene_metrics.csv` — flag scenes that are
   systematically worse than the chassis baseline.
4. `<output_dir>/analysis/val_bad_cases_topK.csv` — pick a few of the largest
   endpoint errors and inspect them by token.
5. The PNG diagnostics already produced in `<output_dir>/` for residual /
   delta-vy / pred-vs-target shapes.
6. Only when needed, run `compare_checkpoints.py` to confirm `best.pth` is
   actually better than `last.pth` on your selection metric.

## Notes

- The CLIs honor everything in the YAML config (frame cache, max_scenes,
  pose_assisted, flow, etc.), so the same command works across all 4
  experiments.
- `--max_eval_batches N` is available for smoke-style sanity checks (matches
  `train.py limit_eval_batches`).
- `postprocess.py` is intentionally framework-free (no torch, no nuscenes), so
  you can scp the eval folder off the server and finish the analysis on a
  laptop. The unit tests in `tests/test_analysis_postprocess.py` exercise it
  with synthetic predictions.
