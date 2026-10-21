from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.v3 import V3VelocityLoss  # noqa: E402
from train_v3 import (  # noqa: E402
    build_dataset,
    build_loader,
    build_model,
    evaluate,
    geometry_audit,
    load_config,
    write_evaluation,
)
from utils.seed import set_seed  # noqa: E402


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen strict visual-only V3 checkpoint on stage1_test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--stage",
        choices=["geometry_audit", "direct_pretrain", "fusion"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataroot")
    parser.add_argument("--flow_root")
    parser.add_argument("--frame_cache_root")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The full V3 holdout evaluator requires CUDA.")

    cfg = load_config(args.config)
    for key in ("dataroot", "flow_root", "frame_cache_root"):
        value = getattr(args, key)
        if value:
            cfg["data"][key] = value
    cfg["data"]["max_scenes"] = None
    cfg["model"]["mode"] = args.stage
    cfg["train"]["seed"] = args.seed
    set_seed(args.seed)

    device = torch.device("cuda")
    dataset = build_dataset(cfg, "stage1_test")
    loader = build_loader(dataset, cfg, device, shuffle=False)
    criterion = V3VelocityLoss(**cfg.get("loss", {})).to(device)
    output_dir = Path(args.output_dir)
    if args.stage == "geometry_audit":
        geometry_audit(cfg, loader, criterion, device, output_dir, max_batches=0)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for learned V3 stages.")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    if args.stage == "direct_pretrain":
        for name in (
            "geometry",
            "rotation",
            "reprojection",
            "residual",
            "support",
            "tv",
        ):
            criterion.weights[name] = 0.0

    metrics, rows = evaluate(model, loader, criterion, device)
    metrics.update(
        {
            "partition": "stage1_test",
            "seed": args.seed,
            "stage": args.stage,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sample_count": len(rows),
            "scene_count": len({str(row["scene_name"]) for row in rows}),
        }
    )
    write_evaluation(metrics, rows, output_dir)
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "partition": "stage1_test",
                "split_manifest_path": cfg["data"].get("split_manifest_path"),
                "stage": args.stage,
                "seed": args.seed,
                "checkpoint": str(Path(args.checkpoint).resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
