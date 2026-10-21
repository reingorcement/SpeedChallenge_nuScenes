from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_v2 import build_dataset, build_model, forward_batch, load_config, move_batch  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 image/flow/CAN counterfactual diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/v2_counterfactual.json")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual nuScenes inference is server-only and requires CUDA.")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, "val")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 8)),
        pin_memory=True,
    )
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval().to(device)

    sums = {"image_shuffle_delta_mps": 0.0, "flow_shuffle_delta_mps": 0.0, "can_shuffle_delta_mps": 0.0}
    sample_count = 0
    for step, raw_batch in enumerate(loader):
        if step >= args.batches:
            break
        batch = move_batch(raw_batch, device)
        batch_size = batch["images"].shape[0]
        if batch_size < 2:
            continue
        permutation = torch.roll(torch.arange(batch_size, device=device), shifts=1)
        baseline = forward_batch(model, batch).predicted_velocity

        image_batch = dict(batch)
        image_batch["images"] = batch["images"][permutation]
        image_prediction = forward_batch(model, image_batch).predicted_velocity

        flow_batch = dict(batch)
        flow_batch["flow_v2"] = batch["flow_v2"][permutation]
        flow_batch["flow_confidence"] = batch["flow_confidence"][permutation]
        flow_prediction = forward_batch(model, flow_batch).predicted_velocity

        can_batch = dict(batch)
        for key in ("kinematic_values", "kinematic_valid", "kinematic_age"):
            can_batch[key] = batch[key][permutation]
        can_prediction = forward_batch(model, can_batch).predicted_velocity

        sums["image_shuffle_delta_mps"] += float(
            torch.linalg.vector_norm(image_prediction - baseline, dim=-1).sum()
        )
        sums["flow_shuffle_delta_mps"] += float(
            torch.linalg.vector_norm(flow_prediction - baseline, dim=-1).sum()
        )
        sums["can_shuffle_delta_mps"] += float(
            torch.linalg.vector_norm(can_prediction - baseline, dim=-1).sum()
        )
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Counterfactual evaluation requires batches containing at least two samples.")
    report = {key: value / sample_count for key, value in sums.items()}
    report["sample_count"] = sample_count
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
