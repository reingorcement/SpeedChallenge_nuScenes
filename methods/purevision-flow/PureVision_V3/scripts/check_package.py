"""Check standalone imports and bundled checkpoint compatibility without a dataset."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    for name in (
        "torch", "torchvision", "numpy", "cv2", "yaml", "matplotlib",
        "nuscenes.nuscenes", "pytest", "tqdm",
        "train_v3", "scripts.run_v3_ablation", "scripts.evaluate_v3_holdout",
        "scripts.precompute_frames", "scripts.precompute_v2_optical_flow",
        "scripts.create_v2_split_manifest", "scripts.analyze_v3_results",
    ):
        importlib.import_module(name)
    import torch
    from scripts.run_v3_ablation import load_existing_model
    from train_v3 import build_model

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected Python environment.")
    report = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "checkpoints": [],
    }
    for directory, mode in (("direct_seed_42", "direct_pretrain"), ("fusion_seed_42", "fusion")):
        for filename in ("best.pth", "last.pth"):
            path = ROOT / "checkpoints" / "pretrained" / directory / filename
            package = torch.load(path, map_location="cpu")
            model = build_model(package["config"])
            model.load_state_dict(package["model"], strict=True)
            if package["config"]["model"]["mode"] != mode:
                raise ValueError(f"Unexpected checkpoint mode: {path}")
            report["checkpoints"].append({
                "path": path.relative_to(ROOT).as_posix(),
                "mode": mode, "epoch": package.get("epoch"),
                "early_stopped": package.get("early_stopped", False),
            })
        load_existing_model(
            ROOT / "checkpoints" / "pretrained" / directory / "best.pth",
            expected_mode=mode,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
