"""Isolated V4 inference; no truth inputs are accepted."""
import argparse
from pathlib import Path
from purevision_v4.io import ROOT
from purevision_v4.pipeline import METHODS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["F_RC", "F_R"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--fusion-checkpoint", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--image-root", type=Path, help="Explicit read-only image root, e.g. nuScenes dataroot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run(args.input, args.output, methods=args.methods, device=args.device, config_path=args.config,
                 fusion_checkpoint=args.fusion_checkpoint, start=args.start_frame, count=args.frames,
                 image_root=args.image_root, resume=args.resume, save_features=args.save_features, threads=args.threads)
    print(result["coverage"])


if __name__ == "__main__":
    main()
