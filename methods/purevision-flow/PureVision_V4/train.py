"""Train V4 RGB or compensated fusion heads; labels never enter inference."""
import argparse
from pathlib import Path
from purevision_v4.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["direct", "fusion"], default="fusion")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = vars(parser.parse_args())
    args["manifest_path"] = args.pop("manifest")
    print(train(**args))


if __name__ == "__main__":
    main()
