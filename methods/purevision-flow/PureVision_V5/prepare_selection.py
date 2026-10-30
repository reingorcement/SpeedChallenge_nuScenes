"""Write a reproducible, explicit resolution-pilot scene selection."""
import argparse
from pathlib import Path

from purevision_v5.common import write_json
from purevision_v5.selection import make_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-scenes", type=int, default=32)
    parser.add_argument("--dev-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Selection output must be new")
    value = make_selection(args.manifest, args.train_scenes, args.dev_scenes, args.seed)
    write_json(args.output, value)
    print(value)


if __name__ == "__main__":
    main()
