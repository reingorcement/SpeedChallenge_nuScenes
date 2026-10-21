"""Verify transfer integrity against package_manifest.json."""
import argparse
from pathlib import Path

from roll_velocity.io import ROOT, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    records = read_json(root / "package_manifest.json")["files"]
    errors = []
    for record in records:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            errors.append(record["path"])
    if errors:
        raise SystemExit("Transfer verification failed:\n" + "\n".join(errors))
    print(f"PASS: {len(records)} files verified")


if __name__ == "__main__":
    main()
