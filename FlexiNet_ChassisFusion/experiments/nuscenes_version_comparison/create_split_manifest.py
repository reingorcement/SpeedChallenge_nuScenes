from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nuscenes_version_comparison.splits import create_split_manifest  # noqa: E402


def _scene_metadata(nusc) -> dict[str, dict[str, str]]:
    logs = {row["token"]: row for row in nusc.log}
    metadata: dict[str, dict[str, str]] = {}
    for scene in nusc.scene:
        description = str(scene.get("description", "")).lower()
        log = logs.get(scene["log_token"], {})
        metadata[scene["name"]] = {
            "location": str(log.get("location", "unknown")),
            "lighting": "night" if "night" in description else "day_or_unknown",
            "weather": "rain" if "rain" in description else "dry_or_unknown",
        }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic KD/VKD scene split manifest.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--can-bus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    official = create_splits_scenes()
    available_can = {
        path.name[:10] for path in Path(args.can_bus_root).glob("scene-*_*.json")
    }
    if not available_can:
        raise FileNotFoundError(f"No nuScenes CAN JSON files found in {args.can_bus_root}.")
    train = sorted(set(official["train"]) & available_can)
    val = sorted(set(official["val"]) & available_can)
    excluded = {
        name: "missing_can"
        for name in sorted((set(official["train"]) | set(official["val"])) - available_can)
    }
    manifest = create_split_manifest(
        train,
        val,
        _scene_metadata(nusc),
        seed=args.seed,
        version=args.version,
        excluded_scenes=excluded,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = {key: len(value) for key, value in manifest["partitions"].items()}
    print(json.dumps({"output": str(target), "partition_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
