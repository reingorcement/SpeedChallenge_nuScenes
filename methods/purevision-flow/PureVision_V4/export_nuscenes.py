import argparse
from pathlib import Path
from purevision_v4.nuscenes_export import export


def main():
    parser = argparse.ArgumentParser(description="Export isolated nuScenes sequence manifests and full 3D body labels")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--partitions", nargs="+", default=["train_core", "dev", "stage1_test"])
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--camera-height", type=float, required=True, help="Nominal metric camera-to-road height; do not infer from ego-origin z")
    parser.add_argument("--limit", type=int, default=0)
    print(export(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
