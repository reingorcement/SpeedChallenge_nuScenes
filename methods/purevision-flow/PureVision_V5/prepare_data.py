"""Prepare the fixed first-round nuScenes PLANAR_PROXY dataset."""
import argparse
from purevision_v5.data import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_dataset(args.dataroot, args.split_manifest, args.output)


if __name__ == "__main__":
    main()
