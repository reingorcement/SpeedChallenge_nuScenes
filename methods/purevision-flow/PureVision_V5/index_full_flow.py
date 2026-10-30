"""Create a label-free all-adjacent nuScenes camera index in a new directory."""
import argparse
import json

from purevision_v5.full_flow_index import build_index, load_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--original-manifest", required=True, help="Existing V5 manifest, read for partition mapping only")
    parser.add_argument("--versions", nargs="+", required=True, choices=["v1.0-trainval", "v1.0-test"])
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Must be a new directory")
    args = parser.parse_args()
    path = build_index(args.dataroot, args.original_manifest, args.output, versions=args.versions, cameras=args.cameras)
    value = load_index(path)
    print(json.dumps(dict(index=str(path), identity=value["identity"], counts=value["counts"],
                          versions=value["versions"], cameras=value["cameras"])))


if __name__ == "__main__":
    main()
