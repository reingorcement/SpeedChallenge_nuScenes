"""Pack completed V5 compact flow caches into immutable shared mmap arrays."""
import argparse

import torch

from purevision_v5.packed_flow import pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partitions", nargs="+", default=["train_core", "dev"])
    parser.add_argument("--require-fb", action="store_true")
    parser.add_argument("--reference-pack", help="Verify every forward flow/valid value against this earlier forward-only pack")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    pack(args.manifest, args.source_root, args.output, partitions=args.partitions,
         require_fb=args.require_fb, reference_pack=args.reference_pack)


if __name__ == "__main__":
    main()
