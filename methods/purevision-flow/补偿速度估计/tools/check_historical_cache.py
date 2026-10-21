"""Optional migration audit against an original v2 project; not needed for inference."""
import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roll_velocity.io import code_hashes, load_sequence, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from roll_velocity.pipeline import prepare_region, infer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This migration audit uses CUDA to match the original evaluator")
    config = read_json(ROOT / "config/default.json")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Migration audit output must be new")
    args.output.mkdir(parents=True, exist_ok=True)
    for raw in sorted((ROOT / "data").iterdir()):
        frames, cal = load_sequence(raw)
        flow = args.reference_root / "19_new_map_validation/development" / raw.name / "raft/flow"
        out = args.output / raw.name
        out.mkdir()
        normals = prepare_region(raw, frames, cal, config, flow, out / "cache/region")
        coverage = infer(raw, frames, cal, config, out / "cache/region", normals,
                         ROOT / "weights/fusion_seed42_best.pth", out,
                         ["F_RC", "F_R", "F_B", "V3_matched_R"], torch.device("cuda"))
        write_json(out / "migration_audit.json", dict(
            truth_input=False, optical_flow_source="original frozen RGB-derived flow",
            flow_sha256={p.name: sha256(p) for p in sorted(flow.glob("*.pt"))},
            code_sha256=code_hashes(), coverage=coverage))


if __name__ == "__main__":
    main()
