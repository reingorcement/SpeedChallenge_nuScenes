"""Server entry point: RGB + timestamps + fixed calibration -> metric velocity."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import random
import sys
import time

from roll_velocity.io import ROOT, code_hashes, load_sequence, object_hash, read_json, sha256, write_json
from roll_velocity.normal_policy import GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing rgb/, frames.json, calibration.json")
    parser.add_argument("--output", type=Path, required=True, help="Dedicated run directory")
    parser.add_argument("--methods", nargs="+", choices=GROUPS, default=["F_RC", "F_R"])
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu (CPU is slow)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="0 = all remaining images; use 8 for smoke testing")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    parser.add_argument("--resume", action="store_true", help="Reuse verified flow cache in the same run")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import scipy
    import torch
    import torchvision
    from PIL import __version__ as pillow_version
    from roll_velocity.flow import prepare_flow
    from roll_velocity.pipeline import prepare_region, infer

    if args.threads < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("threads must be positive and methods must be unique")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error("Requested device is unavailable; install CUDA PyTorch or explicitly use --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    raw, out = args.input.resolve(), args.output.resolve()
    if out == raw or out.is_relative_to(raw) or raw.is_relative_to(out):
        parser.error("Input and output directories must not contain one another")
    print("Checking images, calibration, weights and run identity...", flush=True)
    frames, cal = load_sequence(raw, args.start_frame, args.frames)
    weights = {name: ROOT / "weights" / name for name in
               ("fusion_seed42_best.pth", "raft_large_C_T_SKHT_V2-ff5fadd5.pth")}
    config = read_json(args.config)
    environment = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                       numpy=np.__version__, scipy=scipy.__version__, cv2=cv2.__version__, pillow=pillow_version,
                       device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       platform=platform.platform(), threads=args.threads)
    identity = dict(frames=frames, calibration=cal, config=config, methods=args.methods,
                    weights={k: sha256(v) for k, v in weights.items()}, code=code_hashes(), environment=environment)
    signature = object_hash(identity)
    metadata_path = out / "run.json"
    if out.exists() and any(out.iterdir()):
        if not args.resume or not metadata_path.exists():
            parser.error("Output is not empty. Choose a new output, or use --resume for the same run")
        old = read_json(metadata_path)
        if old["signature"] != signature:
            parser.error("Resume identity differs (input/code/config/weights/environment). Choose a new output")
        if old["status"] == "complete":
            if all((out / p).is_file() and sha256(out / p) == digest for p, digest in old["result_hashes"].items()):
                print(f"Already complete; verified outputs: {out}", flush=True)
                return
            parser.error("Completed outputs changed or are missing; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".running"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"Run is locked: {lock}. Check that the previous process stopped before removing this lock")
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    metadata = dict(signature=signature, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    source=str(raw), identity=identity, allowed_inputs=["RGB", "timestamps", "fixed calibration"],
                    truth_input=False, primary_output="speeds.csv: F_RC vx_mps/vy_mps",
                    coordinate_system="fixed calibrated body axes: x forward, y left, z up; reference at static CG",
                    warmup_frames=4, normal_state="fresh at selected start", argv=sys.argv)
    started = time.perf_counter()
    try:
        write_json(metadata_path, metadata)
        prepare_flow(raw, frames, out / "cache/flow", weights["raft_large_C_T_SKHT_V2-ff5fadd5.pth"], device)
        normals = prepare_region(raw, frames, cal, config, out / "cache/flow", out / "cache/region")
        coverage = infer(raw, frames, cal, config, out / "cache/region", normals,
                         weights["fusion_seed42_best.pth"], out, args.methods, device)
        metadata.update(status="complete", coverage=coverage, elapsed_s=time.perf_counter() - started,
                        result_hashes={p: sha256(out / p) for p in ("speeds.csv", "pairs.csv", "cache/region/normals.json", "cache/region/summary.json")})
        write_json(metadata_path, metadata)
        print(f"Complete: {out / 'speeds.csv'}\nCoverage: {coverage}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_s=time.perf_counter() - started)
        write_json(metadata_path, metadata)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
