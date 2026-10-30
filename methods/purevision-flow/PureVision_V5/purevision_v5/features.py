"""Reusable frozen RGB descriptors. Each cache is bound to its sequence and weights."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import ROOT, code_version, now, object_hash, read_json, sha256, write_json
from .rgb import FrozenRGB, preprocess_rgb


def prepare(manifest_path, output, checkpoint, device="cuda:1", batch_size=64,
            partitions=("train_core", "dev"), limit_scenes=0, precision="float32", decode_workers=8):
    torch.set_num_threads(4)
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    version = code_version(require_pushed=limit_scenes == 0)
    manifest = read_json(manifest_path)
    if manifest["schema"] != "v5_dataset_v1":
        raise ValueError("Expected v5_dataset_v1")
    contract = dict(schema="v5_rgb_features_v1", dataset_sha256=sha256(manifest_path),
                    weights_sha256=sha256(checkpoint), input_size=[5, 3, 72, 128],
                    normalization="PIL RGB bilinear; x/127.5-1", dtype="float32", compute_precision=precision,
                    rgb_code_sha256=sha256(ROOT / "purevision_v5/rgb.py"),
                    backbone_code_sha256=sha256(ROOT / "purevision_v5/_rgb_reference/flexinet_backbone.py"),
                    modules_code_sha256=sha256(ROOT / "purevision_v5/_rgb_reference/modules.py"))
    identity = object_hash(contract)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        if index["identity"] != identity:
            raise ValueError("Frozen descriptor cache contract changed; use a new cache directory")
    else:
        index = dict(identity=identity, contract=contract, code=version, sequences={})
    model = FrozenRGB(checkpoint).to(device).eval()
    selected = [s for s in manifest["sequences"] if s["partition"] in partitions]
    if limit_scenes:
        selected = [record for partition in partitions
                    for record in [s for s in selected if s["partition"] == partition][:limit_scenes]]
    completed = 0
    write_json(output / "status.json", dict(status="running", code=version, completed=0, total=len(selected), updated_at=now()))
    started, processed = time.monotonic(), 0
    for record in selected:
        scene = record["scene"]
        sequence_path = manifest_path.parent / record["file"]
        sequence_hash = sha256(sequence_path)
        previous = index["sequences"].get(scene)
        if previous and previous["sequence_sha256"] == sequence_hash:
            path = output / previous["file"]
            if path.is_file() and sha256(path) == previous["sha256"]:
                completed += 1
                continue
        sequence = read_json(sequence_path)
        ends = np.asarray(sequence["windows"], dtype=np.int64)
        def decode(frame):
            with Image.open(Path(manifest["dataroot"]) / frame["filename"]) as im:
                return preprocess_rgb(im)
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            images = torch.stack(list(pool.map(decode, sequence["frames"])))
        values = []
        with torch.inference_mode():
            for offset in range(0, len(ends), batch_size):
                batch = torch.stack([images[int(end)-4:int(end)+1] for end in ends[offset:offset+batch_size]])
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                    enabled=precision == "float16"):
                    values.append(model(batch.to(device)).float().cpu().numpy())
        features = np.concatenate(values) if values else np.empty((0, 384), np.float32)
        if features.shape != (len(ends), 384) or not np.isfinite(features).all():
            raise ValueError(f"Invalid frozen features: {scene}")
        path = output / f"{scene}.npz"
        with open(str(path) + ".partial", "wb") as stream:
            np.savez(stream, descriptor=features, end_indices=ends)
        Path(str(path) + ".partial").replace(path)
        index["sequences"][scene] = dict(file=path.name, sha256=sha256(path), sequence_sha256=sequence_hash,
                                        count=len(ends), partition=record["partition"])
        index["updated_at"] = now()
        write_json(index_path, index)
        processed += len(ends)
        completed += 1
        write_json(output / "status.json", dict(status="running", code=version, completed=completed,
                    total=len(selected), windows_written=processed, updated_at=now(), elapsed_s=time.monotonic()-started))
        print(f"FEATURES {scene} windows={len(ends)} total={processed} windows_s={processed/(time.monotonic()-started):.2f}", flush=True)
    write_json(output / "status.json", dict(status="completed", code=version, completed=completed,
               total=len(selected), updated_at=now(), elapsed_s=time.monotonic()-started))
    return index_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT.parent / "PureVision_V3/checkpoints/pretrained/direct_seed_42/best.pth"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--partitions", nargs="+", default=["train_core", "dev"])
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args()
    try:
        prepare(args.manifest, args.output, args.checkpoint, args.device, args.batch_size, args.partitions,
                args.limit_scenes, args.precision, args.decode_workers)
    except BaseException as error:
        write_json(Path(args.output) / "status.json", dict(status="failed", updated_at=now(),
                   error=f"{type(error).__name__}: {error}"))
        raise
