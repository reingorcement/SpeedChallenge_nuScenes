"""Prepare label-independent numeric RAFT caches from a V5 dataset manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from purevision_v5.flow import DEFAULT_CHECKPOINT, FrozenRaft, cache_path, sha256
from purevision_v5.common import code_version


def enumerate_pairs(manifest_path: str | Path, partitions: list[str], *, shard_index: int = 0, shard_count: int = 1):
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "v5_dataset_v1" or manifest.get("seq_len") != 5:
        raise ValueError("Expected V5 dataset with five-frame causal windows")
    known = {row["partition"] for row in manifest["sequences"]}
    if not set(partitions) <= known:
        raise ValueError(f"Unknown partitions: {set(partitions) - known}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid scene shard index/count")
    root = Path(manifest["dataroot"]).resolve(strict=True)
    seen = {}
    entries = [entry for entry in manifest["sequences"] if entry["partition"] in partitions]
    for sequence_index, entry in enumerate(entries):
        if sequence_index % shard_count != shard_index:
            continue
        sequence_path = path.parent / entry["file"]
        if "sha256" in entry and sha256(sequence_path) != entry["sha256"]:
            raise ValueError("Sequence manifest checksum mismatch")
        sequence = json.loads(sequence_path.read_text())
        if sequence.get("schema") != "v5_sequence_v1" or any(sequence[key] != entry[key] for key in ("scene", "partition", "log_token")):
            raise ValueError("Sequence manifest identity mismatch")
        frames = sequence["frames"]
        needed = set()
        for end in sequence["windows"]:
            if not isinstance(end, int) or end < 4 or end >= len(frames):
                raise ValueError("Invalid five-frame window end index")
            needed.update(range(end - 4, end))
        for index in sorted(needed):
            source, target = ({"token":frame["token"], "time_s":frame["time_s"],
                               "filename":str((root / frame["filename"]).resolve(strict=True))}
                              for frame in (frames[index], frames[index + 1]))
            key = (source["token"], target["token"])
            signature = (source["time_s"], target["time_s"], source["filename"], target["filename"])
            if key in seen:
                if seen[key] != signature:
                    raise ValueError("Duplicated tokens have conflicting frame metadata")
                continue
            seen[key] = signature
            yield source, target, entry["scene"], entry["partition"]


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def batched_pairs(iterator, batch_size: int, limit: int):
    pending = []
    for index, row in enumerate(iterator):
        pending.append(row)
        if len(pending) == batch_size:
            yield pending
            pending = []
        if limit and index + 1 >= limit:
            break
    if pending:
        yield pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Separate root for each direction contract")
    parser.add_argument("--partitions", nargs="+", default=["train_core", "dev"])
    parser.add_argument("--direction", choices=["forward", "bidirectional"], default="forward")
    parser.add_argument("--reuse-forward-root", type=Path, help="Use exact previously computed forward tensors; all requested pairs must exist")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 prepares every selected pair")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.batch_size < 1:
        parser.error("limit must be nonnegative; threads and batch-size positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard-index/shard-count")
    if args.reuse_forward_root and args.direction != "bidirectional":
        parser.error("--reuse-forward-root requires --direction bidirectional")
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or args.output / f"prepare_{stamp}_{os.getpid()}.json"
    version = code_version(require_pushed=args.limit == 0)
    report = {"status":"running", "started_at_utc":stamp, **version,
              "manifest":str(args.manifest.resolve()), "manifest_sha256":sha256(args.manifest),
              "arguments":{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "counts":{}, "pairs_processed":0}
    started = time.monotonic()
    counts = Counter()
    _write_json(report_path, report)
    print(json.dumps(report), flush=True)
    try:
        engine = FrozenRaft(args.checkpoint, args.device, direction=args.direction, batch_size=args.batch_size)
        iterator = enumerate_pairs(args.manifest, args.partitions, shard_index=args.shard_index, shard_count=args.shard_count)
        for batch in batched_pairs(iterator, args.batch_size, args.limit):
            jobs = []
            for source, target, scene, partition in batch:
                path = cache_path(args.output, source["token"], target["token"])
                reuse = cache_path(args.reuse_forward_root, source["token"], target["token"]) if args.reuse_forward_root else None
                jobs.append((source, target, path, reuse))
            report["last_pair"] = {"source":source["token"], "target":target["token"], "scene":scene, "partition":partition}
            before = time.monotonic()
            outcomes = engine.pairs(jobs)
            report["last_batch_s"] = time.monotonic() - before
            counts.update(outcomes)
            report["counts"] = dict(counts)
            report["pairs_processed"] += len(jobs)
            report["elapsed_s"] = time.monotonic() - started
            report["seconds_per_pair"] = report["elapsed_s"] / report["pairs_processed"]
            if engine.device.type == "cuda":
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(engine.device)
            if report["pairs_processed"] <= 32 or report["pairs_processed"] % 100 < args.batch_size:
                print(json.dumps({k:report[k] for k in ("pairs_processed","counts","elapsed_s","seconds_per_pair","last_batch_s","last_pair")}), flush=True)
                _write_json(report_path, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(report_path, report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
