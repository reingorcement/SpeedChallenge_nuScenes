"""Generate audited forward RAFT caches from a label-free full nuScenes index."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time

import torch

from prepare_flow import batched_pairs, _write_json
from purevision_v5.common import code_version, now, sha256
from purevision_v5.flow import DEFAULT_CHECKPOINT, FrozenRaft, native_contract, derived_contract
from purevision_v5.full_flow_cache import FLOW_SOURCE, FullFlowCache, load_adoption_plan
from purevision_v5.full_flow_index import load_index, iter_groups, iter_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adoption", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-pushed", action="store_true",
                        help="Verify pushed detached code even for a reusable small sample")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.limit < 0:
        parser.error("batch-size/threads must be positive and limit nonnegative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard-index/shard-count")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    report = dict(status="running", started_at_utc=now(), pairs_processed=0,
                  total=None, expected_pairs=None, target_pairs=None, all_total=None,
                  remaining_pairs=None, estimated_remaining_s=None,
                  counts={key:0 for key in ("created", "reused", "recovered", "adopted")},
                  elapsed_s=0.0, pairs_per_second=0.0, throughput=0.0,
                  bytes=0, bytes_written=0, cuda_peak_allocated_bytes=0,
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    _write_json(args.report, report)
    counts = Counter(report["counts"])
    last_written = started
    digest = hashlib.sha256()
    first_receipts, last_receipts = [], deque(maxlen=3)
    report.update(processed_pair_sha256=digest.hexdigest(), sample_receipts=[], last_receipts=[])
    try:
        version = code_version(require_pushed=args.limit == 0 or args.require_pushed)
        index_sha = sha256(args.index)
        if args.index_sha256 is not None and args.index_sha256 != index_sha:
            raise ValueError("Full generation index checksum differs from launch plan")
        index = load_index(args.index)
        total = sum(group["pair_count"] for group in iter_groups(index, shard_index=args.shard_index,
                                                                 shard_count=args.shard_count))
        if total < 1:
            raise ValueError("Full generation shard is empty")
        target = min(args.limit, total) if args.limit else total
        adoption = load_adoption_plan(args.adoption)
        report.update(code=version, index=str(args.index.resolve()), index_sha256=index_sha,
                      index_identity=index["identity"], total=total, expected_pairs=total, target_pairs=target,
                      all_total=index["counts"]["pairs"], remaining_pairs=target,
                      scope="sample" if args.limit else "full_dataset_generation",
                      flow_py_sha256=sha256(FLOW_SOURCE),
                      adoption={key:value for key,value in adoption.items() if key != "origins"},
                      adoption_origins=[{key:value for key,value in origin.items() if key != "pairs"}
                                        for origin in adoption["origins"]],
                      native_contract=native_contract((576,1024)), derived_contract=derived_contract((576,1024)))
        _write_json(args.report, report)
        print(json.dumps(report), flush=True)
        engine = FrozenRaft(args.checkpoint, args.device, direction="forward", batch_size=args.batch_size,
                            raft_size=(576,1024), generation_code=version)
        cache = FullFlowCache(engine, args.output_root, adoption)
        def record_pair(result):
            counts.update([result["outcome"]])
            report["counts"] = dict(counts)
            report["pairs_processed"] += 1
            report["remaining_pairs"] = target - report["pairs_processed"]
            report["bytes"] += result["bytes"]
            report["bytes_written"] += result["bytes_written"]
            report["last_pair"] = {key:result[key] for key in ("source","target","version","camera","scene")}
            report["elapsed_s"] = time.monotonic() - started
            report["pairs_per_second"] = report["pairs_processed"] / report["elapsed_s"]
            report["throughput"] = report["pairs_per_second"]
            report["estimated_remaining_s"] = report["remaining_pairs"] / report["pairs_per_second"]
            digest.update(f"{result['source']} {result['target']}\n".encode("utf-8"))
            report["processed_pair_sha256"] = digest.hexdigest()
            if len(first_receipts) < 3:
                first_receipts.append(dict(result))
            last_receipts.append(dict(result))
            report["sample_receipts"] = list(first_receipts)
            report["last_receipts"] = list(last_receipts)
        iterator = iter_pairs(args.index, shard_index=args.shard_index, shard_count=args.shard_count,
                              verify_sources=False)
        for rows in batched_pairs(iterator, args.batch_size, args.limit):
            cache.prepare_batch(rows, on_pair=record_pair)
            if engine.device.type == "cuda":
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(engine.device)
            if (report["pairs_processed"] <= 32 or report["pairs_processed"] % 100 < args.batch_size
                    or time.monotonic() - last_written >= 30):
                _write_json(args.report, report)
                last_written = time.monotonic()
                print(json.dumps({key:report[key] for key in ("pairs_processed","total","counts","elapsed_s",
                                                              "pairs_per_second","bytes_written","last_pair")}), flush=True)
        if report["pairs_processed"] != target or sum(counts.values()) != target:
            raise ValueError("Full generation coverage differs from fixed index expected count")
        if sha256(args.index) != index_sha:
            raise ValueError("Full generation index changed during execution")
        report["status"] = "complete" if target == total else "sample_complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(elapsed_s=time.monotonic() - started, finished_at_utc=now())
        _write_json(args.report, report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
