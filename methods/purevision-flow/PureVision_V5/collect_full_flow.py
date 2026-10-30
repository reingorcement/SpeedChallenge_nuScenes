"""Audit completed full-camera caches and the unchanged train/dev packed view."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime
import hashlib
from pathlib import Path
import time

import torch

from full_flow_pipeline import validate_config
from pipeline import validate_plan
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.flow import CONFIGURABLE_SCHEMA, FrozenRaft, cache_path, derived_contract, native_contract, native_path
from purevision_v5.full_flow_cache import FullFlowCache, load_adoption_plan
from purevision_v5.full_flow_index import iter_groups, iter_pairs, load_index
from purevision_v5.packed_flow import PackedFlow, _manifest_pairs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _worker(job, commit):
    value = read_json(job["status_path"])
    require(value.get("status") == "completed" and value.get("exit_code") == 0
            and value.get("commit") == commit, f"Worker did not complete on fixed commit: {job['name']}")
    return value


def packet_verifier(plan, adoption):
    # No constructor, weights, model, CUDA context or RAFT prediction is created.
    # _verify uses only this shell's provenance() and CPU file/tensor validators.
    engine = FrozenRaft.__new__(FrozenRaft)
    engine.schema, engine.direction = CONFIGURABLE_SCHEMA, "forward"
    engine.raft_size, engine.batch_size = (576, 1024), plan["config"]["batch_size"]
    engine.generation_code = plan["code"]
    return FullFlowCache(engine, plan["output_root"], adoption)


def verify_receipt(verifier, row, receipt):
    source, target, group = row
    expected = dict(source=source["token"], target=target["token"],
                    **{k: group[k] for k in ("version", "camera", "scene")})
    require(all(receipt.get(k) == v for k, v in expected.items()), "Sample receipt pair/group differs from index")
    require(receipt.get("outcome") in {"created", "reused", "recovered", "adopted"}, "Invalid sample outcome")
    path = verifier.path(source, target, group)
    require(sha256(path) == receipt["compact_sha256"] and sha256(native_path(path)) == receipt["native_sha256"],
            "Sample native/compact receipt checksum mismatch")
    checked = verifier._verify(path, source, target, recover=False)
    for key in ("native_sha256", "compact_sha256", "identity", "generation_code", "bytes"):
        require(checked[key] == receipt[key], f"Sample receipt {key} mismatch")
    return dict(expected, path=str(path), native_sha256=checked["native_sha256"],
                compact_sha256=checked["compact_sha256"], generation_code=checked["generation_code"],
                bytes=checked["bytes"], outcome=receipt["outcome"],
                validation="CPU native/compact load, image hashes/times/context, contracts, legal mask and exact derived values")


def _coverage(index):
    result = {"versions": defaultdict(Counter), "partitions": defaultdict(Counter)}
    for group in index["groups"]:
        for category, key in (("versions", group["version"]), ("partitions", group["partition"])):
            result[category][key].update(groups=1, frames=group["frame_count"], pairs=group["pair_count"])
    return {kind: {name: dict(value) for name, value in groups.items()} for kind, groups in result.items()}


def inspect(plan, progress):
    validate_plan(plan)
    require(plan.get("stage") == "full_nuscenes_flow" and plan.get("full_generation_authorized") is True
            and plan.get("training_authorized") is False, "Not an approved cache-only plan")
    commit = plan["code"]["commit"]
    require(plan["code"] == dict(commit=commit, dirty=False, verified_remote_commit=commit), "Generation code was not fixed and pushed")
    for path_key, hash_key in (("manifest", "manifest_sha256"), ("data_manifest", "data_manifest_sha256"),
                               ("adoption", "adoption_sha256"), ("config_path", "config_sha256")):
        require(sha256(plan[path_key]) == plan[hash_key], f"Immutable {path_key} changed")
    validate_config(plan["config"])
    require(read_json(plan["config_path"]) == plan["config"], "Actual configuration differs from plan")
    # Metadata sources are hashed once. Shard enumeration below skips that repeat.
    index = load_index(plan["manifest"])
    require(index["identity"] == plan["index_identity"] and index["counts"] == plan["counts"], "Full index differs from plan")
    require(index["cameras"] == plan["config"]["cameras"] and index["versions"] == plan["config"]["versions"], "Changed camera/version coverage")
    require(index["partition_source"]["manifest_sha256"] == plan["data_manifest_sha256"]
            and Path(index["partition_source"]["manifest"]).resolve() == Path(plan["data_manifest"]).resolve(), "Changed partition source")
    flow_sha = sha256(Path(plan["code_root"]) / "purevision_v5/flow.py")
    adoption = load_adoption_plan(plan["adoption"], flow_sha)
    verifier = packet_verifier(plan, adoption)
    jobs = plan["jobs"]
    shard_count = len(plan["config"]["devices"])
    require(shard_count in (2, 3) and [j["kind"] for j in jobs] == ["flow"] * shard_count + ["pack", "verify"],
            "Expected the configured generators, pack and verification")
    flow_jobs = jobs[:shard_count]
    shards, full_pairs, all_tokens, totals = [], {}, set(), Counter()
    progress(phase="verifying_flow_shards", coverage=_coverage(index), counts=index["counts"], shards=shards)
    for shard_number, job in enumerate(flow_jobs):
        _worker(job, commit)
        report = read_json(job["progress_path"])
        args = report["arguments"]
        expected_count = sum(g["pair_count"] for g in iter_groups(index, shard_index=shard_number, shard_count=shard_count))
        require(report.get("status") == "complete" and report.get("scope") == "full_dataset_generation"
                and report.get("code") == plan["code"], "Incomplete/nonformal shard or mismatched generation code")
        require(report["index_identity"] == index["identity"] and report["index_sha256"] == plan["manifest_sha256"]
                and Path(report["index"]).resolve() == Path(plan["manifest"]).resolve()
                and report["flow_py_sha256"] == flow_sha, "Shard index/numerical source mismatch")
        require(args["limit"] == 0 and args["shard_index"] == shard_number and args["shard_count"] == shard_count
                and args["batch_size"] == plan["config"]["batch_size"] and args["device"] == job["device"]
                and Path(args["output_root"]).resolve() == Path(plan["output_root"]).resolve()
                and Path(args["index"]).resolve() == Path(plan["manifest"]).resolve()
                and args["index_sha256"] == plan["manifest_sha256"], "Shard launch scope differs from plan")
        require(Path(args["adoption"]).resolve() == Path(plan["adoption"]).resolve()
                and report["adoption"].get("sha256") == plan["adoption_sha256"], "Shard adoption source differs")
        require(report["native_contract"] == native_contract((576, 1024))
                and report["derived_contract"] == derived_contract((576, 1024)), "Shard resolution/units contract differs")
        require(all(report[key] == expected_count for key in ("pairs_processed", "expected_pairs", "target_pairs", "total"))
                and job["expected_pairs"] == expected_count and report["remaining_pairs"] == 0
                and report["all_total"] == index["counts"]["pairs"], "Shard processed/expected count mismatch")
        outcomes = report["counts"]
        require(set(outcomes) == {"created", "reused", "recovered", "adopted"}
                and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in outcomes.values())
                and sum(outcomes.values()) == expected_count, "Shard outcome accounting mismatch")
        digest, count, observed_bytes = hashlib.sha256(), 0, 0
        first, last = [], deque(maxlen=3)
        for row in iter_pairs(plan["manifest"], shard_index=shard_number, shard_count=shard_count, verify_sources=False):
            source, target, group = row
            key = source["token"], target["token"]
            require(key not in full_pairs, "Duplicate pair across full shards")
            full_pairs[key] = (source["time_s"], target["time_s"], group["partition"])
            all_tokens.update(key)
            digest.update(f"{key[0]} {key[1]}\n".encode())
            count += 1
            if len(first) < 3:
                first.append(row)
            last.append(row)
            # Stat every declared packet, without rereading the large native data.
            path = cache_path(Path(plan["output_root"]) / group["version"] / group["camera"], *key, schema=CONFIGURABLE_SCHEMA)
            sizes = path.stat().st_size, native_path(path).stat().st_size
            require(min(sizes) > 0, "Empty native/compact packet")
            observed_bytes += sum(sizes)
            if count % 5000 == 0:
                progress(active_shard=job["name"], active_pairs_checked=count, active_pairs_total=expected_count)
        require(count == expected_count and digest.hexdigest() == report["processed_pair_sha256"], "Shard ordered pair digest mismatch")
        require(observed_bytes == report["bytes"] and 0 <= report["bytes_written"] <= observed_bytes, "Shard actual file byte accounting mismatch")
        require(len(report["sample_receipts"]) == len(first) and len(report["last_receipts"]) == len(last), "Missing first/last sample receipts")
        checked, seen_receipts = [], set()
        for rows, receipts in ((first, report["sample_receipts"]), (list(last), report["last_receipts"])):
            for row, receipt in zip(rows, receipts):
                item = verify_receipt(verifier, row, receipt)
                key = item["source"], item["target"]
                if key not in seen_receipts:
                    checked.append(item)
                    seen_receipts.add(key)
        shard = dict(name=job["name"], device=job["device"], status="complete", expected_pairs=count,
                     processed_pair_sha256=digest.hexdigest(), report=job["progress_path"], report_sha256=sha256(job["progress_path"]),
                     counts=outcomes, bytes=observed_bytes, bytes_written=report["bytes_written"],
                     elapsed_s=report["elapsed_s"], pairs_per_second=count/report["elapsed_s"],
                     started_at_utc=report["started_at_utc"], finished_at_utc=report["finished_at_utc"],
                     sampled_pairs=checked, cuda_peak_allocated_bytes=report["cuda_peak_allocated_bytes"])
        shards.append(shard)
        totals.update(outcomes)
        progress(shards=shards, active_shard=None, active_pairs_checked=0)
    require(len(full_pairs) == index["counts"]["pairs"] and len(all_tokens) == index["counts"]["frames"], "Full pair/frame coverage differs")
    progress(phase="hashing_packed_arrays", packed_array_hash_note="Hashes all packed arrays, approximately 8.77 GB for 158513 pairs; native corpus is not reread")
    _worker(jobs[shard_count], commit)
    pack_started = time.monotonic()
    packed = PackedFlow(plan["packed_train_dev"], dataset_sha256=plan["data_manifest_sha256"])
    pack = packed.index
    contract = pack["contract"]
    require(pack["code"]["commit"] == commit and pack["code"]["dirty"] is False
            and pack["code"].get("verified_remote_commit") in (None, commit), "Packed view was made by another/dirty commit")
    require(contract["partitions"] == ["train_core", "dev"] and "selection" not in contract
            and "selection_sha256" not in contract and contract["fb_available"] is False
            and Path(contract["source_root"]).resolve() == Path(plan["output_root"]).resolve()/"v1.0-trainval/CAM_FRONT",
            "Packed view scope changed or includes heldout/selected data")
    require(packed.flow_contract["schema"] == CONFIGURABLE_SCHEMA
            and packed.flow_contract["native_contract"] == native_contract((576,1024))
            and packed.flow_contract["derived_contract"] == derived_contract((576,1024)), "Packed flow frontend differs")
    expected = _manifest_pairs(plan["data_manifest"], ("train_core", "dev"))
    actual = [(r["source"], r["target"], r["source_time"], r["target_time"]) for r in pack["pairs"]]
    require(actual == expected, "Packed pair set/timestamps differ from complete original train_core/dev windows")
    for source, target, source_time, target_time in actual:
        require((source, target) in full_pairs and full_pairs[(source,target)][:2] == (source_time,target_time)
                and full_pairs[(source,target)][2] in {"train_core", "dev"}, "Packed pair is absent from allowed full index")
    packed_by_key = {(r["source"],r["target"]):r for r in pack["pairs"]}
    sampled_packed = 0
    for shard in shards:
        for receipt in shard["sampled_pairs"]:
            row = packed_by_key.get((receipt["source"],receipt["target"]))
            if row:
                require(row["source_sha256"] == receipt["compact_sha256"], "Packed sampled compact checksum differs")
                sampled_packed += 1
    arrays = {name: dict(info, bytes=(Path(plan["packed_train_dev"])/info["file"]).stat().st_size)
              for name,info in pack["arrays"].items()}
    pack_evidence = dict(root=plan["packed_train_dev"], index_sha256=packed.index_sha256, identity=packed.identity,
                         pairs=len(actual), partitions=contract["partitions"], arrays=arrays,
                         hashed_array_bytes=sum(a["bytes"] for a in arrays.values()),
                         validation_elapsed_s=time.monotonic()-pack_started, complete_original_pair_set_and_times=True,
                         sampled_source_checksums_matched=sampled_packed, flow_contract=packed.flow_contract)
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    wall = (max(parse(s["finished_at_utc"]) for s in shards)-min(parse(s["started_at_utc"]) for s in shards)).total_seconds()
    require(wall > 0, "Invalid shard generation wall duration")
    return dict(index=plan["manifest"], index_sha256=plan["manifest_sha256"], index_identity=index["identity"],
                counts=index["counts"], cameras=index["cameras"], versions=index["versions"], coverage=_coverage(index),
                generation_code=plan["code"], flow_py_sha256=flow_sha, shards=shards, packed_train_dev=pack_evidence,
                outcomes=dict(totals), cache_bytes=sum(s["bytes"] for s in shards),
                bytes_written=sum(s["bytes_written"] for s in shards), generation_wall_s=wall,
                worker_elapsed_s_sum=sum(s["elapsed_s"] for s in shards),
                generation_wall_pairs_per_second=index["counts"]["pairs"]/wall,
                adoption={k:v for k,v in adoption.items() if k != "origins"},
                adoption_origins=[{k:v for k,v in origin.items() if k != "pairs"} for origin in adoption["origins"]],
                validation_scope=dict(all_index_sources_and_group_checksums=True, ordered_pair_digests=f"all {shard_count} shards independently recomputed",
                                      native_and_compact_file_presence_and_sizes="every indexed pair", full_native_content_reread=False,
                                      native_and_compact_checksum_tensor_samples=sum(len(s["sampled_pairs"]) for s in shards),
                                      sample_policy="first three and last three pairs per shard, deduplicated",
                                      packed_array_hashes="all arrays", packed_pair_set_times="complete original train_core/dev",
                                      generation_validation="Fresh packets use unchanged flow.save_pair metadata/tensor construction and native-byte SHA binding; adoption/resume validate native/compact identity, source images and exact derivation per pair",
                                      limitation="Final native content audit is sampled; file presence/sizes and ordered coverage are complete, not a fresh hash pass over the entire native corpus"),
                model_training_started=False, model_metrics=None,
                boundary="Cache-only holdout, unassigned and official test scenes remain excluded from train/dev packing and model selection")


def markdown(report):
    lines = ["# Full nuScenes CAM_FRONT optical-flow cache", "", f"Status: **{report['status']}**.",
             "", "This report covers preprocessing and storage. It contains no model-training or accuracy result."]
    if report.get("error"):
        lines += ["", "Failure: " + report["error"]]
    if report.get("coverage"):
        lines += ["", "| Scope | Groups/scenes | Frames | Adjacent pairs |", "|---|---:|---:|---:|"]
        for kind in ("versions", "partitions"):
            for name, row in report["coverage"][kind].items():
                lines.append(f"| {name} | {row['groups']} | {row['frames']} | {row['pairs']} |")
    if report["status"] == "complete":
        lines += ["", f"Generation commit: `{report['generation_code']['commit']}`.",
                  f"Native + compact bytes present: {report['cache_bytes']:,}; bytes written in this invocation: {report['bytes_written']:,}.",
                  f"Generation wall time: {report['generation_wall_s']:.1f} s; wall throughput: {report['generation_wall_pairs_per_second']:.3f} pairs/s.",
                  "Outcome counts: " + str(report["outcomes"]),
                  f"Packed original train_core/dev pairs: {report['packed_train_dev']['pairs']:,}.",
                  f"Packed array bytes hashed: {report['packed_train_dev']['hashed_array_bytes']:,}; packed validation took {report['packed_train_dev']['validation_elapsed_s']:.1f} s.",
                  "", "Validation:", "", "- The complete index, group checksums, every shard's ordered pair digest, and every expected packet's presence/size were checked.",
                  "- Fresh generation validates packet construction and binds native bytes by SHA; adoption/resume also checks source image provenance and exact native-to-compact derivation per pair.",
                  f"- Final native/compact checksum and CPU tensor reload covers {report['validation_scope']['native_and_compact_checksum_tensor_samples']} distinct first/last samples; the entire native corpus was not read again.",
                  "- Every packed array checksum and the complete original train/dev pair set and timestamps were checked.",
                  "- Holdout, unassigned and official test caching does not authorize model selection or training on those scenes.",
                  "", "Approved reuse origins:"]
        lines += [f"- `{origin['commit']}`: {origin.get('native_root', 'explicit compatible existing destination only')}" for origin in report["adoption_origins"]]
    return "\n".join(lines) + "\n"


def collect(plan_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(schema="v5_full_flow_report_v1", status="running", started_at=now(), updated_at=now(),
                  phase="validating_inputs", plan=str(Path(plan_path).resolve()), model_training_started=False)
    def progress(**values):
        report.update(values, updated_at=now(), verification_elapsed_s=time.monotonic()-started)
        write_json(output/"summary.json", report)
    progress()
    try:
        plan = read_json(plan_path)
        progress(plan_identity=plan.get("identity"), plan_sha256=sha256(plan_path))
        report.update(inspect(plan, progress), status="complete", phase="complete")
    except Exception as error:
        report.update(status="failed", phase="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report.update(finished_at=now())
        progress()
        (output/"summary.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = collect(args.plan, args.output)
    print({"status":report["status"], "output":args.output, "error":report.get("error")}, flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
