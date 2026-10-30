"""Four additional R02 runs on existing pilot caches; no RAFT or full-data jobs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

from collect_results import require
from pipeline import initialize, lock, tick, validate_plan
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256
from purevision_v5.selection import load_selection
from resolution_pilot import collect_finished_plan, computation_identity, validate_adopted_source


SEEDS = (43, 44)
WIDTHS = (512, 1024)


def validate_multiseed_inputs(plan):
    """Small immutable inputs, callable before every queued worker command."""
    require(plan.get("stage") == "raft_resolution_multiseed", "Not a resolution seed-sweep plan")
    require(plan.get("new_training_seeds") == list(SEEDS) and plan.get("all_training_seeds") == [42, 43, 44],
            "Training seeds differ from the approved sweep")
    require(plan.get("full_generation_authorized") is False and plan.get("raft_generation_authorized") is False,
            "This sweep does not authorize data generation")
    require(sha256(plan["manifest"]) == plan["manifest_sha256"]
            and sha256(plan["selection_path"]) == plan["selection_sha256"], "Seed sweep data/selection changed")
    selection = load_selection(plan["manifest"], plan["selection_path"])
    require(selection == plan["selection"] and selection["seed"] == 42, "Scene selection must stay at seed 42")
    source = plan["seed42_source"]
    require(sha256(source["path"]) == source["sha256"], "Seed-42 source plan bytes changed")
    original = read_json(source["path"])
    validate_plan(original)
    require(original["identity"] == source["identity"] and original["selection"] == selection,
            "Seed-42 source plan identity/selection changed")
    validate_adopted_source(original)
    require(computation_identity(plan["code_root"]) == source["computation"], "Numerical code changed since seed 42")
    for width in WIDTHS:
        pack = plan["locked_packs"][str(width)]
        path = Path(pack["root"]) / "packed_index.json"
        require(sha256(path) == pack["index_sha256"], "Existing packed flow index changed")
        index = read_json(path)
        require(index["identity"] == pack["identity"] == object_hash({k:v for k,v in index.items() if k != "identity"}),
                "Existing packed flow identity changed")
        require(index["contract"]["dataset_sha256"] == plan["manifest_sha256"] and index["contract"]["fb_available"] is False,
                "Only the fixed forward-only pilot inputs are allowed")
    require(len(plan["jobs"]) == 4 and all(job["kind"] == "training" for job in plan["jobs"]),
            "Only four additional training jobs are allowed")
    require({(job["raft_width"],job["training_seed"]) for job in plan["jobs"]} == {(w,s) for w in WIDTHS for s in SEEDS},
            "Missing, duplicate or unexpected seed/resolution job")
    for job in plan["jobs"]:
        width, seed = job["raft_width"], job["training_seed"]
        conf = read_json(job["config_path"])
        require(sha256(job["config_path"]) == plan["config_hashes"][job["name"]], "Seed-sweep configuration changed")
        reference = dict(source["rows"][f"R02_{width}"]["config"], seed=seed)
        require(conf == reference, "New configuration differs beyond the approved training seed")
    return original


def build_plan(root, source_plan, prefix, version, *, code_root=ROOT, python=sys.executable):
    from collect_resolution import collect
    root, source_plan, code_root = [Path(p).resolve() for p in (root, source_plan, code_root)]
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", prefix)), "Invalid seed-sweep prefix")
    source = read_json(source_plan)
    validate_plan(source)
    require(source["root"] == str(root) and source["prefix"] != prefix, "Reuse requires the same artifact root and a new prefix")
    report = collect(source_plan)
    require(report["comparison"]["status"] == "comparable" and all(r["status"] == "completed" for r in report["rows"].values()),
            "Both seed-42 pilot runs must be complete and verified")
    selection = load_selection(source["manifest"], source["selection_path"])
    require(selection["seed"] == 42 and selection["counts"]["train_core"]["scenes"] <= 32
            and selection["counts"]["dev"]["scenes"] <= 8, "The fixed small-scene pilot is required")
    computation = computation_identity(code_root)
    require(all(row["computation"] == computation for row in report["rows"].values()), "Numerical code differs from seed 42")
    base = root / "pipeline" / prefix
    rows, packs, jobs = {}, {}, []
    for width in WIDTHS:
        row = report["rows"][f"R02_{width}"]
        rows[f"R02_{width}"] = {k: row[k] for k in ("run_id", "training_commit", "best_checkpoint_sha256", "predictions_sha256", "target_identity", "config")}
        pack_root = Path(row["data"]["flow_root"])
        index_path = pack_root / "packed_index.json"
        index = read_json(index_path)
        packs[str(width)] = dict(root=str(pack_root), index_sha256=sha256(index_path), identity=index["identity"])
    for seed in SEEDS:
        for width, device in ((512, "cuda:1"), (1024, "cuda:2")):
            name = f"R02_{width}_s{seed}"
            config = code_root / "configs/next" / f"R02_raft{width}_s{seed}.json"
            conf = read_json(config)
            require(conf == dict(rows[f"R02_{width}"]["config"], seed=seed), "Configuration differs beyond training seed")
            output = root / "runs" / f"{prefix}_{name}"
            evaluation = root / "evaluations" / output.name
            inputs = ["--manifest", source["manifest"], "--selection", source["selection_path"],
                      "--flow", packs[str(width)]["root"], "--device", device, "--workers", "4"]
            def command(script, *args):
                return [str(python), "-B", str(code_root / script), *map(str, args)]
            jobs.append(dict(name=name, job_id=output.name, kind="training", device=device, experiment="R02",
                training_seed=seed, raft_width=width, requires=[], after=[] if seed == 43 else [f"R02_{width}_s43"],
                config_path=str(config), commands=[command("train.py", "--config", config, *inputs, "--output", output),
                    command("evaluate.py", "--checkpoint", output / "best.pt", *inputs, "--output", evaluation)],
                session=f"v5next_{prefix}_{name}", log_path=str(root / "logs" / f"{output.name}.log"),
                status_path=str(base / "jobs" / f"{name}.json"), progress_path=str(output / "status.json"),
                run_dir=str(output), fresh_output=str(output),
                checks=[dict(path=str(output / "status.json"), field="status", equals="completed"),
                        dict(path=str(evaluation / "evaluation.json"), field="status", equals="completed"),
                        dict(path=str(output / "last.pt")), dict(path=str(output / "best.pt"))]))
    plan = dict(schema="v5_pipeline_v1", scope="resolution_pilot", stage="raft_resolution_multiseed",
                root=str(root), code_root=str(code_root), python=str(python), prefix=prefix, code=version,
                manifest=source["manifest"], manifest_sha256=source["manifest_sha256"], selection_path=source["selection_path"],
                selection=selection, selection_sha256=source["selection_sha256"], features=None, features_status=None,
                low_flow_root=packs["512"]["root"], low_flow_index_sha256=packs["512"]["index_sha256"], locked_packs=packs,
                full_generation_authorized=False, raft_generation_authorized=False,
                new_training_seeds=list(SEEDS), all_training_seeds=[42,43,44],
                seed42_source=dict(path=str(source_plan), sha256=sha256(source_plan), identity=source["identity"],
                                   code=source["code"], computation=computation, rows=rows),
                config_hashes={job["name"]:sha256(job["config_path"]) for job in jobs}, jobs=jobs)
    validate_multiseed_inputs(plan)
    return {**plan, "identity": object_hash(plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-plan", "run-prefix"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.root, args.source_plan, args.run_prefix, code_version(require_pushed=not args.dry_run))
    if args.dry_run:
        import json
        print(json.dumps(plan, indent=2))
        return
    with lock(Path(args.root).resolve() / "pipeline/coordinator.lock"):
        path = initialize(plan)
        print({"time":now(), "plan":str(path), "identity":plan["identity"], "stage":plan["stage"]}, flush=True)
        while True:
            done, launched = tick(plan, path)
            if launched:
                print({"time":now(), "launched":launched}, flush=True)
            if done:
                collect_finished_plan(plan, path, collector_script="collect_resolution_multiseed.py", report_prefix="resolution_multiseed_")
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
