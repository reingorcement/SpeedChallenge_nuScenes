"""Fixed small-scene R02 resolution comparison; never launches full-cache generation.

Reuse the existing worker, locks, observation rules and tmux launcher. A user
decision after this pilot is required before any full-resolution dataset job.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

from pipeline import initialize, lock, tick
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256
from purevision_v5.selection import load_selection


def build_plan(root, manifest, selection_path, low_flow, prefix, version, *, high_batch=8,
               code_root=ROOT, python=sys.executable):
    root, manifest, selection_path, low_flow, code_root = [Path(p).resolve() for p in
        (root, manifest, selection_path, low_flow, code_root)]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", prefix):
        raise ValueError("Invalid pilot run prefix")
    if not isinstance(high_batch, int) or isinstance(high_batch, bool) or high_batch < 1:
        raise ValueError("High-resolution generation batch must be positive")
    selection = load_selection(manifest, selection_path)
    if selection is None or selection["scope"] != "resolution_pilot":
        raise ValueError("An explicit validated pilot selection is mandatory")
    if selection["seed"] != 42:
        raise ValueError("Pilot selection must use the locked seed 42")
    if (selection["counts"]["train_core"]["scenes"] > 32
            or selection["counts"]["dev"]["scenes"] > 8):
        raise ValueError("This launcher is capped at 32 train / 8 dev scenes; full generation needs a new decision")
    base = root / "pipeline" / prefix
    high_flow = root / "flows" / f"{prefix}_raft1024_native"
    high_pack = root / "flows" / f"{prefix}_raft1024_packed"
    jobs = []

    def command(script, *args):
        return [str(python), "-B", str(code_root / script), *map(str, args)]

    def add(name, kind, device, requires, commands, *, output=None, progress=None,
            run_dir=None, checks=(), experiment=None, config_path=None):
        jobs.append(dict(name=name, job_id=f"{prefix}_{name}", kind=kind, device=device,
                         requires=list(requires), after=[], commands=commands,
                         session=f"v5next_{prefix}_{name}",
                         log_path=str(root / "logs" / f"{prefix}_{name}.log"),
                         status_path=str(base / "jobs" / f"{name}.json"),
                         progress_path=str(progress) if progress else None,
                         fresh_output=str(output) if output else None,
                         run_dir=str(run_dir) if run_dir else None, checks=list(checks),
                         experiment=experiment, config_path=str(config_path) if config_path else None))

    def training(width, device, flow, requires):
        name = f"R02_{width}"
        config = code_root / "configs" / "next" / f"R02_raft{width}.json"
        conf = read_json(config)
        if (conf["experiment"] != "R02" or conf["epochs"] != 40 or conf["seed"] != 42
                or conf["batch_size"] != 64 or conf.get("early_stopping") is not False):
            raise ValueError("Pilot requires the locked 40-epoch R02 protocol")
        output = root / "runs" / f"{prefix}_{name}"
        evaluation = root / "evaluations" / output.name
        inputs = ["--manifest", manifest, "--selection", selection_path,
                  "--flow", flow, "--device", device, "--workers", 4]
        add(name, "training", device, requires,
            [command("train.py", "--config", config, *inputs, "--output", output),
             command("evaluate.py", "--checkpoint", output / "best.pt", *inputs, "--output", evaluation)],
            output=output, progress=output / "status.json", run_dir=output,
            experiment="R02", config_path=config,
            checks=[dict(path=str(output / "status.json"), field="status", equals="completed"),
                    dict(path=str(evaluation / "evaluation.json"), field="status", equals="completed"),
                    dict(path=str(output / "last.pt")), dict(path=str(output / "best.pt"))])

    training(512, "cuda:1", low_flow, [])
    # Two disjoint scene shards share only immutable pair files. They have
    # separate progress receipts, locks and devices; neither owns the root.
    for shard, device in ((0, "cuda:0"), (1, "cuda:2")):
        name = f"flow1024_{shard}"
        report = base / "reports" / f"{name}.json"
        add(name, "flow", device, [],
            [command("prepare_flow.py", "--manifest", manifest, "--selection", selection_path,
                     "--output", high_flow, "--direction", "forward", "--raft-width", 1024,
                     "--raft-height", 576, "--batch-size", high_batch, "--device", device,
                     "--shard-index", shard, "--shard-count", 2, "--report", report)],
            progress=report, checks=[dict(path=str(report), field="status", equals="complete")])
    add("pack1024", "pack", None, ["flow1024_0", "flow1024_1"],
        [command("pack_flow.py", "--manifest", manifest, "--selection", selection_path,
                 "--source-root", high_flow, "--output", high_pack)], output=high_pack,
        checks=[dict(path=str(high_pack / "packed_index.json"), field="schema", equals="v5_packed_flow_v1")])
    training(1024, "cuda:1", high_pack, ["pack1024"])
    configs = {job["name"]: read_json(job["config_path"]) for job in jobs if job["kind"] == "training"}
    common = [{k: v for k, v in conf.items() if k != "expected_flow"} for conf in configs.values()]
    if common[0] != common[1]:
        raise ValueError("Resolution comparisons may only change expected frontend size")
    plan = dict(schema="v5_pipeline_v1", scope="resolution_pilot", stage="raft_resolution",
                root=str(root), code_root=str(code_root), python=str(python), prefix=prefix,
                code=version, manifest=str(manifest), manifest_sha256=sha256(manifest),
                selection_path=str(selection_path), selection=selection, selection_sha256=sha256(selection_path),
                features=None, features_status=None, low_flow_root=str(low_flow),
                low_flow_index_sha256=sha256(low_flow / "packed_index.json"),
                config_hashes={job["name"]: sha256(job["config_path"]) for job in jobs if job["kind"] == "training"},
                full_generation_authorized=False, high_batch=high_batch, jobs=jobs)
    return {**plan, "identity": object_hash(plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "selection", "low-flow", "run-prefix"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--high-batch", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.root, args.manifest, args.selection, args.low_flow, args.run_prefix,
                      code_version(require_pushed=not args.dry_run), high_batch=args.high_batch)
    if args.dry_run:
        import json
        print(json.dumps(plan, indent=2))
        return
    with lock(Path(args.root).resolve() / "pipeline/coordinator.lock"):
        path = initialize(plan)
        print({"time": now(), "plan": str(path), "identity": plan["identity"], "scope": plan["scope"]}, flush=True)
        while True:
            done, launched = tick(plan, path)
            if launched:
                print({"time": now(), "launched": launched}, flush=True)
            if done:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
