"""Immutable cache -> three frozen-readout trainings -> evaluation/comparison DAG.

Reuse the existing conservative tmux worker and GPU locks. Every GPU command
binds the planned physical UUID; cuda:0 is its process-local device. This module
does not restore, move or stop any pre-existing experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import pipeline
from fusion_abc_launch import ART, FLOW, GPU, MANIFEST, PY, SOURCE
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256, write_json

STAGE = "frozen_fusion_abc_pilot"
SOURCE_SHA = "ab25fd6c4a7cd14a99a189f56e6b47913ed0f946a60cff2d0c9102479bc30ff8"
GPU_NAMES = {0: "RTX PRO 6000", 1: "RTX 5090", 2: "RTX 5090"}


def gpu_inventory():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = {}
    for line in text.splitlines():
        index, uuid, name = [v.strip() for v in line.split(",", 2)]
        rows[int(index)] = dict(index=int(index), uuid=uuid, name=name)
    for index in GPU:
        if rows.get(index, {}).get("uuid") != GPU[index] or GPU_NAMES[index] not in rows[index]["name"]:
            raise ValueError(f"Physical GPU {index} no longer matches authorized UUID/model")
    return rows


def validate_inputs(plan, job=None):
    pipeline.validate_plan(plan)
    if plan.get("stage") != STAGE:
        raise ValueError("Not an authorized frozen A/B/C plan")
    checks = {"manifest": "manifest_sha256", "selection_path": "selection_sha256",
              "source_checkpoint": "source_checkpoint_sha256", "verification": "verification_sha256",
              "run_spec": "run_spec_sha256"}
    for path_key, hash_key in checks.items():
        if sha256(plan[path_key]) != plan[hash_key]:
            raise ValueError(f"Locked input changed: {path_key}")
    if plan["source_checkpoint_sha256"] != SOURCE_SHA:
        raise ValueError("Wrong trained appearance/difference encoder checkpoint")
    spec = read_json(plan["run_spec"])
    for plan_key, spec_key in (("manifest_sha256", "manifest_sha256"), ("selection_sha256", "selection_sha256"),
                               ("source_checkpoint_sha256", "source_sha256"), ("config_hashes", "config_sha256")):
        if plan[plan_key] != spec[spec_key]:
            raise ValueError(f"Runtime identity differs from the committed execution spec: {plan_key}")
    if spec["jobs"] != {job["name"]: job["job_id"] for job in plan["jobs"]}:
        raise ValueError("Run IDs differ from the pre-published execution spec")
    if read_json(plan["verification"]).get("status") != "passed":
        raise ValueError("ABC real-data preflight is not passed")
    for name in "ABC":
        config = Path(plan["code_root"]) / f"configs/next/FUSION_ABC_{name}.json"
        if sha256(config) != plan["config_hashes"][name]:
            raise ValueError(f"Locked {name} configuration changed")
    for path, expected in ((Path(plan["flow_root"]) / "packed_index.json", plan["flow_index_sha256"]),
                           (Path(plan["rgb_frames"]) / "index.json", plan["rgb_frames_index_sha256"])):
        if sha256(path) != expected:
            raise ValueError("An existing flow/RGB cache index changed")
    # Preserve the explicitly interrupted historical control. No old job is adopted.
    if read_json(Path(plan["root"]) / "runs/20261031_r03_flex_s42_R03_FT/status.json").get("status") != "interrupted":
        raise ValueError("Historical R03_FT is no longer paused; recheck task boundaries")
    if [j["name"] for j in plan["jobs"] if j["kind"] == "training"] != ["A", "B", "C"]:
        raise ValueError("This pilot authorizes exactly A/B/C training jobs")
    if job and job.get("physical_gpu"):
        actual = gpu_inventory()[job["physical_gpu"]["index"]]
        if actual != job["physical_gpu"]:
            raise ValueError("Physical GPU identity changed since planning")


def build_plan(run_spec, version, *, code_root=ROOT, python=PY):
    spec_path = Path(run_spec).resolve()
    spec = read_json(spec_path)
    prefix = spec["prefix"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", prefix):
        raise ValueError("Invalid unique experiment prefix")
    root = Path(spec.get("root", ART)).resolve()
    code_root = Path(code_root).resolve()
    base = root / "pipeline" / prefix
    manifest = Path(spec.get("manifest", MANIFEST)).resolve()
    selection = Path(spec.get("selection_path", root / "plans/resolution_pilot_s42_32t8v.json")).resolve()
    source = Path(spec.get("source_checkpoint", SOURCE)).resolve()
    flow = Path(spec.get("flow_root", FLOW)).resolve()
    rgb = Path(spec.get("rgb_frames", root / "rgb_frames/v3_uint8_train_dev_v1")).resolve()
    cache = Path(spec.get("cache_root", root / "fusion_abc_cache" / prefix)).resolve()
    verification = Path(spec.get("verification", root / "verification/fusion_abc_final_20261101/receipt.json")).resolve()
    report = root / "reports" / f"fusion_abc_{prefix}"
    inventory = gpu_inventory()
    jobs, evaluation_paths = [], {}

    def command(script, *args, gpu=None):
        cmd = [str(python), "-B", str(code_root / script), *map(str, args)]
        if gpu is not None:
            cmd = ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={GPU[gpu]}", *cmd]
        return cmd

    def add(name, kind, gpu, requires, commands, *, run_dir=None, progress=None, fresh_output=None, checks=(), **extra):
        jobs.append(dict(name=name, job_id=f"{prefix}_{name}", kind=kind,
                         device=None if gpu is None else f"cuda:{gpu}",
                         physical_gpu=inventory[gpu] if gpu is not None else None,
                         process_device="cuda:0" if gpu is not None else "cpu",
                         requires=list(requires), after=[], commands=commands,
                         session=f"v5abc_{prefix}_{name}", log_path=str(root / "logs" / f"{prefix}_{name}.log"),
                         status_path=str(base / "jobs" / f"{name}.json"),
                         run_dir=str(run_dir) if run_dir else None, progress_path=str(progress) if progress else None,
                         fresh_output=str(fresh_output) if fresh_output else None, checks=list(checks), **extra))

    add("cache", "cache", 0, [], [command("fusion_abc_extract.py", "--checkpoint", source,
        "--manifest", manifest, "--selection", selection, "--flow", flow, "--rgb-frames", rgb,
        "--scope", "pilot", "--allow-pilot-cache", "--batch-size", spec.get("cache_batch_size", 2),
        "--device", "cuda:0", "--output", cache, gpu=0)], progress=cache / "status.json", fresh_output=cache,
        checks=[dict(path=str(cache / "index.json"), field="schema", equals="v5_fusion_abc_cache_v1"),
                dict(path=str(cache / "status.json"), field="status", equals="completed")])
    for name, gpu in (("A", 2), ("B", 1), ("C", 0)):
        out = root / "runs" / f"{prefix}_{name}"
        config = code_root / f"configs/next/FUSION_ABC_{name}.json"
        evaluation_paths[name] = {p: str(out / f"eval_{p}") for p in ("dev", "train_core")}
        commands = [command("fusion_abc_train.py", "--config", config, "--cache", cache,
                            "--output", out, "--device", "cuda:0", gpu=gpu)]
        commands.extend(command("fusion_abc_evaluate.py", "--checkpoint", out / "best.pt", "--cache", cache,
            "--partition", p, "--output", evaluation_paths[name][p], "--device", "cuda:0",
            "--require-completed-run", out, gpu=gpu) for p in ("dev", "train_core"))
        add(name, "training", gpu, ["cache"], commands, run_dir=out, progress=out / "status.json", fresh_output=out,
            config_path=str(config), variant=name,
            checks=[dict(path=str(out / "status.json"), field="status", equals="completed"),
                    dict(path=str(out / "status.json"), field="completed_epochs", equals=20),
                    dict(path=str(out / "status.json"), field="optimizer_updates", equals=2300),
                    *[dict(path=str(out / f"eval_{p}" / "evaluation.json"), field="status", equals="completed")
                      for p in ("dev", "train_core")]])
    add("compare", "comparison", None, ["A", "B", "C"],
        [command("fusion_abc_comparison.py", "--plan", base / "plan.json", "--output", report)],
        fresh_output=report, checks=[dict(path=str(report / "summary.json"), field="status", equals="completed")])
    plan = dict(schema="v5_pipeline_v1", stage=STAGE, root=str(root), code_root=str(code_root), python=str(python),
                prefix=prefix, code=version, jobs=jobs, features=None, features_status=None,
                manifest=str(manifest), manifest_sha256=sha256(manifest), selection_path=str(selection), selection_sha256=sha256(selection),
                source_checkpoint=str(source), source_checkpoint_sha256=sha256(source),
                flow_root=str(flow), flow_index_sha256=sha256(flow / "packed_index.json"),
                rgb_frames=str(rgb), rgb_frames_index_sha256=sha256(rgb / "index.json"),
                verification=str(verification), verification_sha256=sha256(verification),
                run_spec=str(spec_path), run_spec_sha256=sha256(spec_path), cache_root=str(cache),
                expected_counts=dict(train_core=dict(windows=7324, scenes=32), dev=dict(windows=1826, scenes=8)),
                expected_batches_per_epoch=115, expected_optimizer_updates_per_run=2300,
                config_hashes={v: sha256(code_root / f"configs/next/FUSION_ABC_{v}.json") for v in "ABC"},
                evaluation_paths=evaluation_paths, completion_report=str(report / "summary.json"),
                gpu_inventory=inventory, worker_network_file=str(root / "monitoring/network.json"))
    plan["identity"] = object_hash(plan)
    validate_inputs(plan)
    return plan


class Tmux(pipeline.Tmux):
    """Run the existing worker through this plan-specific input-validation adapter."""
    def launch(self, plan_path, plan, job):
        if self.exists(job["session"]):
            raise RuntimeError("Named tmux session already exists")
        Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
        command = [plan["python"], "-B", str(Path(plan["code_root"]) / "fusion_abc_pipeline.py"),
                   "--worker", "--plan", str(plan_path), "--job", job["name"]]
        # Match the established private transport; never serialize proxy values.
        command = [plan["python"], "-B", str(Path(plan["code_root"]) / "scripts/secure_exec.py"),
                   "--network", plan["worker_network_file"], "--", *command]
        shell = "exec " + shlex.join(command) + " >> " + shlex.quote(job["log_path"]) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", job["session"], "-c", plan["code_root"], shell], check=True)


def worker(plan_path, name):
    plan = read_json(plan_path)
    validate_inputs(plan)
    job = next(j for j in plan["jobs"] if j["name"] == name)
    # A newly occupied target remains queued; never select another GPU.
    if job["device"] and job["device"] in pipeline.busy_gpus():
        state = read_json(job["status_path"])
        state.update(status="queued", reason="Target physical GPU occupied; waiting without migration", updated_at=now())
        write_json(job["status_path"], state)
        pipeline.register(plan, job, state)
        return 0
    original = pipeline.check_planned_inputs
    pipeline.check_planned_inputs = validate_inputs
    try:
        exit_code = pipeline.worker(plan_path, name)
        state = read_json(job["status_path"])
        if (state.get("status") == "failed" and not state.get("child_pid") and
                state.get("error") == "RuntimeError: GPU became occupied by another compute process before launch"):
            # The original worker repeats its occupancy check under the GPU lock.
            # Preserve that protection while treating this launch race as a queue,
            # not as a failed experiment. No numerical subprocess has been started.
            for key in ("error", "exit_code", "finished_at", "pid", "pid_start_ticks"):
                state.pop(key, None)
            state.update(status="queued", reason="Target GPU became occupied; waiting without migration", updated_at=now())
            write_json(job["status_path"], state)
            pipeline.register(plan, job, state)
            return 0
        return exit_code
    finally:
        pipeline.check_planned_inputs = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.plan, args.job))
    version = code_version(require_pushed=True)
    if args.materialize:
        if not args.run_spec:
            parser.error("--materialize requires the tracked --run-spec")
        plan = build_plan(args.run_spec, version)
        path = pipeline.initialize(plan)
        print(json.dumps(dict(plan=str(path), identity=plan["identity"], code=version)), flush=True)
        return
    if not args.plan:
        parser.error("--plan is required")
    plan = read_json(args.plan)
    validate_inputs(plan)
    if version["commit"] != plan["code"]["commit"]:
        raise ValueError("Coordinator must execute the planned fixed training commit")
    with pipeline.lock(Path(plan["root"]) / "pipeline" / f"coordinator_{plan['prefix']}.lock"):
        pipeline.initialize(plan)
        while True:
            done, launched = pipeline.tick(plan, args.plan, tmux=Tmux())
            if launched:
                print(json.dumps(dict(time=now(), launched=launched)), flush=True)
            if done:
                return
            time.sleep(10)


if __name__ == "__main__":
    main()
