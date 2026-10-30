"""Fixed S2 job DAG: immutable code, tmux workers, explicit data dependencies.

This coordinator never resumes failed training or changes its algorithms. A
worker owns its job/GPU locks throughout training and independent evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from monitor import process_info, write_json
from purevision_v5.common import ROOT, code_version, now, object_hash, sha256


TERMINAL = {"completed", "failed", "interrupted", "cancelled", "blocked"}
ACTIVE = {"launching", "running"}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


@contextmanager
def lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def resolve_input(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def build_plan(root, manifest, features, prefix, version, *, code_root=ROOT, python=sys.executable):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", prefix) is None:
        raise ValueError("run-prefix must be 1-64 letters, digits, underscores or hyphens")
    root, code_root = Path(root).resolve(), Path(code_root).resolve()
    manifest, features = resolve_input(root, manifest), resolve_input(root, features)
    base = root / "pipeline" / prefix
    flow, fb = root / "flows/forward", root / "flows/bidirectional"
    fpack, bpack = root / "flows/forward_packed", root / "flows/bidirectional_packed"
    jobs = []

    def command(script, *arguments):
        return [str(python), "-B", str(code_root / script), *map(str, arguments)]

    def add(name, kind, device, requires, commands, *, after=(), run_dir=None, progress=None, output=None, checks=()):
        jobs.append(dict(name=name, job_id=f"{prefix}_{name}", kind=kind, device=device,
                         requires=list(requires), after=list(after), commands=commands,
                         session=f"v5_{prefix}_{name}", log_path=str(root / "logs" / f"{prefix}_{name}.log"),
                         status_path=str(base / "jobs" / f"{name}.json"),
                         run_dir=str(run_dir) if run_dir else None,
                         progress_path=str(progress) if progress else None,
                         fresh_output=str(output) if output else None, checks=list(checks)))

    for index, device in ((0,"cuda:0"),(1,"cuda:2")):
        name = f"forward{index}"
        report = base / "reports" / f"{name}.json"
        args = ["--manifest",manifest,"--output",flow,"--direction","forward","--device",device,
                "--batch-size",16,"--shard-count",2,"--shard-index",index,"--report",report]
        add(name,"flow",device,[],[command("prepare_flow.py",*args)],progress=report,
            checks=[{"path":str(report),"field":"status","equals":"complete"}])
    report = base / "reports/fb0.json"
    add("fb0","flow","cuda:0",["forward0"],[command("prepare_flow.py","--manifest",manifest,
        "--output",fb,"--direction","bidirectional","--reuse-forward-root",flow,"--device","cuda:0",
        "--batch-size",16,"--shard-count",2,"--shard-index",0,"--report",report)],progress=report,
        checks=[{"path":str(report),"field":"status","equals":"complete"}])
    add("forward_pack","pack",None,["forward0","forward1"],[command("pack_flow.py","--manifest",manifest,
        "--source-root",flow,"--output",fpack)],output=fpack,
        checks=[{"path":str(fpack / "packed_index.json"),"field":"schema","equals":"v5_packed_flow_v1"}])

    def training(name, device, requires, after=()):
        output, evaluation = root / "runs" / f"{prefix}_{name}", root / "evaluations" / f"{prefix}_{name}"
        inputs = ["--manifest",manifest,"--device",device,"--workers",4]
        if name != "R02":
            inputs += ["--features",features]
        if name != "R00":
            inputs += ["--flow",bpack if name in {"R04","R05"} else fpack]
        add(name,"training",device,requires,[command("train.py","--config",code_root / f"configs/{name}.json",
            *inputs,"--output",output),command("evaluate.py","--checkpoint",output / "best.pt",*inputs,"--output",evaluation)],
            after=after,run_dir=output,progress=output / "status.json",output=output,
            checks=[{"path":str(output / "status.json"),"field":"status","equals":"completed"},
                    {"path":str(evaluation / "evaluation.json"),"field":"status","equals":"completed"},
                    {"path":str(output / "best.pt")},{"path":str(evaluation / "metrics.json")}])

    training("R00","cuda:1",["features"])
    training("R01","cuda:1",["features","forward_pack"],["R00"])
    training("R03","cuda:2",["features","forward_pack"])
    training("R02","cuda:2",["forward_pack"],["R03"])
    report = base / "reports/fb1.json"
    add("fb1","flow","cuda:1",["forward1"],[command("prepare_flow.py","--manifest",manifest,
        "--output",fb,"--direction","bidirectional","--reuse-forward-root",flow,"--device","cuda:1",
        "--batch-size",16,"--shard-count",2,"--shard-index",1,"--report",report)],after=["R01"],progress=report,
        checks=[{"path":str(report),"field":"status","equals":"complete"}])
    add("fb_pack","pack",None,["fb0","fb1","forward_pack"],[command("pack_flow.py","--manifest",manifest,
        "--source-root",fb,"--output",bpack,"--require-fb","--reference-pack",fpack)],output=bpack,
        checks=[{"path":str(bpack / "packed_index.json"),"field":"schema","equals":"v5_packed_flow_v1"}])
    training("R04","cuda:1",["features","fb_pack"])
    training("R05","cuda:0",["features","fb_pack"])
    config_hashes = {name:sha256(code_root / f"configs/{name}.json") for name in ("R00","R01","R02","R03","R04","R05")}
    plan = dict(schema="v5_pipeline_v1", root=str(root), code_root=str(code_root), python=str(python),
                prefix=prefix, code=version, manifest=str(manifest), manifest_sha256=sha256(manifest),
                features=str(features), features_status=str(features / "status.json"),
                config_hashes=config_hashes, jobs=jobs)
    return {**plan,"identity":object_hash(plan)}


def validate_plan(plan):
    if plan.get("schema") != "v5_pipeline_v1" or plan.get("identity") != object_hash({k:v for k,v in plan.items() if k != "identity"}):
        raise ValueError("Pipeline plan identity mismatch")


def registry_rows(root):
    value = read_json(Path(root) / "monitoring/jobs.json", {"jobs":[]})
    rows = value if isinstance(value,list) else value.get("jobs",[])
    if isinstance(rows,dict):
        rows = [{"job_id":key,**row} for key,row in rows.items()]
    return value, rows


def register(plan, job, state):
    root = Path(plan["root"])
    with lock(root / "monitoring/jobs.lock",blocking=True):
        value, rows = registry_rows(root)
        by_id = {row["job_id"]:row for row in rows}
        old = by_id.get(job["job_id"],{})
        if old and (old.get("pipeline_id") != plan["prefix"] or old.get("plan_identity") != plan["identity"]):
            raise ValueError(f"Registry job already belongs to another launch: {job['job_id']}")
        by_id[job["job_id"]] = {**old,**{key:job[key] for key in
            ("job_id","kind","device","run_dir","log_path","status_path","session")},
            "pipeline_id":plan["prefix"],"plan_identity":plan["identity"],"commit":plan["code"]["commit"],
            **{key:state.get(key) for key in ("status","pid","pid_start_ticks","child_pid","child_pid_start_ticks","exit_code")}}
        # Preserve all external rows and unrelated registry top-level fields.
        result = {**value,"jobs":list(by_id.values())} if isinstance(value,dict) else list(by_id.values())
        write_json(root / "monitoring/jobs.json",result)


def initialize(plan):
    validate_plan(plan)
    base = Path(plan["root"]) / "pipeline" / plan["prefix"]
    path = base / "plan.json"
    existing = read_json(path)
    if existing is not None and existing != plan:
        raise ValueError("Pipeline prefix already has a different immutable plan; use a new prefix")
    if existing is None:
        write_json(path,plan)
    for job in plan["jobs"]:
        state = read_json(job["status_path"])
        if state is None:
            state = dict(status="queued" if existing is None else "interrupted",updated_at=now(),launch_count=0,
                         reason=None if existing is None else "Missing state for an already registered pipeline")
            write_json(job["status_path"],state)
        register(plan,job,state)
    return path


def external_features(plan):
    status = read_json(plan["features_status"],{"status":"not_started"})
    if status.get("status") == "completed":
        index = read_json(Path(plan["features"]) / "index.json",{})
        if index.get("contract",{}).get("dataset_sha256") != plan["manifest_sha256"]:
            return dict(status="failed",reason="External features missing matching manifest contract")
    if status.get("status") == "running":
        _,rows = registry_rows(plan["root"])
        for job in rows:
            if job.get("status_path") == plan["features_status"] or job.get("run_dir") == plan["features"]:
                if job.get("pid") and not process_info(job["pid"],job.get("pid_start_ticks"))["alive"]:
                    return dict(status="interrupted",reason="External feature process disappeared")
    return status


def dependency_state(job, states):
    failed = [name for name in job["requires"] if states[name]["status"] in TERMINAL - {"completed"}]
    if failed:
        return "blocked",failed
    if any(states[name]["status"] != "completed" for name in job["requires"]):
        return "waiting",[]
    if any(states[name]["status"] not in TERMINAL for name in job["after"]):
        return "waiting",[]
    return "ready",[]


def busy_gpus():
    gpu_rows = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True,timeout=10)
    devices = {row.split(",")[1].strip():f"cuda:{row.split(',')[0].strip()}" for row in gpu_rows.splitlines() if row.strip()}
    apps = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True,timeout=10)
    return {devices[row.split(",")[0].strip()] for row in apps.splitlines() if row.split(",")[0].strip() in devices}


class Tmux:
    def exists(self, session):
        return subprocess.run(["tmux","has-session","-t",session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0

    def launch(self, plan_path, plan, job):
        if self.exists(job["session"]):
            raise RuntimeError("Named tmux session already exists")
        Path(job["log_path"]).parent.mkdir(parents=True,exist_ok=True)
        command = [plan["python"],"-B",str(Path(plan["code_root"])/"pipeline.py"),"--worker","--plan",str(plan_path),"--job",job["name"]]
        shell = "exec " + shlex.join(command) + " >> " + shlex.quote(job["log_path"]) + " 2>&1"
        subprocess.run(["tmux","new-session","-d","-s",job["session"],"-c",plan["code_root"],shell],check=True)
        result = subprocess.run(["tmux","display-message","-p","-t",job["session"],"#{pane_pid}"],text=True,capture_output=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
        return None  # A very short worker may already have written its terminal state.


def tick(plan, plan_path, *, tmux=None, gpu_probe=busy_gpus, alive=process_info):
    tmux = tmux or Tmux()
    states = {job["name"]:read_json(job["status_path"],{"status":"interrupted"}) for job in plan["jobs"]}
    states["features"] = external_features(plan)
    for job in plan["jobs"]:
        state = states[job["name"]]
        if state["status"] == "running" and state.get("pid") and not alive(state["pid"],state.get("pid_start_ticks"))["alive"]:
            state.update(status="interrupted",reason="Worker disappeared; automatic restart is disabled",updated_at=now())
            write_json(job["status_path"],state)
        elif (state["status"] == "launching" and time.time() - state.get("launched_at_epoch",time.time()) > 30
              and not tmux.exists(job["session"])):
            state.update(status="interrupted",reason="Worker never established a live tmux session",updated_at=now())
            write_json(job["status_path"],state)
    probe_error = None
    try:
        occupied = set(gpu_probe())
    except (OSError,subprocess.SubprocessError) as error:
        occupied = {"cuda:0","cuda:1","cuda:2"}
        probe_error = str(error)
    occupied.update(job["device"] for job in plan["jobs"] if job["device"] and states[job["name"]]["status"] in ACTIVE)
    launched = []
    for job in plan["jobs"]:
        state = states[job["name"]]
        if state["status"] != "queued":
            register(plan,job,state)
            continue
        disposition, failed = dependency_state(job,states)
        if disposition == "blocked":
            state.update(status="blocked",blocked_by=failed,updated_at=now())
            write_json(job["status_path"],state)
        elif disposition == "ready" and job["device"] not in occupied:
            output = Path(job["fresh_output"]) if job["fresh_output"] else None
            if output is not None and output.exists():
                state.update(status="failed",reason="Output already exists; select a new run/pack path",updated_at=now())
                write_json(job["status_path"],state)
            else:
                state.update(status="launching",launch_count=state.get("launch_count",0)+1,
                             launched_at_epoch=time.time(),updated_at=now())
                write_json(job["status_path"],state)
                try:
                    pid = tmux.launch(plan_path,plan,job)
                    # Do not overwrite a status written by an already-started worker.
                    current = read_json(job["status_path"],state)
                    if current["status"] == "launching" and pid is not None:
                        info = alive(pid)
                        current.update(pid=pid,pid_start_ticks=info.get("start_ticks"))
                        write_json(job["status_path"],current)
                    state = states[job["name"]] = current
                    launched.append(job["name"])
                    if job["device"]:
                        occupied.add(job["device"])
                except BaseException as error:
                    state.update(status="failed",reason=f"tmux launch failed: {type(error).__name__}: {error}",updated_at=now())
                    write_json(job["status_path"],state)
        register(plan,job,state)
    done = all(states[job["name"]]["status"] in TERMINAL for job in plan["jobs"])
    status = {"status":"completed" if done and all(states[job["name"]]["status"] == "completed" for job in plan["jobs"])
              else "finished_with_failures" if done else "running", "updated_at":now(),"plan_identity":plan["identity"],
              "code":plan["code"],"jobs":states,"gpu_probe_error":probe_error}
    write_json(Path(plan["root"])/"pipeline"/plan["prefix"]/"status.json",status)
    return done, launched


def check_outputs(job):
    for check in job["checks"]:
        path = Path(check["path"])
        if not path.is_file():
            raise RuntimeError(f"Expected output is missing: {path}")
        if "field" in check and read_json(path,{}).get(check["field"]) != check["equals"]:
            raise RuntimeError(f"Output did not finish successfully: {path}")


def worker(plan_path, name):
    plan = read_json(plan_path)
    validate_plan(plan)
    if Path(plan["code_root"]) != Path(__file__).resolve().parent:
        raise ValueError("Worker is not executing the immutable planned checkout")
    job = next(job for job in plan["jobs"] if job["name"] == name)
    root = Path(plan["root"])
    child, stopped, stop_time = None, False, None
    state = read_json(job["status_path"],{})

    def stop(signum, frame):
        nonlocal stopped, stop_time
        stopped, stop_time = True, stop_time or time.monotonic()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)

    with ExitStack() as stack:
        stack.enter_context(lock(root/"pipeline"/"locks"/f"job_{job['job_id']}.lock"))
        if state.get("status") in TERMINAL:
            return 0
        try:
            current = code_version()
            if (current["dirty"] or current["commit"] != plan["code"]["commit"]
                    or plan["code"].get("verified_remote_commit") != current["commit"]):
                raise ValueError("Worker code is not the clean verified-pushed planned version")
            if job["device"]:
                stack.enter_context(lock(root/"pipeline"/"locks"/f"gpu_{job['device'].split(':')[1]}.lock"))
                if job["device"] in busy_gpus():
                    raise RuntimeError("GPU became occupied by another compute process before launch")
            info = process_info(os.getpid())
            state.update(status="running",pid=os.getpid(),pid_start_ticks=info["start_ticks"],
                         started_at=now(),updated_at=now(),commit=current["commit"])
            write_json(job["status_path"],state)
            register(plan,job,state)
            previous = {sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
            try:
                for index, command in enumerate(job["commands"]):
                    if stopped:
                        raise InterruptedError("Worker received a stop signal")
                    state.update(phase="evaluation" if index else job["kind"],command_index=index,command=command,updated_at=now())
                    print(json.dumps({"time":now(),"job":name,"command":command}),flush=True)
                    child = subprocess.Popen(command,cwd=plan["code_root"],start_new_session=True)
                    child_info = process_info(child.pid)
                    state.update(child_pid=child.pid,child_pid_start_ticks=child_info.get("start_ticks"))
                    while child.poll() is None:
                        if stopped and time.monotonic() - stop_time > 60:
                            os.killpg(child.pid,signal.SIGKILL)
                        progress = read_json(job["progress_path"],{}) if job["progress_path"] else {}
                        state.update({key:progress[key] for key in ("epoch","global_step","loss","validation","completed","total","elapsed_seconds") if key in progress})
                        if "pairs_processed" in progress:
                            state["completed"] = progress["pairs_processed"]
                        state["updated_at"] = now()
                        write_json(job["status_path"],state)
                        time.sleep(5)
                    if stopped:
                        raise InterruptedError("Worker stopped; no automatic training resume")
                    if child.returncode != 0:
                        raise RuntimeError(f"Command {index} exited with {child.returncode}")
                    # A signal-aware train command may return zero after saving an
                    # interrupted checkpoint; never evaluate that as completed.
                    if job["kind"] == "training" and index == 0:
                        if read_json(Path(job["run_dir"])/"status.json",{}).get("status") != "completed":
                            raise InterruptedError("Training stopped before completion")
                check_outputs(job)
                state.update(status="completed",exit_code=0,finished_at=now(),updated_at=now())
            finally:
                for sig,handler in previous.items():
                    signal.signal(sig,handler)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error,(InterruptedError,KeyboardInterrupt)) else "failed",
                         exit_code=130 if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 1,
                         error=f"{type(error).__name__}: {error}",finished_at=now(),updated_at=now())
            print(json.dumps({"job":name,"error":state["error"]}),flush=True)
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            write_json(job["status_path"],state)
            register(plan,job,state)
    return state.get("exit_code",1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--features")
    parser.add_argument("--run-prefix")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--plan",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--job",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.plan or not args.job:
            parser.error("worker requires plan and job")
        raise SystemExit(worker(args.plan,args.job))
    if not all((args.root,args.manifest,args.features,args.run_prefix)):
        parser.error("root, manifest, features and run-prefix are required")
    plan = build_plan(args.root,args.manifest,args.features,args.run_prefix,code_version(require_pushed=not args.dry_run))
    if args.dry_run:
        print(json.dumps(plan,indent=2))
        return
    with lock(args.root.resolve()/"pipeline/coordinator.lock"):
        path = initialize(plan)
        print(json.dumps({"plan":str(path),"identity":plan["identity"],"code":plan["code"]}),flush=True)
        while True:
            done, launched = tick(plan,path)
            if launched:
                print(json.dumps({"time":now(),"launched":launched}),flush=True)
            if done:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
