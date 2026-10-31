"""Inspect formal startup evidence on CPU; never control training processes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import subprocess
import time
from datetime import datetime, timezone


TERMINAL = {"completed", "failed", "interrupted", "cancelled", "blocked"}
FAILED = TERMINAL - {"completed"}
EXPERIMENTS = {f"R{i:02d}" for i in range(6)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def records(path):
    """Read complete JSON records only; an unfinished final line is pending."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            if not line.lstrip().startswith("{"):
                continue
            yield json.loads(line)


def first_steps(path):
    values = []
    for value in records(path):
        if value.get("event") != "step":
            continue
        if not integer(value.get("step"), 1) or not isinstance(value.get("loss"), (float, int)) or not finite(value):
            raise ValueError("Nonfinite or invalid startup step")
        if value["loss"] < 0 or (values and value["step"] <= values[-1]["step"]):
            raise ValueError("Startup steps must advance with nonnegative loss")
        if value["step"] != len(values) + 1:
            raise ValueError("Fresh formal run requires actual startup steps 1, 2, 3")
        values.append(value)
        if len(values) == 3:
            return values
    return values


def first_validation(path):
    for value in records(path):
        overall = value.get("validation", {}).get("overall", {})
        if not integer(value.get("epoch")) or not integer(value.get("global_step"), 1) or not finite(value):
            raise ValueError("Nonfinite or invalid first validation")
        if not integer(overall.get("count"), 1) or not all(isinstance(overall.get(k), (int, float)) for k in ("epe", "mae_vx", "mae_vy")):
            raise ValueError("First validation lacks velocity metrics")
        return value
    return None


def validate_run(run, job, plan):
    if run.get("smoke") or run.get("contract", {}).get("execution", {}).get("smoke"):
        raise ValueError("Smoke run excluded from formal startup proof")
    if run.get("run_id") != job["job_id"] or run.get("experiment") != job.get("experiment", job["name"]):
        raise ValueError("Run identity does not match declared job")
    code = run.get("code", {})
    if code.get("commit") != plan["code"]["commit"] or code.get("dirty") is not False or code.get("verified_remote_commit") != code.get("commit"):
        raise ValueError("Formal run does not match the clean pushed plan commit")
    contract = run.get("contract", {})
    if digest(contract) != run.get("contract_hash") or contract.get("config") != run.get("config"):
        raise ValueError("Run contract/config hash mismatch")
    if contract.get("data", {}).get("dataset_sha256") != plan["manifest_sha256"]:
        raise ValueError("Run manifest differs from declared plan")
    if not contract.get("train_selection") or not contract.get("dev_selection") or contract.get("execution", {}).get("max_batches") != 0:
        raise ValueError("Formal run lacks full train/dev selection provenance")
    code_root = Path(plan["code_root"]).resolve()
    config_path = Path(job.get("config_path", code_root / "configs" / f"{job['name']}.json")).resolve()
    if not config_path.is_relative_to(code_root):
        raise ValueError("Configuration escaped immutable plan code root")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != plan["config_hashes"][job["name"]] or read(config_path) != run["config"]:
        raise ValueError("Run config differs from immutable plan configuration")


def check_checkpoint(path, run, validation):
    """Load one immutable inode, so a concurrent atomic save cannot mix versions."""
    import numpy as np
    import torch
    from purevision_v5.model import build_model

    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        state = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Checkpoint inode changed during inspection; retry")
        stream.seek(0)
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "epoch", "next_batch", "global_step", "best_epe", "best_epoch", "bad_epochs", "running_loss", "running_count"}
    if state.get("schema") != "v5_training_state_v1" or not required <= state.keys():
        raise ValueError("Checkpoint is not a complete resumable training state")
    for name in ("run_id", "code", "contract_hash", "contract", "config"):
        if state.get(name) != run.get(name):
            raise ValueError(f"Checkpoint {name} differs from formal run")
    for name in ("epoch", "next_batch", "global_step", "bad_epochs", "running_count"):
        if not integer(state[name]):
            raise ValueError(f"Invalid checkpoint {name}")
    if state["epoch"] <= validation["epoch"] or state["global_step"] < validation["global_step"]:
        raise ValueError("Checkpoint predates first validation")
    if not integer(state["best_epoch"]) or not finite([state["best_epe"], state["running_loss"]]):
        raise ValueError("Invalid checkpoint best metric or accumulated loss")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("Checkpoint contains nonfinite/non-CPU tensor")
        elif isinstance(value, dict):
            for item in value.values():
                tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                tensors(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Checkpoint contains nonfinite optimizer/scaler value")
    for name in ("model", "optimizer", "scheduler", "scaler"):
        tensors(state[name])
    model = build_model(run["config"]).cpu()
    model.load_state_dict(state["model"], strict=True)
    config = run["config"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer.load_state_dict(state["optimizer"])
    if set(optimizer.state) != set(model.parameters()):
        raise ValueError("Checkpoint optimizer lacks trained state for model parameters")
    for parameter, value in optimizer.state.items():
        if not {"step", "exp_avg", "exp_avg_sq"} <= value.keys() or any(value[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Optimizer moments are incomplete or incompatible")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if not {"T_max", "last_epoch", "base_lrs"} <= state["scheduler"].keys():
        raise ValueError("Checkpoint scheduler is incomplete")
    if state["scheduler"]["T_max"] != config["epochs"] or state["scheduler"]["last_epoch"] != state["epoch"]:
        raise ValueError("Checkpoint scheduler disagrees with completed epochs")
    scheduler.load_state_dict(state["scheduler"])
    scaler = torch.amp.GradScaler("cpu", enabled=bool(state["scaler"]))
    scaler.load_state_dict(state["scaler"])
    rng = state["rng"]
    if not {"python", "numpy", "torch", "cuda"} <= rng.keys() or not isinstance(rng["cuda"], list):
        raise ValueError("Checkpoint RNG state is incomplete")
    random.Random().setstate(rng["python"])
    value = rng["numpy"]
    np.random.RandomState().set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.Generator(device="cpu").set_state(rng["torch"])
    if any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1 for item in rng["cuda"]):
        raise ValueError("Invalid saved CUDA RNG bytes")
    return dict(path=str(path), sha256=checksum, size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                inode=before.st_ino, epoch=state["epoch"], next_batch=state["next_batch"],
                global_step=state["global_step"], checked_at=now(), load_device="cpu", resumable_state_verified=True)


def effective_wrapper(raw, job, plan):
    """Accept terminal coordinator evidence only for this exact unchanged worker."""
    base = Path(plan["root"]) / "pipeline" / plan["prefix"]
    pipeline = read(base / "status.json") or {}
    if pipeline.get("plan_identity") != plan["identity"] or raw["status"] in TERMINAL:
        return raw
    effective = pipeline.get("jobs", {}).get(job["name"], {})
    if effective.get("status") != "interrupted" or not effective.get("observation_path"):
        return raw
    path = Path(effective["observation_path"]).resolve()
    if path != (base / "observations" / f"{job['name']}.json").resolve():
        return raw
    observation = read(path) or {}
    confirmed = (observation.get("effective_status") == "interrupted"
                 and observation.get("dead_observations", 0) >= 2
                 and observation.get("last_worker_state") == raw
                 and observation.get("wrapper", {}).get("verdict") == "dead"
                 and observation.get("child", {}).get("verdict") == "dead"
                 and observation.get("tmux_exists") is False
                 and observation.get("worker_lock_free") is True)
    return effective if confirmed else raw


def inspect(job, plan, previous=None):
    proof = dict(previous or {})
    proof.update(schema="v5_startup_proof_v1", run_id=job["job_id"], plan_identity=plan["identity"],
                 expected_commit=plan["code"]["commit"], updated_at=now(), errors=[])
    try:
        if previous and (previous.get("plan_identity") != plan["identity"] or previous.get("expected_commit") != plan["code"]["commit"]):
            raise ValueError("Existing startup proof belongs to another plan/commit")
        wrapper = read(job["status_path"]) or {"status": "queued"}
        proof["raw_job_status"] = wrapper["status"]
        wrapper = effective_wrapper(wrapper, job, plan)
        proof["effective_observation_path"] = wrapper.get("observation_path")
        run_dir = Path(job["run_dir"])
        run, state = read(run_dir / "run.json"), read(run_dir / "status.json") or {}
        proof["job_status"], proof["training_status"] = wrapper["status"], state.get("status")
        if wrapper["status"] in FAILED or state.get("status") in FAILED or (run and run.get("status") in FAILED):
            proof["status"] = "excluded_terminal_failure"
            return proof
        if run is None:
            proof["status"] = "pending_run"
            return proof
        validate_run(run, job, plan)
        if previous and previous.get("contract_hash") not in (None, run["contract_hash"]):
            raise ValueError("Formal run contract changed after startup observation")
        proof["contract_hash"] = run["contract_hash"]
        proof["first_steps"] = proof.get("first_steps") or first_steps(job["log_path"])
        if len(proof["first_steps"]) < 3:
            proof["first_steps"] = first_steps(job["log_path"])
        proof["first_validation"] = proof.get("first_validation") or first_validation(run_dir / "metrics.jsonl")
        if len(proof["first_steps"]) != 3 or proof["first_validation"] is None:
            proof["status"] = "pending_steps_or_validation"
            return proof
        checkpoints = proof.setdefault("checkpoints", {})
        for name in ("last.pt", "best.pt"):
            if name not in checkpoints:
                if not (run_dir / name).exists():
                    proof["status"] = "pending_checkpoint"
                    return proof
                checkpoints[name] = check_checkpoint(run_dir / name, run, proof["first_validation"])
        proof["status"] = "verified"
        proof.setdefault("verified_at", now())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        proof["status"] = "rejected"
        proof["errors"] = [f"{type(error).__name__}: {error}"]
    return proof


def patrol_result(path, run_id=None):
    for line in reversed(Path(path).read_text().splitlines() if Path(path).exists() else []):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if (value.get("trigger") == "startup" and value.get("success") is True
                and value.get("actual_shell_read_observed") is True and value.get("read_evidence", {}).get("verified") is True
                and (run_id is None or run_id in value.get("run_ids", []))):
            if Path(value.get("answer", "")).is_file():
                return value
    return None


def trigger_patrol(proof, proof_path, root, script, runner=subprocess.run):
    if proof.get("patrol", {}).get("success") or proof["status"] != "verified":
        return proof
    attempts = proof.setdefault("patrol_attempts", [])
    for attempt in attempts:
        if result := patrol_result(attempt["stdout_path"], proof["run_id"]):
            proof["patrol"] = result
            return proof
    output = proof_path.parent / f"patrol_attempt_{len(attempts) + 1}.log"
    attempts.append(dict(started_at=now(), stdout_path=str(output), script=str(script)))
    write(proof_path, proof)
    try:
        with output.open("w") as stream:
            result = runner(["/usr/bin/bash", str(script), str(root), "startup", str(proof_path)], stdout=stream,
                            stderr=subprocess.STDOUT, timeout=360, check=False)
        attempts[-1].update(exit_code=result.returncode, finished_at=now())
    except (OSError, subprocess.TimeoutExpired) as error:
        attempts[-1].update(error=str(error), finished_at=now())
    proof["patrol"] = patrol_result(output, proof["run_id"]) or dict(success=False, status="retry_pending_no_verified_report")
    return proof


def tick(plan, root, script, *, runner=subprocess.run):
    jobs = [job for job in plan["jobs"] if job["kind"] == "training"]
    summaries = []
    for job in jobs:
        path = root / "monitoring/startup" / job["job_id"] / "startup_proof.json"
        proof = inspect(job, plan, read(path))
        proof = trigger_patrol(proof, path, root, script, runner)
        write(path, proof)
        summaries.append(dict(run_id=job["job_id"], job_status=proof.get("job_status"),
                              startup_status=proof["status"], patrol_success=proof.get("patrol", {}).get("success", False),
                              proof_path=str(path), errors=proof["errors"]))
    terminal = all(item["job_status"] in TERMINAL for item in summaries)
    result = dict(schema="v5_startup_watcher_v1", status="completed" if terminal else "running", pid=os.getpid(),
                  updated_at=now(), plan_identity=plan["identity"], jobs=summaries,
                  all_startups_verified=all(item["startup_status"] == "verified" and item["patrol_success"] for item in summaries))
    write(root / "monitoring/startup" / f"watcher_{plan['prefix']}.json", result)
    return result


def validate_training_jobs(plan, interval):
    """Named pilot jobs may share a model experiment; each run remains unique."""
    jobs = [job for job in plan["jobs"] if job["kind"] == "training"]
    if (not jobs or interval <= 0
            or len({job["name"] for job in jobs}) != len(jobs)
            or len({job["job_id"] for job in jobs}) != len(jobs)
            or any(job.get("experiment", job["name"]) not in EXPERIMENTS for job in jobs)):
        raise ValueError("Watcher requires nonempty unique training jobs, supported model experiments and positive interval")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--patrol-script", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    root, plan = args.root.resolve(), read(args.plan)
    if plan.get("identity") != digest({key: value for key, value in plan.items() if key != "identity"}) or Path(plan["root"]).resolve() != root:
        raise ValueError("Invalid plan identity or artifact root")
    validate_training_jobs(plan, args.interval)
    script = args.patrol_script
    if script is None:
        script = Path(__file__).resolve().parent / "scripts/astra_patrol.sh"
        if not script.is_file():
            schedule = read(root / "monitoring/schedule.json") or {}
            script = Path(schedule["code_root"]) / "scripts/astra_patrol.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    base = root / "monitoring/startup"
    base.mkdir(parents=True, exist_ok=True)
    with (base / f"watcher_{plan['prefix']}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = tick(plan, root, script)
            print(json.dumps(result), flush=True)
            if args.once or result["status"] == "completed":
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
