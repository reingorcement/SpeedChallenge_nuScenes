"""Read-only job inspection; writes only monitoring records and explicitly managed cron entries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_ROOT = Path("/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5")
PYTHON = "/home/zhangyu/.conda/envs/speedest/bin/python"
TERMINAL = {"completed", "failed", "interrupted", "cancelled", "blocked"}
ACTIVE = {"launching", "running", "running_unverified", "queued", "preparing", "not_started", "unknown"}
FLOW_PROGRESS_FIELDS = ("completed", "total", "target_pairs", "expected_pairs", "pairs", "pairs_processed",
                        "counts", "scenes", "progress", "elapsed_s", "elapsed_seconds", "seconds_per_pair",
                        "pairs_per_second", "throughput", "bytes", "bytes_written", "last_batch_s",
                        "cuda_peak_allocated_bytes", "last_pair")
ERROR_PATTERN = re.compile(r"Traceback|(?:Runtime|Value|FileNotFound|Assertion)Error|CUDA.*out of memory|non.?finite|Segmentation fault|(?:^|[\s:=,])(nan|[+-]?inf)(?:$|[\s,}])", re.I)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def tail(path, max_bytes=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - max_bytes))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-100:]
    except OSError:
        return []


def age_seconds(path, now=None):
    try:
        return max(0., (now or datetime.now().timestamp()) - Path(path).stat().st_mtime)
    except OSError:
        return None


def process_info(pid, expected_start=None):
    if not pid:
        return {"pid": None, "alive": None, "verdict": "unknown", "identity_verified": False, "reason": "pid unavailable"}
    try:
        pid = int(pid)
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        start_ticks = int(fields[19])
        alive = fields[0] not in {"Z", "X"}
        identity_matches = expected_start is None or int(expected_start) == start_ticks
        return {"pid": pid, "alive": alive and identity_matches,
            "verdict": "alive" if alive and identity_matches else "dead", "start_ticks": start_ticks,
            "identity_verified": expected_start is not None, "pid_reused": not identity_matches}
    except FileNotFoundError:
        return {"pid": pid, "alive": False, "verdict": "dead", "identity_verified": expected_start is not None,
                "reason": "proc stat absent"}
    except (OSError, ValueError, IndexError) as error:
        return {"pid": pid, "alive": None, "verdict": "unknown", "identity_verified": False,
                "reason": type(error).__name__}


def confirmed_pipeline_terminal(job, root, raw):
    """Reuse the coordinator's existing two-observation proof; never mutate worker state."""
    prefix = job.get("pipeline_id")
    if not prefix or raw.get("status") in TERMINAL:
        return None
    base = (root / "pipeline" / prefix).resolve()
    pipeline = read_json(base / "status.json")
    plan = read_json(base / "plan.json")
    if not plan.get("identity") or pipeline.get("plan_identity") != plan["identity"]:
        return None
    names = [entry["name"] for entry in plan.get("jobs", []) if entry.get("job_id") == job["job_id"]]
    if len(names) != 1:
        return None
    effective = pipeline.get("jobs", {}).get(names[0], {})
    path = effective.get("observation_path")
    if effective.get("status") != "interrupted" or not path:
        return None
    expected = base / "observations" / f"{names[0]}.json"
    if Path(path).resolve() != expected.resolve():
        return None
    observed = read_json(expected)
    if (observed.get("effective_status") == "interrupted" and observed.get("dead_observations", 0) >= 2
            and observed.get("last_worker_state") == raw
            and observed.get("wrapper", {}).get("verdict") == "dead"
            and observed.get("child", {}).get("verdict") == "dead"
            and observed.get("tmux_exists") is False and observed.get("worker_lock_free") is True):
        return str(expected)
    return None


def last_metric(path):
    for line in reversed(tail(path)):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except ValueError:
            continue
    return {}


def inspect_job(job, root):
    root = Path(root).resolve()
    notes = []
    run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = root / run_dir
    launch = read_json(run_dir / "launch.json") if run_dir else {}
    run = read_json(run_dir / "run.json") if run_dir else {}
    status_path = Path(job["status_path"]) if job.get("status_path") else (run_dir / "status.json" if run_dir else None)
    status = read_json(status_path) if status_path else {}
    status_unreadable = bool(status_path and not status and (status_path.exists() or job.get("status") in {"running", "preparing", "launching", "unknown"}))
    training_status = read_json(run_dir / "status.json") if run_dir and job.get("kind", "training") == "training" else {}
    progress_path, flow_progress = None, {}
    if job.get("progress_path"):
        candidate = Path(job["progress_path"])
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or resolved.name in {"network.json", "network.env"}:
            notes.append("progress_path_outside_artifacts_or_private_requires_recheck")
        else:
            progress_path = candidate
            try:
                if candidate.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("oversized progress report")
                flow_progress = json.loads(candidate.read_text())
                if not isinstance(flow_progress, dict):
                    raise ValueError("progress report is not an object")
            except (OSError, ValueError):
                flow_progress = {}
                if status.get("status") not in {"queued", "not_started", "launching"}:
                    notes.append("progress_report_missing_or_unreadable_requires_recheck")
    progress_status = {**status, **flow_progress, **training_status}
    if "pairs_processed" in flow_progress:
        progress_status["completed"] = flow_progress["pairs_processed"]
    if job.get("expected_pairs") is not None:
        progress_status["expected_pairs"] = job["expected_pairs"]
        if flow_progress.get("total") not in (None, job["expected_pairs"]):
            notes.append("progress_total_differs_from_planned_pairs_requires_recheck")
    state = status.get("status", run.get("status", job.get("status", "not_started")))
    state = {"done": "completed", "complete": "completed", "pending": "queued"}.get(state, state)
    if state not in ACTIVE | TERMINAL:
        state = "unknown"
    if job.get("pipeline_id"):
        # The wrapper owns training AND its following independent evaluation.
        process = process_info(status.get("pid") or job.get("pid"), status.get("pid_start_ticks") or job.get("pid_start_ticks"))
    else:
        process = process_info(launch.get("pid") or job.get("pid"), launch.get("pid_start_ticks") or job.get("pid_start_ticks"))
    child_process = process_info(status.get("child_pid") or job.get("child_pid"), status.get("child_pid_start_ticks") or job.get("child_pid_start_ticks"))
    log = Path(job.get("log_path", str(run_dir / "stdout.log") if run_dir else "/nonexistent"))
    checkpoint = Path(job.get("checkpoint_path", str(run_dir / "last.pt") if run_dir else "/nonexistent"))
    metric = last_metric(run_dir / "metrics.jsonl") if run_dir else {}
    if state == "blocked":
        notes.append(str(status.get("reason", job.get("reason", "dependency_blocked"))))
    confirmation = confirmed_pipeline_terminal(job, root, status)
    if status_unreadable:
        state = "unknown"
        notes.append("status_missing_or_unreadable_requires_recheck")
    elif state in ACTIVE and confirmation:
        state = "interrupted"
        notes.append("coordinator_confirmed_two_absent_worker_child_tmux_and_released_lock")
    elif state in {"running", "preparing", "running_unverified", "unknown"} and process["alive"] is not True:
        state = "unknown"
        notes.append("worker_liveness_unknown_keep_monitoring")
        if child_process["alive"] is True:
            notes.append("child_process_alive")
    if state in TERMINAL and any(p.get("alive") is True and p.get("identity_verified") for p in (process, child_process)):
        state = "unknown"
        notes.append("terminal_status_but_verified_process_still_alive")
    if process.get("pid_reused"):
        notes.append("pid_reused")
    if state in ACTIVE and process["alive"] and age_seconds(log) is not None and age_seconds(log) > 900:
        notes.append("log_unchanged_over_15min_inspect_before_action")
    if state in ACTIVE and run_dir and (progress_status.get("global_step") or 0) > 0:
        ckpt_age = age_seconds(checkpoint)
        if ckpt_age is not None and ckpt_age > 900:
            notes.append("checkpoint_older_than_15min")
        if ckpt_age is None and progress_status.get("elapsed_seconds", 0) > 300:
            notes.append("no_recovery_checkpoint_after_5min")
    if state in ACTIVE and process["alive"] and status_path and age_seconds(status_path) is not None and age_seconds(status_path) > 900:
        notes.append("progress_status_unchanged_over_15min")
    if state in ACTIVE and process["alive"] and progress_path and age_seconds(progress_path) is not None and age_seconds(progress_path) > 900:
        notes.append("progress_report_unchanged_over_15min")
    anomalies = [line[-500:] for line in tail(log) if ERROR_PATTERN.search(line)][-8:]
    if anomalies:
        notes.append("anomaly_text_in_log_requires_diagnosis")
    if status.get("error") or run.get("error"):
        notes.append(str(status.get("error", run.get("error")))[:500])
    if flow_progress.get("error"):
        notes.append(str(flow_progress["error"])[:500])
    validation = progress_status.get("validation") or metric.get("validation", {}).get("overall", {})
    return {"job_id": job["job_id"], "kind": job.get("kind", "training"), "status": state,
        "run_dir": str(run_dir) if run_dir else None, "session": launch.get("session", job.get("session")),
        "device": launch.get("device", job.get("device")), "process": process, "child_process": child_process,
        "phase": status.get("phase"), "training_status": training_status.get("status"),
        "status_path": str(status_path) if status_path else None, "terminal_observation": confirmation,
        "commit": run.get("code", {}).get("commit", job.get("commit")),
        "epoch": progress_status.get("epoch", metric.get("epoch")), "global_step": progress_status.get("global_step", metric.get("global_step")),
        "loss": progress_status.get("loss", metric.get("train_loss")), "validation": validation,
        "progress": {k: progress_status[k] for k in FLOW_PROGRESS_FIELDS if k in progress_status},
        "progress_path": str(progress_path) if progress_path else None,
        "progress_report_status": flow_progress.get("status"), "progress_age_s": age_seconds(progress_path) if progress_path else None,
        "status_age_s": age_seconds(status_path) if status_path else None,
        "log_path": str(log) if log.exists() else None, "log_age_s": age_seconds(log),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "checkpoint_age_s": age_seconds(checkpoint), "anomaly_lines": anomalies, "notes": notes}


def collect_jobs(root):
    registry_path = root / "monitoring/jobs.json"
    # A malformed/inaccessible registry must fail the check, never hide active jobs.
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"jobs": []}
    jobs = registry if isinstance(registry, list) else registry.get("jobs", [])
    if isinstance(jobs, dict):
        jobs = [{"job_id": name, **value} for name, value in jobs.items()]
    by_id = {job.get("job_id", job.get("run_id")): dict(job) for job in jobs}
    if None in by_id:
        raise ValueError("Every registry job needs job_id or run_id")
    for name, job in by_id.items():
        job["job_id"] = name
    runs = root / "runs"
    if runs.is_dir():
        for directory in sorted(runs.iterdir()):
            if directory.is_dir() and any((directory / name).exists() for name in ("run.json", "launch.json", "status.json")):
                previous = by_id.get(directory.name, {})
                by_id[directory.name] = {**previous, "job_id": directory.name, "kind": "training", "run_dir": str(directory)}
    return list(by_id.values())


def disk_fields(usage):
    return {"total_gb": usage.total / 1e9, "free_gb": usage.free / 1e9,
        "used_percent": 100 * usage.used / usage.total,
        "low_space": usage.free < 10e9 or usage.free / usage.total < .02}


def storage_volumes(root, *, jobs=(), device_for=None, usage_for=None):
    """Measure the resolved cache filesystem; do not count one device twice."""
    root = Path(root).resolve()
    device_for = device_for or (lambda path: path.stat().st_dev)
    usage_for = usage_for or shutil.disk_usage
    volumes = {}
    paths = [("artifact_root", root), ("flow_cache", root / "flows")]
    for job in jobs:
        configured = job.get("storage_paths", [])
        if not isinstance(configured, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in configured):
            raise ValueError("Registered storage_paths must be a list of absolute paths")
        paths.extend(("job_cache", Path(p)) for p in configured)
    for role, path in paths:
        if not path.exists() and role != "job_cache":
            continue
        # A not-yet-created output still consumes its existing parent's filesystem.
        resolved = path.resolve()
        while not resolved.exists() and resolved != resolved.parent:
            resolved = resolved.parent
        device = int(device_for(resolved))
        if device not in volumes:
            volumes[device] = {"device_id": device, "roles": [], "paths": [], "resolved_paths": [],
                **disk_fields(usage_for(resolved))}
        volume = volumes[device]
        if role not in volume["roles"]:
            volume["roles"].append(role)
        if str(path) not in volume["paths"]:
            volume["paths"].append(str(path))
        if str(resolved) not in volume["resolved_paths"]:
            volume["resolved_paths"].append(str(resolved))
    return list(volumes.values())


def snapshot(root):
    root = Path(root).resolve()
    registered = collect_jobs(root)
    jobs = [inspect_job(job, root) for job in registered]
    gpu = {}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        gpu = {"returncode": result.returncode, "csv": result.stdout.strip(), "error": result.stderr.strip()[:300]}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"error": type(error).__name__}
    disk = shutil.disk_usage(root)
    return {"schema": "v5_patrol_snapshot_v1", "time_utc": timestamp(),
        "time_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "hostname": os.uname().nodename,
        "root": str(root), "jobs": jobs, "active_count": sum(job["status"] in ACTIVE for job in jobs),
        "unknown_count": sum(job["status"] == "unknown" for job in jobs),
        "all_terminal": bool(jobs) and all(job["status"] in TERMINAL for job in jobs),
        "gpu": gpu, "disk": disk_fields(disk), "storage_volumes": storage_volumes(root, jobs=registered)}


def table(value):
    lines = [f"# V5 状态 — {value['time_shanghai']}", "", "| 作业 | 状态 | PID | epoch/step 或缓存进度 | loss | dev EPE | checkpoint 年龄(s) | 信号 |",
        "|---|---|---:|---|---:|---:|---:|---|"]
    def cell(item):
        if item is None:
            return "—"
        return f"{item:.5g}" if isinstance(item, float) else str(item).replace("|", "/")
    for job in value["jobs"]:
        progress = f"{job['epoch']} / {job['global_step']}" if job["kind"] == "training" else json.dumps(job["progress"], ensure_ascii=False)
        display_state = f"{job['status']} ({job['phase']})" if job.get("phase") else job["status"]
        values = [job["job_id"], display_state, job["process"]["pid"], progress,
            job["loss"], job["validation"].get("epe"), job["checkpoint_age_s"], "; ".join(job["notes"]) or "—"]
        lines.append("| " + " | ".join(cell(item) for item in values) + " |")
    lines.extend(["", f"活跃/排队：{value['active_count']}；artifact root 磁盘剩余 {value['disk']['free_gb']:.1f} GB。"])
    for volume in value.get("storage_volumes", []):
        lines.append(f"存储 {','.join(volume['roles'])}：剩余 {volume['free_gb']:.1f} GB，使用 {volume['used_percent']:.1f}%，低空间告警={volume['low_space']}。")
    lines.extend(["",
        "异常文本、日志停更或损失波动只触发人工定位；巡视不会改代码、参数或重启作业。", ""])
    return "\n".join(lines)


def cron_marker(root):
    return "PUREVISION_V5_" + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]


def filter_crontab(text, marker, probe_only=False, publisher_only=False):
    suffixes = ("_PUBLISHER",) if publisher_only else (("_PROBE",) if probe_only else ("_PROBE", "_SERVER", "_ASTRA"))
    # Monitoring cleanup must not remove the final, still-pending result publication.
    return "\n".join(line for line in text.splitlines() if not any(marker + suffix in line for suffix in suffixes)) + "\n"


def modify_cron(root, additions=None, probe_only=False, publisher_only=False):
    root = Path(root).resolve()
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("Unable to read crontab")
    existing = result.stdout
    backup = root / "monitoring/crontab_backups" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".txt")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(existing)
    updated = filter_crontab(existing, cron_marker(root), probe_only, publisher_only)
    if additions:
        updated += "\n".join(additions) + "\n"
    subprocess.run(["crontab", "-"], input=updated, text=True, check=True)
    return backup


def install_cron(root, code_root, probe=False):
    import shlex
    root, code_root = Path(root).resolve(), Path(code_root).resolve()
    for name in ("monitor.py", "scripts/server_check.sh", "scripts/astra_patrol.sh"):
        if not (code_root / name).is_file():
            raise FileNotFoundError(code_root / name)
    commit = "probe_external_snapshot"
    if not probe:
        repository = code_root.parents[2]
        base = ["git", "-c", f"safe.directory={repository}", "-C", str(repository)]
        if subprocess.check_output(base + ["rev-parse", "--abbrev-ref", "HEAD"], text=True).strip() != "HEAD":
            raise ValueError("Formal monitoring requires a detached fixed worktree")
        if subprocess.check_output(base + ["status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise ValueError("Formal monitoring worktree contains tracked modifications")
        commit = subprocess.check_output(base + ["rev-parse", "HEAD"], text=True).strip()
    quote = shlex.quote
    marker = cron_marker(root)
    env = "/usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin TZ=Asia/Shanghai"
    def command(script, trigger):
        return f"{env} /usr/bin/bash {quote(str(code_root / 'scripts' / script))} {quote(str(root))} {trigger} >> {quote(str(root / 'monitoring/cron.log'))} 2>&1"
    lines = [f"* * * * * {command('astra_patrol.sh', 'cron_probe')} # {marker}_PROBE"] if probe else [
        f"*/5 * * * * {command('server_check.sh', 'scheduled')} # {marker}_SERVER",
        f"*/30 * * * * {command('astra_patrol.sh', 'scheduled')} # {marker}_ASTRA"]
    backup = modify_cron(root, lines, probe_only=probe)
    record = {"installed_at": timestamp(), "code_root": str(code_root), "commit": commit,
        "probe": probe, "crontab_backup": str(backup), "lines": lines,
        "status": "installed_pending_observed_trigger", "model": "gpt-6-astra"}
    write_json(root / "monitoring" / ("probe_schedule.json" if probe else "schedule.json"), record)
    return record


def install_publisher(root, code_root, command_file):
    import shlex
    root, code_root, command_file = Path(root).resolve(), Path(code_root).resolve(), Path(command_file).resolve()
    repository = code_root.parents[2]
    base = ["git", "-c", f"safe.directory={repository}", "-C", str(repository)]
    if subprocess.check_output(base + ["rev-parse", "--abbrev-ref", "HEAD"], text=True).strip() != "HEAD":
        raise ValueError("Publisher requires a detached fixed code worktree")
    if subprocess.check_output(base + ["status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Publisher code worktree contains tracked modifications")
    if not command_file.is_relative_to(root / "monitoring") or not command_file.is_file():
        raise ValueError("Publisher command file must exist under task monitoring directory")
    command = json.loads(command_file.read_text()).get("command", [])
    if command[:3] != [PYTHON, "-B", str(code_root / "publish_results.py")]:
        raise ValueError("Publisher command must target the fixed script")
    if command_file.stat().st_uid != os.getuid() or command_file.stat().st_mode & 0o077:
        raise ValueError("Publisher command file must be task-owned and private")
    env = "/usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin TZ=Asia/Shanghai"
    command_line = f"{env} /usr/bin/bash {shlex.quote(str(code_root / 'scripts/publish_results.sh'))} {shlex.quote(str(root))} {shlex.quote(str(command_file))} scheduled >> {shlex.quote(str(root / 'monitoring/publisher_cron.log'))} 2>&1"
    lines = [f"17 * * * * {command_line} # {cron_marker(root)}_PUBLISHER"]
    backup = modify_cron(root, lines, publisher_only=True)
    record = dict(installed_at=timestamp(), code_root=str(code_root),
                  commit=subprocess.check_output(base + ["rev-parse", "HEAD"], text=True).strip(),
                  lines=lines, crontab_backup=str(backup), command_file=str(command_file),
                  status="installed_pending_observed_trigger")
    write_json(root / "monitoring/publisher_schedule.json", record)
    return record


def normalized_command(command):
    import shlex
    try:
        args = shlex.split(command)
        if args and Path(args[0]).name in {"bash", "sh"} and len(args) >= 3 and args[1] in {"-lc", "-c"}:
            args = shlex.split(args[2])
        return args
    except ValueError:
        return []


def patrol_evidence(prefix, log_text):
    """Require successful CLI command events for the exact deterministic read helper."""
    request_path = prefix.with_suffix(".request.json")
    request = read_json(request_path)
    if not request or not request.get("command"):
        return {"verified": False, "reason": "missing_read_request", "reads": []}
    matches = []
    for line in log_text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get("item", {})
        if (event.get("type") != "item.completed" or item.get("type") != "command_execution"
                or item.get("exit_code") != 0 or item.get("status") != "completed"
                or normalized_command(item.get("command", "")) != request["command"]):
            continue
        for output in item.get("aggregated_output", "").splitlines():
            if not output.startswith("V5_FILE_READ_RECEIPT "):
                continue
            try:
                receipt = json.loads(output[len("V5_FILE_READ_RECEIPT "):])
            except ValueError:
                continue
            if receipt.get("request_id") == request.get("request_id"):
                matches.append(receipt)
    expected = {(t["path"], t.get("run_id")) for t in request.get("targets", [])}
    reads = [r for r in matches if r.get("status") == "read_success" and r.get("bytes_read", 0) > 0 and r.get("sha256")]
    observed = {(r.get("path"), r.get("run_id")) for r in reads}
    return {"verified": bool(expected) and expected <= observed,
            "request_id": request.get("request_id"), "required_targets": request.get("targets", []),
            "code": request.get("code"), "reads": reads, "missing": [list(v) for v in sorted(expected - observed, key=str)]}


def record_patrol(root, prefix, exit_code, trigger):
    prefix = Path(prefix)
    answer = prefix.with_suffix(".txt")
    log = prefix.with_suffix(".log")
    answer_text = answer.read_text() if answer.exists() else ""
    log_text = log.read_text(errors="replace") if log.exists() else ""
    evidence = patrol_evidence(prefix, log_text)
    success = exit_code == 0 and "ASTRA_PATROL_OK" in answer_text.splitlines() and evidence["verified"]
    record = {"time_utc": timestamp(), "trigger": trigger, "model": "gpt-6-astra", "exit_code": exit_code,
        "success": success, "actual_shell_read_observed": evidence["verified"], "read_evidence": evidence,
        "run_ids": sorted({r["run_id"] for r in evidence.get("reads", []) if r.get("run_id")}),
        "answer": str(answer), "log": str(log), "evidence_schema": "v5_patrol_read_evidence_v2"}
    write_json(prefix.with_suffix(".meta.json"), record)
    write_json(Path(root) / "monitoring/last_astra.json", record)
    schedule_path = Path(root) / "monitoring" / ("probe_schedule.json" if trigger == "cron_probe" else "schedule.json")
    if trigger in {"cron_probe", "scheduled"} and schedule_path.exists():
        schedule = read_json(schedule_path)
        schedule.update(status="verified_trigger" if success else "trigger_failed", last_result=record)
        write_json(schedule_path, schedule)
    if trigger == "cron_probe":
        modify_cron(root, probe_only=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--auto-disable", action="store_true")
    parser.add_argument("--install-cron", type=Path)
    parser.add_argument("--install-probe", type=Path)
    parser.add_argument("--remove-cron", action="store_true")
    parser.add_argument("--install-publisher", type=Path)
    parser.add_argument("--publisher-command-file", type=Path)
    parser.add_argument("--remove-publisher", action="store_true")
    parser.add_argument("--patrol-result", type=Path)
    parser.add_argument("--patrol-exit-code", type=int, default=0)
    parser.add_argument("--trigger", default="manual")
    args = parser.parse_args()
    if args.install_publisher:
        if args.publisher_command_file is None:
            parser.error("--install-publisher requires --publisher-command-file")
        print(json.dumps(install_publisher(args.root, args.install_publisher, args.publisher_command_file), indent=2))
        return
    if args.remove_publisher:
        print(modify_cron(args.root, publisher_only=True))
        return
    if args.install_cron or args.install_probe:
        print(json.dumps(install_cron(args.root, args.install_cron or args.install_probe, bool(args.install_probe)), indent=2))
        return
    if args.remove_cron:
        print(modify_cron(args.root))
        return
    if args.patrol_result:
        record = record_patrol(args.root, args.patrol_result, args.patrol_exit_code, args.trigger)
        print(json.dumps(record))
        raise SystemExit(0 if record["success"] else 1)
    value = snapshot(args.root)
    if args.write:
        monitoring = args.root / "monitoring"
        filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
        write_json(monitoring / "snapshots" / filename, value)
        write_json(monitoring / "latest.json", value)
        (monitoring / "STATUS.md").write_text(table(value))
    if args.auto_disable and value["all_terminal"]:
        modify_cron(args.root)
        write_json(args.root / "monitoring/auto_disabled.json", {"time_utc": timestamp(), "reason": "all_registered_jobs_terminal"})
    print(json.dumps(value, ensure_ascii=False, indent=2) if args.json else table(value))


if __name__ == "__main__":
    main()
