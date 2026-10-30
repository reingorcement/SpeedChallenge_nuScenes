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
ACTIVE = {"launching", "running", "running_unverified", "queued", "preparing", "not_started"}
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
        return {"pid": None, "alive": None, "identity_verified": False}
    try:
        pid = int(pid)
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        start_ticks = int(fields[19])
        alive = fields[0] not in {"Z", "X"}
        identity_matches = expected_start is None or int(expected_start) == start_ticks
        return {"pid": pid, "alive": alive and identity_matches, "start_ticks": start_ticks,
            "identity_verified": expected_start is not None, "pid_reused": not identity_matches}
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "alive": False, "identity_verified": expected_start is not None}


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
    run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = root / run_dir
    launch = read_json(run_dir / "launch.json") if run_dir else {}
    run = read_json(run_dir / "run.json") if run_dir else {}
    status_path = Path(job["status_path"]) if job.get("status_path") else (run_dir / "status.json" if run_dir else None)
    status = read_json(status_path) if status_path else {}
    training_status = read_json(run_dir / "status.json") if run_dir and job.get("kind", "training") == "training" else {}
    progress_status = {**status, **training_status}
    state = status.get("status", run.get("status", job.get("status", "not_started")))
    state = {"done": "completed", "complete": "completed", "pending": "queued"}.get(state, state)
    if job.get("pipeline_id"):
        # The wrapper owns training AND its following independent evaluation.
        process = process_info(status.get("pid") or job.get("pid"), status.get("pid_start_ticks") or job.get("pid_start_ticks"))
    else:
        process = process_info(launch.get("pid") or job.get("pid"), launch.get("pid_start_ticks") or job.get("pid_start_ticks"))
    child_process = process_info(status.get("child_pid") or job.get("child_pid"), status.get("child_pid_start_ticks") or job.get("child_pid_start_ticks"))
    log = Path(job.get("log_path", str(run_dir / "stdout.log") if run_dir else "/nonexistent"))
    checkpoint = Path(job.get("checkpoint_path", str(run_dir / "last.pt") if run_dir else "/nonexistent"))
    metric = last_metric(run_dir / "metrics.jsonl") if run_dir else {}
    notes = []
    if state == "blocked":
        notes.append(str(status.get("reason", job.get("reason", "dependency_blocked"))))
    if state in {"running", "preparing"} and process["alive"] is False:
        state = "failed" if launch.get("exit_code", job.get("exit_code", 0)) else "interrupted"
        notes.append("process_missing_without_terminal_status")
    elif state == "running" and process["alive"] is None:
        state = "running_unverified"
        notes.append("launch_pid_not_recorded")
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
    anomalies = [line[-500:] for line in tail(log) if ERROR_PATTERN.search(line)][-8:]
    if anomalies:
        notes.append("anomaly_text_in_log_requires_diagnosis")
    if status.get("error") or run.get("error"):
        notes.append(str(status.get("error", run.get("error")))[:500])
    validation = progress_status.get("validation") or metric.get("validation", {}).get("overall", {})
    return {"job_id": job["job_id"], "kind": job.get("kind", "training"), "status": state,
        "run_dir": str(run_dir) if run_dir else None, "session": launch.get("session", job.get("session")),
        "device": launch.get("device", job.get("device")), "process": process, "child_process": child_process,
        "phase": status.get("phase"), "training_status": training_status.get("status"),
        "commit": run.get("code", {}).get("commit", job.get("commit")),
        "epoch": progress_status.get("epoch", metric.get("epoch")), "global_step": progress_status.get("global_step", metric.get("global_step")),
        "loss": progress_status.get("loss", metric.get("train_loss")), "validation": validation,
        "progress": {k: status[k] for k in ("completed", "total", "pairs", "pairs_processed", "counts", "scenes", "progress") if k in status},
        "status_age_s": age_seconds(status_path) if status_path else None,
        "log_path": str(log) if log.exists() else None, "log_age_s": age_seconds(log),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "checkpoint_age_s": age_seconds(checkpoint), "anomaly_lines": anomalies, "notes": notes}


def collect_jobs(root):
    registry = read_json(root / "monitoring/jobs.json", {"jobs": []})
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


def snapshot(root):
    root = Path(root).resolve()
    jobs = [inspect_job(job, root) for job in collect_jobs(root)]
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
        "all_terminal": bool(jobs) and all(job["status"] in TERMINAL for job in jobs),
        "gpu": gpu, "disk": {"total_gb": disk.total / 1e9, "free_gb": disk.free / 1e9,
            "used_percent": 100 * disk.used / disk.total, "low_space": disk.free < 10e9 or disk.free / disk.total < .02}}


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
    lines.extend(["", f"活跃/排队：{value['active_count']}；磁盘剩余 {value['disk']['free_gb']:.1f} GB。", "",
        "异常文本、日志停更或损失波动只触发人工定位；巡视不会改代码、参数或重启作业。", ""])
    return "\n".join(lines)


def cron_marker(root):
    return "PUREVISION_V5_" + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]


def filter_crontab(text, marker, probe_only=False):
    suffix = marker + "_PROBE" if probe_only else marker
    return "\n".join(line for line in text.splitlines() if suffix not in line) + "\n"


def modify_cron(root, additions=None, probe_only=False):
    root = Path(root).resolve()
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("Unable to read crontab")
    existing = result.stdout
    backup = root / "monitoring/crontab_backups" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".txt")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(existing)
    updated = filter_crontab(existing, cron_marker(root), probe_only)
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
        f"0 * * * * {command('astra_patrol.sh', 'scheduled')} # {marker}_ASTRA"]
    backup = modify_cron(root, lines, probe_only=probe)
    record = {"installed_at": timestamp(), "code_root": str(code_root), "commit": commit,
        "probe": probe, "crontab_backup": str(backup), "lines": lines,
        "status": "installed_pending_observed_trigger", "model": "gpt-6-astra"}
    write_json(root / "monitoring" / ("probe_schedule.json" if probe else "schedule.json"), record)
    return record


def record_patrol(root, prefix, exit_code, trigger):
    prefix = Path(prefix)
    answer = prefix.with_suffix(".txt")
    log = prefix.with_suffix(".log")
    answer_text = answer.read_text() if answer.exists() else ""
    log_text = log.read_text(errors="replace") if log.exists() else ""
    success = exit_code == 0 and "ASTRA_PATROL_OK" in answer_text and "\nexec\n" in log_text
    record = {"time_utc": timestamp(), "trigger": trigger, "model": "gpt-6-astra", "exit_code": exit_code,
        "success": success, "actual_shell_read_observed": "\nexec\n" in log_text, "answer": str(answer), "log": str(log)}
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
    parser.add_argument("--patrol-result", type=Path)
    parser.add_argument("--patrol-exit-code", type=int, default=0)
    parser.add_argument("--trigger", default="manual")
    args = parser.parse_args()
    if args.install_cron or args.install_probe:
        print(json.dumps(install_cron(args.root, args.install_cron or args.install_probe, bool(args.install_probe)), indent=2))
        return
    if args.remove_cron:
        print(modify_cron(args.root))
        return
    if args.patrol_result:
        print(json.dumps(record_patrol(args.root, args.patrol_result, args.patrol_exit_code, args.trigger)))
        return
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
