"""Bounded ABC startup checks, real cron probes and independent result publication.

This process never changes a model/configuration or starts/stops a training job.
The coordinator owns the cache/training DAG. Existing cron lines are preserved;
only exact lines installed by this lifecycle are removed after publication.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from monitor import cron_marker, install_cron, install_publisher, publication_allows_stop
from pipeline import lock, process_observation, validate_plan
from purevision_v5.common import ROOT, now, read_json, sha256, write_json


def optional(path):
    return read_json(path) if Path(path).is_file() else {}


def stage_files(plan):
    root = Path(plan["root"])
    return root, root / "monitoring" / f"fusion_abc_{plan['prefix']}"


def read_crontab():
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect current crontab")
    return result.stdout


def remove_exact_lines(lines):
    current = read_crontab()
    remove = set(lines)
    updated = "\n".join(line for line in current.splitlines() if line not in remove) + "\n"
    if updated != current:
        subprocess.run(["crontab", "-"], input=updated, text=True, check=True)


def next_trigger(minutes, *, hourly_minute=None):
    candidate = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(minutes=1)
    while (candidate.minute % minutes != 0 if hourly_minute is None else candidate.minute != hourly_minute):
        candidate += timedelta(minutes=1)
    return candidate.isoformat()


def install(plan_path, command_file):
    plan = read_json(plan_path)
    validate_plan(plan)
    root, directory = stage_files(plan)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "schedule_install.json"
    existing = optional(receipt_path)
    if existing:
        if existing["plan_identity"] != plan["identity"]:
            raise ValueError("Schedule belongs to another plan")
        if not all(line in read_crontab().splitlines() for line in existing["formal_lines"]):
            raise ValueError("Existing stage schedules changed; do not silently reinstall")
        return existing
    previous_guard = optional(root / "monitoring/lifecycle_guard.json")
    if previous_guard and previous_guard.get("plan_identity") != plan["identity"]:
        old = optional(previous_guard.get("plan", ""))
        if old and any(optional(job["status_path"]).get("status") not in
                       {"completed", "failed", "interrupted", "cancelled", "blocked"} for job in old.get("jobs", [])):
            raise ValueError("A previous guarded stage is still active; preserve its scheduling")
    before = read_crontab()
    # Existing shared-root V5 schedules could belong to another task. Never replace them.
    if any(cron_marker(root) in line for line in before.splitlines()):
        raise ValueError("Existing V5 schedule requires explicit reconciliation; no duplicate/replacement installed")
    write_json(directory / "previous_guard.json", previous_guard)
    write_json(root / "monitoring/lifecycle_guard.json", dict(plan=str(Path(plan_path).resolve()),
               plan_identity=plan["identity"], completion_requires_success=True))
    server_astra = install_cron(root, ROOT)
    publisher = install_publisher(root, ROOT, command_file)
    command = ["/usr/bin/env", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "TZ=Asia/Shanghai",
               plan["python"], "-B", str(ROOT / "fusion_abc_lifecycle.py"), "--plan", str(Path(plan_path).resolve()),
               "--publisher-command", str(Path(command_file).resolve()), "--probe"]
    probe_line = "* * * * * " + shlex.join(command) + " >> " + shlex.quote(str(directory / "probe.log")) + \
                 " 2>&1 # " + cron_marker(root) + "_ABC_PROBE_" + plan["prefix"]
    current = read_crontab()
    subprocess.run(["crontab", "-"], input=current.rstrip() + "\n" + probe_line + "\n", text=True, check=True)
    formal_lines = server_astra["lines"] + publisher["lines"]
    receipt = dict(status="installed_pending_actual_trigger", installed_at=now(), plan_identity=plan["identity"],
                   run_ids=[job["job_id"] for job in plan["jobs"]], code=plan["code"],
                   formal_lines=formal_lines, probe_line=probe_line, crontab_before=before,
                   crontab_after=read_crontab(), previous_guard=previous_guard,
                   next_server_utc=next_trigger(5), next_astra_utc=next_trigger(30),
                   next_publisher_utc=next_trigger(60, hourly_minute=17), next_probe_utc=next_trigger(1))
    write_json(receipt_path, receipt)
    write_json(root / "monitoring/probe_schedule.json", dict(status="installed_pending_observed_trigger",
               installed_at=receipt["installed_at"], plan_identity=plan["identity"],
               run_ids=receipt["run_ids"], lines=[probe_line], model="gpt-6-astra"))
    return receipt


def publish(plan, command_file, trigger="manual"):
    root, directory = stage_files(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "publication.log").open("a") as stream:
        try:
            result = subprocess.run(["bash", str(ROOT / "scripts/publish_results.sh"), str(root),
                                     str(command_file), trigger], stdout=stream, stderr=subprocess.STDOUT, timeout=650)
            receipt = optional(root / "monitoring/publication.json")
            success = (result.returncode == 0 and receipt.get("status") == "published" and
                       plan["identity"] in receipt.get("plan_identities", []) and
                       receipt.get("report_commit") == receipt.get("verified_remote_commit"))
            return dict(time=now(), success=success, exit_code=result.returncode,
                        report_commit=receipt.get("report_commit"), error=receipt.get("error"))
        except (OSError, subprocess.TimeoutExpired) as error:
            return dict(time=now(), success=False, error=f"{type(error).__name__}: {error}")


def probe(plan_path, command_file):
    """Invoked by cron, not by the interactive launch check."""
    plan = read_json(plan_path)
    root, directory = stage_files(plan)
    with lock(directory / "probe.lock"):
        install_record = read_json(directory / "schedule_install.json")
        previous = optional(directory / "schedule_trigger.json")
        if previous.get("status") == "verified":
            remove_exact_lines([install_record["probe_line"]])
            return
        started = now()
        result = subprocess.run(["bash", str(ROOT / "scripts/server_check.sh"), str(root), "abc_cron_probe"],
                                capture_output=True, text=True, timeout=90)
        snapshot = optional(root / "monitoring/latest.json")
        ids = {j["job_id"] for j in plan["jobs"]}
        snapshot_ids = {j["job_id"] for j in snapshot.get("jobs", [])}
        server_ok = result.returncode == 0 and ids <= snapshot_ids
        def astra_probe():
            try:
                process = subprocess.run(["bash", str(ROOT / "scripts/astra_patrol.sh"), str(root), "cron_probe"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=400)
                return dict(exit_code=process.returncode, patrol=optional(root / "monitoring/last_astra.json"))
            except (OSError, subprocess.TimeoutExpired) as error:
                return dict(exit_code=None, patrol={}, error=f"{type(error).__name__}: {error}")

        # Network publication and AI access are independent. A GitHub timeout
        # must not delay the real AI trigger. Retain each successful probe so a
        # publisher retry does not repeatedly spend tokens on the same startup.
        with ThreadPoolExecutor(max_workers=2) as executor:
            publication_future = None if previous.get("publisher", {}).get("success") else executor.submit(
                publish, plan, command_file, "scheduled")
            astra_future = None if previous.get("astra_success") else executor.submit(astra_probe)
            publication = publication_future.result() if publication_future else previous["publisher"]
            astra_result = astra_future.result() if astra_future else dict(exit_code=0, patrol=previous["astra"])
        patrol = astra_result["patrol"]
        astra_triggered_at = previous.get("astra_triggered_at", previous.get("triggered_at")) if astra_future is None else started
        actual_ids = set(patrol.get("run_ids", []))
        request = optional(Path(patrol.get("answer", "missing")).with_suffix(".request.json"))
        all_done = all(optional(job["status_path"]).get("status") == "completed" for job in plan["jobs"])
        scope_verified = set(request.get("scope_run_ids") or []) == ids
        astra_ok = (astra_result["exit_code"] == 0 and patrol.get("success") is True and
                    patrol.get("time_utc", "") >= astra_triggered_at[:19] and scope_verified and
                    (bool(actual_ids & ids) or all_done))
        receipt = dict(status="verified" if server_ok and publication["success"] and astra_ok else "retry_pending",
                       triggered_at=started, finished_at=now(), plan_identity=plan["identity"],
                       trigger="actual_crontab_execution", expected_run_ids=sorted(ids),
                       server=dict(success=server_ok, exit_code=result.returncode,
                                   snapshot_time=snapshot.get("time_utc"), observed_run_ids=sorted(snapshot_ids & ids)),
                       publisher=publication, astra=patrol, astra_success=astra_ok,
                       astra_request_scope_verified=scope_verified, stage_already_complete=all_done,
                       astra_triggered_at=astra_triggered_at, astra_error=astra_result.get("error"),
                       concurrent_independent_probes=True)
        write_json(directory / "schedule_trigger.json", receipt)
        if receipt["status"] == "verified":
            remove_exact_lines([install_record["probe_line"]])
        return receipt


def startup_proof(plan, job):
    root, directory = stage_files(plan)
    state = optional(job["status_path"])
    path = directory / "startup" / f"{job['name']}.json"
    previous = optional(path)
    if previous.get("checkpoint_verified"):
        return previous
    progress = optional(job["progress_path"]) if job.get("progress_path") else {}
    log = Path(job["log_path"])
    process = process_observation(state.get("child_pid"), state.get("child_pid_start_ticks"))
    proof = dict(run_id=job["job_id"], time=now(), phase=state.get("phase"), worker_status=state.get("status"),
                 process=process, log_exists=log.is_file(), log_bytes=log.stat().st_size if log.is_file() else 0,
                 physical_gpu=job.get("physical_gpu"), process_device=job.get("process_device"),
                 progress=progress, checkpoint_verified=False)
    if job["kind"] == "training" and progress.get("completed_epochs", 0) >= 1:
        import torch
        torch.set_num_threads(2)
        run = Path(job["run_dir"])
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        from purevision_v5.fusion_abc_training import validate_checkpoint
        index = read_json(Path(plan["cache_root"]) / "index.json")
        validate_checkpoint(checkpoint, index)
        keys = {"model", "optimizer", "scheduler", "rng", "completed_epochs", "epoch_index", "next_batch",
                "processed_batches", "optimizer_updates", "amp_skips", "initialization", "scaler"}
        if not keys <= set(checkpoint) or checkpoint["code"]["commit"] != plan["code"]["commit"]:
            raise ValueError("Checkpoint lacks required exact-version full recovery state")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if not metrics or metrics[0]["epoch"] != 1 or metrics[0]["dev"]["overall"]["count"] != 1826:
            raise ValueError("First complete validation does not match the fixed dev selection")
        proof.update(checkpoint_verified=True, checkpoint_sha256=sha256(run / "last.pt"),
                     checkpoint_keys=sorted(keys), first_validation=metrics[0])
    write_json(path, proof)
    return proof


def run(plan_path, command_file, interval=30):
    plan = read_json(plan_path)
    validate_plan(plan)
    root, directory = stage_files(plan)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "lifecycle.json"
    state = optional(status_path) or dict(plan_identity=plan["identity"], started_at=now(), publications=[])
    if state["plan_identity"] != plan["identity"]:
        raise ValueError("Lifecycle state belongs to another plan")
    with lock(directory / "lifecycle.lock"):
        while True:
            if state.get("schedules_removed_after_verified_publication"):
                # The experiment/report already reached remote GitHub. Publish
                # the immutable cleanup receipt too, retrying only publication
                # if the network fails; no patrol or experiment is restarted.
                outcome = publish(plan, command_file)
                state["publications"].append(outcome)
                state.update(updated_at=now(), final_cleanup_publication=outcome,
                             status="completed" if outcome["success"] else "cleanup_publication_pending")
                write_json(status_path, state)
                if outcome["success"]:
                    return state
                time.sleep(300)
                continue
            milestones, proofs = [], []
            for job in plan["jobs"]:
                raw = optional(job["status_path"])
                progress = optional(job["progress_path"]) if job.get("progress_path") else {}
                milestones.append([job["name"], raw.get("status"),
                                   bool(progress.get("completed_epochs", 0) >= 1),
                                   progress.get("completed_epochs", 0) // 5])
                try:
                    proofs.append(startup_proof(plan, job))
                except Exception as error:
                    proofs.append(dict(run_id=job["job_id"], status="requires_investigation",
                                       error=f"{type(error).__name__}: {error}"))
            terminal = all(optional(j["status_path"]).get("status") in
                           {"completed", "failed", "interrupted", "cancelled", "blocked"} for j in plan["jobs"])
            due = (milestones != state.get("published_milestones") or time.time() - state.get("last_success_epoch", 0) >= 3600)
            if due and time.time() - state.get("last_attempt_epoch", 0) >= 60:
                state["last_attempt_epoch"] = time.time()
                outcome = publish(plan, command_file)
                state["publications"].append(outcome)
                if outcome["success"]:
                    state.update(published_milestones=milestones, last_success_epoch=time.time())
            state.update(updated_at=now(), pid=os.getpid(), milestones=milestones,
                         startup_proofs={p["run_id"]: p.get("checkpoint_verified", False) for p in proofs},
                         anomalies=[p for p in proofs if p.get("status") == "requires_investigation"])
            if terminal and publication_allows_stop(root):
                install_record = optional(directory / "schedule_install.json")
                remove_exact_lines(install_record.get("formal_lines", []) + [install_record.get("probe_line", "")])
                trigger = optional(directory / "schedule_trigger.json")
                state.update(status="completed", finished_at=now(),
                             schedules_removed_after_verified_publication=True, final_crontab=read_crontab(),
                             actual_schedule_trigger_verified=trigger.get("status") == "verified",
                             schedule_completion_note=("verified current-stage cron probe" if trigger.get("status") == "verified" else
                                 "finished before a complete probe receipt; do not claim an unobserved trigger or extend the run"))
                write_json(directory / "cleanup_receipt.json", dict(status="completed", time=now(),
                    plan_identity=plan["identity"], run_ids=[j["job_id"] for j in plan["jobs"]],
                    all_authorized_budgets_completed=True, comparison=str(plan["completion_report"]),
                    final_report_publication=optional(root / "monitoring/publication.json"),
                    actual_schedule_trigger_verified=state["actual_schedule_trigger_verified"],
                    schedule_completion_note=state["schedule_completion_note"],
                    removed_lines=install_record.get("formal_lines", []) + [install_record.get("probe_line", "")],
                    remaining_crontab=state["final_crontab"], no_budget_extension=True))
                write_json(status_path, state)
                continue
            state["status"] = "running" if not terminal else "publication_or_completion_pending"
            write_json(status_path, state)
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--publisher-command", required=True, type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--install", action="store_true")
    action.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    value = install(args.plan, args.publisher_command) if args.install else (
        probe(args.plan, args.publisher_command) if args.probe else run(args.plan, args.publisher_command))
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
