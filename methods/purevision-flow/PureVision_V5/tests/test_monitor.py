import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("v5_monitor", Path(__file__).resolve().parents[1] / "monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_cron_preserves_other_entries_and_probe_cleanup():
    text = "MAILTO=someone@example.com\n15 1 * * * /other/job\n* * * * * probe # TAG_PROBE\n0 * * * * patrol # TAG_ASTRA\n"
    probe_removed = monitor.filter_crontab(text, "TAG", probe_only=True)
    assert "/other/job" in probe_removed and "MAILTO=" in probe_removed
    assert "TAG_PROBE" not in probe_removed and "TAG_ASTRA" in probe_removed
    all_removed = monitor.filter_crontab(text, "TAG")
    assert "TAG" not in all_removed and "/other/job" in all_removed


def test_job_terminal_status_and_missing_process(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.json").write_text(json.dumps({"pid": 999999999}))
    (run / "status.json").write_text(json.dumps({"status": "running", "global_step": 17}))
    job = {"job_id": "r00", "run_dir": str(run)}
    observed = monitor.inspect_job(job, tmp_path)
    assert observed["status"] == "unknown"  # One missing PID is not a terminal job receipt.
    (run / "status.json").write_text(json.dumps({"status": "completed", "global_step": 20}))
    assert monitor.inspect_job(job, tmp_path)["status"] == "completed"
    (run / "status.json").write_text(json.dumps({"status": "failed", "error": "bad loss"}))
    assert monitor.inspect_job(job, tmp_path)["status"] == "failed"


def test_pid_identity_and_queued_job(tmp_path):
    actual = monitor.process_info(os.getpid())
    assert actual["alive"]
    assert monitor.process_info(os.getpid(), actual["start_ticks"])["identity_verified"]
    assert not monitor.process_info(os.getpid(), actual["start_ticks"] + 1)["alive"]
    job = monitor.inspect_job({"job_id": "queued", "status": "queued"}, tmp_path)
    assert job["status"] == "queued"


def test_cache_status_alias_progress_and_blocked_terminal(tmp_path):
    path = tmp_path / "flow_status.json"
    path.write_text(json.dumps({"status": "complete", "pairs_processed": 100, "counts": {"train_core": 90, "dev": 10}}))
    job = monitor.inspect_job({"job_id": "flow", "kind": "cache", "status_path": str(path)}, tmp_path)
    assert job["status"] == "completed"
    assert job["progress"]["pairs_processed"] == 100
    assert job["progress"]["counts"]["dev"] == 10
    assert "blocked" in monitor.TERMINAL
    blocked = monitor.inspect_job({"job_id": "dependent", "status": "blocked"}, tmp_path)
    assert blocked["status"] == "blocked"


def test_partial_metric_line_and_error_log(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"epoch":1,"train_loss":0.1}\n{"partial":')
    assert monitor.last_metric(tmp_path / "metrics.jsonl")["epoch"] == 1
    assert monitor.ERROR_PATTERN.search("RuntimeError: CUDA out of memory")
    assert not monitor.ERROR_PATTERN.search('{"event":"step","loss":0.1}')


def test_latest_complete_validation_overrides_stale_worker_cache(tmp_path):
    run=tmp_path/"run";run.mkdir()
    monitor.write_json(run/"status.json",dict(status="running",epoch=8,global_step=100))
    wrapper=tmp_path/"worker.json"
    monitor.write_json(wrapper,dict(status="running",pid=os.getpid(),validation=dict(epe=.9)))
    (run/"metrics.jsonl").write_text(json.dumps(dict(epoch=7,validation=dict(overall=dict(epe=.2))))+'\n{"partial":')
    observed=monitor.inspect_job(dict(job_id="current",kind="training",pipeline_id="test",
                                     run_dir=str(run),status_path=str(wrapper)),tmp_path)
    assert observed["validation"]==dict(epe=.2)
    assert observed["validation_epoch_zero_based"]==7
    assert observed["epoch"]==8


def test_patrol_success_requires_actual_tool_read_and_exit_zero(tmp_path):
    prefix = tmp_path / "patrol"
    prefix.with_suffix(".txt").write_text("ASTRA_PATROL_OK\n")
    prefix.with_suffix(".log").write_text("No tool call\n")
    assert not monitor.record_patrol(tmp_path, prefix, 0, "manual")["success"]
    prefix.with_suffix(".log").write_text("header\nexec\nread current snapshot\n")
    assert not monitor.record_patrol(tmp_path, prefix, 0, "manual")["success"]
    assert not monitor.record_patrol(tmp_path, prefix, 124, "manual")["success"]


def test_unknown_proc_read_and_surviving_child_keep_monitoring(tmp_path, monkeypatch):
    original = Path.read_text
    def denied(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            raise PermissionError("isolated probe")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", denied)
    assert monitor.process_info(123, 17)["verdict"] == "unknown"
    (tmp_path / "status.json").write_text('{"status":"running","pid":123}')
    observed = monitor.inspect_job(dict(job_id="x", status="running", status_path=str(tmp_path / "status.json")), tmp_path)
    assert observed["status"] == "unknown" and observed["status"] in monitor.ACTIVE
    monkeypatch.setattr(Path, "read_text", original)
    (tmp_path / "status.json").write_text(json.dumps(dict(status="running", pid=999999999, child_pid=os.getpid())))
    job = monitor.inspect_job(dict(job_id="x", pipeline_id="test", status_path=str(tmp_path / "status.json")), tmp_path)
    assert job["status"] == "unknown" and job["child_process"]["alive"]


def test_patrol_requires_successful_exact_read_command_targets_and_run(tmp_path):
    prefix = tmp_path / "patrol"
    command = ["/python", "-B", "/fixed/scripts/patrol_read.py", "--request", str(prefix.with_suffix(".request.json"))]
    target = {"path": "/artifacts/runs/one/status.json", "run_id": "one"}
    prefix.with_suffix(".request.json").write_text(json.dumps(dict(request_id="nonce", command=command, targets=[target])))
    prefix.with_suffix(".txt").write_text("ASTRA_PATROL_OK\n")
    import shlex
    receipt = dict(**target, request_id="nonce", status="read_success", bytes_read=10, sha256="a" * 64)
    item = dict(type="command_execution", command=shlex.join(command), status="completed", exit_code=0,
                aggregated_output="V5_FILE_READ_RECEIPT " + json.dumps(receipt))
    def result():
        prefix.with_suffix(".log").write_text(json.dumps(dict(type="item.completed", item=item)) + "\n")
        return monitor.record_patrol(tmp_path, prefix, 0, "manual")
    assert result()["success"] and result()["run_ids"] == ["one"]
    item["exit_code"] = 1
    assert not result()["success"]
    item["exit_code"] = 0
    receipt["run_id"] = "another"
    item["aggregated_output"] = "V5_FILE_READ_RECEIPT " + json.dumps(receipt)
    assert not result()["success"]


def test_monitor_cleanup_preserves_independent_publisher():
    value = "*/5 * * * * check # TASK_SERVER\n*/30 * * * * ai # TASK_ASTRA\n17 * * * * publish # TASK_PUBLISHER\n0 2 * * * other\n"
    cleaned = monitor.filter_crontab(value, "TASK")
    assert "TASK_PUBLISHER" in cleaned and "other" in cleaned and "TASK_ASTRA" not in cleaned
    publish_removed = monitor.filter_crontab(value, "TASK", publisher_only=True)
    assert "TASK_ASTRA" in publish_removed and "TASK_PUBLISHER" not in publish_removed


def test_proc_parse_failure_is_unknown_and_not_dead(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "malformed stat")
    assert monitor.process_info(123)["verdict"] == "unknown"


def test_pipeline_terminal_requires_same_worker_and_two_observations(tmp_path):
    base = tmp_path / "pipeline/pilot"
    (base / "observations").mkdir(parents=True)
    raw = {"status": "running", "pid": 99999999}
    job = dict(job_id="pilot_R02", name="R02", pipeline_id="pilot")
    monitor.write_json(base / "plan.json", dict(identity="plan", jobs=[job]))
    path = base / "observations/R02.json"
    monitor.write_json(base / "status.json", dict(plan_identity="plan", jobs={"R02": dict(status="interrupted", observation_path=str(path))}))
    proof = dict(effective_status="interrupted", dead_observations=1, last_worker_state=raw,
                 wrapper={"verdict": "dead"}, child={"verdict": "dead"}, tmux_exists=False, worker_lock_free=True)
    monitor.write_json(path, proof)
    assert monitor.confirmed_pipeline_terminal(job, tmp_path, raw) is None
    proof["dead_observations"] = 2
    monitor.write_json(path, proof)
    assert monitor.confirmed_pipeline_terminal(job, tmp_path, raw) == str(path)
    assert monitor.confirmed_pipeline_terminal(job, tmp_path, {**raw, "heartbeat": 2}) is None
    proof["child"] = {"verdict": "unknown"}
    monitor.write_json(path, proof)
    assert monitor.confirmed_pipeline_terminal(job, tmp_path, raw) is None


def test_snapshot_unknown_job_never_auto_terminal(tmp_path, monkeypatch):
    monitoring = tmp_path / "monitoring"
    monitoring.mkdir()
    monitor.write_json(monitoring / "jobs.json", {"jobs": [{"job_id": "unknown_run", "status": "unknown"}]})
    monkeypatch.setattr(monitor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    value = monitor.snapshot(tmp_path)
    assert value["unknown_count"] == value["active_count"] == 1 and not value["all_terminal"]


def test_pipeline_evaluation_keeps_wrapper_lifecycle_and_fresh_train_progress(tmp_path):
    run = tmp_path / "runs" / "prefix_R00"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "completed", "epoch": 20, "global_step": 800}))
    (run / "launch.json").write_text(json.dumps({"pid": 999999999}))
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text(json.dumps({"status": "running", "phase": "evaluation", "pid": os.getpid(), "epoch": 19, "global_step": 799}))
    job = {"job_id": "prefix_R00", "kind": "training", "pipeline_id": "prefix", "run_dir": str(run), "status_path": str(wrapper)}
    result = monitor.inspect_job(job, tmp_path)
    assert result["status"] == "running" and result["phase"] == "evaluation"
    assert result["process"]["pid"] == os.getpid() and result["process"]["alive"]
    assert result["global_step"] == 800 and result["training_status"] == "completed"
    registry = tmp_path / "monitoring/jobs.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"jobs": [job]}))
    jobs = monitor.collect_jobs(tmp_path)
    assert len(jobs) == 1 and jobs[0]["status_path"] == str(wrapper)
    assert "launching" in monitor.ACTIVE


def test_flow_symlink_reports_separate_storage_and_low_space(tmp_path):
    root, nvme = tmp_path / "artifacts", tmp_path / "nvme"
    root.mkdir()
    nvme.mkdir()
    (root / "flows").symlink_to(nvme, target_is_directory=True)
    volumes = monitor.storage_volumes(root, device_for=lambda path: 2 if path == nvme else 1,
        usage_for=lambda path: SimpleNamespace(total=1000e9, used=995e9 if path == nvme else 100e9,
                                              free=5e9 if path == nvme else 900e9))
    assert len(volumes) == 2
    cache = next(volume for volume in volumes if "flow_cache" in volume["roles"])
    assert cache["free_gb"] == 5 and cache["low_space"]
    assert cache["resolved_paths"] == [str(nvme)]
    assert cache["paths"] == [str(root / "flows")]


def test_storage_deduplicates_same_device(tmp_path):
    (tmp_path / "flows").mkdir()
    calls = []
    def usage(path):
        calls.append(path)
        return SimpleNamespace(total=1000e9, used=100e9, free=900e9)
    volumes = monitor.storage_volumes(tmp_path, device_for=lambda path: 1, usage_for=usage)
    assert len(volumes) == 1 and len(calls) == 1
    assert volumes[0]["roles"] == ["artifact_root", "flow_cache"]


def test_external_job_cache_disk_is_observed_once_without_traversal(tmp_path, monkeypatch):
    root, disk = tmp_path / "private", tmp_path / "dataset"
    root.mkdir(); disk.mkdir()
    jobs = [{"storage_paths": [str(disk / "new/cache")]}, {"storage_paths": [str(disk / "other")]}]
    calls = []
    def usage(path):
        calls.append(path)
        return SimpleNamespace(total=1000e9, used=995e9 if path == disk else 10e9,
                               free=5e9 if path == disk else 990e9)
    monkeypatch.setattr(Path, "rglob", lambda *args: (_ for _ in ()).throw(AssertionError("no recursive scan")))
    result = monitor.storage_volumes(root, jobs=jobs, device_for=lambda path: 2 if path == disk else 1, usage_for=usage)
    assert calls == [root, disk]
    external = next(x for x in result if x["device_id"] == 2)
    assert external["roles"] == ["job_cache"] and external["low_space"]
    assert external["resolved_paths"] == [str(disk)]
    assert external["paths"] == [str(disk / "new/cache"), str(disk / "other")]
    with pytest.raises(ValueError, match="absolute"):
        monitor.storage_volumes(root, jobs=[{"storage_paths": ["relative"]}])


def test_flow_report_is_fresh_progress_but_not_worker_terminal_status(tmp_path):
    wrapper, report = tmp_path / "worker.json", tmp_path / "progress.json"
    wrapper.write_text(json.dumps({"status": "running", "pid": os.getpid(), "completed": 3}))
    report.write_text(json.dumps({"status": "complete", "pairs_processed": 10, "total": 10,
        "counts": {"created": 8, "reused": 2}, "throughput": 5., "bytes_written": 100}))
    job = dict(job_id="flow", kind="flow", pipeline_id="flow_only", status_path=str(wrapper),
               progress_path=str(report), expected_pairs=10)
    observed = monitor.inspect_job(job, tmp_path)
    assert observed["status"] == "running" and observed["checkpoint"] is None
    assert observed["progress"]["pairs_processed"] == 10 and observed["progress"]["counts"]["reused"] == 2
    assert observed["progress"]["completed"] == 10  # Fresh report supersedes a slower wrapper heartbeat.
    assert observed["progress"]["bytes_written"] == 100 and observed["progress_path"] == str(report)
    assert observed["progress_report_status"] == "complete"
    report.unlink()
    missing = monitor.inspect_job(job, tmp_path)
    assert missing["status"] == "running" and missing["progress_path"] == str(report)
    assert "progress_report_missing_or_unreadable_requires_recheck" in missing["notes"]
    job["progress_path"] = str(tmp_path.parent / "private.json")
    refused = monitor.inspect_job(job, tmp_path)
    assert refused["status"] == "running" and refused["progress_path"] is None
    assert "progress_path_outside_artifacts_or_private_requires_recheck" in refused["notes"]
    private = tmp_path / "network.json"
    private.write_text('{"not_progress":"must not read"}')
    report.symlink_to(private)
    job["progress_path"] = str(report)
    assert monitor.inspect_job(job, tmp_path)["progress_path"] is None
