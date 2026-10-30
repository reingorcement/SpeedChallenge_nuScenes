import importlib.util
import json
import os
from pathlib import Path

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
    assert observed["status"] == "interrupted"
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


def test_patrol_success_requires_actual_tool_read_and_exit_zero(tmp_path):
    prefix = tmp_path / "patrol"
    prefix.with_suffix(".txt").write_text("ASTRA_PATROL_OK\n")
    prefix.with_suffix(".log").write_text("No tool call\n")
    assert not monitor.record_patrol(tmp_path, prefix, 0, "manual")["success"]
    prefix.with_suffix(".log").write_text("header\nexec\nread current snapshot\n")
    assert monitor.record_patrol(tmp_path, prefix, 0, "manual")["success"]
    assert not monitor.record_patrol(tmp_path, prefix, 124, "manual")["success"]


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
