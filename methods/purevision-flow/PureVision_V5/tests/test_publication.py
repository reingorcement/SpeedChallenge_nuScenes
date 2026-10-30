import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

SOURCE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


publisher = module("v5_publisher", SOURCE / "publish_results.py")
secure = module("v5_secure", SOURCE / "scripts/secure_exec.py")
reader = module("v5_reader", SOURCE / "scripts/patrol_read.py")
preparer = module("v5_patrol_preparer", SOURCE / "scripts/prepare_patrol.py")


def test_proxy_json_is_whitelisted_data_not_shell(tmp_path, monkeypatch):
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps({"HTTPS_PROXY": "$(touch /should-not-exist)"}))
    monkeypatch.setattr(secure, "trusted_file", lambda p: Path(p))
    env = secure.proxy_environment(path, {})
    assert env["HTTPS_PROXY"] == "$(touch /should-not-exist)"
    path.write_text('{"LD_PRELOAD":"/bad"}')
    with pytest.raises(ValueError, match="unsupported"):
        secure.proxy_environment(path, {})


def test_reader_emits_real_file_bytes_and_refuses_escape(tmp_path, capsys):
    path = tmp_path / "status.json"
    path.write_text('{"status":"running","global_step":3}')
    request = dict(root=str(tmp_path), request_id="request", targets=[dict(path=str(path), run_id="pilot", mode="head")])
    reader.read_targets(request)
    lines = capsys.readouterr().out.splitlines()
    receipt = json.loads(lines[0].removeprefix("V5_FILE_READ_RECEIPT "))
    assert receipt["bytes_read"] == path.stat().st_size and receipt["sha256"] == publisher.checksum(path)
    request["targets"][0]["path"] = str(tmp_path.parent / "secret")
    with pytest.raises(ValueError, match="outside"):
        reader.read_targets(request)
    private = tmp_path / "network.json"
    private.write_text('{"not_progress":"must not read"}')
    alias = tmp_path / "progress_alias.json"
    alias.symlink_to(private)
    request["targets"][0]["path"] = str(alias)
    with pytest.raises(ValueError, match="private"):
        reader.read_targets(request)


def test_flow_patrol_requires_current_report_and_preserves_snapshot_progress(tmp_path, capsys):
    prefix, report = tmp_path / "patrol", tmp_path / "flow_progress.json"
    report.write_text('{"pairs_processed":7,"total":10,"counts":{"reused":2}}')
    job = dict(job_id="full_flow_0", kind="flow", status="running", progress_path=str(report),
               progress={"pairs_processed": 7, "total": 10})
    snapshot = dict(jobs=[job], storage_volumes=[{"roles": ["job_cache"], "free_gb": 18000}])
    prefix.with_suffix(".snapshot.json").write_text(json.dumps(snapshot))
    request = preparer.prepare(tmp_path, prefix, snapshot)
    assert any(t["path"] == str(report) and t["run_id"] == "full_flow_0" for t in request["targets"])
    reader.read_targets(request)
    content = [json.loads(line.removeprefix("V5_FILE_CONTENT ")) for line in capsys.readouterr().out.splitlines()
               if line.startswith("V5_FILE_CONTENT ")]
    assert content[0]["content"]["jobs"][0]["kind"] == "flow"
    assert content[0]["content"]["jobs"][0]["progress"]["pairs_processed"] == 7
    assert content[0]["content"]["storage_volumes"][0]["free_gb"] == 18000
    report.unlink()
    request = preparer.prepare(tmp_path, prefix, snapshot)
    with pytest.raises(FileNotFoundError):
        reader.read_targets(request)
    job["status"] = "queued"
    request = preparer.prepare(tmp_path, prefix, snapshot)
    assert not any(t.get("run_id") for t in request["targets"])
    job.update(status="running", progress_path=str(tmp_path.parent / "private.json"))
    with pytest.raises(ValueError, match="outside"):
        preparer.prepare(tmp_path, prefix, snapshot)
    private = tmp_path / "network.json"
    private.write_text('{"not_progress":"must not read"}')
    report.symlink_to(private)
    job["progress_path"] = str(report)
    with pytest.raises(ValueError, match="private"):
        preparer.prepare(tmp_path, prefix, snapshot)


def make_plan(root):
    run_id = "pilot_R02_512"
    run = root / "runs" / run_id
    run.mkdir(parents=True)
    state = root / "pipeline/pilot/jobs/R02_512.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"status":"running"}')
    (run / "run.json").write_text(json.dumps(dict(run_id=run_id, code={"commit": "a" * 40}, config={"experiment": "R02", "epochs": 40})))
    (run / "status.json").write_text('{"status":"running","epoch":1,"global_step":30,"optimizer_updates":29,"amp_skips":1}')
    (run / "metrics.jsonl").write_text('{"epoch":0,"validation":{"overall":{"epe":0.3}}}\n{"partial":')
    plan = dict(prefix="pilot", code={"commit": "a" * 40}, jobs=[dict(name="R02_512", experiment="R02", job_id=run_id,
                kind="training", status_path=str(state), run_dir=str(run), commands=[])])
    path = root / "pipeline/pilot/plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_complete_jsonl_metadata_identity_and_no_weights(tmp_path):
    path = make_plan(tmp_path)
    files, summary = publisher.collect(tmp_path, [path])
    assert summary["rows"][0]["experiment"] == "R02" and not summary["all_terminal"]
    assert summary["rows"][0]["progress"]["optimizer_updates"] == 29
    assert summary["rows"][0]["progress"]["amp_skips"] == 1
    lines = files["experiments/pilot_R02_512/metrics.jsonl"].decode().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["epoch"] == 0
    assert not any(name.endswith(".pt") for name in files)
    index = json.loads(files["source_index.json"])
    assert all("source_sha256" in v for v in index["sources"])
    assert all("export_sha256" in v for v in index["exports"])


def test_local_remote_publication_no_empty_commit_and_dirty_refusal(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    plan = make_plan(root)
    remote, repo = tmp_path / "remote.git", root / "report-worktree"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", publisher.REPORT_BRANCH, str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("test\n")
    (repo / ".gitignore").write_text("**/runs/\n")
    publisher.git(repo, "add", "README.md", ".gitignore")
    publisher.git(repo, "commit", "-m", "initial")
    first = publisher.publish(root, repo, [plan], push_url=str(remote))
    second = publisher.publish(root, repo, [plan], push_url=str(remote))
    assert first["status"] == second["status"] == "published"
    assert first["report_commit"] == second["report_commit"] and not second["changed"]
    (repo / "unrelated.txt").write_text("unknown change")
    with pytest.raises(ValueError, match="unknown changes"):
        publisher.publish(root, repo, [plan], push_url=str(remote))
    assert (repo / "unrelated.txt").read_text() == "unknown change"


def test_failed_push_is_pending_retry_and_remote_conflict_is_not_rewritten(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    plan = make_plan(root)
    remote, repo = tmp_path / "remote.git", root / "report-worktree"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", publisher.REPORT_BRANCH, str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("test\n")
    publisher.git(repo, "add", "README.md")
    publisher.git(repo, "commit", "-m", "initial")
    publisher.publish(root, repo, [plan], push_url=str(remote))
    state = root / "runs/pilot_R02_512/status.json"
    state.write_text('{"status":"running","global_step":40}')
    actual_git = publisher.git
    def failed_push(repo, *args, **kwargs):
        if args[0] == "push":
            raise RuntimeError("isolated simulated network outage")
        return actual_git(repo, *args, **kwargs)
    monkeypatch.setattr(publisher, "git", failed_push)
    with pytest.raises(RuntimeError, match="outage"):
        publisher.publish(root, repo, [plan], push_url=str(remote))
    receipt = json.loads((root / "monitoring/publication.json").read_text())
    assert receipt["status"] == "pending" and receipt["report_commit"]
    monkeypatch.setattr(publisher, "git", actual_git)
    recovered = publisher.publish(root, repo, [plan], push_url=str(remote))
    assert recovered["status"] == "published" and recovered["report_commit"] == receipt["report_commit"]
    outsider = tmp_path / "other"
    subprocess.run(["git", "clone", "-b", publisher.REPORT_BRANCH, str(remote), str(outsider)], check=True, capture_output=True)
    (outsider / "new.txt").write_text("another writer\n")
    actual_git(outsider, "add", "new.txt")
    actual_git(outsider, "commit", "-m", "external publication")
    actual_git(outsider, "push", "origin", publisher.REPORT_BRANCH)
    with pytest.raises(ValueError, match="manual reconciliation"):
        publisher.publish(root, repo, [plan], push_url=str(remote))
    assert actual_git(outsider, "rev-parse", "HEAD") == actual_git(repo, "ls-remote", str(remote), f"refs/heads/{publisher.REPORT_BRANCH}").split()[0]


def test_unknown_registry_is_not_terminal_and_source_escape_refused(tmp_path):
    plan = make_plan(tmp_path)
    job_path = tmp_path / "pipeline/pilot/jobs/R02_512.json"
    job_path.write_text('{"status":"unknown"}')
    _, summary = publisher.collect(tmp_path, [plan])
    assert not summary["all_terminal"]
    with pytest.raises(ValueError, match="escaped"):
        publisher.read_source(tmp_path, tmp_path.parent / "private.json")


def test_coordinator_unknown_requires_same_plan_identity(tmp_path):
    path = make_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["identity"] = "expected"
    path.write_text(json.dumps(plan))
    coordinator = path.parent / "status.json"
    coordinator.write_text(json.dumps(dict(plan_identity="expected", jobs={"R02_512": {"status": "unknown"}})))
    _, summary = publisher.collect(tmp_path, [path])
    assert summary["rows"][0]["status"] == "unknown"
    coordinator.write_text(json.dumps(dict(plan_identity="wrong", jobs={"R02_512": {"status": "completed"}})))
    _, summary = publisher.collect(tmp_path, [path])
    assert summary["rows"][0]["status"] == "running"


def test_additional_evidence_is_explicitly_allowlisted(tmp_path):
    plan = make_plan(tmp_path)
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/baselines.json").write_text('{"train_median_vy":0.01}')
    report = tmp_path / "reports/resolution_pilot"
    report.mkdir(parents=True)
    (report / "summary.md").write_text("# pending comparison\n")
    (report / "summary.json").write_text('{"status":"pending"}')
    (tmp_path / "monitoring").mkdir()
    (tmp_path / "monitoring/network.json").write_text('{"HTTPS_PROXY":"private"}')
    files, _ = publisher.collect(tmp_path, [plan])
    assert "evidence/baselines.json" in files and "comparisons/resolution_pilot/summary.md" in files
    assert not any("network" in k or "monitoring" in k for k in files)


def test_git_proxy_is_scoped_to_transport_command(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0, stdout="", stderr="")))
    publisher.git(tmp_path, "ls-remote", "git@github.com:example/repo.git")
    publisher.git(tmp_path, "status", "--porcelain")
    assert any(v.startswith("core.sshCommand=") for v in calls[0])
    assert not any(v.startswith("core.sshCommand=") for v in calls[1])
    assert not any("--global" in command or "config" in command for command in calls)


def test_completed_prediction_crlf_bytes_survive_real_git_publication(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    path = make_plan(root)
    plan = json.loads(path.read_text())
    job = plan["jobs"][0]
    evaluation = root / "evaluations/pilot_R02_512"
    evaluation.mkdir(parents=True)
    job["commands"] = [["python", "/fixed/evaluate.py", "--output", str(evaluation)]]
    path.write_text(json.dumps(plan))
    Path(job["status_path"]).write_text('{"status":"completed"}')
    (Path(job["run_dir"]) / "status.json").write_text('{"status":"completed","epoch":40}')
    (evaluation / "evaluation.json").write_text('{"status":"completed"}')
    (evaluation / "metrics.json").write_text('{"overall":{"epe":0.1}}')
    original = b"scene,token,vx,vy\r\nscene-one,token-one,1.0,0.1\r\n"
    (evaluation / "predictions.csv").write_bytes(original)
    remote, repo = tmp_path / "remote.git", root / "report-worktree"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", publisher.REPORT_BRANCH, str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("test\n")
    publisher.git(repo, "add", "README.md")
    publisher.git(repo, "commit", "-m", "initial")
    receipt = publisher.publish(root, repo, [path], push_url=str(remote))
    assert receipt["status"] == "published" and receipt["all_terminal"]
    copied = repo / publisher.REPORT_DIRECTORY / "experiments/pilot_R02_512/evaluation/predictions.csv"
    assert copied.read_bytes() == original
    summary = json.loads((repo / publisher.REPORT_DIRECTORY / "status.json").read_text())
    assert summary["rows"][0]["predictions"]["sha256"] == publisher.checksum(copied)
    assert publisher.git(repo, "status", "--porcelain") == ""


def test_flow_only_report_publication_and_external_cache_exclusion(tmp_path):
    root, outside = tmp_path / "artifacts", tmp_path / "cache"
    root.mkdir(); outside.mkdir()
    base = root / "pipeline/full_flow"
    base.mkdir(parents=True)
    worker, progress = base / "worker.json", base / "progress.json"
    worker.write_text('{"status":"running","completed":9}')
    progress.write_text('{"status":"complete","pairs_processed":10,"total":10,"counts":{"created":10},"bytes":1234}')
    (outside / "large.pt").write_bytes(b"never export")
    job = dict(job_id="full_flow_shard0", name="shard0", kind="flow", status_path=str(worker),
               progress_path=str(progress), expected_pairs=10, storage_paths=[str(outside)])
    plan = base / "plan.json"
    plan.write_text(json.dumps(dict(prefix="full_flow", code={"commit": "b" * 40}, jobs=[job])))
    report = root / "reports/full_flow_new"
    report.mkdir(parents=True)
    (report / "summary.json").write_text('{"pairs":10}')
    (report / "summary.md").write_text("Full flow progress\n")
    other = root / "reports/unrelated"
    other.mkdir(); (other / "summary.json").write_text('{"private":true}')
    files, summary = publisher.collect(root, [plan])
    assert not summary["all_terminal"]  # A report complete is not a completed worker.
    assert summary["rows"][0]["progress"]["bytes"] == 1234
    assert "experiments/full_flow_shard0/progress_report.json" in files
    assert "comparisons/full_flow_new/summary.json" in files and "comparisons/full_flow_new/summary.md" in files
    assert not any("large.pt" in name or "unrelated" in name for name in files)
    source = next(x for x in json.loads(files["source_index.json"])["sources"] if x["path"] == str(progress))
    assert source["source_sha256"] == publisher.checksum(progress)
    worker.write_text('{"status":"completed","completed":10}')
    remote, repo = tmp_path / "remote.git", root / "report-worktree"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", publisher.REPORT_BRANCH, str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("test\n")
    publisher.git(repo, "add", "README.md"); publisher.git(repo, "commit", "-m", "initial")
    receipt = publisher.publish(root, repo, [plan], push_url=str(remote))
    assert receipt["all_terminal"] and receipt["verified_remote_commit"] == receipt["report_commit"]
    job["progress_path"] = str(outside / "progress.json")
    plan.write_text(json.dumps(dict(prefix="full_flow", code={"commit": "b" * 40}, jobs=[job])))
    with pytest.raises(ValueError, match="escaped"):
        publisher.collect(root, [plan])
