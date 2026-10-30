"""Scheduling decisions are tested without launching tmux, CUDA or training."""
import copy
import json
from pathlib import Path

import pytest

from pipeline import (build_plan, dependency_state, initialize, lock, read_json,
                      process_observation, register, tick, validate_plan, write_json)


@pytest.fixture
def plan(tmp_path):
    manifest=tmp_path/"data/dataset.json"
    write_json(manifest,{"schema":"v5_dataset_v1","sequences":[]})
    version={"commit":"a"*40,"dirty":False,"verified_remote_commit":"a"*40}
    return build_plan(tmp_path,manifest,"features/v2","test_s42",version)


class MockTmux:
    def __init__(self):
        self.launched=[]
    def exists(self,session):
        return session in [job["session"] for job in self.launched]
    def launch(self,path,plan,job):
        assert not self.exists(job["session"])
        self.launched.append(job)
        return 10000+len(self.launched)


def alive(pid,expected_start=None):
    return {"alive":True,"start_ticks":12345}


def job(plan,name):
    return next(job for job in plan["jobs"] if job["name"]==name)


def mark(plan,name,status,**extra):
    selected=job(plan,name)
    state=read_json(selected["status_path"],{})
    write_json(selected["status_path"],{**state,"status":status,**extra})


def features(plan,status):
    write_json(plan["features_status"],{"status":status})
    if status=="completed":
        write_json(Path(plan["features"])/"index.json",{"contract":{"dataset_sha256":plan["manifest_sha256"]}})


def run_tick(plan,path,tmux,busy=()):
    return tick(plan,path,tmux=tmux,gpu_probe=lambda:set(busy),alive=alive)


def test_initial_flow_jobs_do_not_wait_for_features_and_repeated_tick_is_idempotent(plan):
    root=Path(plan["root"])
    external={"job_id":"existing_external_feature","status":"running","custom":"preserve","device":"cuda:1"}
    write_json(root/"monitoring/jobs.json",{"owner":"original","jobs":[external]})
    features(plan,"running")
    path=initialize(plan);tmux=MockTmux()
    done,launched=run_tick(plan,path,tmux,busy=["cuda:1"])
    assert not done and launched==["forward0","forward1"]
    assert [row["device"] for row in tmux.launched]==["cuda:0","cuda:2"]
    assert run_tick(plan,path,tmux,busy=["cuda:1"])[1]==[]
    assert len(tmux.launched)==2
    assert not (root/"runs").exists()
    registry=read_json(root/"monitoring/jobs.json")
    assert registry["owner"]=="original"
    assert next(row for row in registry["jobs"] if row["job_id"]==external["job_id"])==external
    for name in launched:
        row=read_json(job(plan,name)["status_path"])
        assert row["launch_count"]==1 and "pid_start_ticks" not in row


def test_flow_progress_and_external_storage_are_registered_without_lifecycle_override(plan):
    from pipeline import copy_worker_progress
    selected = job(plan, "forward0")
    selected.update(storage_paths=["/dataset/nuscenes/flow-cache"], expected_pairs=123)
    register(plan, selected, {"status": "running", "pid": 41})
    row = next(x for x in read_json(Path(plan["root"]) / "monitoring/jobs.json")["jobs"]
               if x["job_id"] == selected["job_id"])
    assert row["progress_path"] == selected["progress_path"]
    assert row["storage_paths"] == selected["storage_paths"] and row["expected_pairs"] == 123
    state = {"status": "running", "pid": 41}
    copy_worker_progress(state, {"status": "complete", "pid": 99, "pairs_processed": 123,
        "total": 123, "target_pairs": 123, "counts": {"created": 100, "reused": 23},
        "elapsed_s": 20., "throughput": 6.15, "pairs_per_second": 6.15,
        "bytes": 4567, "bytes_written": 4000, "cuda_peak_allocated_bytes": 9000}, selected)
    assert state["status"] == "running" and state["pid"] == 41
    assert state["completed"] == state["expected_pairs"] == state["total"] == 123
    assert state["counts"]["reused"] == 23 and state["bytes_written"] == 4000
    assert state["elapsed_s"] == 20. and state["throughput"] == 6.15


def test_full_flow_stage_validates_its_bound_inputs_without_affecting_other_stages(plan, monkeypatch):
    import sys
    from types import SimpleNamespace
    from pipeline import check_planned_inputs
    calls = []
    monkeypatch.setitem(sys.modules, "full_flow_pipeline", SimpleNamespace(validate_inputs=lambda value: calls.append(value)))
    check_planned_inputs(plan, job(plan, "forward0"))
    assert calls == []
    plan["stage"] = "full_nuscenes_flow"
    check_planned_inputs(plan, job(plan, "forward0"))
    assert calls == [plan]
    def rejected(value):
        raise ValueError("Immutable full-flow index changed")
    monkeypatch.setitem(sys.modules, "full_flow_pipeline", SimpleNamespace(validate_inputs=rejected))
    with pytest.raises(ValueError, match="index changed"):
        check_planned_inputs(plan, job(plan, "forward0"))


def test_software_queue_edges_are_soft_and_failed_rgb_does_not_block_flow_only(plan):
    path=initialize(plan);tmux=MockTmux()
    features(plan,"failed")
    for name in ("forward0","forward1","forward_pack"):
        mark(plan,name,"completed")
    _,launched=run_tick(plan,path,tmux)
    assert set(launched)=={"fb0","R02","fb1"}
    for name in ("R00","R01","R03","R04","R05"):
        state=read_json(job(plan,name)["status_path"])
        assert state["status"]=="blocked" and "features" in state["blocked_by"]
    assert "--features" not in job(plan,"R02")["commands"][0]
    assert "--features" not in job(plan,"R02")["commands"][1]


def test_training_failure_releases_soft_order_but_does_not_restart_job(plan):
    path=initialize(plan);tmux=MockTmux();features(plan,"completed")
    for name in ("forward0","forward1","forward_pack"):
        mark(plan,name,"completed")
    mark(plan,"R00","failed")
    mark(plan,"R03","failed")
    assert set(run_tick(plan,path,tmux)[1])=={"fb0","R01","R02"}
    assert run_tick(plan,path,tmux)[1]==[]
    mark(plan,"R01","failed")
    assert run_tick(plan,path,tmux)[1]==["fb1"]
    assert not any(row["name"] in {"R00","R03"} for row in tmux.launched)


def test_real_data_dependency_failure_blocks_only_its_descendants(plan):
    path=initialize(plan);tmux=MockTmux();features(plan,"completed")
    mark(plan,"forward0","failed")
    _,launched=run_tick(plan,path,tmux)
    assert set(launched)=={"forward1","R00"}
    assert read_json(job(plan,"fb0")["status_path"])["status"]=="blocked"
    assert read_json(job(plan,"forward_pack")["status_path"])["blocked_by"]==["forward0"]
    assert read_json(job(plan,"R03")["status_path"])["status"]=="blocked"


def test_gpu_occupancy_prevents_overlap_but_allows_cpu_pack(plan):
    path=initialize(plan);tmux=MockTmux();features(plan,"completed")
    for name in ("forward0","forward1"):
        mark(plan,name,"completed")
    assert run_tick(plan,path,tmux,busy=["cuda:0","cuda:1","cuda:2"])[1]==["forward_pack"]
    assert all(row["device"] is None for row in tmux.launched)
    assert read_json(job(plan,"R00")["status_path"])["status"]=="queued"


def test_existing_run_output_is_preserved_and_not_launched(plan):
    path=initialize(plan);tmux=MockTmux();features(plan,"completed")
    directory=Path(job(plan,"R00")["run_dir"])
    directory.mkdir(parents=True)
    (directory/"user.txt").write_text("do not overwrite")
    _,launched=run_tick(plan,path,tmux)
    assert "R00" not in launched
    assert read_json(job(plan,"R00")["status_path"])["status"]=="failed"
    assert (directory/"user.txt").read_text()=="do not overwrite"


def test_immutable_plan_singleton_lock_and_missing_state_never_restarts(plan):
    path=initialize(plan)
    original=path.read_text()
    initialize(plan)
    changed=copy.deepcopy(plan);changed["features"]+="_changed"
    from purevision_v5.common import object_hash
    changed["identity"]=object_hash({k:v for k,v in changed.items() if k!="identity"})
    with pytest.raises(ValueError,match="different immutable plan"):
        initialize(changed)
    assert path.read_text()==original
    target=Path(job(plan,"forward0")["status_path"]);target.unlink()
    initialize(plan)
    assert not target.exists()  # Never manufacture a terminal worker state from a missing observation.
    assert next(row for row in read_json(Path(plan["root"])/"monitoring/jobs.json")["jobs"]
                if row["job_id"]==job(plan,"forward0")["job_id"])["status"]=="unknown"
    with lock(Path(plan["root"])/"lock"):
        with pytest.raises(BlockingIOError):
            with lock(Path(plan["root"])/"lock"):
                pass


def test_dead_worker_is_terminal_not_automatically_restarted(plan):
    path=initialize(plan);features(plan,"running");tmux=MockTmux()
    mark(plan,"forward0","running",pid=9999,pid_start_ticks=10)
    probe=lambda *args:{"alive":False,"verdict":"dead"}
    _,launched=tick(plan,path,tmux=tmux,gpu_probe=lambda:set(),alive=probe)
    assert "forward0" not in launched
    assert read_json(job(plan,"fb0")["status_path"])["status"]=="queued"
    tick(plan,path,tmux=tmux,gpu_probe=lambda:set(),alive=probe)
    # Worker-owned JSON remains intact; the separate confirmed observation drives dependencies.
    assert read_json(job(plan,"forward0")["status_path"])["status"]=="running"
    assert read_json(job(plan,"fb0")["status_path"])["status"]=="blocked"
    state=read_json(Path(plan["root"])/"pipeline"/plan["prefix"]/"status.json")
    assert state["jobs"]["forward0"]["status"]=="interrupted"


@pytest.mark.parametrize("failure", ["legacy_false", "unknown", "missing", "malformed"])
def test_transient_observation_never_blocks_live_preparation_descendants(plan,failure):
    path=initialize(plan);features(plan,"running");tmux=MockTmux()
    mark(plan,"forward1","running",pid=9999,pid_start_ticks=10)
    target=Path(job(plan,"forward1")["status_path"])
    original=target.read_text()
    if failure=="missing":
        target.unlink()
    elif failure=="malformed":
        target.write_text("{")
    probe=lambda *args:{"alive":False,**({"verdict":"unknown","reason":"read error"} if failure=="unknown" else {})}
    tick(plan,path,tmux=tmux,gpu_probe=lambda:{"cuda:0","cuda:1","cuda:2"},alive=probe)
    assert all(read_json(job(plan,name)["status_path"])["status"]=="queued"
               for name in ("forward_pack","R01","R03","R02","fb1","fb_pack","R04","R05"))
    target.write_text(original)
    run_tick(plan,path,tmux,busy=["cuda:0","cuda:1","cuda:2"])
    assert read_json(job(plan,"forward_pack")["status_path"])["status"]=="queued"


def test_live_child_or_worker_lock_corroborates_worker_survival(plan):
    path=initialize(plan);features(plan,"running");tmux=MockTmux()
    mark(plan,"forward1","running",pid=9999,pid_start_ticks=10,child_pid=8888)
    def probe(pid,*args):
        return {"alive":pid==8888,"verdict":"alive" if pid==8888 else "dead"}
    for _ in range(3):
        tick(plan,path,tmux=tmux,gpu_probe=lambda:{"cuda:0","cuda:1","cuda:2"},alive=probe)
    assert read_json(job(plan,"forward_pack")["status_path"])["status"]=="queued"
    mark(plan,"forward1","running",pid=9999,child_pid=None)
    with lock(Path(plan["root"])/"pipeline/locks"/f"job_{job(plan,'forward1')['job_id']}.lock"):
        for _ in range(3):
            tick(plan,path,tmux=tmux,gpu_probe=lambda:{"cuda:0","cuda:1","cuda:2"},alive=probe)
    assert read_json(job(plan,"forward_pack")["status_path"])["status"]=="queued"


def test_changed_heartbeat_resets_death_confirmation(plan):
    path=initialize(plan);tmux=MockTmux();features(plan,"running")
    mark(plan,"forward1","running",pid=9999,updated_at="first")
    for heartbeat in ("first","second","third"):
        mark(plan,"forward1","running",pid=9999,updated_at=heartbeat)
        tick(plan,path,tmux=tmux,gpu_probe=lambda:{"cuda:0","cuda:1","cuda:2"},
             alive=lambda *args:{"alive":False,"verdict":"dead"})
    assert read_json(job(plan,"forward_pack")["status_path"])["status"]=="queued"


def test_recovery_adopts_exact_running_forward_jobs_without_launch_or_registry_overwrite(plan):
    old_path=initialize(plan)
    mark(plan,"forward0","running",pid=111,launch_count=1)
    mark(plan,"forward1","running",pid=222,launch_count=1)
    register(plan,job(plan,"forward0"),read_json(job(plan,"forward0")["status_path"]))
    old_row=next(row for row in read_json(Path(plan["root"])/"monitoring/jobs.json")["jobs"]
                 if row["job_id"]==job(plan,"forward0")["job_id"])
    new_version={"commit":"b"*40,"dirty":False,"verified_remote_commit":"b"*40}
    recovery=build_plan(plan["root"],plan["manifest"],plan["features"],"test_s42_r1",new_version,preparation_plan=old_path)
    assert len(recovery["jobs"])==10 and len(recovery["external_preparation"])==2
    assert {row["job_id"] for row in recovery["external_preparation"]}=={"test_s42_forward0","test_s42_forward1"}
    assert recovery["preparation_source"]["code"]==plan["code"]
    new_path=initialize(recovery);tmux=MockTmux();features(recovery,"completed")
    assert run_tick(recovery,new_path,tmux,busy=["cuda:0","cuda:2"])[1]==["R00"]
    current=next(row for row in read_json(Path(plan["root"])/"monitoring/jobs.json")["jobs"]
                 if row["job_id"]==old_row["job_id"])
    assert current==old_row
    assert all(row["name"] not in {"forward0","forward1"} for row in tmux.launched)
    mark(plan,"forward0","completed");mark(plan,"forward1","completed")
    assert set(run_tick(recovery,new_path,tmux)[1])=={"fb0","forward_pack"}
    assert all(row["job_id"].startswith("test_s42_r1_") for row in recovery["jobs"])


def test_process_probe_distinguishes_absence_from_read_failure(monkeypatch):
    def inaccessible(*args, **kwargs):
        raise PermissionError("test permission failure")
    monkeypatch.setattr(Path,"read_text",inaccessible)
    assert process_observation(123,1)["verdict"]=="unknown"
    def absent(*args, **kwargs):
        raise FileNotFoundError("test disappearance")
    monkeypatch.setattr(Path,"read_text",absent)
    assert process_observation(123,1)["verdict"]=="dead"


def test_failed_startup_needs_two_observations_and_keeps_worker_status(plan):
    path=initialize(plan);features(plan,"running");tmux=MockTmux()
    mark(plan,"forward1","launching",launched_at_epoch=0,launch_count=1)
    for expected in ("queued","blocked"):
        tick(plan,path,tmux=tmux,gpu_probe=lambda:{"cuda:0","cuda:1","cuda:2"},
             alive=lambda *args:{"alive":None,"verdict":"unknown"})
        assert read_json(job(plan,"forward_pack")["status_path"])["status"]==expected
    assert read_json(job(plan,"forward1")["status_path"])["status"]=="launching"


def test_launch_never_overwrites_worker_first_status(plan):
    path=initialize(plan);features(plan,"running")
    class FastWorker(MockTmux):
        def launch(self,path,plan,selected):
            result=super().launch(path,plan,selected)
            mark(plan,selected["name"],"running",pid=123,pid_start_ticks=456)
            return result
    tmux=FastWorker()
    run_tick(plan,path,tmux,busy=["cuda:1"])
    for name in ("forward0","forward1"):
        state=read_json(job(plan,name)["status_path"])
        assert state["status"]=="running" and state["pid"]==123 and state["pid_start_ticks"]==456


def test_plan_declares_evaluation_and_packing_reference_and_rejects_tampering(plan):
    assert len(plan["jobs"])==12
    for name in ("R00","R01","R02","R03","R04","R05"):
        selected=job(plan,name)
        assert len(selected["commands"])==2
        assert selected["commands"][1][2].endswith("evaluate.py")
        assert "--checkpoint" in selected["commands"][1]
    assert "--reference-pack" in job(plan,"fb_pack")["commands"][0]
    assert job(plan,"R03")["requires"]==["features","forward_pack"]
    validate_plan(plan)
    changed=copy.deepcopy(plan);changed["jobs"][0]["device"]="cuda:1"
    with pytest.raises(ValueError,match="identity"):
        validate_plan(changed)
