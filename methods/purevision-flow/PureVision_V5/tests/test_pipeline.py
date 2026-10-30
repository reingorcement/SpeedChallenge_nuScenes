"""Scheduling decisions are tested without launching tmux, CUDA or training."""
import copy
import json
from pathlib import Path

import pytest

from pipeline import (build_plan, dependency_state, initialize, lock, read_json,
                      register, tick, validate_plan, write_json)


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
        assert row["launch_count"]==1 and row["pid_start_ticks"]==12345


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
    assert read_json(target)["status"]=="interrupted"
    with lock(Path(plan["root"])/"lock"):
        with pytest.raises(BlockingIOError):
            with lock(Path(plan["root"])/"lock"):
                pass


def test_dead_worker_is_terminal_not_automatically_restarted(plan):
    path=initialize(plan);features(plan,"running");tmux=MockTmux()
    mark(plan,"forward0","running",pid=9999,pid_start_ticks=10)
    _,launched=tick(plan,path,tmux=tmux,gpu_probe=lambda:set(),alive=lambda *args:{"alive":False})
    assert "forward0" not in launched
    assert read_json(job(plan,"forward0")["status_path"])["status"]=="interrupted"


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
