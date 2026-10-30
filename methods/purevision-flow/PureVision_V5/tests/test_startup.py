import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from purevision_v5.model import VelocityModel


spec = importlib.util.spec_from_file_location("v5_startup", Path(__file__).resolve().parents[1] / "startup_watch.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


@pytest.fixture
def formal(tmp_path):
    torch.set_num_threads(1)
    code = {"commit": "a" * 40, "dirty": False, "verified_remote_commit": "a" * 40}
    source = tmp_path / "fixed"
    (source / "configs").mkdir(parents=True)
    config = dict(experiment="R00", epochs=3, learning_rate=.001, weight_decay=.0001)
    startup.write(source / "configs/R00.json", config)
    run_dir = tmp_path / "runs/formal_R00"
    run_dir.mkdir(parents=True)
    job = dict(name="R00", job_id="formal_R00", kind="training", run_dir=str(run_dir),
               status_path=str(tmp_path / "wrapper.json"), log_path=str(tmp_path / "train.log"))
    plan = dict(identity="test-plan", prefix="formal", root=str(tmp_path), code=code,
                code_root=str(source), manifest_sha256="dataset-hash", jobs=[job],
                config_hashes={"R00": hashlib.sha256((source / "configs/R00.json").read_bytes()).hexdigest()})
    contract = dict(config=config, data={"dataset_sha256": "dataset-hash"}, train_selection="train", dev_selection="dev",
                    execution={"smoke": False, "max_batches": 0})
    run = dict(run_id=job["job_id"], experiment="R00", smoke=False, code=code, config=config,
               contract=contract, contract_hash=startup.digest(contract), status="running")
    startup.write(job["status_path"], {"status": "running"})
    startup.write(run_dir / "run.json", run)
    startup.write(run_dir / "status.json", {"status": "running", "global_step": 3})
    model = VelocityModel("R00")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    events = []
    for step in range(1, 4):
        optimizer.zero_grad()
        loss = model({"descriptor": torch.ones(2, 384), "pair_dt": torch.full((2, 4), .1)}).square().mean()
        loss.backward()
        optimizer.step()
        events.append(dict(event="step", step=step, epoch=0, loss=loss.item()))
    Path(job["log_path"]).write_text("library warning\n" + "\n".join(map(json.dumps, events)) + "\n")
    validation = dict(epoch=0, global_step=3, train_loss=.1,
                      validation={"overall": dict(count=2, epe=.2, mae_vx=.1, mae_vy=.1)})
    (run_dir / "metrics.jsonl").write_text(json.dumps(validation) + "\n")
    scheduler.step()
    numpy_rng = np.random.get_state()
    checkpoint = dict(schema="v5_training_state_v1", run_id=run["run_id"], code=code, contract=contract,
                      contract_hash=run["contract_hash"], config=config, model=model.state_dict(),
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler={},
                      rng=dict(python=random.getstate(), numpy=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                               torch=torch.get_rng_state(), cuda=[]),
                      epoch=1, next_batch=0, global_step=3, best_epe=.2, best_epoch=0, bad_epochs=0,
                      running_loss=0., running_count=0)
    return SimpleNamespace(root=tmp_path, job=job, plan=plan, run=run, run_dir=run_dir,
                           checkpoint=checkpoint, validation=validation)


def save_checkpoints(formal, value=None):
    for name in ("last.pt", "best.pt"):
        torch.save(formal.checkpoint if value is None else value, formal.run_dir / name)


def test_normal_startup_requires_steps_validation_and_resumable_state(formal):
    pending = startup.inspect(formal.job, formal.plan)
    assert pending["status"] == "pending_checkpoint"
    save_checkpoints(formal)
    proof = startup.inspect(formal.job, formal.plan, pending)
    assert proof["status"] == "verified"
    assert [item["step"] for item in proof["first_steps"]] == [1, 2, 3]
    assert set(proof["checkpoints"]) == {"last.pt", "best.pt"}
    assert all(item["load_device"] == "cpu" and item["sha256"] for item in proof["checkpoints"].values())
    assert not (formal.run_dir / "startup_proof.json").exists()


def test_pending_partial_log_and_missing_run_are_not_failure(formal):
    (formal.run_dir / "run.json").unlink()
    assert startup.inspect(formal.job, formal.plan)["status"] == "pending_run"
    startup.write(formal.run_dir / "run.json", formal.run)
    Path(formal.job["log_path"]).write_text('{"event":"step","step":1,"loss":0.1}\n{"event":"step"')
    proof = startup.inspect(formal.job, formal.plan)
    assert proof["status"] == "pending_steps_or_validation"
    assert len(proof["first_steps"]) == 1


@pytest.mark.parametrize("fault", ["weights_only", "wrong_contract", "wrong_code", "nonfinite", "optimizer_missing", "predates_validation"])
def test_reject_unresumable_or_semantically_wrong_checkpoint(formal, fault):
    bad = copy.deepcopy(formal.checkpoint)
    if fault == "weights_only":
        bad = {"model": bad["model"]}
    elif fault == "wrong_contract":
        bad["contract_hash"] = "other"
    elif fault == "wrong_code":
        bad["code"]["commit"] = "b" * 40
    elif fault == "nonfinite":
        next(iter(bad["model"].values())).flatten()[0] = float("nan")
    elif fault == "optimizer_missing":
        bad["optimizer"]["state"] = {}
    else:
        bad["epoch"] = 0
    save_checkpoints(formal, bad)
    proof = startup.inspect(formal.job, formal.plan)
    assert proof["status"] == "rejected" and proof["errors"]


def test_nonfinite_first_step_smoke_and_failed_runs_cannot_pass(formal):
    Path(formal.job["log_path"]).write_text('{"event":"step","step":1,"loss":NaN}\n')
    assert startup.inspect(formal.job, formal.plan)["status"] == "rejected"
    smoke = {**formal.run, "smoke": True}
    startup.write(formal.run_dir / "run.json", smoke)
    assert "Smoke" in startup.inspect(formal.job, formal.plan)["errors"][0]
    startup.write(formal.job["status_path"], {"status": "failed"})
    assert startup.inspect(formal.job, formal.plan)["status"] == "excluded_terminal_failure"


def test_missing_first_steps_cannot_be_replaced_by_later_logs(formal):
    Path(formal.job["log_path"]).write_text("\n".join(json.dumps(dict(event="step", step=step, loss=.1)) for step in (20, 40, 60)) + "\n")
    proof = startup.inspect(formal.job, formal.plan)
    assert proof["status"] == "rejected"
    assert "actual startup steps 1, 2, 3" in proof["errors"][0]


def test_patrol_lock_collision_retries_and_success_is_not_duplicated(formal):
    save_checkpoints(formal)
    proof = startup.inspect(formal.job, formal.plan)
    path = formal.root / "monitoring/startup/formal_R00/startup_proof.json"
    calls = []
    answer = formal.root / "report.txt"
    answer.write_text("actual report\nASTRA_PATROL_OK\n")
    def runner(command, *, stdout, **kwargs):
        calls.append(command)
        assert command[-2:] == ["startup", str(path)]
        if len(calls) > 1:
            stdout.write(json.dumps(dict(trigger="startup", success=True, actual_shell_read_observed=True,
                                         read_evidence={"verified": True}, run_ids=[formal.job["job_id"]], answer=str(answer))) + "\n")
        return SimpleNamespace(returncode=0)
    proof = startup.trigger_patrol(proof, path, formal.root, Path("/fixed/astra_patrol.sh"), runner)
    assert not proof["patrol"]["success"]
    proof = startup.trigger_patrol(proof, path, formal.root, Path("/fixed/astra_patrol.sh"), runner)
    assert proof["patrol"]["success"]
    startup.trigger_patrol(proof, path, formal.root, Path("/fixed/astra_patrol.sh"), runner)
    assert len(calls) == 2


def test_named_pilot_config_keeps_model_experiment_and_path_boundary(formal):
    job = {**formal.job, "name": "R00_512", "experiment": "R00", "config_path": str(Path(formal.plan["code_root"]) / "configs/R00.json")}
    plan = {**formal.plan, "config_hashes": {"R00_512": formal.plan["config_hashes"]["R00"]}}
    startup.validate_run(formal.run, job, plan)
    job["config_path"] = str(formal.root / "outside.json")
    with pytest.raises(ValueError, match="escaped"):
        startup.validate_run(formal.run, job, plan)


def test_actual_cli_accepts_two_named_pilot_jobs_without_running_astra(formal):
    plan = copy.deepcopy(formal.plan)
    plan["jobs"] = []
    for size in (512, 1024):
        job = {**formal.job, "name": f"R02_{size}", "experiment": "R02", "job_id": f"pilot_R02_{size}",
               "status_path": str(formal.root / f"job_{size}.json"), "run_dir": str(formal.root / "runs" / f"pilot_R02_{size}")}
        startup.write(job["status_path"], {"status": "failed", "reason": "isolated CLI test, no real training"})
        plan["jobs"].append(job)
    plan["identity"] = startup.digest({k: v for k, v in plan.items() if k != "identity"})
    plan_path = formal.root / "plan.json"
    startup.write(plan_path, plan)
    result = subprocess.run([sys.executable, "-B", str(Path(startup.__file__)), "--plan", str(plan_path),
                             "--root", str(formal.root), "--once"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status"] == "completed" and not report["all_startups_verified"]
    assert {j["run_id"] for j in report["jobs"]} == {"pilot_R02_512", "pilot_R02_1024"}


def test_watcher_rejects_duplicate_run_id_or_unsupported_model(formal):
    plan = {**formal.plan, "jobs": [formal.job, {**formal.job, "name": "another", "experiment": "R00"}]}
    with pytest.raises(ValueError, match="unique"):
        startup.validate_training_jobs(plan, 30)
    plan["jobs"] = [{**formal.job, "experiment": "unsupported"}]
    with pytest.raises(ValueError, match="supported"):
        startup.validate_training_jobs(plan, 30)


def test_watcher_records_failed_terminal_without_model_patrol(formal):
    startup.write(formal.job["status_path"], {"status": "blocked"})
    def unexpected(*args, **kwargs):
        raise AssertionError("No model patrol for failed/blocked startup")
    result = startup.tick(formal.plan, formal.root, Path("/fixed/astra_patrol.sh"), runner=unexpected)
    assert result["status"] == "completed" and not result["all_startups_verified"]
    assert Path(result["jobs"][0]["proof_path"]).is_file()


def test_confirmed_effective_interruption_excludes_live_raw_status(formal):
    base = formal.root / "pipeline" / formal.plan["prefix"]
    observation = base / "observations/R00.json"
    raw = startup.read(formal.job["status_path"])
    startup.write(observation, dict(effective_status="interrupted", dead_observations=2,
                  last_worker_state=raw, wrapper={"verdict": "dead"}, child={"verdict": "dead"},
                  tmux_exists=False, worker_lock_free=True))
    pipeline = dict(plan_identity=formal.plan["identity"], jobs={"R00": {
                    "status": "interrupted", "observation_path": str(observation)}})
    startup.write(base / "status.json", pipeline)
    proof = startup.inspect(formal.job, formal.plan)
    assert proof["raw_job_status"] == "running" and proof["job_status"] == "interrupted"
    assert proof["status"] == "excluded_terminal_failure"
    result = startup.tick(formal.plan, formal.root, Path("/fixed/astra_patrol.sh"))
    assert result["status"] == "completed" and not result["all_startups_verified"]
    pipeline["plan_identity"] = "another-plan"
    startup.write(base / "status.json", pipeline)
    ignored = startup.inspect(formal.job, formal.plan)
    assert ignored["job_status"] == "running" and ignored["status"] == "pending_checkpoint"
