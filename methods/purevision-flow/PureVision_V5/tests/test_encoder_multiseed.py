import json
from pathlib import Path
import pytest

from encoder_lifecycle import startup_gate
from encoder_multiseed import numerical_identity
from purevision_v5.common import ROOT


def test_four_seed_configs_only_change_seed_and_explicit_head_seed():
    for seed in (43,44):
        for name in ("BASE","FLEX"):
            old=json.loads((ROOT/f"configs/next/R02_{name}.json").read_text())
            new=json.loads((ROOT/f"configs/next/R02_{name}_s{seed}.json").read_text())
            assert new==dict(old,seed=seed,head_seed=seed)


def test_initial_wave_gate_does_not_claim_queued_runs_verified():
    plan=dict(jobs=[dict(name=n,job_id=n,kind="training") for n in ("b43","f43","b44","f44")],
              startup_gate_jobs=["b43","b44"])
    proofs=[dict(run_id=n,status="verified" if n.startswith("b") else "pending") for n in ("b43","f43","b44","f44")]
    assert startup_gate(plan,proofs)
    assert not all(p["status"]=="verified" for p in proofs)
    proofs[2]["status"]="invalid"
    assert not startup_gate(plan,proofs)
    plan["startup_gate_jobs"]=["not_a_run"]
    with pytest.raises(ValueError):startup_gate(plan,proofs)


def test_numerical_identity_binds_imported_flex_modules():
    identity=numerical_identity(ROOT)
    assert "purevision_v5/_rgb_reference/modules.py" in identity["files"]
    assert "purevision_v5/training.py" in identity["files"]


def test_light_startup_does_not_wait_for_validation(tmp_path,monkeypatch):
    import encoder_lifecycle as lifecycle
    status=tmp_path/"worker.json";log=tmp_path/"run.log"
    status.write_text(json.dumps(dict(status="running",child_pid=123,child_pid_start_ticks=456)))
    log.write_text("training process started\n")
    plan=dict(jobs=[dict(name="BASE",job_id="BASE",kind="training",status_path=str(status),log_path=str(log))])
    monkeypatch.setattr(lifecycle,"process_observation",lambda *args:dict(alive=True))
    assert lifecycle.lightweight_startup(plan)  # No dev metric/checkpoint exists.
    monkeypatch.setattr(lifecycle,"process_observation",lambda *args:dict(alive=False))
    assert not lifecycle.lightweight_startup(plan)


def test_tmux_worker_loads_private_network_independently_of_server_env(tmp_path,monkeypatch):
    import subprocess
    from pipeline import Tmux
    calls=[]
    def fake_run(args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,stdout="123\n")
    monkeypatch.setattr(subprocess,"run",fake_run)
    tmux=Tmux();monkeypatch.setattr(tmux,"exists",lambda _:False)
    plan=dict(root=str(tmp_path),python="/python",code_root="/code",worker_network_file=str(tmp_path/"monitoring/network.json"))
    job=dict(name="R02",session="run",log_path=str(tmp_path/"run.log"))
    assert tmux.launch(tmp_path/"plan.json",plan,job)==123
    command=calls[0][-1]
    assert "scripts/secure_exec.py --network" in command
    assert str(tmp_path/"monitoring/network.json") in command
    assert "-- /python -B /code/pipeline.py --worker" in command
    plan["worker_network_file"]="/another/private.json"
    with pytest.raises(ValueError):tmux.launch(tmp_path/"plan.json",plan,job)
