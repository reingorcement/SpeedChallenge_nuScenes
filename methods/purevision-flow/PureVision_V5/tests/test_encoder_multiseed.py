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
