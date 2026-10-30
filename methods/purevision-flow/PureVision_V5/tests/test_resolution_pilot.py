"""Pilot dependency and scope checks run without actual CUDA/tmux jobs."""
import pytest

from pipeline import check_planned_inputs, dependency_state, initialize, tick, write_json
from purevision_v5.common import ROOT
from purevision_v5.selection import make_selection
from resolution_pilot import build_plan
from test_pipeline import MockTmux, alive
from test_training import make_case


@pytest.fixture
def pilot(tmp_path):
    args = make_case(tmp_path / "data")
    selection = tmp_path / "selection.json"
    write_json(selection, make_selection(args.manifest, 1, 1))
    flow = tmp_path / "old_flow"
    write_json(flow / "packed_index.json", {"fixture": True})
    version = dict(commit="a" * 40, dirty=False, verified_remote_commit="a" * 40)
    return build_plan(tmp_path / "art", args.manifest, selection, flow, "pilot_s42", version)


def test_low_training_does_not_wait_for_high_generation(pilot):
    path = initialize(pilot)
    tmux = MockTmux()
    done, launched = tick(pilot, path, tmux=tmux, gpu_probe=lambda: set(), alive=alive)
    assert not done and set(launched) == {"R02_512", "flow1024_0", "flow1024_1"}
    assert len({j["device"] for j in tmux.launched}) == 3
    assert pilot["features_status"] is None and not pilot["full_generation_authorized"]
    for job in pilot["jobs"]:
        for command in job["commands"]:
            assert "--selection" in command
            assert "--features" not in command and "bidirectional" not in command


def test_high_training_requires_both_shards_and_pack(pilot):
    states = {j["name"]: {"status": "queued"} for j in pilot["jobs"]}
    states["R02_512"]["status"] = "completed"
    states["flow1024_0"]["status"] = "completed"
    pack = next(j for j in pilot["jobs"] if j["name"] == "pack1024")
    high = next(j for j in pilot["jobs"] if j["name"] == "R02_1024")
    assert dependency_state(pack, states)[0] == "waiting"
    assert dependency_state(high, states)[0] == "waiting"
    states["flow1024_1"]["status"] = "completed"
    assert dependency_state(pack, states)[0] == "ready"
    states["pack1024"]["status"] = "completed"
    assert dependency_state(high, states)[0] == "ready"
    assert high["experiment"] == "R02" and high["config_path"].startswith(str(ROOT))


def test_oversized_selection_cannot_launch_full_generation(pilot, monkeypatch):
    import resolution_pilot
    selection = dict(pilot["selection"], counts={"train_core": {"scenes": 582}, "dev": {"scenes": 103}})
    monkeypatch.setattr(resolution_pilot, "load_selection", lambda *a: selection)
    with pytest.raises(ValueError, match="capped"):
        build_plan(pilot["root"], pilot["manifest"], pilot["selection_path"], pilot["low_flow_root"],
                   "too_big", pilot["code"])


def test_changed_external_selection_is_rejected_before_worker_command(pilot):
    from pathlib import Path
    job = next(j for j in pilot["jobs"] if j["name"] == "R02_512")
    check_planned_inputs(pilot, job)
    Path(pilot["selection_path"]).write_text('{"a":"different but parseable selection"}\n')
    with pytest.raises(ValueError, match="selection changed"):
        check_planned_inputs(pilot, job)
