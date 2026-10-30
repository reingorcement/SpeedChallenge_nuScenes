"""Exercise the full-cache DAG with real immutable inputs and fake processes."""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import full_flow_pipeline as full
import pipeline
from purevision_v5.common import ROOT, read_json, sha256, write_json
from purevision_v5.full_flow_cache import ADOPTION_SCHEMA
from purevision_v5.full_flow_index import build_index, iter_pairs


VERSION = dict(commit="a" * 40, dirty=False, verified_remote_commit="a" * 40)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    """Six groups include supervised, held-out and unlabelled scenes."""
    monkeypatch.setattr("purevision_v5.full_flow_index.code_version", lambda **kwargs: dict(VERSION))
    raw = tmp_path / "dataset"
    scene_number = 0
    for version, names in (("v1.0-trainval", ["train", "dev", "heldout", "extra"]),
                           ("v1.0-test", ["test-a", "test-b"])):
        scenes, samples, frames = [], [], []
        for scene in names:
            count = scene_number + 3
            scene_number += 1
            scenes.append(dict(token=scene, name=scene))
            samples.append(dict(token=scene + "-sample", scene_token=scene))
            for i in range(count):
                filename = f"samples/CAM_FRONT/{scene}-{i}.jpg"
                path = raw / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture: indexing does not decode images")
                frames.append(dict(token=f"{scene}-{i}", sample_token=scene + "-sample",
                                   calibrated_sensor_token="front", timestamp=1_000_000 + i * 100_000,
                                   filename=filename, height=900, width=1600,
                                   prev=f"{scene}-{i-1}" if i else "",
                                   next=f"{scene}-{i+1}" if i + 1 < count else ""))
        for name, rows in dict(scene=scenes, sample=samples, sample_data=frames,
                               sensor=[dict(token="front", channel="CAM_FRONT")],
                               calibrated_sensor=[dict(token="front", sensor_token="front")]).items():
            write_json(raw / version / f"{name}.json", rows)
    original = tmp_path / "art/data/dataset.json"
    write_json(original, dict(schema="v5_dataset_v1", version="v1.0-trainval", dataroot=str(raw),
                              split_sha256="b" * 64, sequences=[
                                  dict(scene="train", partition="train_core"),
                                  dict(scene="dev", partition="dev"),
                                  dict(scene="heldout", partition="final_system_test")]))
    output = raw / "derived/optical_flow/full1024"
    index = build_index(raw, original, output / "index", versions=["v1.0-test", "v1.0-trainval"],
                        cameras=["CAM_FRONT"])
    adoption = tmp_path / "art/plans/adoption.json"
    write_json(adoption, dict(schema=ADOPTION_SCHEMA, origins=[]))
    # Planning's capacity gate must not depend on the test machine's filesystem.
    monkeypatch.setattr(full.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    return full.build_plan(tmp_path / "art", index, original, output, adoption, "full_fixture", VERSION)


class FakeTmux:
    def __init__(self):
        self.launched = []

    def exists(self, session):
        return any(job["session"] == session for job in self.launched)

    def launch(self, path, plan, job):
        assert Path(path).is_file() and not self.exists(job["session"])
        self.launched.append(job)


def selected(plan, name):
    return next(job for job in plan["jobs"] if job["name"] == name)


def mark(plan, name, status, **extra):
    path = selected(plan, name)["status_path"]
    write_json(path, {**read_json(path), "status": status, **extra})


def tick(plan, path, tmux, busy=()):
    return pipeline.tick(plan, path, tmux=tmux, gpu_probe=lambda: set(busy),
                         alive=lambda *args: dict(alive=True, verdict="alive"))


def test_exact_visual_scope_shards_and_runtime_arguments(plan, monkeypatch):
    pipeline.validate_plan(plan)
    full.validate_inputs(plan)
    assert plan["full_generation_authorized"] and plan["training_authorized"] is False
    assert plan["features"] is plan["features_status"] is None
    assert [job["kind"] for job in plan["jobs"]] == ["flow"] * 3 + ["pack", "verify"]
    assert all(job["run_dir"] is None for job in plan["jobs"])
    flows = plan["jobs"][:3]
    assert [job["device"] for job in flows] == ["cuda:0", "cuda:1", "cuda:2"]
    shard_keys = []
    for i, job in enumerate(flows):
        args = job["commands"][0]
        pairs = list(iter_pairs(plan["manifest"], shard_index=i, shard_count=3))
        shard_keys.extend((a["token"], b["token"]) for a, b, _ in pairs)
        assert job["expected_pairs"] == len(pairs)
        for option, value in (("--device", job["device"]), ("--batch-size", "8"),
                              ("--threads", "4"), ("--shard-index", str(i)), ("--shard-count", "3")):
            assert args[args.index(option) + 1] == value
        assert args[args.index("--report") + 1] == job["progress_path"]
        assert job["fresh_output"] is None  # Existing verified pair packets can resume.
        assert job["storage_paths"] == [plan["output_root"]]
        assert Path(job["progress_path"]).is_relative_to(Path(plan["root"]))
    assert len(shard_keys) == len(set(shard_keys)) == plan["counts"]["pairs"] == 27
    assert sum(job["expected_pairs"] for job in flows) == plan["counts"]["pairs"]
    # Invoke the actual pack CLI parser to check its effective partition default.
    import pack_flow
    calls = []
    monkeypatch.setattr(pack_flow, "pack", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(pack_flow.torch, "set_num_threads", lambda _: None)
    pack = selected(plan, "pack_train_dev")
    monkeypatch.setattr(sys, "argv", pack["commands"][0][2:])
    pack_flow.main()
    assert calls[0][0][0] == plan["data_manifest"]
    assert calls[0][0][1] == str(Path(plan["output_root"]) / "v1.0-trainval/CAM_FRONT")
    assert calls[0][1]["partitions"] == plan["config"]["pack_partitions"] == ["train_core", "dev"]
    assert calls[0][1]["selection_path"] is None and calls[0][1]["require_fb"] is False


def test_actual_scheduler_three_gpus_then_pack_then_verification(plan):
    path = pipeline.initialize(plan)
    tmux = FakeTmux()
    assert tick(plan, path, tmux, busy=["cuda:1"]) == (False, ["flow1024_0", "flow1024_2"])
    assert tick(plan, path, tmux) == (False, ["flow1024_1"])
    assert tick(plan, path, tmux) == (False, [])
    for job in plan["jobs"][:3]:
        mark(plan, job["name"], "completed")
    assert tick(plan, path, tmux, busy=["cuda:0", "cuda:1", "cuda:2"]) == (False, ["pack_train_dev"])
    mark(plan, "pack_train_dev", "completed")
    assert tick(plan, path, tmux) == (False, ["verify_full_cache"])
    assert tick(plan, path, tmux) == (False, [])
    mark(plan, "verify_full_cache", "completed")
    assert tick(plan, path, tmux) == (True, [])
    coordinator = read_json(path.parent / "status.json")
    assert coordinator["status"] == "completed" and coordinator["plan_identity"] == plan["identity"]
    assert len(tmux.launched) == 5


@pytest.mark.parametrize("name", ["pack_train_dev", "verify_full_cache"])
def test_fresh_derived_output_is_never_silently_overwritten(plan, name):
    path = pipeline.initialize(plan)
    target = selected(plan, name)
    for dependency in target["requires"]:
        mark(plan, dependency, "completed")
    output = Path(target["fresh_output"])
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve existing output")
    tmux = FakeTmux()
    tick(plan, path, tmux)
    state = read_json(target["status_path"])
    assert state["status"] == "failed" and "already exists" in state["reason"]
    assert name not in [job["name"] for job in tmux.launched]
    assert sentinel.read_text() == "preserve existing output"


def test_partial_or_sample_report_is_not_a_successful_worker_output(plan):
    job = plan["jobs"][0]
    for status in ("running", "sample_complete", "failed"):
        write_json(job["progress_path"], dict(status=status, pairs_processed=1))
        with pytest.raises(RuntimeError, match="did not finish"):
            pipeline.check_outputs(job)
    write_json(job["progress_path"], dict(status="complete", pairs_processed=job["expected_pairs"]))
    pipeline.check_outputs(job)
    verify = selected(plan, "verify_full_cache")
    assert Path(verify["progress_path"]).parent == Path(verify["fresh_output"])
    assert verify["checks"] == [dict(path=verify["progress_path"], field="status", equals="complete")]


@pytest.mark.parametrize("field", ["manifest", "data_manifest", "adoption"])
def test_worker_checks_immutable_full_flow_dependencies(plan, field):
    path = Path(plan[field])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Immutable"):
        pipeline.check_planned_inputs(plan, plan["jobs"][0])


def test_full_flow_hook_does_not_change_other_stage_input_contract(plan):
    Path(plan["data_manifest"]).write_bytes(Path(plan["data_manifest"]).read_bytes() + b"\n")
    old = copy.deepcopy(plan)
    old.pop("stage")
    pipeline.check_planned_inputs(old, old["jobs"][0])
    with pytest.raises(ValueError, match="Immutable data_manifest"):
        pipeline.check_planned_inputs(plan, plan["jobs"][0])


def test_coordinator_resume_uses_stored_plan_and_never_restarts_terminal_workers(plan, monkeypatch):
    path = pipeline.initialize(plan)
    before = path.read_bytes()
    for job in plan["jobs"][:3]:
        mark(plan, job["name"], "completed")
    mark(plan, "flow1024_1", "failed")
    monkeypatch.setattr(full.shutil, "disk_usage", lambda _: (_ for _ in ()).throw(AssertionError("must not rebuild plan")))
    resumed = full.resume_plan(path, dict(VERSION))
    assert resumed == plan and pipeline.initialize(resumed) == path and path.read_bytes() == before
    tmux = FakeTmux()
    done, launched = tick(resumed, path, tmux)
    if not done:
        done, launched = tick(resumed, path, tmux)
    assert done and not launched and not tmux.launched
    assert read_json(selected(plan, "flow1024_1")["status_path"])["status"] == "failed"
    assert read_json(path.parent / "status.json")["status"] == "finished_with_failures"


@pytest.mark.parametrize("change", ["commit", "dirty", "unverified", "checkout", "copy"])
def test_resume_rejects_changed_runtime_or_copied_plan(plan, tmp_path, change):
    path = pipeline.initialize(plan)
    version, code_root = dict(VERSION), ROOT
    if change == "commit":
        version.update(commit="c" * 40, verified_remote_commit="c" * 40)
    elif change == "dirty":
        version["dirty"] = True
    elif change == "unverified":
        version["verified_remote_commit"] = None
    elif change == "checkout":
        code_root = tmp_path / "other-checkout"
    else:
        copied = tmp_path / "copied-plan.json"
        copied.write_bytes(path.read_bytes())
        path = copied
    with pytest.raises(ValueError, match="Resume"):
        full.resume_plan(path, version, code_root=code_root)


def test_resume_cli_uses_immutable_plan_without_building_new_one(plan, monkeypatch):
    path = pipeline.initialize(plan)
    calls = []
    monkeypatch.setattr(full, "code_version", lambda **kw: (calls.append(kw) or dict(VERSION)))
    monkeypatch.setattr(full, "build_plan", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected rebuild")))
    monkeypatch.setattr(full, "tick", lambda value, actual: (calls.append((value, actual)) or (True, [])))
    monkeypatch.setattr(sys, "argv", ["full_flow_pipeline.py", "--resume-plan", str(path)])
    full.main()
    assert calls[0] == dict(require_pushed=True) and calls[1] == (plan, path)
    monkeypatch.setattr(sys, "argv", ["full_flow_pipeline.py", "--resume-plan", str(path), "--dry-run"])
    with pytest.raises(SystemExit):
        full.main()


@pytest.fixture
def completed_cache(plan, monkeypatch):
    """Real coverage/byte fixtures; tensor/mmap validation has separate tests."""
    import collect_full_flow as collector
    from purevision_v5.flow import CONFIGURABLE_SCHEMA, cache_path, derived_contract, native_contract, native_path
    path = pipeline.initialize(plan)
    expected_pack = []
    packed_pairs = []
    for i, job in enumerate(plan["jobs"][:3]):
        pairs = list(iter_pairs(plan["manifest"], shard_index=i, shard_count=3))
        receipts = []
        for source, target, group in pairs:
            compact = cache_path(Path(plan["output_root"]) / group["version"] / group["camera"],
                                 source["token"], target["token"], schema=CONFIGURABLE_SCHEMA)
            compact.parent.mkdir(parents=True, exist_ok=True)
            compact.write_bytes(b"compact fixture")
            native_path(compact).write_bytes(b"native fixture")
            receipts.append(dict(source=source["token"], target=target["token"],
                                 **{key: group[key] for key in ("version", "camera", "scene")},
                                 outcome="created", compact_sha256=sha256(compact),
                                 native_sha256=sha256(native_path(compact)), bytes=29,
                                 generation_code=dict(VERSION)))
            if group["partition"] in {"train_core", "dev"}:
                expected_pack.append((source["token"], target["token"], source["time_s"], target["time_s"]))
                packed_pairs.append(dict(source=source["token"], target=target["token"],
                                         source_time=source["time_s"], target_time=target["time_s"],
                                         source_sha256=sha256(compact)))
        digest = hashlib.sha256("".join(f"{a['token']} {b['token']}\n" for a, b, _ in pairs).encode()).hexdigest()
        report = dict(status="complete", scope="full_dataset_generation", code=dict(VERSION),
                      index_identity=plan["index_identity"], index_sha256=plan["manifest_sha256"],
                      index=plan["manifest"], flow_py_sha256=sha256(ROOT / "purevision_v5/flow.py"),
                      arguments=dict(limit=0, shard_index=i, shard_count=3, batch_size=8, threads=4,
                                     device=job["device"], output_root=plan["output_root"], index=plan["manifest"],
                                     index_sha256=plan["manifest_sha256"], adoption=plan["adoption"]),
                      adoption=dict(sha256=plan["adoption_sha256"]), native_contract=native_contract((576, 1024)),
                      derived_contract=derived_contract((576, 1024)), pairs_processed=len(pairs),
                      expected_pairs=len(pairs), target_pairs=len(pairs), total=len(pairs), remaining_pairs=0,
                      all_total=plan["counts"]["pairs"], counts=dict(created=len(pairs), reused=0, recovered=0, adopted=0),
                      processed_pair_sha256=digest, bytes=29 * len(pairs), bytes_written=29 * len(pairs),
                      sample_receipts=receipts[:3], last_receipts=receipts[-3:], elapsed_s=10.,
                      started_at_utc="2026-10-30T00:00:00Z", finished_at_utc="2026-10-30T00:00:10Z",
                      cuda_peak_allocated_bytes=0)
        write_json(job["progress_path"], report)
        mark(plan, job["name"], "completed", exit_code=0, commit=VERSION["commit"])
    mark(plan, "pack_train_dev", "completed", exit_code=0, commit=VERSION["commit"])
    mark(plan, "verify_full_cache", "running", exit_code=None, commit=VERSION["commit"])
    pack_root = Path(plan["packed_train_dev"])
    pack_root.mkdir(parents=True)
    (pack_root / "flow.npy").write_bytes(b"array fixture")
    contract = dict(partitions=["train_core", "dev"], fb_available=False,
                    source_root=str(Path(plan["output_root"]) / "v1.0-trainval/CAM_FRONT"))
    fake_pack = SimpleNamespace(index=dict(code=dict(VERSION), contract=contract, pairs=packed_pairs,
        arrays=dict(flow=dict(file="flow.npy", sha256=sha256(pack_root / "flow.npy")))),
        index_sha256="f" * 64, identity="e" * 64,
        flow_contract=dict(schema=CONFIGURABLE_SCHEMA, native_contract=native_contract((576, 1024)),
                           derived_contract=derived_contract((576, 1024))))
    monkeypatch.setattr(collector, "PackedFlow", lambda actual, **kwargs: fake_pack)
    monkeypatch.setattr(collector, "_manifest_pairs", lambda actual, partitions: expected_pack)
    def verify_receipt(verifier, row, receipt):
        source, target, group = row
        assert (receipt["source"], receipt["target"], receipt["scene"]) == (source["token"], target["token"], group["scene"])
        return dict(receipt)
    monkeypatch.setattr(collector, "verify_receipt", verify_receipt)
    return plan, path, fake_pack


def test_collector_finishes_while_its_own_worker_is_running_and_reports_progress(completed_cache, monkeypatch):
    import collect_full_flow as collector
    plan, path, _ = completed_cache
    own = selected(plan, "verify_full_cache")
    output = Path(own["fresh_output"])
    inspect = collector.inspect
    observations = []
    def inspected(value, progress):
        initial = read_json(own["progress_path"])
        assert initial["status"] == "running" and initial["plan_identity"] == plan["identity"]
        assert read_json(own["status_path"])["status"] == "running"
        def observed(**values):
            progress(**values)
            observations.append(read_json(own["progress_path"]))
        return inspect(value, observed)
    monkeypatch.setattr(collector, "inspect", inspected)
    report = collector.collect(path, output)
    assert report["status"] == "complete", report.get("error")
    assert report["counts"]["pairs"] == 27 and report["outcomes"]["created"] == 27
    assert report["cache_bytes"] == 27 * 29 and report["generation_wall_pairs_per_second"] == 2.7
    assert report["model_training_started"] is False and report["model_metrics"] is None
    assert report["coverage"]["partitions"]["official_test_cache_only"]["pairs"] == 13
    assert report["packed_train_dev"]["pairs"] == 5
    assert any(item["phase"] == "hashing_packed_arrays" for item in observations)
    assert all(item["status"] == "running" for item in observations)
    assert read_json(own["status_path"])["status"] == "running"  # Only worker owns its lifecycle.
    pipeline.check_outputs(own)
    assert "no model-training" in (output / "summary.md").read_text()
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        collector.collect(path, output)
    assert (output / "summary.json").read_bytes() == before


@pytest.mark.parametrize("corruption", ["digest", "bytes", "sample_status", "heldout_pack", "worker_not_done"])
def test_collector_records_failure_and_never_certifies_partial_or_changed_cache(completed_cache, corruption):
    import collect_full_flow as collector
    plan, path, packed = completed_cache
    job = plan["jobs"][0]
    report = read_json(job["progress_path"])
    if corruption == "digest":
        report["processed_pair_sha256"] = "0" * 64
    elif corruption == "bytes":
        report["bytes"] += 1
    elif corruption == "sample_status":
        report["status"] = "sample_complete"
    elif corruption == "heldout_pack":
        packed.index["contract"]["partitions"].append("final_system_test")
    else:
        mark(plan, "flow1024_0", "running")
    write_json(job["progress_path"], report)
    own = selected(plan, "verify_full_cache")
    result = collector.collect(path, own["fresh_output"])
    assert result["status"] == "failed" and result["error"].startswith("ValueError:")
    assert read_json(own["progress_path"])["status"] == "failed"
    assert "Failure:" in (Path(own["fresh_output"]) / "summary.md").read_text()
    with pytest.raises(RuntimeError, match="did not finish"):
        pipeline.check_outputs(own)


def test_collector_packet_verifier_checks_real_cpu_packets_without_constructing_raft(plan, tmp_path, monkeypatch):
    import collect_full_flow as collector
    from PIL import Image
    import torch
    from purevision_v5.flow import FrozenRaft, native_path, save_pair
    from purevision_v5.full_flow_cache import load_adoption_plan
    def forbidden(*args, **kwargs):
        raise AssertionError("Final verification must not construct a RAFT network")
    monkeypatch.setattr(FrozenRaft, "__init__", forbidden)
    group = dict(version="v1.0-trainval", camera="CAM_FRONT", scene="train")
    frames = []
    for i in range(2):
        image = tmp_path / f"verify{i}.jpg"
        Image.new("RGB", (16, 9), (25 + i, 40, 50)).save(image)
        frames.append(dict(token=f"verify{i}", filename=str(image), time_s=1. + .1 * i,
                           dataset="nuScenes", **group))
    verifier = collector.packet_verifier(plan, load_adoption_plan(plan["adoption"]))
    compact = verifier.path(*frames, group)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        save_pair(compact, torch.zeros(2, 576, 1024), None, verifier.engine.provenance(*frames))
        validated = verifier._verify(compact, *frames, recover=False)
        receipt = dict(validated, source=frames[0]["token"], target=frames[1]["token"],
                       **group, outcome="created")
        result = collector.verify_receipt(verifier, (*frames, group), receipt)
        assert result["generation_code"] == VERSION and result["compact_sha256"] == sha256(compact)
        assert result["native_sha256"] == sha256(native_path(compact))
        broken = dict(receipt, compact_sha256="0" * 64)
        with pytest.raises(ValueError, match="checksum mismatch"):
            collector.verify_receipt(verifier, (*frames, group), broken)
    finally:
        torch.set_num_threads(previous)
