"""Reporting must never turn smoke, partial, or incomparable data into a gain."""
import copy
import csv
from pathlib import Path

import pytest
import torch

from collect_results import collect, compare, export, markdown
from pipeline import build_plan
from purevision_v5.common import ROOT, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import summarize
from purevision_v5.labels import LABEL_CONTRACT, LABEL_SHA256


@pytest.fixture
def records(tmp_path):
    root, code = tmp_path / "artifacts", tmp_path / "repo/methods/purevision-flow/PureVision_V5"
    data = root / "data"
    predictions, sequences = [], []
    for scene_index in range(5):
        scene = f"scene-{scene_index}"
        frames = [dict(token=f"{scene}-{i}", time_s=float(i), target=[5., .25]) for i in range(24)]
        sequence = dict(schema="v5_sequence_v1", scene=scene, partition="dev", log_token=scene,
                        frames=frames, windows=list(range(4, 24)))
        path = data / f"{scene}.json"
        write_json(path, sequence)
        sequences.append(dict(scene=scene, partition="dev", log_token=scene, file=path.name, sha256=sha256(path)))
        for frame in frames[4:]:
            predictions.append(dict(scene=scene, token=frame["token"], time_s=frame["time_s"],
                                    vx=6., vy=.25, target_vx=5., target_vy=.25))
    manifest = data / "dataset.json"
    write_json(manifest, dict(schema="v5_dataset_v1", seq_len=5, label_contract=LABEL_CONTRACT,
                             label_sha256=LABEL_SHA256, sequences=sequences))
    for i in range(6):
        write_json(code / f"configs/R{i:02d}.json", read_json(ROOT / f"configs/R{i:02d}.json"))
    version = dict(commit="a" * 40, dirty=False, verified_remote_commit="a" * 40)
    plan = build_plan(root, manifest, "features/rgb", "test", version, code_root=code)
    plan_path = root / "pipeline/test/plan.json"
    write_json(plan_path, plan)
    feature = dict(schema="v5_rgb_features_v1", dataset_sha256=sha256(manifest), weights_sha256="w" * 64)
    write_json(Path(plan["features"]) / "index.json", dict(contract=feature, identity=object_hash(feature)))
    pairs = [dict(source="s", target="t", source_time=1., target_time=2., source_sha256="c" * 64)]
    packed = {}
    for name, fb in (("forward_packed", False), ("bidirectional_packed", True)):
        path = root / "flows" / name / "packed_index.json"
        index = dict(schema="v5_packed_flow_v1", contract=dict(dataset_sha256=sha256(manifest), fb_available=fb), pairs=pairs)
        if fb:
            index["forward_equality"] = dict(reference_index_sha256=sha256(packed["forward_packed"]),
                                             compared_pairs=1, flow_and_valid_exact=True)
        index["identity"] = object_hash(index)
        write_json(path, index)
        packed[name] = path

    def complete(name, *, vx_error=1., smoke=False):
        job = next(j for j in plan["jobs"] if j["name"] == name)
        output = Path(job["run_dir"])
        evaluation = Path(job["commands"][1][-1])
        config = read_json(code / f"configs/{name}.json")
        flow_path = packed["bidirectional_packed" if name in {"R04", "R05"} else "forward_packed"]
        data_contract = dict(dataset_sha256=sha256(manifest), label=LABEL_CONTRACT,
                             features=None if name == "R02" else feature,
                             flow_root=None if name == "R00" else str(flow_path.parent),
                             flow=None if name == "R00" else dict(packed_index_sha256=sha256(flow_path)))
        contract = dict(config=config, data=data_contract, train_selection="train", dev_selection="dev",
                        execution=dict(smoke=smoke, max_batches=0))
        run = dict(run_id=output.name, experiment=name, status="completed", config=config,
                   contract=contract, contract_hash=object_hash(contract), code=version, smoke=smoke,
                   runtime=dict(features=plan["features"]), dev_windows=100, trainable_parameters=75074)
        write_json(output / "run.json", run)
        write_json(output / "status.json", dict(status="completed", best_epoch=2, global_step=20))
        write_json(job["status_path"], dict(status="completed"))
        packet = dict(schema="v5_training_state_v1", contract=contract, contract_hash=object_hash(contract),
                      config=config, code=version, run_id=output.name, best_epoch=2, model={"weight": torch.ones(1)})
        torch.save(packet, output / "best.pt")
        rows = [dict(r, vx=5. + vx_error) for r in predictions]
        write_json(evaluation / "metrics.json", summarize([[r["vx"], r["vy"]] for r in rows],
                                                          [[r["target_vx"], r["target_vy"]] for r in rows],
                                                          [r["scene"] for r in rows]))
        with (evaluation / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(evaluation / "evaluation.json", dict(status="completed", checkpoint=str(output / "best.pt"),
                   checkpoint_sha256=sha256(output / "best.pt"), training_run=output.name,
                   training_code=version, evaluation_code=version, contract_hash=object_hash(contract),
                   data=data_contract, selection_hash="dev", count=100))
        return job, output, evaluation

    return plan, plan_path, complete


def test_complete_metrics_comparisons_and_external_snapshot(records, tmp_path):
    _, path, complete = records
    complete("R00")
    complete("R01", vx_error=.5)
    report = export(path, tmp_path / "reports/snapshot1")
    base, candidate = report["rows"][:2]
    assert len(report["rows"]) == 6
    assert base["metrics"]["overall"]["epe"] == 1.
    assert base["metrics"]["overall"]["zero_vy_mae"] == .25
    assert base["best_epoch"] == 3
    assert candidate["vs_R00"]["improvement"] is True
    assert candidate["vs_R00"]["lateral_acceptance"]["acceptable"] is True
    assert report["rows"][2]["status"] == "not_started"
    assert (tmp_path / "reports/snapshot1/results.json").is_file()
    assert "constant-zero lateral predictor" in markdown(report)
    with pytest.raises(ValueError, match="new report directory"):
        export(path, tmp_path / "reports/snapshot1")


@pytest.mark.parametrize("kind,expected", [("smoke", "excluded_smoke"), ("failed", "failed"),
                                           ("pending", "evaluation_pending")])
def test_smoke_failure_and_pending_never_expose_formal_metrics(records, kind, expected):
    _, path, complete = records
    job, output, evaluation = complete("R00", smoke=kind == "smoke")
    if kind == "failed":
        write_json(job["status_path"], dict(status="failed", error="evaluation crashed"))
    elif kind == "pending":
        (evaluation / "evaluation.json").unlink()
    row = collect(path)["rows"][0]
    assert row["status"] == expected and row["metrics"] is None


@pytest.mark.parametrize("kind", ["data", "targets", "metrics", "checkpoint"])
def test_mismatched_data_or_changed_results_are_rejected(records, kind):
    _, path, complete = records
    _, output, evaluation = complete("R00")
    if kind == "data":
        value = read_json(output / "run.json")
        value["contract"]["data"]["dataset_sha256"] = "different"
        value["contract_hash"] = object_hash(value["contract"])
        write_json(output / "run.json", value)
    elif kind == "targets":
        csv_path = evaluation / "predictions.csv"
        csv_path.write_text(csv_path.read_text().replace(",5.0,0.25", ",5.5,0.25", 1))
    elif kind == "metrics":
        value = read_json(evaluation / "metrics.json")
        value["overall"]["epe"] = .001
        write_json(evaluation / "metrics.json", value)
    else:
        with (output / "best.pt").open("ab") as stream:
            stream.write(b"changed")
    row = collect(path)["rows"][0]
    assert row["status"] == "invalid" and row["metrics"] is None


def test_comparison_guards_and_insufficient_lateral_data(records):
    _, path, complete = records
    complete("R00")
    complete("R01", vx_error=.5)
    base, candidate = collect(path)["rows"][:2]
    for key in ("common_config_identity", "target_window_identity", "rgb_identity"):
        changed = copy.deepcopy(candidate)
        changed[key] = "different"
        comparison = compare(changed, base)
        assert comparison["status"] == "incomparable" and comparison["improvement"] is None
    candidate["metrics"]["nonzero_vy"]["adequate"] = False
    comparison = compare(candidate, base)
    assert comparison["conclusion"] == "insufficient_data" and comparison["improvement"] is None


def test_fb_contrasts_require_same_cache_and_matching_forward_proof(records):
    _, path, complete = records
    for name in ("R03", "R04", "R05"):
        complete(name)
    report = collect(path)
    assert [c["status"] for c in report["flow_contrasts"][1:]] == ["comparable"] * 3
    rows = {r["experiment"]: r for r in report["rows"]}
    changed = copy.deepcopy(rows["R05"])
    changed["data"]["flow"]["packed_index_sha256"] = "wrong"
    assert compare(changed, rows["R04"], flow_check=True)["status"] == "incomparable"
    fb_path = Path(rows["R04"]["data"]["flow_root"]) / "packed_index.json"
    index = read_json(fb_path)
    index["forward_equality"]["reference_index_sha256"] = "unrelated"
    index["identity"] = object_hash({k: v for k, v in index.items() if k != "identity"})
    write_json(fb_path, index)
    changed = copy.deepcopy(rows["R04"])
    changed["data"]["flow"]["packed_index_sha256"] = sha256(fb_path)
    comparison = compare(changed, rows["R03"], flow_check=True)
    assert comparison["status"] == "incomparable" and "proof" in comparison["reason"]


def test_manifest_changes_abort_collection_and_output_cannot_touch_runs(records):
    plan, path, complete = records
    _, output, _ = complete("R00")
    with pytest.raises(ValueError, match="external"):
        export(path, output / "reports")
    manifest_path = Path(plan["manifest"])
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest checksum"):
        collect(path)


@pytest.mark.parametrize("matching", [True, False])
def test_confirmed_coordinator_failure_keeps_raw_worker_evidence(records, matching):
    plan, path, complete = records
    job, _, evaluation = complete("R00")
    write_json(job["status_path"], dict(status="running"))
    (evaluation / "evaluation.json").unlink()
    write_json(path.parent / "status.json", dict(plan_identity=plan["identity"] if matching else "another-plan",
        jobs={"R00": dict(status="interrupted", reason="Confirmed absent worker/child/tmux; lock released",
                          observation_path=str(path.parent / "observations/R00.json"))}))
    row = collect(path)["rows"][0]
    assert row["observed_status"]["job"] == "running" and row["metrics"] is None
    assert row["status"] == ("failed" if matching else "evaluation_pending")
    assert row["observed_status"]["coordinator"] == ("interrupted" if matching else None)
