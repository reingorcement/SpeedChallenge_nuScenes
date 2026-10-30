import copy
import csv
import json
import shutil
from pathlib import Path

import pytest
import torch

from collect_resolution import collect, compare_rows, markdown
from purevision_v5.common import ROOT, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import summarize
from purevision_v5.flow import PREPROCESSING, RAFT_SHA256, SCHEMA, CONFIGURABLE_SCHEMA, native_contract, derived_contract
from purevision_v5.labels import LABEL_CONTRACT
from purevision_v5.selection import label_rows, make_selection
from purevision_v5.training import epoch_order
from resolution_pilot import COMPUTATION_FILES, build_plan, collect_finished_plan, validate_adopted_source
from test_training import make_case


@pytest.fixture
def pilot_records(tmp_path):
    args = make_case(tmp_path / "data", experiment="R02")
    root, code = tmp_path / "artifacts", tmp_path / "code"
    version = dict(commit="a"*40, dirty=False, verified_remote_commit="a"*40)
    selection_path = root / "selection.json"
    selection = make_selection(args.manifest, 1, 1)
    write_json(selection_path, selection)
    for width in (512, 1024):
        write_json(code / f"configs/next/R02_raft{width}.json", read_json(ROOT / f"configs/next/R02_raft{width}.json"))
    for name in COMPUTATION_FILES:
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, code / name)
    low = root / "low"

    def pack(path, width):
        flow = (dict(schema=SCHEMA, raft_sha256=RAFT_SHA256, preprocessing=PREPROCESSING) if width == 512 else
                dict(schema=CONFIGURABLE_SCHEMA, raft_sha256=RAFT_SHA256,
                     native_contract=native_contract((576, 1024)), derived_contract=derived_contract((576, 1024))))
        contract = {k: v for k, v in flow.items() if k != "schema"}
        contract.update(flow_schema=flow["schema"], dataset_sha256=sha256(args.manifest), fb_available=False)
        write_json(path / "array.json", {})
        index = dict(schema="v5_packed_flow_v1", contract=contract, code=version, pairs=[], arrays={"flow": {"file": "array.json"}})
        index["identity"] = object_hash(index)
        write_json(path / "packed_index.json", index)
        return dict(flow, packed_index_sha256=sha256(path / "packed_index.json"))

    contracts = {512: pack(low, 512)}
    plan = build_plan(root, args.manifest, selection_path, low, "test", version, code_root=code)
    high = root / "flows/test_raft1024_packed"
    contracts[1024] = pack(high, 1024)
    plan_path = root / "pipeline/test/plan.json"
    write_json(plan_path, plan)
    train, _ = label_rows(args.manifest, "train_core", selection_path)
    dev, _ = label_rows(args.manifest, "dev", selection_path)

    def complete(width):
        job = next(j for j in plan["jobs"] if j["name"] == f"R02_{width}")
        output, evaluation = Path(job["run_dir"]), Path(job["commands"][1][-1])
        config = read_json(job["config_path"])
        data = dict(dataset_sha256=sha256(args.manifest), label=LABEL_CONTRACT, features=None,
                    scene_selection=selection, flow_root=str(low if width == 512 else high), flow=contracts[width])
        contract = dict(config=config, data=data, train_selection="train", dev_selection="dev",
                        execution=dict(smoke=False, max_batches=0, scope="resolution_pilot", scene_selection_identity=selection["identity"]))
        initialization = dict(policy="seeded_fresh_model_v1", seed=42, model_state_sha256="0"*64)
        sample_order = dict(policy="torch_randperm_seed_plus_zero_based_epoch_v1", seed=42,
                            epoch_index_dtype="little_endian_int64", training_samples_sha256=object_hash([
                                [r["scene"], r["token"], r["time_s"], [r["target_vx"], r["target_vy"]]] for r in train]))
        run = dict(run_id=output.name, experiment="R02", status="completed", config=config, contract=contract,
                   contract_hash=object_hash(contract), code=version, smoke=False, scope="resolution_pilot", runtime={},
                   train_windows=len(train), dev_windows=len(dev), trainable_parameters=74354,
                   initialization=initialization, sample_order=sample_order, effective_lateral_rule=config["lateral_rule"])
        write_json(output / "run.json", run)
        rows = [dict(r, vx=r["target_vx"] + (1. if width == 512 else .8), vy=r["target_vy"]) for r in dev]
        metrics = summarize([[r["vx"], r["vy"]] for r in rows], [[r["target_vx"], r["target_vy"]] for r in rows], [r["scene"] for r in rows], config["lateral_rule"])
        epe = metrics["overall"]["epe"]
        counters = dict(global_step=40, processed_batches=40, optimizer_updates=40, amp_skips=0)
        write_json(output / "status.json", dict(status="completed", epoch=40, stop_reason="epoch_budget", best_epoch=39,
                                                best_epe=epe, elapsed_seconds=1., **counters))
        packet = dict(schema="v5_training_state_v1", contract=contract, contract_hash=run["contract_hash"], config=config,
                      code=version, run_id=output.name, initialization=initialization, sample_order=sample_order,
                      model={"w": torch.ones(1)}, optimizer={}, scheduler={}, scaler={}, rng={}, epoch=40, next_batch=0,
                      best_epoch=39, best_epe=epe, **counters)
        torch.save(packet, output / "best.pt")
        torch.save(packet, output / "last.pt")
        history = []
        for epoch in range(40):
            summary = copy.deepcopy(metrics)
            summary["overall"]["epe"] += .01 * (39-epoch)
            history.append(dict(epoch=epoch, sample_order_sha256=epoch_order(len(train), 42, epoch)[1],
                                global_step=epoch+1, processed_batches=epoch+1, optimizer_updates=epoch+1, amp_skips=0,
                                validation=summary))
        (output / "metrics.jsonl").write_text("".join(json.dumps(h)+"\n" for h in history))
        write_json(output / "best_metrics.json", metrics)
        write_json(evaluation / "metrics.json", metrics)
        with (evaluation / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(evaluation / "evaluation.json", dict(status="completed", scope="resolution_pilot", training_run=output.name,
                   training_code=version, evaluation_code=version, contract_hash=run["contract_hash"], data=data,
                   selection_hash="dev", checkpoint=str(output / "best.pt"), checkpoint_sha256=sha256(output / "best.pt"),
                   count=len(rows), effective_lateral_rule=config["lateral_rule"]))
        write_json(job["status_path"], dict(status="completed"))
        return output, evaluation, job
    return plan_path, complete


def test_complete_pilot_is_checked_and_never_selects_a_frontend(pilot_records):
    plan, complete = pilot_records
    assert all(r["status"] == "not_started" for r in collect(plan)["rows"].values())
    complete(512); complete(1024)
    report = collect(plan)
    assert report["comparison"]["status"] == "comparable", report
    assert report["comparison"]["decision"] == "pending_user_choice"
    assert report["comparison"]["delta_high_minus_low"]["epe"] == pytest.approx(-.2)
    assert "pending_user_choice" in markdown(report)
    assert report["lateral_baselines"]["fit"]["partition"] == "train_core"


@pytest.mark.parametrize("mutation", ["short_budget", "wrong_targets", "wrong_order", "changed_checkpoint"])
def test_bad_evidence_is_not_promoted_to_a_completed_comparison(pilot_records, mutation):
    plan, complete = pilot_records
    output, evaluation, job = complete(512)
    complete(1024)
    if mutation == "short_budget":
        value = read_json(output / "status.json"); value["epoch"] = 39; write_json(output / "status.json", value)
    elif mutation == "wrong_targets":
        text = (evaluation / "predictions.csv").read_text(); (evaluation / "predictions.csv").write_text(text.replace("scene-dev", "wrong-dev"))
    elif mutation == "wrong_order":
        values = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        values[0]["sample_order_sha256"] = "f"*64
        (output / "metrics.jsonl").write_text("".join(json.dumps(v)+"\n" for v in values))
    else:
        with (output / "best.pt").open("ab") as stream: stream.write(b"changed")
    report = collect(plan)
    assert report["rows"]["R02_512"]["status"] == "invalid"
    assert report["comparison"]["status"] == "pending"


def test_different_initialization_prevents_resolution_attribution(pilot_records):
    plan, complete = pilot_records
    complete(512); complete(1024)
    rows = collect(plan)["rows"]
    rows["R02_1024"]["initialization"]["model_state_sha256"] = "f"*64
    assert compare_rows(rows)["status"] == "incomparable"


def recovery_plan(plan_path, **kwargs):
    source = read_json(plan_path)
    return build_plan(source["root"], source["manifest"], source["selection_path"], source["low_flow_root"], "recovery",
                      dict(commit="b"*40, dirty=False, verified_remote_commit="b"*40),
                      code_root=kwargs.get("code_root", source["code_root"]), reuse_low_plan=plan_path)


def test_adopt_completed_low_preserves_source_and_actual_commit(pilot_records):
    source_path, complete = pilot_records
    output, _, job = complete(512)
    original_status = Path(job["status_path"]).read_bytes()
    plan = recovery_plan(source_path)
    assert "R02_512" not in {j["name"] for j in plan["jobs"]}
    assert plan["external_preparation"][0]["run_dir"] == str(output)
    assert plan["external_preparation"][0]["source_commit"] == "a"*40
    path = source_path.parent.parent / "recovery/plan.json"
    write_json(path, plan)
    report = collect(path)
    assert report["rows"]["R02_512"]["status"] == "completed"
    assert report["rows"]["R02_512"]["training_commit"] == "a"*40
    assert report["rows"]["R02_1024"]["training_commit"] == "b"*40
    assert Path(job["status_path"]).read_bytes() == original_status
    assert report["comparison"]["status"] == "pending"


def test_adoption_rejects_failed_or_changed_protocol_and_mutated_source(pilot_records, tmp_path):
    source_path, complete = pilot_records
    _, _, job = complete(512)
    write_json(job["status_path"], dict(status="failed", error="fixture failure"))
    with pytest.raises(ValueError, match="not a verified completed"):
        recovery_plan(source_path)
    complete(512)
    source = read_json(source_path)
    changed_code = tmp_path / "changed_code"
    shutil.copytree(source["code_root"], changed_code)
    for width in (512, 1024):
        path = changed_code / f"configs/next/R02_raft{width}.json"
        value = read_json(path); value["learning_rate"] /= 2; write_json(path, value)
    with pytest.raises(ValueError, match="configuration differs"):
        recovery_plan(source_path, code_root=changed_code)
    plan = recovery_plan(source_path)
    source["prefix"] = "mutated"
    source["identity"] = object_hash({k:v for k,v in source.items() if k != "identity"})
    write_json(source_path, source)
    with pytest.raises(ValueError, match="source plan bytes changed"):
        validate_adopted_source(plan)


def test_coordinator_collection_has_a_receipt_and_never_overwrites(pilot_records, monkeypatch):
    from types import SimpleNamespace
    import resolution_pilot
    path, _ = pilot_records
    plan = read_json(path)
    def successful_collect(command, **kwargs):
        output = Path(command[-1])
        write_json(output / "summary.json", dict(plan_identity=plan["identity"],
                   comparison=dict(status="pending", decision="pending_user_choice")))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(resolution_pilot.subprocess, "run", successful_collect)
    result = collect_finished_plan(plan, path)
    assert result["status"] == "completed" and result["decision"] == "pending_user_choice"
    receipt = (path.parent / "collection.json").read_bytes()
    repeat = collect_finished_plan(plan, path)
    assert repeat["status"] == "failed"
    assert (path.parent / "collection.json").read_bytes() == receipt


def test_coordinator_collection_failure_is_not_comparison_success(pilot_records, monkeypatch):
    from types import SimpleNamespace
    import resolution_pilot
    path, _ = pilot_records
    plan = read_json(path)
    monkeypatch.setattr(resolution_pilot.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=2))
    result = collect_finished_plan(plan, path)
    assert result["status"] == "failed" and result["exit_code"] == 2
    assert "comparison_status" not in result
    assert read_json(path.parent / "collection.json")["status"] == "failed"
