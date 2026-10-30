import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from collect_resolution_multiseed import collect, mean_sd, markdown
from pipeline import check_planned_inputs
from purevision_v5.common import ROOT, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import summarize
from purevision_v5.training import epoch_order
from resolution_multiseed import build_plan, validate_multiseed_inputs
from test_collect_resolution import pilot_records


@pytest.fixture
def sweep(pilot_records):
    source_path, complete = pilot_records
    complete(512); complete(1024)
    source = read_json(source_path)
    code = Path(source["code_root"])
    for width in (512,1024):
        for seed in (43,44):
            shutil.copy2(ROOT / f"configs/next/R02_raft{width}_s{seed}.json", code / f"configs/next/R02_raft{width}_s{seed}.json")
    plan = build_plan(source["root"], source_path, "sweep", dict(commit="b"*40, dirty=False, verified_remote_commit="b"*40), code_root=code)
    plan_path = Path(source["root"]) / "pipeline/sweep/plan.json"
    write_json(plan_path,plan)

    def finish(width, seed, error):
        original_job = next(j for j in source["jobs"] if j["name"] == f"R02_{width}")
        original = Path(original_job["run_dir"])
        original_eval = Path(original_job["commands"][1][-1])
        job = next(j for j in plan["jobs"] if j["name"] == f"R02_{width}_s{seed}")
        output, evaluation = Path(job["run_dir"]), Path(job["commands"][1][-1])
        run = read_json(original / "run.json")
        config = read_json(job["config_path"])
        run.update(run_id=output.name, config=config, code=plan["code"])
        run["contract"]["config"] = config
        run["contract_hash"] = object_hash(run["contract"])
        run["initialization"].update(seed=seed,model_state_sha256=object_hash(["initialization",seed]))
        run["sample_order"]["seed"] = seed
        write_json(output / "run.json",run)
        with (original_eval / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["vx"] = float(row["target_vx"]) + error
        summary = summarize([[float(r["vx"]),float(r["vy"])] for r in rows],
                            [[float(r["target_vx"]),float(r["target_vy"])] for r in rows], [r["scene"] for r in rows], config["lateral_rule"])
        write_json(output / "best_metrics.json",summary)
        write_json(evaluation / "metrics.json",summary)
        with (evaluation / "predictions.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        status=read_json(original / "status.json");status["best_epe"]=summary["overall"]["epe"]
        write_json(output / "status.json",status)
        history=[]
        for epoch in range(40):
            value=copy.deepcopy(summary);value["overall"]["epe"] += (39-epoch)*.01
            history.append(dict(epoch=epoch,global_step=epoch+1,processed_batches=epoch+1,optimizer_updates=epoch+1,amp_skips=0,
                                sample_order_sha256=epoch_order(run["train_windows"],seed,epoch)[1],validation=value))
        (output / "metrics.jsonl").write_text("".join(json.dumps(h)+"\n" for h in history))
        for name in ("best.pt","last.pt"):
            packet=torch.load(original/name,weights_only=True)
            packet.update(run_id=output.name,code=run["code"],config=config,contract=run["contract"],contract_hash=run["contract_hash"],
                          initialization=run["initialization"],sample_order=run["sample_order"],best_epe=status["best_epe"])
            torch.save(packet,output/name)
        receipt=read_json(original_eval / "evaluation.json")
        receipt.update(training_run=output.name,training_code=run["code"],evaluation_code=run["code"],contract_hash=run["contract_hash"],
                       checkpoint=str(output/"best.pt"),checkpoint_sha256=sha256(output/"best.pt"))
        write_json(evaluation / "evaluation.json",receipt)
        write_json(job["status_path"],dict(status="completed"))
        return output,job
    return plan_path,finish


def test_four_training_jobs_reuse_fixed_data_and_wait_on_same_gpu(sweep):
    path,_=sweep;plan=read_json(path)
    assert len(plan["jobs"]) == 4 and all(j["kind"] == "training" for j in plan["jobs"])
    assert plan["selection"]["seed"] == 42
    assert {j["training_seed"] for j in plan["jobs"]} == {43,44}
    for job in plan["jobs"]:
        assert job["device"] == ("cuda:1" if job["raft_width"] == 512 else "cuda:2")
        assert job["after"] == ([] if job["training_seed"] == 43 else [f"R02_{job['raft_width']}_s43"])
        assert all("prepare_flow.py" not in " ".join(command) for command in job["commands"])
    report=collect(path)
    assert report["rows"]["R02_512_s42"]["status"] == "completed"
    assert all(report["rows"][f"R02_{w}_s{s}"]["status"] == "not_started" for w in (512,1024) for s in (43,44))
    assert report["comparison"]["status"] == "pending"


def test_three_seed_paired_statistics_use_sample_sd_and_keep_selection(sweep):
    path,finish=sweep
    for width,seed,error in [(512,43,1.1),(1024,43,.7),(512,44,.9),(1024,44,1.0)]: finish(width,seed,error)
    report=collect(path)
    result=report["comparison"]
    assert result["status"] == "comparable",report
    expected=np.asarray([-.2,-.4,.1])
    assert result["paired_high_minus_low"]["epe"]["values"] == pytest.approx(expected)
    assert result["paired_high_minus_low"]["epe"]["mean"] == pytest.approx(expected.mean())
    assert result["paired_high_minus_low"]["epe"]["sample_sd"] == pytest.approx(expected.std(ddof=1))
    assert result["paired_high_minus_low"]["epe"]["ddof"] == 1
    assert result["epe_improved_seed_count"] == 2
    assert result["decision"] == "pending_user_choice"
    subset=report["rows"]["R02_512_s42"]["metrics"]["nonzero_vy"]
    assert result["lateral"]["unique_nonzero_windows"] == subset["count"]
    assert report["incremental_raft_generation"] == dict(pairs=0,seconds=0,new_cache_bytes=0)
    assert "sample SD" in markdown(report)


def test_seed_order_mismatch_and_failed_job_block_aggregate(sweep):
    path,finish=sweep
    for width,seed,error in [(512,43,1.1),(1024,43,.7),(512,44,.9),(1024,44,1.0)]: finish(width,seed,error)
    plan=read_json(path);job=plan["jobs"][0]
    history_path=Path(job["run_dir"])/"metrics.jsonl"
    history=[json.loads(line) for line in history_path.read_text().splitlines()]
    history[0]["sample_order_sha256"]=epoch_order(12,42,0)[1]
    history_path.write_text("".join(json.dumps(h)+"\n" for h in history))
    report=collect(path)
    assert report["rows"][job["name"]]["status"] == "invalid"
    assert report["comparison"]["status"] == "pending"
    write_json(job["status_path"],dict(status="failed",error="fixture failure"))
    assert collect(path)["rows"][job["name"]]["status"] == "failed"


def test_worker_entry_refuses_changed_high_cache(sweep):
    path,_=sweep;plan=read_json(path)
    job=next(j for j in plan["jobs"] if j["raft_width"] == 1024)
    check_planned_inputs(plan,job)
    pack=Path(plan["locked_packs"]["1024"]["root"])/"packed_index.json"
    pack.write_text(pack.read_text()+"\n")
    with pytest.raises(ValueError,match="packed flow index changed"):
        check_planned_inputs(plan,job)


def test_changed_source_and_changed_protocol_are_rejected(sweep):
    path,_=sweep;plan=read_json(path)
    config_path=Path(plan["jobs"][0]["config_path"])
    conf=read_json(config_path);conf["epochs"]=41;write_json(config_path,conf)
    with pytest.raises(ValueError,match="configuration changed"):
        validate_multiseed_inputs(plan)
    source=Path(plan["seed42_source"]["path"]);source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError,match="source plan bytes changed"):
        validate_multiseed_inputs(plan)


def test_statistics_cannot_summarize_missing_seed_as_three():
    with pytest.raises(ValueError,match="Exactly three"):
        mean_sd([1.,2.])
