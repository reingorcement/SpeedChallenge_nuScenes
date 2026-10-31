"""Verify the two small-scene R02 runs and report evidence, never choose a frontend."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from collect_results import check_summary, command_value, require, target_identity
from lateral_baselines import baseline_report
from pipeline import validate_plan
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import lateral_acceptance, summarize, validate_lateral_rule
from purevision_v5.flow import validate_flow_contract
from purevision_v5.labels import LABEL_CONTRACT
from purevision_v5.selection import label_rows, load_selection
from purevision_v5.training import epoch_order, validate_expected_flow
from resolution_pilot import computation_identity, validate_adopted_source


def optional(path):
    return read_json(path) if Path(path).is_file() else {}


def inspect_run(plan, job, train_rows, dev_rows, *, expected_seed=42):
    # Shared receipt/40-epoch/prediction validation; selection scope is explicit.
    fusion = plan.get("stage")=="r03_flex_continuation"
    full = fusion or plan.get("stage") in {"r02_encoder_comparison", "r02_encoder_multiseed"}
    scope = "full_train_dev" if full else "resolution_pilot"
    output = Path(job["run_dir"])
    evaluation = Path(command_value(job["commands"][1], "--output"))
    row = dict(name=job["name"], experiment=job.get("experiment","R02"), run_id=output.name, status="not_started",
               scope=scope, training_commit=plan["code"]["commit"], run_dir=str(output),
               evaluation_dir=str(evaluation), log=job["log_path"], metrics=None)
    try:
        run, status = optional(output / "run.json"), optional(output / "status.json")
        worker, receipt = optional(job["status_path"]), optional(evaluation / "evaluation.json")
        observed = dict(run=run.get("status"), training=status.get("status"), worker=worker.get("status"),
                        evaluation=receipt.get("status"))
        row.update(observed_status=observed, progress={k: status[k] for k in
                   ("epoch", "processed_batches", "optimizer_updates", "amp_skips", "best_epe", "updated_at") if k in status})
        if any(s in {"failed", "interrupted", "blocked", "cancelled"} for s in observed.values()):
            return {**row, "status": "failed", "reason": status.get("error") or worker.get("error")
                    or worker.get("reason") or receipt.get("error") or "A recorded stage did not complete"}
        if run.get("smoke"):
            return {**row, "status": "excluded_smoke", "reason": "Smoke is not the explicit 40-epoch pilot"}
        if run.get("status") != "completed" or status.get("status") != "completed":
            row["status"] = "running" if any(s in {"running", "launching"} for s in observed.values()) else "not_started"
            if "unknown" in observed.values():
                row["status"] = "unknown"
            return row
        if receipt.get("status") != "completed" or worker.get("status") != "completed":
            return {**row, "status": "evaluation_pending"}
        config, contract = run["config"], run["contract"]
        require(config["experiment"] == run["experiment"] == (job["experiment"] if fusion else "R02"), "Unexpected experiment")
        require(config["epochs"] == 40 and config["batch_size"] == 64 and config["seed"] == expected_seed
                and config.get("early_stopping") is False, "Changed pilot budget/batch/seed/stopping policy")
        require(read_json(job["config_path"]) == config and sha256(job["config_path"]) == plan["config_hashes"][job["name"]],
                "Actual configuration differs from fixed plan")
        execution = dict(smoke=False, max_batches=0)
        if not full:
            execution.update(scope="resolution_pilot", scene_selection_identity=plan["selection"]["identity"])
        require(run["scope"] == receipt["scope"] == scope and contract["execution"] == execution,
                "Pilot scope/selection mismatch")
        require(not any(run["runtime"].get(k, 0) for k in ("limit_scenes", "limit_windows", "max_batches")),
                "Pilot cannot use additional unregistered subset limits")
        require(contract["config"] == config and object_hash(contract) == run["contract_hash"], "Training contract mismatch")
        data = contract["data"]
        require(data["dataset_sha256"] == plan["manifest_sha256"] and data["label"] == LABEL_CONTRACT
                and data.get("scene_selection") == plan.get("selection"), "Data identity mismatch")
        expected_features=plan["rgb_features_contract"] if fusion and job["name"]=="R03_Frozen" else None
        require(data.get("features")==expected_features,"Frozen RGB feature identity mismatch")
        if fusion and job["name"]=="R03_FT":
            require(data["rgb_images"]["view_identity"]==plan["rgb_frames_identity"],"FT image view identity mismatch")
        require(Path(data["flow_root"]).resolve() == Path(command_value(job["commands"][0], "--flow")).resolve(),
                "Actual flow root differs from launch command")
        validate_expected_flow(config, data)
        flow_contract = {k: v for k, v in data["flow"].items() if k != "packed_index_sha256"}
        validate_flow_contract(flow_contract)
        pack_path = Path(data["flow_root"]) / "packed_index.json"
        pack = read_json(pack_path)
        require(sha256(pack_path) == data["flow"]["packed_index_sha256"]
                and pack["identity"] == object_hash({k: v for k, v in pack.items() if k != "identity"}), "Packed index identity mismatch")
        require(pack["contract"]["dataset_sha256"] == plan["manifest_sha256"]
                and pack["contract"]["fb_available"] is False, "Pilot must use the declared forward-only cache")
        actual_pack_contract = {k: pack["contract"][k] for k in flow_contract if k != "schema"}
        actual_pack_contract["schema"] = pack["contract"]["flow_schema"]
        require(actual_pack_contract == flow_contract, "Training flow contract differs from actual pack")
        commit = plan["code"]["commit"]
        for version in (run["code"], receipt["training_code"], receipt["evaluation_code"]):
            require(version["commit"] == commit and version["dirty"] is False, "Training/evaluation code mismatch")
        require(run["code"]["verified_remote_commit"] == commit, "Training code was not verified pushed")
        require(receipt["training_run"] == run["run_id"] and receipt["contract_hash"] == run["contract_hash"]
                and receipt["data"] == data and receipt["selection_hash"] == contract["dev_selection"], "Evaluation receipt contract mismatch")
        best_path, last_path = output / "best.pt", output / "last.pt"
        require(Path(receipt["checkpoint"]).resolve() == best_path.resolve()
                and sha256(best_path) == receipt["checkpoint_sha256"], "Best checkpoint/evaluation checksum mismatch")
        best, last = [torch.load(path, map_location="cpu", weights_only=True) for path in (best_path, last_path)]
        for packet in (best, last):
            require(packet["schema"] == "v5_training_state_v1" and packet["config"] == config
                    and packet["contract"] == contract and packet["contract_hash"] == run["contract_hash"]
                    and packet["code"] == run["code"] and packet["run_id"] == run["run_id"], "Checkpoint contract/code mismatch")
            require(packet["initialization"] == run["initialization"] and packet["sample_order"] == run["sample_order"],
                    "Checkpoint initialization/sample-order provenance mismatch")
            require(all(key in packet for key in ("optimizer", "scheduler", "scaler", "rng")), "Incomplete resumable state")
        require(len(train_rows) == run["train_windows"] and len(dev_rows) == run["dev_windows"] == receipt["count"],
                "Selected train/dev counts mismatch")
        expected_steps = math.ceil(len(train_rows) / config["batch_size"]) * 40
        require(last["epoch"] == status["epoch"] == 40 and last["next_batch"] == 0
                and status["stop_reason"] == "epoch_budget", "Pilot did not complete all 40 epochs")
        for state in (status, last):
            require(state["global_step"] == state["processed_batches"] == expected_steps
                    and state["processed_batches"] == state["optimizer_updates"] + state["amp_skips"]
                    and state["optimizer_updates"] > 0 and state["amp_skips"] >= 0, "Invalid processed/update/skip counters")
            if fusion:
                require(state["processed_microbatches"]==math.ceil(len(train_rows)/config["micro_batch_size"])*40,
                        "Microbatch counter differs from complete epoch budget")
        require(status["optimizer_updates"] == last["optimizer_updates"] and status["amp_skips"] == last["amp_skips"],
                "Last checkpoint counters differ from final status")
        history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
        require([h["epoch"] for h in history] == list(range(40)), "Missing or repeated complete epoch metrics")
        expected_samples = object_hash([[r["scene"], r["token"], r["time_s"], [r["target_vx"], r["target_vy"]]] for r in train_rows])
        require(run["sample_order"]["training_samples_sha256"] == expected_samples, "Training window/label order differs")
        orders = []
        rule = validate_lateral_rule(config.get("lateral_rule"))
        require(run["effective_lateral_rule"] == receipt["effective_lateral_rule"] == rule, "Effective lateral rule differs")
        for index, entry in enumerate(history):
            order_hash = epoch_order(len(train_rows), expected_seed, index)[1]
            require(entry["sample_order_sha256"] == order_hash, "Training epoch sample order differs")
            require(entry["processed_batches"] == math.ceil(len(train_rows) / 64) * (index + 1)
                    and entry["global_step"] == entry["processed_batches"]
                    and entry["processed_batches"] == entry["optimizer_updates"] + entry["amp_skips"], "Epoch counter mismatch")
            require(entry["validation"]["effective_lateral_rule"] == rule, "Epoch lateral rule mismatch")
            orders.append(order_hash)
        best_epoch = best["best_epoch"]
        require(best_epoch == last["best_epoch"] == status["best_epoch"] and 0 <= best_epoch < 40,
                "Best epoch mismatch")
        require(np.isclose(history[best_epoch]["validation"]["overall"]["epe"], status["best_epe"], rtol=1e-10),
                "Best metric differs from recorded epoch")
        require(config["min_delta"] == 0 and best_epoch == int(np.argmin([h["validation"]["overall"]["epe"] for h in history])),
                "Checkpoint is not the strict best development EPE")
        prediction_path = evaluation / "predictions.csv"
        with prediction_path.open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        require(target_identity(predictions) == target_identity(dev_rows), "Predictions do not cover the exact selected dev labels")
        actual = summarize([[float(r["vx"]), float(r["vy"])] for r in predictions],
                           [[float(r["target_vx"]), float(r["target_vy"])] for r in predictions],
                           [r["scene"] for r in predictions], rule)
        metrics = read_json(evaluation / "metrics.json")
        check_summary(metrics, actual)
        check_summary(read_json(output / "best_metrics.json"), actual)
        require(np.isclose(actual["overall"]["epe"], status["best_epe"], rtol=1e-10), "Recomputed EPE differs from selected best")
        row.update(status="completed", config=config, data=data, metrics=metrics, best_epoch_one_based=best_epoch + 1,
                   best_checkpoint_sha256=sha256(best_path), last_checkpoint_sha256=sha256(last_path),
                   predictions=str(prediction_path), predictions_sha256=sha256(prediction_path),
                   target_identity=target_identity(predictions), initialization=run["initialization"], sample_order=run["sample_order"],
                   epoch_order_sha256=orders, trainable_parameters=run["trainable_parameters"],
                   processed_batches=last["processed_batches"], optimizer_updates=last["optimizer_updates"], amp_skips=last["amp_skips"],
                   elapsed_seconds=status["elapsed_seconds"], common_config_identity=object_hash({k: v for k, v in config.items() if k != "expected_flow"}),
                   cuda_peak_allocated_bytes=status.get("cuda_peak_allocated_bytes"),
                   packed_bytes=sum((pack_path.parent / a["file"]).stat().st_size for a in pack["arrays"].values()))
        if fusion:
            from purevision_v5.model import build_model
            from purevision_v5.training import model_state_sha256
            model=build_model(config);model.load_state_dict(last["model"],strict=True)
            row.update(total_parameters=run["total_parameters"],processed_microbatches=last["processed_microbatches"],
                final_rgb_state_sha256=model_state_sha256(model.rgb_encoder) if model.rgb_encoder is not None else None,
                optimizer_groups=run["optimizer_groups"],micro_batch_size=run["micro_batch_size"])
        row["computation"] = computation_identity(plan["code_root"])
    except Exception as error:
        row.update(status="invalid", metrics=None, reason=f"{type(error).__name__}: {error}")
    return row


def compare_rows(rows):
    low, high = [rows[name] for name in ("R02_512", "R02_1024")]
    result = dict(status="pending", decision="pending_user_choice", requires_full_generation_approval=True)
    if low["status"] != "completed" or high["status"] != "completed":
        return result
    try:
        for key in ("common_config_identity", "target_identity", "initialization", "sample_order", "epoch_order_sha256", "computation", "trainable_parameters"):
            require(low[key] == high[key], f"Resolution comparison differs in {key}")
        for key in ("dataset_sha256", "label", "scene_selection"):
            require(low["data"][key] == high["data"][key], f"Resolution comparison differs in {key}")
        rule = validate_lateral_rule(low["config"]["lateral_rule"])
        result.update(status="comparable", delta_high_minus_low={key: high["metrics"]["overall"][key] - low["metrics"]["overall"][key]
                      for key in ("epe", "mae_vx", "mae_vy", "rmse_vx", "rmse_vy", "bias_vx", "bias_vy")},
                      lateral_acceptance=lateral_acceptance(high["metrics"], low["metrics"], rule),
                      training_commits={r["name"]: r["training_commit"] for r in (low, high)},
                      numerical_code_byte_identical=True,
                      commit_note="Actual commits are reported per run; numerical code hashes match. Network/queue/reporting recovery may differ.")
    except (KeyError, ValueError, TypeError) as error:
        result.update(status="incomparable", reason=str(error))
    return result


def collect(plan_path):
    plan = read_json(plan_path)
    validate_plan(plan)
    require(plan.get("scope") == "resolution_pilot" and plan.get("full_generation_authorized") is False,
            "This collector only accepts the explicit small-scene pilot")
    require(sha256(plan["manifest"]) == plan["manifest_sha256"] and sha256(plan["selection_path"]) == plan["selection_sha256"],
            "Planned dataset/selection changed")
    selection = load_selection(plan["manifest"], plan["selection_path"])
    require(selection == plan["selection"], "Plan scene selection differs")
    train_rows, _ = label_rows(plan["manifest"], "train_core", plan["selection_path"])
    dev_rows, _ = label_rows(plan["manifest"], "dev", plan["selection_path"])
    jobs = [job for job in plan["jobs"] + plan.get("external_preparation", []) if job["kind"] == "training"]
    require({j["name"] for j in jobs} == {"R02_512", "R02_1024"} and len(jobs) == 2, "Expected exactly two pilot training jobs")
    source = validate_adopted_source(plan)
    rows = {}
    for job in jobs:
        if job["name"] == "R02_512" and source is not None:
            old = next(j for j in source["jobs"] if j["name"] == "R02_512")
            require(all(job.get(k) == v for k, v in old.items()), "Adopted job fields differ from original source")
            row = inspect_run(source, old, train_rows, dev_rows)
            if row["status"] == "completed":
                require(row["best_checkpoint_sha256"] == plan["low_run_source"]["best_checkpoint_sha256"]
                        and row["predictions_sha256"] == plan["low_run_source"]["predictions_sha256"], "Adopted low evidence changed")
            row["adopted_from_plan"] = plan["low_run_source"]
        else:
            row = inspect_run(plan, job, train_rows, dev_rows)
        rows[job["name"]] = row
    rows = {name: rows[name] for name in ("R02_512", "R02_1024")}
    rule = validate_lateral_rule(read_json(jobs[0]["config_path"])["lateral_rule"])
    baselines = baseline_report(plan["manifest"], plan["selection_path"], rule,
                                [r["predictions"] for r in rows.values() if r["status"] == "completed"])
    flow_jobs = [job for job in plan["jobs"] if job["kind"] == "flow"]
    costs = dict(shards=[], elapsed_seconds_sum_workers=0., pairs_processed=0,
                 note="Sum of worker elapsed time is compute occupancy, not parallel wall time")
    roots = set()
    for job in flow_jobs:
        flow_report = optional(job["progress_path"])
        item = {key: flow_report[key] for key in ("status", "pairs_processed", "elapsed_s", "seconds_per_pair", "cuda_peak_allocated_bytes", "counts", "started_at_utc", "finished_at_utc") if key in flow_report}
        item.update(job=job["name"], device=job["device"], report=job["progress_path"],
                    report_sha256=sha256(job["progress_path"]) if flow_report else None)
        costs["shards"].append(item)
        costs["elapsed_seconds_sum_workers"] += item.get("elapsed_s", 0.)
        costs["pairs_processed"] += item.get("pairs_processed", 0)
        roots.add(command_value(job["commands"][0], "--output"))
    if costs["shards"] and all(r.get("status") == "complete" for r in costs["shards"]):
        costs["native_and_view_file_bytes"] = sum(p.stat().st_size for root in roots for p in Path(root).rglob("*") if p.is_file())
    return dict(schema="v5_resolution_pilot_report_v1", generated_at=now(), scope="resolution_pilot",
                plan=str(Path(plan_path).resolve()), plan_sha256=sha256(plan_path), plan_identity=plan["identity"],
                training_commit=plan["code"]["commit"], training_commits={k:r["training_commit"] for k,r in rows.items()},
                low_run_source=plan.get("low_run_source"), collector_sha256=sha256(__file__), selection=selection,
                rows=rows, comparison=compare_rows(rows), lateral_baselines=baselines, high_flow_cost=costs,
                low_flow_cost=dict(source="existing_512x288_cache_reused", incremental_raft_generation_seconds=0,
                                   note="Earlier generation cost is not measured by this pilot"),
                metric_units="m/s",
                limitations=[f"{selection['counts']['train_core']['scenes']} train / {selection['counts']['dev']['scenes']} dev scenes only; this is not a full-dataset result.",
                             "Single seed 42, dev selection and dev recomputation; no independent held-out test.",
                             "PLANAR_PROXY rear-axle yaw-frame labels; weak lateral evidence may remain insufficient.",
                             "This report verifies small metadata and checkpoint/prediction bytes, not multi-GB flow array bytes.",
                             "The user chooses the frontend before any full-cache generation; no automatic promotion."])


def markdown(report):
    counts = report["selection"]["counts"]
    lines = ["# R02 resolution pilot", "", "Actual training commits: " + "; ".join(f"{k} `{v}`" for k,v in report["training_commits"].items()) + ".",
             "", f"Scope: fixed {counts['train_core']['scenes']} train / {counts['dev']['scenes']} dev scene pilot, seed 42, 40 complete epochs, batch 64.", "",
             "| Run | Status | Dev EPE | MAE vx | MAE vy | Best epoch | Updates / AMP skips | Seconds |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"].values():
        if row["status"] == "completed":
            m = row["metrics"]["overall"]
            lines.append(f"| {row['run_id']} | completed | {m['epe']:.6f} | {m['mae_vx']:.6f} | {m['mae_vy']:.6f} | {row['best_epoch_one_based']} | {row['optimizer_updates']} / {row['amp_skips']} | {row['elapsed_seconds']:.2f} |")
        else:
            lines.append(f"| {row['run_id']} | {row['status']} | — | — | — | — | — | — |")
    baseline = report["lateral_baselines"]
    lines += ["", f"Train-fitted constant vy: {baseline['fit']['constant_vy_m_s']:.9f} m/s; "
              f"dev MAE zero={baseline['zero_vy']['overall']['mae_vy']:.6f}, median={baseline['train_median_vy']['overall']['mae_vy']:.6f}.",
              "", f"Comparison: **{report['comparison']['status']}**. Decision: **pending_user_choice**.",
              "The two inputs have the same 128×72 downstream view; only RAFT generation resolution changes.", ""]
    if "delta_high_minus_low" in report["comparison"]:
        lines += [f"High minus low dev EPE: {report['comparison']['delta_high_minus_low']['epe']:.6f} m/s.", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    for row in report["rows"].values():
        if row.get("reason"):
            lines.append(f"- {row['name']}: {row['reason']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Each report snapshot needs a new output directory")
    report = collect(args.plan)
    write_json(output / "summary.json", report)
    (output / "summary.md").write_text(markdown(report))
    print({"report": str(output), "statuses": {k: r["status"] for k, r in report["rows"].items()},
           "comparison": report["comparison"]})


if __name__ == "__main__":
    main()
