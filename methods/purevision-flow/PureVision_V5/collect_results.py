"""Export six honest S2 rows from a fixed pipeline and independent evaluations.

This reads records and small checkpoints on CPU. It never runs a model, modifies
training artifacts, or promotes smoke/partial results into formal metrics.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from pipeline import validate_plan
from purevision_v5.common import ROOT, now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import lateral_acceptance, summarize
from purevision_v5.labels import LABEL_CONTRACT, LABEL_SHA256


EXPERIMENTS = tuple(f"R{i:02d}" for i in range(6))
LATERAL_RULE = dict(max_relative_mae_increase=.05, max_absolute_mae_increase_m_s=.01,
                    nonzero_threshold_m_s=.2, min_subset_windows=100, min_subset_scenes=5)
LIMITS = [
    "PLANAR_PROXY rear-axle planar labels; these are not true 3D body-axis velocity results.",
    "One seed and development-set checkpoint selection; no held-out or statistical improvement claim.",
    "The nonzero-vy subset needs >=100 windows across >=5 scenes; otherwise improvement is undetermined.",
    "R04/R05 can use different GPU models; wall time is not an isolated algorithm comparison.",
    "Cache provenance is checked against recorded index identities; this report does not rehash multi-GB arrays.",
]


def optional_json(path):
    return read_json(path) if Path(path).is_file() else {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command_value(command, flag):
    return command[command.index(flag) + 1]


def target_identity(rows):
    values = sorted((r["scene"], r["token"], float(r["time_s"]),
                     float(r["target_vx"]), float(r["target_vy"])) for r in rows)
    require(bool(values), "Empty dev predictions")
    require(len({v[:3] for v in values}) == len(values), "Duplicate dev prediction windows")
    require(np.isfinite([v[2:] for v in values]).all(), "Nonfinite dev targets/timestamps")
    return object_hash(values)


@lru_cache(maxsize=2)
def expected_dev_identity(manifest_path, manifest_sha256):
    manifest_path = Path(manifest_path)
    require(sha256(manifest_path) == manifest_sha256, "Manifest changed during collection")
    manifest, rows = read_json(manifest_path), []
    for record in manifest["sequences"]:
        if record["partition"] != "dev":
            continue
        path = (manifest_path.parent / record["file"]).resolve()
        require(path.is_relative_to(manifest_path.parent), "Sequence escaped manifest directory")
        require(sha256(path) == record["sha256"], "Dev sequence checksum mismatch")
        sequence = read_json(path)
        require(all(sequence.get(k) == record[k] for k in ("scene", "partition", "log_token")),
                "Dev sequence identity mismatch")
        for end in sequence["windows"]:
            frame = sequence["frames"][end]
            target = np.asarray(frame["target"], dtype=np.float32)
            rows.append(dict(scene=record["scene"], token=frame["token"], time_s=frame["time_s"],
                             target_vx=float(target[0]), target_vy=float(target[1])))
    return target_identity(rows), len(rows)


def check_summary(recorded, actual, prefix="metrics"):
    """All recomputed metric fields must agree; recorded flow coverage is extra."""
    for key, value in actual.items():
        observed = recorded.get(key)
        if isinstance(value, dict):
            require(isinstance(observed, dict), f"Missing {prefix}.{key}")
            check_summary(observed, value, f"{prefix}.{key}")
        elif isinstance(value, float):
            require(isinstance(observed, (int, float)) and np.isfinite(observed)
                    and np.isclose(observed, value, rtol=1e-9, atol=1e-10), f"Changed {prefix}.{key}")
        else:
            require(observed == value, f"Changed {prefix}.{key}")


def inspect_completed(row, run, status, receipt, plan, job):
    config, contract = run["config"], run["contract"]
    require(run["contract_hash"] == object_hash(contract) and contract["config"] == config,
            "Training contract hash/config mismatch")
    require(config["experiment"] == row["experiment"] == run["experiment"], "Experiment mismatch")
    require(contract["execution"] == {"smoke": False, "max_batches": 0}, "Not a full formal execution")
    require(not any(run["runtime"].get(k, 0) for k in ("limit_scenes", "limit_windows", "max_batches")),
            "Subset limits exclude formal results")
    config_path = Path(command_value(job["commands"][0], "--config"))
    require(sha256(config_path) == plan["config_hashes"][row["experiment"]]
            and read_json(config_path) == config, "Configuration differs from the fixed plan")
    require(config.get("lateral_rule") == LATERAL_RULE, "Lateral rule differs from the fixed implementation")
    data = contract["data"]
    require(data["dataset_sha256"] == plan["manifest_sha256"], "Training dataset differs from pipeline manifest")
    require(data["label"] == LABEL_CONTRACT, "Label contract mismatch")
    commit = plan["code"]["commit"]
    for version in (run["code"], receipt["training_code"], receipt["evaluation_code"]):
        require(version["commit"] == commit and version.get("dirty") is False,
                "Training/evaluation code differs from the clean planned commit")
    require(run["code"].get("verified_remote_commit") == commit, "Training code lacks pushed-commit verification")
    require(receipt["contract_hash"] == run["contract_hash"] and receipt["data"] == data
            and receipt["selection_hash"] == contract["dev_selection"], "Independent evaluation data/contract mismatch")
    require(Path(receipt["checkpoint"]).resolve() == Path(row["checkpoint"]).resolve(), "Wrong evaluated checkpoint")
    require(sha256(row["checkpoint"]) == receipt["checkpoint_sha256"], "Evaluated checkpoint checksum mismatch")
    packet = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
    require(packet.get("schema") == "v5_training_state_v1" and packet["contract"] == contract
            and packet["contract_hash"] == run["contract_hash"] and packet["config"] == config,
            "Best checkpoint contract mismatch")
    require(packet["code"] == run["code"] and receipt["training_run"] == packet["run_id"]
            and packet["run_id"] in {run["run_id"], run.get("parent_run")}, "Best checkpoint run/code mismatch")
    require(packet["best_epoch"] == status["best_epoch"], "Best epoch differs from final training state")
    feature = data.get("features")
    if row["experiment"] == "R02":
        require(feature is None, "Flow-only run unexpectedly has an RGB contract")
    else:
        index = read_json(Path(plan["features"]) / "index.json")
        require(feature == index["contract"] and index["identity"] == object_hash(feature), "RGB cache identity mismatch")
        require(Path(run["runtime"]["features"]).resolve() == Path(plan["features"]).resolve(), "Different RGB cache directory")
    with Path(row["predictions"]).open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    identity = target_identity(predictions)
    expected_identity, expected_count = expected_dev_identity(plan["manifest"], plan["manifest_sha256"])
    require(identity == expected_identity, "Prediction target/window identity differs from the complete dev set")
    require(len(predictions) == receipt["count"] == run["dev_windows"] == expected_count, "Incomplete dev prediction count")
    actual = summarize([[float(r["vx"]), float(r["vy"])] for r in predictions],
                       [[float(r["target_vx"]), float(r["target_vy"])] for r in predictions],
                       [r["scene"] for r in predictions])
    summary = read_json(Path(row["evaluation_dir"]) / "metrics.json")
    check_summary(summary, actual)
    row.update(status="completed", metrics=summary, best_epoch=packet["best_epoch"] + 1,
               checkpoint_sha256=receipt["checkpoint_sha256"], target_window_identity=identity,
               predictions_sha256=sha256(row["predictions"]), contract_hash=run["contract_hash"],
               common_config_identity=object_hash({k: v for k, v in config.items() if k != "experiment"}),
               rgb_identity=object_hash(feature) if feature else None, data=data)


def collect_row(plan, job):
    name, output = job["name"], Path(job["run_dir"])
    evaluation = Path(command_value(job["commands"][1], "--output"))
    row = dict(experiment=name, status="not_started", metrics=None, reason=None, best_epoch=None,
               trainable_parameters=None, run_id=output.name, commit=plan["code"]["commit"],
               device=job["device"], run_dir=str(output), evaluation_dir=str(evaluation),
               checkpoint=str(output / "best.pt"), log_path=job["log_path"],
               predictions=str(evaluation / "predictions.csv"))
    try:
        job_state = optional_json(job["status_path"])
        run, status = optional_json(output / "run.json"), optional_json(output / "status.json")
        receipt = optional_json(evaluation / "evaluation.json")
        row["observed_status"] = dict(job=job_state.get("status"), training=status.get("status"),
                                      run=run.get("status"), evaluation=receipt.get("status"))
        row["progress"] = {k: status[k] for k in ("epoch", "global_step", "updated_at") if k in status}
        if run:
            row.update(trainable_parameters=run.get("trainable_parameters"), commit=run.get("code", {}).get("commit"))
        if run.get("smoke") or run.get("contract", {}).get("execution", {}).get("smoke"):
            row.update(status="excluded_smoke", reason="Smoke checks are not formal experiment results")
        elif any(s in {"failed", "interrupted", "cancelled", "blocked"} for s in row["observed_status"].values()):
            row.update(status="failed", reason=job_state.get("error") or job_state.get("reason") or receipt.get("error")
                       or status.get("error") or "A job/training/evaluation record is failed, interrupted or blocked")
        elif run.get("status") == status.get("status") == "completed":
            row["status"] = "evaluation_pending"
            if receipt.get("status") == job_state.get("status") == "completed":
                inspect_completed(row, run, status, receipt, plan, job)
        elif any(s in {"running", "launching"} for s in row["observed_status"].values()):
            row["status"] = "running"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as error:
        row.update(status="invalid", metrics=None, reason=f"{type(error).__name__}: {error}")
    return row


def checked_pack(row):
    data = row["data"]
    path = Path(data["flow_root"]) / "packed_index.json"
    index = read_json(path)
    require(sha256(path) == data["flow"]["packed_index_sha256"], "Recorded packed index checksum differs")
    require(index["schema"] == "v5_packed_flow_v1"
            and index["identity"] == object_hash({k: v for k, v in index.items() if k != "identity"}),
            "Packed index identity mismatch")
    require(index["contract"]["dataset_sha256"] == data["dataset_sha256"], "Packed dataset mismatch")
    return index


def compare(candidate, baseline, *, flow_check=False):
    result = dict(candidate=candidate["experiment"], baseline=baseline["experiment"],
                  status="pending", improvement=None)
    if candidate["status"] != "completed" or baseline["status"] != "completed":
        result["reason"] = "Both runs need valid complete formal training and independent evaluation"
        return result
    try:
        for key in ("common_config_identity", "target_window_identity", "commit"):
            require(candidate[key] == baseline[key], f"Comparison mismatch: {key}")
        for key in ("dataset_sha256", "label"):
            require(candidate["data"][key] == baseline["data"][key], f"Comparison mismatch: {key}")
        if candidate["experiment"] != "R02" and baseline["experiment"] != "R02":
            require(candidate["rgb_identity"] == baseline["rgb_identity"], "Comparison RGB identity mismatch")
        if flow_check:
            first, second = checked_pack(candidate), checked_pack(baseline)
            if {candidate["experiment"], baseline["experiment"]} == {"R04", "R05"}:
                require(first["identity"] == second["identity"] and first["contract"]["fb_available"],
                        "R04/R05 need the same FB cache identity")
            elif candidate["experiment"] in {"R04", "R05"} and baseline["experiment"] == "R03":
                proof = first.get("forward_equality", {})
                require(first["contract"]["fb_available"] and not second["contract"]["fb_available"],
                        "Wrong forward/FB cache roles")
                require(proof.get("flow_and_valid_exact") is True
                        and proof.get("reference_index_sha256") == baseline["data"]["flow"]["packed_index_sha256"]
                        and proof.get("compared_pairs") == len(first["pairs"]) == len(second["pairs"]),
                        "Missing/mismatched forward equality proof for FB comparison")
                pair_keys = lambda index: [(p["source"], p["target"], p["source_time"], p["target_time"]) for p in index["pairs"]]
                require(pair_keys(first) == pair_keys(second), "FB proof has different source pairs")
            else:
                require(first["identity"] == second["identity"], "Flow comparison uses different forward caches")
        a, b = candidate["metrics"], baseline["metrics"]
        result.update(status="comparable", delta_epe=a["overall"]["epe"] - b["overall"]["epe"],
                      delta_mae_vx=a["overall"]["mae_vx"] - b["overall"]["mae_vx"],
                      delta_mae_vy=a["overall"]["mae_vy"] - b["overall"]["mae_vy"])
        if baseline["experiment"] == "R00":
            acceptance = lateral_acceptance(a, b)
            result["lateral_acceptance"] = acceptance
            if any(c["status"] == "insufficient_data" for c in acceptance["checks"].values()):
                result["conclusion"] = "insufficient_data"
            else:
                result["improvement"] = bool(result["delta_epe"] < 0 and acceptance["acceptable"])
                result["conclusion"] = "exploratory_epe_gain_with_lateral_acceptance" if result["improvement"] else "no_accepted_gain"
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="incomparable", reason=str(error))
    return result


def collect(plan_path):
    plan = read_json(plan_path)
    validate_plan(plan)
    manifest = read_json(plan["manifest"])
    require(sha256(plan["manifest"]) == plan["manifest_sha256"], "Pipeline manifest checksum mismatch")
    require(manifest.get("label_contract") == LABEL_CONTRACT and manifest.get("label_sha256") == LABEL_SHA256,
            "Pipeline label contract mismatch")
    jobs = {j["name"]: j for j in plan["jobs"] if j["name"] in EXPERIMENTS}
    require(set(jobs) == set(EXPERIMENTS), "Pipeline must declare all six experiments")
    rows = [collect_row(plan, jobs[name]) for name in EXPERIMENTS]
    by_name = {r["experiment"]: r for r in rows}
    for row in rows[1:]:
        row["vs_R00"] = compare(row, rows[0])
    contrasts = [compare(by_name[a], by_name[b], flow_check=True) for a, b in
                 (("R03", "R01"), ("R04", "R03"), ("R05", "R03"), ("R05", "R04"))]
    return dict(schema="v5_results_v1", generated_at=now(), plan=str(Path(plan_path).resolve()),
                plan_identity=plan["identity"], training_commit=plan["code"]["commit"],
                collector_sha256=sha256(__file__), dataset_sha256=plan["manifest_sha256"],
                metric_units="m/s", best_epoch_numbering="one-based", rows=rows,
                flow_contrasts=contrasts, limitations=LIMITS)


def markdown(report):
    lines = ["# S2 experiment status", "", f"Snapshot: {report['generated_at']}", "",
             f"Training commit: `{report['training_commit']}`. Metrics: m/s. Best epoch: one-based.", "",
             "| Experiment | Status | EPE | MAE vx | MAE vy | Nonzero-vy MAE (n/scenes) | Zero-vy MAE | Best epoch | Trainable parameters |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    number = lambda value: "—" if value is None else f"{value:.6f}"
    for row in report["rows"]:
        summary = row["metrics"] or {}
        overall, nonzero = summary.get("overall", {}), summary.get("nonzero_vy", {})
        subset = "—" if not summary else f"{number(nonzero.get('metrics', {}).get('mae_vy'))} ({nonzero['count']}/{nonzero['scene_count']}; adequate={nonzero['adequate']})"
        lines.append(f"| {row['experiment']} | {row['status']} | {number(overall.get('epe'))} | {number(overall.get('mae_vx'))} | {number(overall.get('mae_vy'))} | {subset} | {number(overall.get('zero_vy_mae'))} | {row['best_epoch'] or '—'} | {row['trainable_parameters'] or '—'} |")
    lines += ["", "Zero-vy MAE is the constant-zero lateral predictor, not a zero-target subset.", "",
              "## Comparisons", ""]
    for row in report["rows"][1:]:
        comparison = row["vs_R00"]
        lines.append(f"- {row['experiment']} vs R00: {comparison.get('conclusion', comparison['status'])}; {comparison.get('reason', 'exploratory single-seed dev comparison')}.")
    for contrast in report["flow_contrasts"]:
        lines.append(f"- {contrast['candidate']} vs {contrast['baseline']} flow provenance: {contrast['status']}; {contrast.get('reason', 'recorded cache identity/equality proof validated')}.")
    lines += ["", "## Records", ""]
    for row in report["rows"]:
        lines += [f"- {row['experiment']}: commit `{row['commit']}`; checkpoint `{row['checkpoint']}`; log `{row['log_path']}`; evaluation `{row['evaluation_dir']}`."]
        if row["reason"]:
            lines.append(f"  Reason: {row['reason']}")
    lines += ["", "## Limits", "", *[f"- {value}" for value in report["limitations"]], ""]
    return "\n".join(lines)


def export(plan_path, output):
    plan, output = read_json(plan_path), Path(output).resolve()
    protected = [ROOT.parents[2], Path(plan["code_root"]).parents[2], Path(plan["manifest"]).parent,
                 Path(plan["features"]), Path(plan["root"]) / "flows", Path(plan_path).resolve().parent]
    for job in plan["jobs"]:
        if job.get("run_dir"):
            protected.append(Path(job["run_dir"]))
            protected.append(Path(command_value(job["commands"][1], "--output")))
    require(not any(output.is_relative_to(p.resolve()) for p in protected), "Report output must be external to source/input/run directories")
    require(not output.exists(), "Use a new report directory to preserve previous snapshots")
    report = collect(plan_path)
    output.mkdir(parents=True)
    write_json(output / "results.json", report)
    (output / "results.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True, help="New external snapshot directory")
    args = parser.parse_args()
    report = export(args.plan, args.output)
    print({r["experiment"]: r["status"] for r in report["rows"]})


if __name__ == "__main__":
    main()
