"""Three paired training seeds on one fixed pilot selection, with sample SD."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_resolution import collect as collect_seed42, compare_rows, inspect_run
from collect_results import require
from lateral_baselines import baseline_report
from pipeline import validate_plan
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.selection import label_rows
from resolution_multiseed import validate_multiseed_inputs


METRICS = ("epe", "mae_vx", "mae_vy", "rmse_vx", "rmse_vy", "bias_vx", "bias_vy")


def mean_sd(values):
    values = np.asarray(values, float)
    require(values.shape == (3,) and np.isfinite(values).all(), "Exactly three finite paired seeds are required")
    return dict(n=3, values=values.tolist(), mean=float(values.mean()), sample_sd=float(values.std(ddof=1)), ddof=1)


def aggregate(rows, pairs):
    result = dict(status="pending", decision="pending_user_choice", requires_full_generation_approval=True,
                  training_seeds=[42,43,44], note="Training-seed variability on one fixed scene selection; no significance claim")
    if any(pair["status"] != "comparable" for pair in pairs.values()):
        return result
    try:
        reference = rows["R02_512_s42"]
        protocol = lambda r: {k:v for k,v in r["config"].items() if k not in {"seed", "expected_flow"}}
        for row in rows.values():
            require(protocol(row) == protocol(reference) and row["target_identity"] == reference["target_identity"]
                    and row["computation"] == reference["computation"], "Cross-seed protocol/targets/numerical code mismatch")
        resolutions = {}
        for width in (512,1024):
            resolutions[str(width)] = {metric: mean_sd([rows[f"R02_{width}_s{seed}"]["metrics"]["overall"][metric]
                                                       for seed in (42,43,44)]) for metric in METRICS}
        paired = {metric:mean_sd([pairs[str(seed)]["delta_high_minus_low"][metric] for seed in (42,43,44)]) for metric in METRICS}
        scene_deltas = {}
        for scene in sorted(reference["metrics"]["per_scene"]):
            values = [rows[f"R02_1024_s{seed}"]["metrics"]["per_scene"][scene]["epe"] -
                      rows[f"R02_512_s{seed}"]["metrics"]["per_scene"][scene]["epe"] for seed in (42,43,44)]
            scene_deltas[scene] = dict(mean_sd(values), windows=reference["metrics"]["per_scene"][scene]["count"],
                                      improved_seed_count=sum(value < 0 for value in values))
        subset = reference["metrics"]["nonzero_vy"]
        result.update(status="comparable", by_resolution=resolutions, paired_high_minus_low=paired, per_scene_epe_delta=scene_deltas,
                      epe_improved_seed_count=sum(value < 0 for value in paired["epe"]["values"]),
                      scenes_improved_on_mean=sum(value["mean"] < 0 for value in scene_deltas.values()),
                      lateral=dict(unique_nonzero_windows=subset["count"], unique_nonzero_scenes=subset["scene_count"],
                                   adequate=subset["adequate"], effective_rule=reference["metrics"]["effective_lateral_rule"],
                                   per_seed={seed:pair["lateral_acceptance"] for seed,pair in pairs.items()},
                                   note="The same lateral windows repeat across seeds; 13 windows do not become 39 independent samples"))
    except (KeyError, ValueError, TypeError) as error:
        result.update(status="incomparable", reason=str(error))
    return result


def collect(plan_path):
    plan = read_json(plan_path)
    validate_plan(plan)
    validate_multiseed_inputs(plan)
    original = collect_seed42(plan["seed42_source"]["path"])
    require(original["comparison"]["status"] == "comparable", "Seed-42 source no longer validates")
    train_rows, _ = label_rows(plan["manifest"], "train_core", plan["selection_path"])
    dev_rows, _ = label_rows(plan["manifest"], "dev", plan["selection_path"])
    rows = {}
    for width in (512,1024):
        old = original["rows"][f"R02_{width}"]
        expected = plan["seed42_source"]["rows"][f"R02_{width}"]
        require(all(old[key] == value for key,value in expected.items()), "Seed-42 source evidence changed")
        name = f"R02_{width}_s42"
        rows[name] = dict(old, name=name, training_seed=42, adopted_from_source_plan=plan["seed42_source"]["path"])
    for job in plan["jobs"]:
        row = inspect_run(plan, job, train_rows, dev_rows, expected_seed=job["training_seed"])
        row["training_seed"] = job["training_seed"]
        if row["status"] == "completed":
            locked = plan["locked_packs"][str(job["raft_width"])]
            if row["data"]["flow"]["packed_index_sha256"] != locked["index_sha256"]:
                row.update(status="invalid", reason="Actual run uses a different locked flow pack", metrics=None)
        rows[job["name"]] = row
    pairs = {str(seed):compare_rows({f"R02_{width}":rows[f"R02_{width}_s{seed}"] for width in (512,1024)}) for seed in (42,43,44)}
    rule = rows["R02_512_s42"]["config"]["lateral_rule"]
    baselines = baseline_report(plan["manifest"], plan["selection_path"], rule,
                                [row["predictions"] for row in rows.values() if row["status"] == "completed"])
    return dict(schema="v5_resolution_multiseed_report_v1", generated_at=now(), scope="resolution_pilot",
                plan=str(Path(plan_path).resolve()), plan_sha256=sha256(plan_path), plan_identity=plan["identity"],
                training_commit=plan["code"]["commit"], training_commits={name:row["training_commit"] for name,row in rows.items()},
                collector_sha256=sha256(__file__), seed42_source=plan["seed42_source"], selection=plan["selection"],
                locked_packs=plan["locked_packs"], rows=rows, paired_by_seed=pairs, comparison=aggregate(rows,pairs),
                lateral_baselines=baselines, incremental_raft_generation=dict(pairs=0, seconds=0, new_cache_bytes=0), metric_units="m/s",
                limitations=["Three training seeds reuse exactly the same 32 train / 8 dev scenes; selection seed remains 42.",
                             "Mean and sample SD use n=3 training runs and ddof=1; paired deltas are high minus low within each seed.",
                             "The nonzero-vy subset still has 13 unique windows in one scene, not 39 independent observations.",
                             "PLANAR_PROXY and same-dev checkpoint selection/recomputation; no held-out test or statistical-significance claim.",
                             "No new RAFT, full cache, Flex encoder or automatic frontend selection is authorized by this report."])


def markdown(report):
    lines = ["# R02 resolution pilot: three paired training seeds", "", "Scene selection remains fixed at seed 42. Training seeds: 42, 43, 44.",
             "", "| Seed | Resolution | Status | EPE | MAE vx | MAE vy | Best epoch | Updates / AMP skips |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for seed in (42,43,44):
        for width in (512,1024):
            row = report["rows"][f"R02_{width}_s{seed}"]
            if row["status"] == "completed":
                m = row["metrics"]["overall"]
                lines.append(f"| {seed} | {width} | completed | {m['epe']:.6f} | {m['mae_vx']:.6f} | {m['mae_vy']:.6f} | {row['best_epoch_one_based']} | {row['optimizer_updates']} / {row['amp_skips']} |")
            else:
                lines.append(f"| {seed} | {width} | {row['status']} | — | — | — | — | — |")
    comparison = report["comparison"]
    lines += ["", f"Comparison: **{comparison['status']}**. Decision: **pending_user_choice**."]
    if comparison["status"] == "comparable":
        lines += ["", "Mean ± sample SD (n=3, ddof=1), m/s:", "", "| Metric | 512 | 1024 | Paired high − low |", "|---|---:|---:|---:|"]
        for metric in METRICS:
            values = [comparison["by_resolution"][str(w)][metric] for w in (512,1024)] + [comparison["paired_high_minus_low"][metric]]
            lines.append(f"| {metric} | " + " | ".join(f"{v['mean']:.6f} ± {v['sample_sd']:.6f}" for v in values) + " |")
        lines += ["", "Paired EPE deltas by seed 42/43/44: " + ", ".join(f"{v:+.6f}" for v in comparison["paired_high_minus_low"]["epe"]["values"]) + ".",
                  "", "| Scene | Mean paired EPE delta | Sample SD | Improved seeds |", "|---|---:|---:|---:|"]
        for scene,v in comparison["per_scene_epe_delta"].items():
            lines.append(f"| {scene} | {v['mean']:+.6f} | {v['sample_sd']:.6f} | {v['improved_seed_count']} / 3 |")
    lines += ["", *[f"- {item}" for item in report["limitations"]], "", "Actual training commits:"]
    lines += [f"- {name}: `{commit}`" for name,commit in report["training_commits"].items()]
    for name,row in report["rows"].items():
        if row.get("reason"):
            lines.append(f"- {name}: {row['reason']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("A new report directory is required")
    report = collect(args.plan)
    write_json(output / "summary.json", report)
    (output / "summary.md").write_text(markdown(report))
    print({"report":str(output), "statuses":{name:row["status"] for name,row in report["rows"].items()},
           "comparison":report["comparison"]["status"], "decision":"pending_user_choice"})


if __name__ == "__main__":
    main()
