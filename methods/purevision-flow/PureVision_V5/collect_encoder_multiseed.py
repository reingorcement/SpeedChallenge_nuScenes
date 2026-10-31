"""Verify six full-budget encoder runs; seeds replicate training, not dev samples."""
import argparse
from pathlib import Path
import numpy as np

from collect_encoders import collect as collect_seed42
from collect_resolution import inspect_run
from collect_results import require
from encoder_multiseed import validate_inputs
from pipeline import validate_plan
from purevision_v5.common import now, read_json, write_json
from purevision_v5.evaluation import lateral_acceptance
from purevision_v5.selection import label_rows

METRICS=("epe","mae_vx","mae_vy","rmse_vx","rmse_vy","bias_vx","bias_vy")


def compare_pair(base, flex):
    common=lambda r:{k:v for k,v in r["config"].items() if k!="encoder_variant"}
    require(common(base)==common(flex),"Pair protocols differ beyond encoder")
    for key in ("data","target_identity","sample_order","epoch_order_sha256","computation"):
        require(base[key]==flex[key],f"Pair differs in {key}")
    require(base["initialization"]["head_state_sha256"]==flex["initialization"]["head_state_sha256"],"Pair head initialization differs")
    b,f=[r["metrics"]["overall"] for r in (base,flex)]
    return dict(delta_flex_minus_base={k:f[k]-b[k] for k in METRICS},
        epe_improvement_percent=100*(b["epe"]-f["epe"])/b["epe"],
        common_head_sha256=base["initialization"]["head_state_sha256"],
        flex_improved_scenes=sum(flex["metrics"]["per_scene"][s]["epe"]<v["epe"] for s,v in base["metrics"]["per_scene"].items()),
        lateral_acceptance=lateral_acceptance(flex["metrics"],base["metrics"],base["config"]["lateral_rule"]))


def collect(plan_path):
    plan=read_json(plan_path);validate_plan(plan);validate_inputs(plan)
    # Recheck old run receipts/checkpoints/predictions, never rerun seed42 training.
    old=collect_seed42(plan["seed42_plan"])
    train,_=label_rows(plan["manifest"],"train_core");dev,_=label_rows(plan["manifest"],"dev")
    rows={f"{name}_s42":row for name,row in old["rows"].items()}
    for job in plan["jobs"]:
        if job["kind"]=="training":
            rows[job["name"]]=inspect_run(plan,job,train,dev,expected_seed=job["training_seed"])
    require(all(r["status"]=="completed" for r in rows.values()),f"Incomplete/invalid comparison: {[(k,r.get('status'),r.get('reason')) for k,r in rows.items()]}")
    reference=rows["R02_BASE_s42"]
    for row in rows.values():
        for key in ("data","target_identity","computation"):
            require(row[key]==reference[key],f"Across-seed data/numerical identity mismatch: {key}")
        row["end_to_end_train_windows_per_second"]=len(train)*40/row["elapsed_seconds"]
        row["cost_note"]="Includes dev/checkpoint IO. Seed43 GPU0 RTX PRO 6000 Max-Q; seed44 GPU2 RTX5090; compare paired cost within seed. Seed42 GPU1 had a concurrent desktop service."
    pairs={str(seed):compare_pair(rows[f"R02_BASE_s{seed}"],rows[f"R02_FLEX_s{seed}"]) for seed in (42,43,44)}
    aggregate={}
    for variant in ("BASE","FLEX"):
        aggregate[variant]={k:dict(mean=float(np.mean(v:=[rows[f'R02_{variant}_s{s}']["metrics"]["overall"][k] for s in (42,43,44)])),
                                   sample_std=float(np.std(v,ddof=1))) for k in METRICS}
    b,f=aggregate["BASE"]["epe"]["mean"],aggregate["FLEX"]["epe"]["mean"]
    return dict(schema="v5_encoder_multiseed_v1",status="completed",generated_at=now(),plan_identity=plan["identity"],
        scope="full_train_dev_PLANAR_PROXY_three_training_seeds",training_commit=plan["code"]["commit"],
        training_commits=sorted({r["training_commit"] for r in rows.values()}),counts=plan["expected_counts"],
        rows=rows,paired_comparisons=pairs,aggregate=aggregate,baselines=old["baselines"],
        comparison=dict(status="comparable",epe_improvement_of_seed_means_percent=100*(b-f)/b,
            flex_epe_wins=sum(p["delta_flex_minus_base"]["epe"]<0 for p in pairs.values()),seeds=3,
            mainline="FLEX",selection_basis="User research decision: accuracy priority; BASE retained as direct reference",
            independent_nonzero_vy_windows=31,independent_nonzero_vy_scenes=5,lateral_evidence="insufficient_data",
            statistical_superiority_claim=False,limits=[
                "Three training seeds share the same 23480 dev windows and dev selection; not independent held-out datasets",
                "Nonzero-vy coverage remains 31 unique windows / 5 scenes, below the unchanged 100 / 5 rule",
                "GPU hardware differs across seeds; paired architectures within each new seed use the same GPU",
                "No RGB, FB or geometry added; no attribution to an individual FLEX submodule"]))


def markdown(report):
    lines=["# R02 BASE/FLEX — full data, 40 epochs, paired seeds42/43/44", "",
        "FLEX is the user-selected development mainline (accuracy priority); BASE remains the direct reference.","",
        "| Run | EPE | vx MAE | vy MAE | vx RMSE | vy RMSE | vx bias | vy bias | Params | Wall s | AMP skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in report["rows"].items():
        fields=[name]+[f'{r["metrics"]["overall"][k]:.6f}' for k in METRICS]
        fields += [str(r["trainable_parameters"]),f'{r["elapsed_seconds"]:.1f}',str(r["amp_skips"])]
        lines.append("| "+" | ".join(fields)+" |")
    c=report["comparison"]
    lines += ["",f'FLEX EPE wins: {c["flex_epe_wins"]}/3 seeds. Improvement of mean EPE: {c["epe_improvement_of_seed_means_percent"]:.3f}%.',
        "", "Nonzero-vy: 31 independent windows / 5 scenes, insufficient_data. Repeated seeds do not add samples.",
        "Three seeds describe initialization sensitivity; no statistical-superiority claim. Costs are recorded, not selection gates.",
        "Full JSON includes paired lateral acceptance, component metrics/bias, per-scene/subset/constant results, hashes and costs.", ""]
    return "\n".join(lines)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=collect(a.plan);write_json(out/"summary.json",result);(out/"summary.md").write_text(markdown(result))
    print(result["comparison"])
