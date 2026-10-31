"""Full-data BASE/FLEX verification, constants, costs and provisional choice."""
import argparse
from pathlib import Path
import numpy as np

from collect_resolution import inspect_run
from collect_results import require
from encoder_comparison import validate_inputs
from lateral_baselines import baseline_report
from pipeline import validate_plan
from purevision_v5.common import now, object_hash, read_json, write_json
from purevision_v5.evaluation import lateral_acceptance, summarize
from purevision_v5.selection import label_rows


def collect(plan_path):
    plan=read_json(plan_path);validate_plan(plan);validate_inputs(plan)
    train,_=label_rows(plan["manifest"],"train_core")
    dev,_=label_rows(plan["manifest"],"dev")
    rows={j["name"]:inspect_run(plan,j,train,dev) for j in plan["jobs"] if j["kind"]=="training"}
    require(all(r["status"]=="completed" for r in rows.values()), f"Run verification failed: {rows}")
    base,flex=rows["R02_BASE"],rows["R02_FLEX"]
    common=lambda r:{k:v for k,v in r["config"].items() if k!="encoder_variant"}
    require(common(base)==common(flex),"Training protocols differ")
    for key in ("data","target_identity","sample_order","epoch_order_sha256"):
        require(base[key]==flex[key],f"Unfair comparison: {key}")
    require(base["initialization"]["head_state_sha256"]==flex["initialization"]["head_state_sha256"],"Common head initialization differs")
    rule=base["config"]["lateral_rule"]
    baseline=baseline_report(plan["manifest"],rule=rule,predictions=[r["predictions"] for r in rows.values()])
    train_y=np.asarray([[r["target_vx"],r["target_vy"]] for r in train])
    target=np.asarray([[r["target_vx"],r["target_vy"]] for r in dev])
    constant=np.median(train_y,axis=0)
    baseline["train_median_xy"]=dict(fit_partition="train_core",constant_m_s=constant.tolist(),
        metrics=summarize(np.broadcast_to(constant,target.shape),target,[r["scene"] for r in dev],rule))
    lateral=lateral_acceptance(flex["metrics"],base["metrics"],rule)
    delta={k:flex["metrics"]["overall"][k]-base["metrics"]["overall"][k]
           for k in ("epe","mae_vx","mae_vy","rmse_vx","rmse_vy","bias_vx","bias_vy")}
    improved=sum(flex["metrics"]["per_scene"][s]["epe"]<v["epe"] for s,v in base["metrics"]["per_scene"].items())
    if delta["epe"]<0 and lateral["checks"]["overall"]["status"]=="pass":
        choice="FLEX is promising for further seeds; retain BASE as the reference until lateral evidence is adequate"
    else:
        choice="retain BASE; current FLEX lacks joint EPE/lateral evidence for adoption"
    for row in rows.values():
        row["end_to_end_train_windows_per_second"]=len(train)*40/row["elapsed_seconds"]
        row["cost_note"]="Training wall time includes dev evaluation/checkpoint IO; not pure GPU kernel throughput"
    return dict(schema="v5_encoder_comparison_v1",status="completed",generated_at=now(),plan_identity=plan["identity"],
        training_commit=plan["code"]["commit"],scope="full_train_dev_PLANAR_PROXY_single_seed",counts=plan["expected_counts"],
        rows=rows,baselines=baseline,comparison=dict(status="comparable",delta_flex_minus_base=delta,
        flex_improved_scenes=improved,total_scenes=103,lateral_acceptance=lateral,recommendation=choice,
        common_head_sha256=base["initialization"]["head_state_sha256"],
        limits=["Dev model selection is not held-out evaluation", "Single seed cannot establish statistical superiority",
                "31 nonzero-vy windows in 5 scenes: insufficient_data; threshold remains 100 windows / 5 scenes",
                "Candidate combines stem width, SFE, MFE, BN and pooling; no attribution to an individual component"],
        next_seeds_if_promising=[43,44]))


def markdown(report):
    lines=["# R02_BASE / R02_FLEX — full train/dev, seed42, 40 epochs", "",f"Training commit: `{report['training_commit']}`", "",
           "| Encoder | EPE | vx MAE | vy MAE | vx RMSE | vy RMSE | vx bias | vy bias | Params | Wall seconds | AMP skips |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in report["rows"].items():
        m=r["metrics"]["overall"]
        values=[name]+[f"{m[k]:.6f}" for k in ("epe","mae_vx","mae_vy","rmse_vx","rmse_vy","bias_vx","bias_vy")]
        values += [str(r["trainable_parameters"]),f"{r['elapsed_seconds']:.1f}",str(r["amp_skips"])]
        lines.append("| "+" | ".join(values)+" |")
    lines += ["",report["comparison"]["recommendation"],"","非零侧向：31窗口/5场景，insufficient_data。单seed，不宣称显著优越。",
              "完整逐场景、常量基线、峰值显存、吞吐、哈希和初始化证据见同目录 summary.json；逐窗口预测在 experiments/ 对应run目录。",""]
    return "\n".join(lines)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=collect(a.plan);write_json(out/"summary.json",result);(out/"summary.md").write_text(markdown(result))
    print(result["comparison"])
