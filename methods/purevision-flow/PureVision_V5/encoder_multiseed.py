"""Paired seed43/44 replication of the completed full-data seed42 encoders."""
import argparse
from pathlib import Path
import re
import sys
import time

from collect_results import require
from encoder_comparison import comparison_training_job, validate_inputs as validate_seed42
from pipeline import initialize, lock, tick, validate_plan
from purevision_v5.common import ROOT, code_version, object_hash, read_json, sha256
from resolution_pilot import computation_identity

SEEDS = (43, 44)
SOURCE_ID = "0f23c724cb8225d60ccfd5b0b1647363479f2bf8c39ab0fb31d912f1092248a1"
SOURCE_REPORT_SHA = "5aa3ffb431408b24e67a42e6655b793ee6550a74e3423812156d8d788474370d"


def numerical_identity(code_root):
    files = computation_identity(code_root)["files"]
    # The FLEX modules are imported code, not merely a bibliographic reference.
    name = "purevision_v5/_rgb_reference/modules.py"
    files[name] = sha256(Path(code_root) / name)
    return dict(files=files, identity=object_hash(files))


def validate_inputs(plan):
    require(plan["stage"] == "r02_encoder_multiseed", "Wrong replication stage")
    require(plan["worker_network_file"] == str(Path(plan["root"])/"monitoring/network.json"),
            "Worker transport must use task-owned private configuration")
    source = read_json(plan["seed42_plan"])
    validate_plan(source); validate_seed42(source)
    require(source["identity"] == SOURCE_ID and sha256(plan["seed42_plan"]) == plan["seed42_plan_sha256"],
            "Wrong seed42 full-data reference")
    require(sha256(plan["seed42_report"]) == SOURCE_REPORT_SHA == plan["seed42_report_sha256"],
            "Seed42 completed report changed")
    for key in ("manifest", "manifest_sha256", "flow_root", "source_summary", "source_summary_sha256", "expected_counts"):
        require(plan[key] == source[key], f"Changed common data: {key}")
    require(numerical_identity(plan["code_root"]) == numerical_identity(source["code_root"]) == plan["numerical_identity"],
            "Numerical training/model/data/evaluation code differs from seed42")
    adopted=plan.get("external_preparation",[])
    recovery=plan.get("recovery_source")
    if recovery:
        require(sha256(recovery["path"]) == recovery["sha256"], "Recovery source plan changed")
        retained=read_json(recovery["path"]);validate_plan(retained)
        require(retained["identity"] == recovery["identity"] and retained["stage"] == "r02_encoder_multiseed",
                "Wrong recovery source")
        require(numerical_identity(retained["code_root"])==plan["numerical_identity"], "Retained FLEX numerical code differs")
        for key in ("manifest_sha256","flow_root","expected_counts"):
            require(retained[key]==plan[key],f"Retained data differs: {key}")
        require({j["name"] for j in adopted}=={"R02_FLEX_s43","R02_FLEX_s44"} and len(adopted)==2,
                "Recovery retains exactly the two running FLEX jobs")
        for job in adopted:
            original=next(j for j in retained["jobs"] if j["name"]==job["name"])
            require({k:v for k,v in job.items() if k not in {"source_plan","source_commit"}}==original
                    and job["source_plan"]==recovery["path"] and job["source_commit"]==retained["code"]["commit"],
                    "Retained job changed")
    else:
        require(not adopted,"Unexpected retained runs")
    jobs = {j["name"]:j for j in plan["jobs"]+adopted}
    names = {f"R02_{v}_s{s}" for s in SEEDS for v in ("BASE", "FLEX")}
    require(set(jobs) == names | {"compare"} and len(plan["jobs"])+len(adopted) == 5, "Unexpected jobs")
    gate_variant="FLEX" if recovery else "BASE"
    require(plan["startup_gate_jobs"] == [f"R02_{gate_variant}_s43",f"R02_{gate_variant}_s44"], "Initial wave changed")
    for seed, device in ((43, "cuda:0"), (44, "cuda:2")):
        for variant in ("BASE", "FLEX"):
            name = f"R02_{variant}_s{seed}"; job = jobs[name]
            old = next(j for j in source["jobs"] if j["name"] == f"R02_{variant}")
            expected = dict(read_json(old["config_path"]), seed=seed, head_seed=seed)
            require(read_json(job["config_path"]) == expected and sha256(job["config_path"]) == plan["config_hashes"][name],
                    f"Protocol differs beyond seed/head_seed: {name}")
            after=([f"R02_FLEX_s{seed}"] if recovery else []) if variant=="BASE" else [f"R02_BASE_s{seed}"]
            require(job["training_seed"] == seed and job["device"] == device and job["kind"] == "training"
                    and job["after"] == after, "Pair scheduling changed")
    require(set(jobs["compare"]["requires"]) == names, "Comparison must wait for all four independent evaluations")


def build_plan(root, source_plan, prefix, version, *, code_root=ROOT, python=sys.executable, recovery_plan=None):
    root, source_plan, code_root = [Path(p).resolve() for p in (root, source_plan, code_root)]
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",prefix)), "Invalid prefix")
    source=read_json(source_plan);validate_plan(source);validate_seed42(source)
    require(source["root"] == str(root) and source["prefix"] != prefix, "Use same artifact root and fresh prefix")
    jobs=[];base=root/"pipeline"/prefix
    for seed,device in ((43,"cuda:0"),(44,"cuda:2")):
        for variant in ("BASE","FLEX"):
            name=f"R02_{variant}_s{seed}"
            job=comparison_training_job(root,code_root,python,prefix,source["manifest"],source["flow_root"],name,device,
                code_root/"configs/next"/f"{name}.json",after=[] if variant=="BASE" else [f"R02_BASE_s{seed}"])
            job["training_seed"]=seed;jobs.append(job)
    report=root/"reports"/f"encoder_comparison_{prefix}"
    jobs.append(dict(name="compare",job_id=f"{prefix}_compare",kind="comparison",device=None,
        requires=[j["name"] for j in jobs],after=[],
        commands=[[str(python),"-B",str(code_root/"collect_encoder_multiseed.py"),"--plan",str(base/"plan.json"),"--output",str(report)]],
        session=f"v5next_{prefix}_compare",log_path=str(root/"logs"/f"{prefix}_compare.log"),
        status_path=str(base/"jobs/compare.json"),progress_path=None,run_dir=None,fresh_output=str(report),
        checks=[dict(path=str(report/"summary.json"),field="status",equals="completed")]))
    report42=root/"reports"/f"encoder_comparison_{source['prefix']}"/"summary.json"
    plan=dict(schema="v5_pipeline_v1",stage="r02_encoder_multiseed",scope="full_train_dev",root=str(root),
        worker_network_file=str(root/"monitoring/network.json"),
        code_root=str(code_root),python=str(python),prefix=prefix,code=version,features=None,features_status=None,
        seed42_plan=str(source_plan),seed42_plan_sha256=sha256(source_plan),seed42_report=str(report42),
        seed42_report_sha256=sha256(report42),numerical_identity=numerical_identity(code_root),
        startup_gate_jobs=["R02_BASE_s43","R02_BASE_s44"],new_seeds=list(SEEDS),all_seeds=[42,43,44],
        config_hashes={j["name"]:sha256(j["config_path"]) for j in jobs if j.get("config_path")},jobs=jobs,
        research_decision=dict(mainline="FLEX",source="user decision after seed42",objective="prediction accuracy",
            cost_is_selection_gate=False,statistical_superiority_claim=False),
        **{k:source[k] for k in ("manifest","manifest_sha256","flow_root","source_summary","source_summary_sha256","expected_counts")})
    if recovery_plan:
        recovery_path=Path(recovery_plan).resolve();retained=read_json(recovery_path);validate_plan(retained)
        require(retained["root"]==str(root) and retained["prefix"]!=prefix,"Recovery requires new ID and same root")
        plan["recovery_source"]=dict(path=str(recovery_path),sha256=sha256(recovery_path),identity=retained["identity"])
        adopted=[dict(j,source_plan=str(recovery_path),source_commit=retained["code"]["commit"])
                 for j in retained["jobs"] if j["name"] in {"R02_FLEX_s43","R02_FLEX_s44"}]
        for job in adopted:
            run=read_json(Path(job["run_dir"])/"run.json")
            state=read_json(Path(job["run_dir"])/"status.json")
            require(run and not run.get("smoke") and run["code"]==retained["code"] and state
                    and state["status"] in {"running","completed"} and state.get("optimizer_updates",0)>0,
                    "Only actual valid running/completed FLEX can be retained")
        for job in retained["jobs"]:
            if job["name"].startswith("R02_BASE"):
                require(read_json(job["status_path"])["status"]=="failed"
                        and not (Path(job["run_dir"])/"run.json").exists(),"Only pre-training BASE failures may be replaced")
        plan["jobs"]=[j for j in jobs if j["name"] not in {"R02_FLEX_s43","R02_FLEX_s44"}]
        for job in plan["jobs"]:
            if job["kind"]=="training":job["after"]=[f"R02_FLEX_s{job['training_seed']}"]
        plan["external_preparation"]=adopted  # Existing DAG API: observe/depend on these jobs; never relaunch them.
        plan["startup_gate_jobs"]=["R02_FLEX_s43","R02_FLEX_s44"]
        plan["config_hashes"]={j["name"]:sha256(j["config_path"]) for j in plan["jobs"]+adopted if j.get("config_path")}
    validate_inputs(plan)
    return dict(plan,identity=object_hash(plan))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","source-plan","run-prefix"):p.add_argument("--"+key,required=True)
    p.add_argument("--recovery-plan")
    p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    plan=build_plan(a.root,a.source_plan,a.run_prefix,code_version(require_pushed=not a.dry_run),recovery_plan=a.recovery_plan)
    if a.dry_run:
        import json
        print(json.dumps(plan,indent=2))
    else:
        with lock(Path(a.root)/"pipeline/coordinator.lock"):
            path=initialize(plan);print(dict(plan=str(path),identity=plan["identity"]),flush=True)
            while True:
                done,launched=tick(plan,path)
                if launched:print(dict(launched=launched),flush=True)
                if done:break
                time.sleep(10)
