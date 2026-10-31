"""Two full-data, 40-epoch flow-only encoders, then a verified comparison."""
import argparse
from pathlib import Path
import re
import sys
import time

from collect_results import require
from pipeline import initialize, lock, tick
from purevision_v5.common import ROOT, code_version, object_hash, read_json, sha256

SOURCE_SHA = "96946ac49805c1d4d57668bd0ff48db7cffa24371e2e277458dadcba8516b25c"
MANIFEST_SHA = "a74f7963f7c9e0b529fefef75f5be1c191a7a2fccbddd3678b82f825ee0de32a"


def validate_inputs(plan):
    require(plan["stage"] == "r02_encoder_comparison", "Wrong comparison stage")
    require(sha256(plan["source_summary"]) == SOURCE_SHA == plan["source_summary_sha256"], "Completed cache evidence changed")
    require(sha256(plan["manifest"]) == MANIFEST_SHA == plan["manifest_sha256"], "Original labels/split changed")
    source=read_json(plan["source_summary"])
    require(source["status"] == "complete" and source["generation_code"]["commit"] ==
            "87d7468f911da5579bbbc39f79ec6bc85fa16dd6", "Wrong generation identity")
    pack=source["packed_train_dev"]
    require(plan["flow_root"] == pack["root"] and sha256(Path(pack["root"])/"packed_index.json") == pack["index_sha256"],
            "Packed cache changed")
    configs=[]
    for name,variant in (("R02_BASE","base"),("R02_FLEX","flex")):
        job=next(j for j in plan["jobs"] if j["name"]==name)
        config=read_json(job["config_path"])
        require(sha256(job["config_path"])==plan["config_hashes"][name], "Configuration changed")
        require(config["encoder_variant"]==variant and config["experiment"]=="R02" and config["seed"]==42
                and config["head_seed"]==42 and config["epochs"]==40 and config["batch_size"]==64
                and config["early_stopping"] is False, "Protocol differs from authorized comparison")
        configs.append({k:v for k,v in config.items() if k!="encoder_variant"})
    require(configs[0]==configs[1], "Protocols differ beyond encoder")
    require(len(plan["jobs"])==3 and {j["name"] for j in plan["jobs"]}=={"R02_BASE","R02_FLEX","compare"}, "Unexpected jobs")


def build_plan(root, manifest, prefix, version, *, code_root=ROOT, python=sys.executable):
    root,manifest,code_root=map(lambda p:Path(p).resolve(),(root,manifest,code_root))
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",prefix)), "Invalid run prefix")
    source=root/"reports/full_flow_20261030_full_nuscenes_front1024/summary.json"
    flow=read_json(source)["packed_train_dev"]["root"]
    base=root/"pipeline"/prefix
    plan_path=base/"plan.json"
    jobs=[]
    def cmd(script,*args): return [str(python),"-B",str(code_root/script),*map(str,args)]
    for name,device in (("R02_BASE","cuda:1"),("R02_FLEX","cuda:2")):
        config=code_root/"configs/next"/f"{name}.json"
        output=root/"runs"/f"{prefix}_{name}"
        evaluation=root/"evaluations"/output.name
        inputs=["--manifest",str(manifest),"--flow",flow,"--device",device,"--workers","4"]
        jobs.append(dict(name=name,job_id=output.name,kind="training",experiment="R02",device=device,requires=[],after=[],
            config_path=str(config),commands=[cmd("train.py","--config",config,*inputs,"--output",output),
                cmd("evaluate.py","--checkpoint",output/"best.pt",*inputs,"--output",evaluation)],
            session=f"v5next_{prefix}_{name}",log_path=str(root/"logs"/f"{output.name}.log"),
            status_path=str(base/"jobs"/f"{name}.json"),progress_path=str(output/"status.json"),
            run_dir=str(output),fresh_output=str(output),storage_paths=[flow],
            checks=[dict(path=str(output/"status.json"),field="status",equals="completed"),
                    dict(path=str(evaluation/"evaluation.json"),field="status",equals="completed"),
                    dict(path=str(output/"last.pt")),dict(path=str(output/"best.pt"))]))
    report=root/"reports"/f"encoder_comparison_{prefix}"
    jobs.append(dict(name="compare",job_id=f"{prefix}_compare",kind="comparison",device=None,
        requires=["R02_BASE","R02_FLEX"],after=[],commands=[cmd("collect_encoders.py","--plan",plan_path,"--output",report)],
        session=f"v5next_{prefix}_compare",log_path=str(root/"logs"/f"{prefix}_compare.log"),
        status_path=str(base/"jobs/compare.json"),progress_path=None,run_dir=None,fresh_output=str(report),
        checks=[dict(path=str(report/"summary.json"),field="status",equals="completed")]))
    plan=dict(schema="v5_pipeline_v1",stage="r02_encoder_comparison",scope="full_train_dev",root=str(root),
        code_root=str(code_root),python=str(python),prefix=prefix,code=version,manifest=str(manifest),
        manifest_sha256=sha256(manifest),flow_root=flow,source_summary=str(source),source_summary_sha256=sha256(source),
        features=None,features_status=None,config_hashes={j["name"]:sha256(j["config_path"]) for j in jobs if j.get("config_path")},
        jobs=jobs,expected_counts=dict(train_core=dict(scenes=582,windows=132978),dev=dict(scenes=103,windows=23480)))
    validate_inputs(plan)
    return dict(plan,identity=object_hash(plan))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","manifest","run-prefix"): p.add_argument("--"+k,required=True)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    plan=build_plan(a.root,a.manifest,a.run_prefix,code_version(require_pushed=not a.dry_run))
    if a.dry_run:
        import json
        print(json.dumps(plan,indent=2));return
    with lock(Path(a.root)/"pipeline/coordinator.lock"):
        path=initialize(plan)
        print(dict(plan=str(path),identity=plan["identity"]),flush=True)
        while True:
            done,launched=tick(plan,path)
            if launched: print(dict(launched=launched),flush=True)
            if done: break
            time.sleep(10)


if __name__=="__main__": main()
