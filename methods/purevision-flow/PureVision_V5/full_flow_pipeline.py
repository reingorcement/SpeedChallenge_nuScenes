"""Generate the explicitly approved full nuScenes CAM_FRONT cache; no training."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys
import time

from pipeline import initialize, lock, tick, validate_plan
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256
from purevision_v5.full_flow_index import iter_groups, load_index


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    require(config.get("schema") == "v5_full_flow_config_v1", "Invalid full-flow configuration")
    require(config["cameras"] == ["CAM_FRONT"] and config["versions"] == ["v1.0-test", "v1.0-trainval"],
            "Only the user-approved complete CAM_FRONT scope is allowed")
    require(config["direction"] == "forward" and config["raft_size_hw"] == [576,1024]
            and config["model_size_hw"] == [72,128], "Changed approved flow computation/view")
    require(config["devices"] == ["cuda:0", "cuda:1", "cuda:2"] and config["batch_size"] == 8,
            "Unexpected GPU scheduling/batch configuration")
    require(config["pack_partitions"] == ["train_core", "dev"]
            and config["labels"] == "unchanged_not_used_for_generation", "Changed data boundary")
    require(config["permission"] == "user_selected_all_1000_scenes_CAM_FRONT_1024x576", "Missing approved scope")
    require(config["threads_per_worker"] == 4 and config["estimated_bytes_per_pair"] > 0
            and config["disk_reserve_bytes"] >= 0 and config["storage_estimate_margin"] >= 1,
            "Invalid resource configuration")


def validate_inputs(plan):
    """Called by workers before launching a command; never changes input files."""
    require(plan.get("stage") == "full_nuscenes_flow" and plan.get("full_generation_authorized") is True,
            "Not an approved full-flow plan")
    require(plan.get("training_authorized") is False, "This plan does not start model training")
    for path_key, hash_key in (("manifest", "manifest_sha256"), ("data_manifest", "data_manifest_sha256"),
                               ("adoption", "adoption_sha256"), ("config_path", "config_sha256")):
        require(sha256(plan[path_key]) == plan[hash_key], f"Immutable {path_key} changed")
    config = read_json(plan["config_path"])
    validate_config(config)
    require(config == plan["config"], "Configuration differs from plan")
    index = load_index(plan["manifest"])
    require(index["identity"] == plan["index_identity"] and index["counts"] == plan["counts"], "Index changed")
    require(index["cameras"] == config["cameras"] and index["versions"] == config["versions"], "Index scope differs")
    require(index["partition_source"]["manifest_sha256"] == plan["data_manifest_sha256"]
            and Path(index["partition_source"]["manifest"]).resolve() == Path(plan["data_manifest"]),
            "Training subset manifest differs from index partition source")
    from purevision_v5.full_flow_cache import load_adoption_plan
    load_adoption_plan(plan["adoption"], sha256(Path(plan["code_root"]) / "purevision_v5/flow.py"))
    require(len(plan["jobs"]) == 5 and [j["kind"] for j in plan["jobs"]] == ["flow"]*3 + ["pack", "verify"],
            "Full-cache plan must have only three generators, one pack and one verification job")
    require(sum(j["expected_pairs"] for j in plan["jobs"] if j["kind"] == "flow") == index["counts"]["pairs"],
            "Shard counts do not cover the full index")
    return index


def build_plan(root, index_path, data_manifest, output_root, adoption, prefix, version, *,
               config_path=None, code_root=ROOT, python=sys.executable):
    root, index_path, data_manifest, output_root, adoption, code_root = [Path(p).resolve() for p in
        (root,index_path,data_manifest,output_root,adoption,code_root)]
    config_path = Path(config_path or code_root / "configs/next/full_nuscenes_flow.json").resolve()
    require(config_path.is_relative_to(code_root), "Configuration must be part of the fixed code tree")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",prefix), "Invalid run prefix")
    config = read_json(config_path)
    validate_config(config)
    index = load_index(index_path)
    dataroot = Path(index["dataroot"]).resolve()
    require(output_root.is_relative_to(dataroot / "derived/optical_flow"), "Full cache must stay in dataset derived/optical_flow")
    require(index_path.is_relative_to(output_root / "index"), "This cache must contain its own immutable index")
    require(index["cameras"] == config["cameras"] and index["versions"] == config["versions"], "Unapproved index scope")
    require(index["partition_source"]["manifest_sha256"] == sha256(data_manifest), "Original manifest changed")
    base = root / "pipeline" / prefix
    jobs = []
    def command(script,*args):
        return [str(python),"-B",str(code_root/script),*map(str,args)]
    def add(name,kind,device,requires,commands,*,report,expected_pairs=None,fresh=None,checks=()):
        jobs.append(dict(name=name,job_id=f"{prefix}_{name}",kind=kind,device=device,requires=list(requires),after=[],
                         commands=commands,session=f"v5next_{prefix}_{name}",
                         log_path=str(root/"logs"/f"{prefix}_{name}.log"),status_path=str(base/"jobs"/f"{name}.json"),
                         progress_path=str(report) if report else None,run_dir=None,fresh_output=str(fresh) if fresh else None,
                         expected_pairs=expected_pairs,storage_paths=[str(output_root)],checks=list(checks)))
    for shard,device in enumerate(config["devices"]):
        name = f"flow1024_{shard}"
        report = base/"reports"/f"{name}.json"
        count = sum(g["pair_count"] for g in iter_groups(index,shard_index=shard,shard_count=3))
        add(name,"flow",device,[],[command("prepare_full_flow.py","--index",index_path,"--index-sha256",sha256(index_path),
            "--output-root",output_root,"--adoption",adoption,"--device",device,"--batch-size",config["batch_size"],
            "--threads",config["threads_per_worker"],"--shard-index",shard,"--shard-count",3,"--report",report)],report=report,expected_pairs=count,
            checks=[dict(path=str(report),field="status",equals="complete")])
    flow_names = [j["name"] for j in jobs]
    pack = output_root/"packed_train_core_dev"
    pack_report = pack/"packed_index.json"
    # The pack is explicitly restricted to the pre-existing supervised manifest;
    # extra all-scene pairs never become train/dev windows through this step.
    add("pack_train_dev","pack",None,flow_names,[command("pack_flow.py","--manifest",data_manifest,
        "--source-root",output_root/"v1.0-trainval/CAM_FRONT","--output",pack)],report=None,
        fresh=pack,checks=[dict(path=str(pack_report),field="schema",equals="v5_packed_flow_v1")])
    summary = root/"reports"/f"full_flow_{prefix}"
    add("verify_full_cache","verify",None,[*flow_names,"pack_train_dev"],
        [command("collect_full_flow.py","--plan",base/"plan.json","--output",summary)],
        report=summary/"summary.json",fresh=summary,checks=[dict(path=str(summary/"summary.json"),field="status",equals="complete")])
    disk = shutil.disk_usage(dataroot)
    estimate = int(index["counts"]["pairs"]*config["estimated_bytes_per_pair"]*config["storage_estimate_margin"])
    require(disk.free > estimate + config["disk_reserve_bytes"], "Insufficient dataset disk headroom for full cache")
    plan = dict(schema="v5_pipeline_v1",stage="full_nuscenes_flow",scope="all_available_nuscenes_CAM_FRONT",
                root=str(root),code_root=str(code_root),python=str(python),prefix=prefix,code=version,
                manifest=str(index_path),manifest_sha256=sha256(index_path),index_identity=index["identity"],counts=index["counts"],
                data_manifest=str(data_manifest),data_manifest_sha256=sha256(data_manifest),output_root=str(output_root),
                adoption=str(adoption),adoption_sha256=sha256(adoption),config_path=str(config_path),
                config_sha256=sha256(config_path),config=config,config_hashes={},features=None,features_status=None,
                full_generation_authorized=True,training_authorized=False,packed_train_dev=str(pack),
                estimated_cache_bytes_with_margin=estimate,observed_free_bytes=disk.free,jobs=jobs)
    validate_inputs(plan)
    return {**plan,"identity":object_hash(plan)}


def resume_plan(path, version, *, code_root=ROOT):
    """Restore only the coordinator; terminal workers are never restarted."""
    plan = read_json(path)
    validate_plan(plan)
    require(Path(plan["code_root"]).resolve() == Path(code_root).resolve(), "Resume requires the fixed original checkout")
    require(version == plan["code"] and not version["dirty"]
            and version.get("verified_remote_commit") == version["commit"], "Resume code is not the original pushed version")
    expected = Path(plan["root"]) / "pipeline" / plan["prefix"] / "plan.json"
    require(Path(path).resolve() == expected.resolve(), "Resume must use the original immutable plan path")
    validate_inputs(plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","index","data-manifest","output-root","adoption","run-prefix"):
        parser.add_argument("--"+name)
    parser.add_argument("--resume-plan",type=Path,help="Restart only the coordinator with its unchanged stored plan")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.resume_plan:
        if any((args.root,args.index,args.data_manifest,args.output_root,args.adoption,args.run_prefix,args.dry_run)):
            parser.error("resume-plan must be used alone")
        plan=resume_plan(args.resume_plan,code_version(require_pushed=True))
    else:
        if not all((args.root,args.index,args.data_manifest,args.output_root,args.adoption,args.run_prefix)):
            parser.error("root/index/data-manifest/output-root/adoption/run-prefix are required")
        plan=build_plan(args.root,args.index,args.data_manifest,args.output_root,args.adoption,args.run_prefix,
                        code_version(require_pushed=not args.dry_run))
    if args.dry_run:
        import json
        print(json.dumps(plan,indent=2))
        return
    with lock(Path(plan["root"])/"pipeline/coordinator.lock"):
        path=initialize(plan)
        print({"time":now(),"plan":str(path),"identity":plan["identity"],"pairs":plan["counts"]["pairs"]},flush=True)
        while True:
            done,launched=tick(plan,path)
            if launched:
                print({"time":now(),"launched":launched},flush=True)
            if done:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
