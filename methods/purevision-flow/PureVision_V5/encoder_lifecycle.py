"""CPU-only startup verification and milestone publication, never training control.

After the declared initial wave passes, install this plan's real cron patrols.
Queued runs are verified separately when they start; never count them as verified.
Network failures retry without touching workers; stop only after terminal remote publication.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time

from monitor import install_cron, modify_cron, publication_allows_stop
from pipeline import lock, validate_plan
from purevision_v5.common import ROOT, now, read_json, write_json
from startup_watch import inspect


def optional(path):
    return read_json(path) if Path(path).is_file() else {}


def startup_gate(plan, proofs):
    jobs={j["name"]:j for j in plan["jobs"] if j["kind"]=="training"}
    names=plan.get("startup_gate_jobs",list(jobs))
    if not names or len(set(names))!=len(names) or not set(names)<=set(jobs):
        raise ValueError("Invalid initial startup wave")
    statuses={p["run_id"]:p["status"] for p in proofs}
    return all(statuses.get(jobs[n]["job_id"])=="verified" for n in names)


def run(plan_path, command_file, interval=30):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    torch.set_num_threads(2)
    plan=read_json(plan_path);validate_plan(plan)
    root=Path(plan["root"])
    status_path=root/"monitoring"/f"lifecycle_{plan['prefix']}.json"
    state=optional(status_path) or dict(plan_identity=plan["identity"],started_at=now(),publication_attempts=[])
    if state["plan_identity"]!=plan["identity"]: raise ValueError("Lifecycle belongs to a different plan")
    write_json(root/"monitoring/lifecycle_guard.json",dict(plan_identity=plan["identity"],plan=str(plan_path)))
    with lock(status_path.with_suffix(".lock")):
        while True:
            proofs=[];milestones=[]
            for job in plan["jobs"]:
                raw=optional(job["status_path"])
                progress=optional(job["progress_path"]) if job.get("progress_path") else {}
                milestones.append([job["name"],raw.get("status"),bool(progress.get("processed_batches",0)>=3),
                                   min(progress.get("epoch",0),1),progress.get("epoch",0)//10])
                if job["kind"]=="training":
                    path=root/"monitoring/startup"/job["job_id"]/"startup_proof.json"
                    proof=inspect(job,plan,optional(path) or None)
                    write_json(path,proof);proofs.append(proof)
            all_verified=bool(proofs) and all(p["status"]=="verified" for p in proofs)
            gate_verified=startup_gate(plan,proofs)
            if gate_verified and not state.get("cron_installed"):
                formal=install_cron(root,ROOT)
                probe=install_cron(root,ROOT,probe=True)
                state.update(cron_installed=True,cron_installed_at=now(),schedule=formal,probe_schedule=probe)
            state.update(updated_at=now(),pid=os.getpid(),all_startups_verified=all_verified,
                         initial_wave_verified=gate_verified,
                         startup_status={p["run_id"]:p["status"] for p in proofs},milestones=milestones)
            terminal=all(optional(j["status_path"]).get("status") in {"completed","failed","interrupted","cancelled","blocked"}
                         for j in plan["jobs"])
            due=(milestones!=state.get("published_milestones") or time.time()-state.get("last_success_epoch",0)>=3600)
            if due and time.time()-state.get("last_attempt_epoch",0)>=300:
                state["last_attempt_epoch"]=time.time()
                logfile=root/"monitoring"/f"lifecycle_publish_{plan['prefix']}.log"
                try:
                    with logfile.open("a") as stream:
                        outcome=subprocess.run(["bash",str(ROOT/"scripts/publish_results.sh"),str(root),str(command_file),"manual"],
                                               stdout=stream,stderr=subprocess.STDOUT,timeout=650)
                    receipt=optional(root/"monitoring/publication.json")
                    success=(outcome.returncode==0 and receipt.get("status")=="published"
                             and plan["identity"] in receipt.get("plan_identities",[]))
                    state["publication_attempts"].append(dict(time=now(),success=success,exit_code=outcome.returncode,
                        report_commit=receipt.get("report_commit"),error=receipt.get("error")))
                    if success:
                        state.update(published_milestones=milestones,last_success_epoch=time.time())
                except (OSError,subprocess.TimeoutExpired) as error:
                    state["publication_attempts"].append(dict(time=now(),success=False,error=str(error)))
            if terminal and publication_allows_stop(root):
                modify_cron(root);modify_cron(root,publisher_only=True)
                state.update(status="completed",finished_at=now(),schedules_removed_after_verified_publication=True)
                write_json(status_path,state)
                return state
            state["status"]="running" if not terminal else "publication_pending"
            write_json(status_path,state)
            time.sleep(interval)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True,type=Path);p.add_argument("--publisher-command",required=True,type=Path)
    a=p.parse_args();print(run(a.plan,a.publisher_command))
