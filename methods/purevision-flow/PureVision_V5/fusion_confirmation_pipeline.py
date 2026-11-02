"""Exactly six full-data seed checks and two matched S23 stage20 runs."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import time

import pipeline
from fusion_abc_launch import FLOW, MANIFEST, PY, SOURCE
from fusion_abc_pipeline import gpu_inventory
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256, write_json
from purevision_v5 import fusion_seed_pipeline as reuse

STAGE = 'fusion_confirmation_tail_v1'
ORDER = (('S23_TAIL_FT',42),('S23_FROZEN_CONT',42)) + tuple(
    (g,s) for s in (43,44) for g in ('S23','J23','F23'))


def validate_inputs(plan, require_ready=False):
    pipeline.validate_plan(plan)
    if plan.get('stage')!=STAGE or [(j['group'],j['seed']) for j in plan['jobs']]!=list(ORDER):
        raise ValueError('Only the eight authorized confirmation/stage20 runs are allowed')
    for path,digest in plan['locked_inputs'].items():
        if sha256(path)!=digest:raise ValueError(f'Locked input changed: {path}')
    for key in ('seed_verification','model_verification','trainer_verification'):
        if read_json(plan[key]).get('status')!='passed':raise ValueError(f'Preflight not passed: {key}')
    for job in plan['jobs']:
        config=read_json(job['config_path'])
        if (config['group'],config['seed'],config['epochs'],config['scheduler_epochs'])!=(job['group'],job['seed'],20,20):
            raise ValueError('Stage20 protocol/config mismatch')
    if require_ready:
        value=read_json(reuse.ready_path(plan))
        if (value.get('status')!='published' or value.get('plan_identity')!=plan['identity']
                or value.get('plan_sha256')!=sha256(reuse.plan_path(plan))
                or value.get('code_commit')!=plan['code']['commit']
                or value.get('locked_inputs_hash')!=object_hash(plan['locked_inputs'])
                or not value.get('verified_report_commit')):
            raise ValueError('Fixed implementation and eight-run evidence not published')


def build_plan(spec_path, version):
    spec=read_json(spec_path);root=Path(spec['root']);prefix=spec['prefix'];verification=Path(spec['verification_root'])
    baseline=read_json(spec['baseline_plan']);cache=Path(spec['cache_full']);index=read_json(cache/'index.json')
    if (baseline['stage']!='frozen_fusion_full_v1' or index['scope']!='full'
            or index['identity']!=object_hash({k:v for k,v in index.items() if k!='identity'})
            or {p:v['count'] for p,v in index['partitions'].items()}!={'train_core':132978,'dev':23480}):
        raise ValueError('Original complete cache/baseline differs')
    evidence=read_json(verification/'model_preflight.json')
    microbatch=evidence['selected_microbatch']
    if type(microbatch) is not int or not 1<=microbatch<=64:raise ValueError('Invalid bounded microbatch selection')
    readout=root/'runs/20261102_0154_fusion_full_s42_S23/best.pt'
    jobs=[];base=root/'pipeline'/prefix
    for group,seed in ORDER:
        tail=group.startswith('S23_');name=group if tail else f'{group}_s{seed}'
        config=ROOT/'configs/next'/ (f'FUSION_TAIL_{group}.json' if tail else f'FUSION_FULL_{group}_s{seed}.json')
        run=root/'runs'/f'{prefix}_{name}'
        if tail:
            command=[PY,'-B',str(ROOT/'train_fusion_tail.py'),'--config',str(config),
                '--encoder-checkpoint',str(SOURCE),'--readout-checkpoint',str(readout),
                '--cache',str(cache),'--manifest',str(MANIFEST),'--flow',str(FLOW),
                '--rgb-frames',str(root/'rgb_frames/v3_uint8_train_dev_v1'),
                '--verification',str(verification/'model_preflight.json'),'--microbatch',str(microbatch),
                '--output',str(run),'--device','cuda:0']
        else:
            command=[PY,'-B',str(ROOT/'train_fusion_grid.py'),'--config',str(config),
                '--cache',str(cache),'--output',str(run),'--device','cuda:0']
        jobs.append(dict(name=name,job_id=run.name,kind='training',group=group,seed=seed,
            protocol='s23_stage20' if tail else 'full_confirmation',
            device=None,physical_gpu=None,process_device='cuda:0',allowed_gpu_indices=[0] if group=='S23_TAIL_FT' else [1,2],
            requires=[],after=[],commands=[command],config_path=str(config),cache_root=str(cache),
            session=f'v5confirm_{run.name}',log_path=str(root/'logs'/f'{run.name}.log'),
            status_path=str(base/'jobs'/f'{name}.json'),run_dir=str(run),progress_path=str(run/'status.json'),
            fresh_output=str(run),checks=[dict(path=str(run/'status.json'),field=k,equals=v)
                for k,v in (('status','completed'),('completed_epochs',20),('optimizer_updates',41560))]))
    report=root/'reports'/f'fusion_confirmation_{prefix}'
    plan=dict(schema='v5_pipeline_v1',stage=STAGE,prefix=prefix,root=str(root),code_root=str(ROOT),python=PY,
        code=version,code_commit=version['commit'],jobs=jobs,baseline_plan=spec['baseline_plan'],
        cache_full=str(cache),history_manifest=baseline['history_manifest'],full_scope=baseline['full_scope'],
        seed_verification=str(verification/'seeds_preflight.json'),model_verification=str(verification/'model_preflight.json'),
        trainer_verification=str(verification/'trainer_preflight.json'),verification_root=str(verification),
        parameter_manifest=str(verification/'parameter_manifest.json'),parents=dict(readout=str(readout),encoders=str(SOURCE)),
        features=None,features_status=None,gpu_inventory=gpu_inventory(),report_root=str(report),completion_report=str(report/'summary.json'),
        scheduling='GPU0 reserved for online tail; GPU1/GPU2 rolling cache jobs; one process per physical GPU',
        stage_budget_epochs=20,stage_updates=41560,epoch0_reference='fixed S23 full seed42 epoch19 best; fresh optimizer and cosine20')
    locked=[Path(spec_path),Path(spec['baseline_plan']),cache/'index.json',Path(plan['history_manifest']),Path(plan['full_scope']),
        SOURCE,readout,Path(MANIFEST),verification/'launch_contract.json',verification/'stage20_revision.json',
        verification/'report_preflight.json',verification/'queue_preflight.json',
        *[Path(plan[k]) for k in ('seed_verification','model_verification','trainer_verification','parameter_manifest')],
        *[Path(j['config_path']) for j in jobs]]
    plan['locked_inputs']={str(p):sha256(p) for p in locked};plan['identity']=object_hash(plan)
    validate_inputs(plan);return plan


def runtime_view(plan,busy):
    view=copy.deepcopy(plan);free={0,1,2}-{int(x.split(':')[1]) for x in busy if x.startswith('cuda:')}
    states={j['name']:pipeline.read_json(j['status_path'],{'status':'unknown'}) for j in plan['jobs']}
    for j in plan['jobs']:
        if states[j['name']]['status'] in pipeline.ACTIVE|{'unknown'} and reuse.assignment_path(plan,j).exists():
            free.discard(reuse.verified_assignment(plan,j)['physical_gpu']['index'])
    resolved=[]
    for j in plan['jobs']:
        state=states[j['name']]['status'];available=free.intersection(j['allowed_gpu_indices'])
        if state=='queued' and available:
            gpu=min(available);free.remove(gpu);inventory=plan['gpu_inventory'].get(str(gpu),plan['gpu_inventory'].get(gpu))
            resolved.append(reuse.apply_assignment(j,{'physical_gpu':inventory}))
        elif state!='queued' and reuse.assignment_path(plan,j).exists():resolved.append(reuse.apply_assignment(j,reuse.verified_assignment(plan,j)))
        else:resolved.append({**j,'device':reuse.WAITING_DEVICE})
    view['jobs']=resolved;return view


class Tmux(pipeline.Tmux):
    def launch(self,path,view,job):
        canonical=read_json(path);validate_inputs(canonical,True)
        original=next(j for j in canonical['jobs'] if j['name']==job['name'])
        if self.exists(job['session']):raise ValueError('Existing session cannot be replaced')
        if (job!=reuse.apply_assignment(original,{'physical_gpu':job['physical_gpu']})
                or job['physical_gpu']['index'] not in job['allowed_gpu_indices']):raise ValueError('Invalid runtime assignment')
        assignment=dict(plan_identity=canonical['identity'],job_id=job['job_id'],physical_gpu=job['physical_gpu'],
            assigned_at=now(),attempt=pipeline.read_json(job['status_path'],{}).get('launch_count',0))
        assignment['identity']=object_hash(assignment);path_a=reuse.assignment_path(canonical,original)
        immutable=path_a.with_name(f"{job['name']}_attempt{assignment['attempt']}.json")
        if immutable.exists():raise ValueError('Duplicate assignment attempt')
        write_json(immutable,assignment);write_json(path_a,assignment)
        command=['/usr/bin/env',f'GIT_SSH_COMMAND={reuse.SSH_COMMAND}',PY,'-B',str(ROOT/'fusion_confirmation_pipeline.py'),
                 '--worker','--plan',str(path),'--job',job['name']]
        Path(job['log_path']).parent.mkdir(parents=True,exist_ok=True)
        shell='exec '+shlex.join(command)+' >> '+shlex.quote(job['log_path'])+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),shell],check=True)


def report_event(plan):
    path=Path(plan['root'])/'logs'/f"{plan['prefix']}_comparison.log"
    with path.open('a') as f:
        result=subprocess.run([PY,'-B',str(ROOT/'fusion_confirmation_comparison.py'),'--plan',str(reuse.plan_path(plan)),
            '--output',plan['report_root']],stdout=f,stderr=subprocess.STDOUT)
    return result.returncode==0


def run(plan):
    validate_inputs(plan,True)
    if code_version(require_pushed=True)['commit']!=plan['code']['commit']:raise ValueError('Wrong fixed implementation')
    publisher=reuse.EventPublisher(plan);observed={}
    with pipeline.lock(Path(plan['root'])/'pipeline'/f"coordinator_{plan['prefix']}.lock"):
        while True:
            busy=set(pipeline.busy_gpus());view=runtime_view(plan,busy)
            done,launched=pipeline.tick(view,reuse.plan_path(plan),tmux=Tmux(),gpu_probe=lambda:busy|{reuse.WAITING_DEVICE})
            if launched:print(json.dumps(dict(time=now(),launched=launched)),flush=True)
            current={}
            for job in plan['jobs']:
                for name in ('dev_ready.json','result.json'):
                    path=Path(job['run_dir'])/name
                    if path.exists():current[str(path)]=sha256(path)
            if current and current!=observed:
                if report_event(plan):observed=current;publisher.request()
            publisher.poll()
            if done:
                if not report_event(plan):raise RuntimeError('Final local report failed; artifacts preserved')
                publisher.request()
                while publisher.pending or publisher.process is not None:publisher.poll();time.sleep(1)
                if publisher.last_returncode:raise RuntimeError('Training finished; final GitHub publication requires retry')
                from monitor import publication_allows_stop,modify_cron
                if publication_allows_stop(plan['root']):
                    modify_cron(plan['root']);modify_cron(plan['root'],publisher_only=True)
                    write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_completion.json",
                        dict(status='completed',time=now(),plan_identity=plan['identity'],schedules_removed=True))
                return
            time.sleep(3)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path);p.add_argument('--run-spec',type=Path)
    p.add_argument('--materialize',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--job')
    p.add_argument('--publish-ready',action='store_true');p.add_argument('--publish',action='store_true');p.add_argument('--install',action='store_true')
    a=p.parse_args()
    # Reuse only the established locked lifecycle adapter; this plan keeps its own validator.
    reuse.validate_inputs=validate_inputs
    if a.materialize:
        plan=build_plan(a.run_spec,code_version(require_pushed=True));print(pipeline.initialize(plan));return
    if a.worker:raise SystemExit(reuse.worker(a.plan,a.job))
    plan=read_json(a.plan)
    if a.publish_ready:print(json.dumps(reuse.publish_ready(plan)));return
    if a.publish:subprocess.run(reuse.publisher_command(plan),env=reuse.publisher_environment(),check=True);return
    if a.install:print(json.dumps(reuse.install(plan)));return
    run(plan)


if __name__=='__main__':main()
