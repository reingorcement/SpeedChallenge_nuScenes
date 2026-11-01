"""Fixed full 2x3 cache, followed by exactly J23/S23/F23 seed42 training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import pipeline
from fusion_abc_launch import SOURCE, MANIFEST, FLOW, PY
from fusion_abc_pipeline import gpu_inventory
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc import SOURCE_SHA
from purevision_v5.fusion_seed_pipeline import SSH_COMMAND, EventPublisher, install, publisher_environment

STAGE = 'frozen_fusion_full_v1'
COUNTS = {'train_core':132978, 'dev':23480}


def plan_path(plan):
    return Path(plan['root'])/'pipeline'/plan['prefix']/'plan.json'


def gate_path(plan, phase):
    return Path(plan['root'])/'evidence'/f"{plan['prefix']}_{phase}_ready.json"


def validate_inputs(plan, job=None):
    pipeline.validate_plan(plan)
    if plan.get('stage') != STAGE or [(j['group'],j['seed']) for j in plan['jobs'] if j['kind']=='training'] != [('S23',42),('J23',42),('F23',42)]:
        raise ValueError('Only the three authorized full-data groups may train')
    for path, digest in plan['locked_inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'Locked full-stage input changed: {path}')
    for key in ('full_scope','history_manifest','extraction_preflight','training_implementation_preflight'):
        if read_json(plan[key]).get('status') != ('locked_verified' if key=='history_manifest' else 'passed'):
            raise ValueError(f'Full-stage evidence has not passed: {key}')
    if job and job.get('physical_gpu') and gpu_inventory()[job['physical_gpu']['index']] != job['physical_gpu']:
        raise ValueError('GPU identity changed')
    if job and job['name'] == 'cache':
        phase = 'cache'
    elif job and job['kind'] == 'training':
        phase = 'train'
    else:
        return
    gate = read_json(gate_path(plan, phase))
    if (gate.get('status') != 'published' or gate.get('plan_identity') != plan['identity']
            or gate.get('plan_sha256') != sha256(plan_path(plan)) or not gate.get('verified_report_commit')):
        raise ValueError('Fixed code/config/source/evidence have not been published for this phase')
    if phase == 'train':
        if gate['cache_index_sha256'] != sha256(Path(plan['cachefull'])/'index.json') or gate['training_preflight_sha256'] != sha256(plan['training_preflight']):
            raise ValueError('Published complete cache or training evidence changed')


def build_plan(spec_path, version):
    spec = read_json(spec_path); root = Path(spec['root']).resolve(); prefix = spec['prefix']
    if not prefix.replace('_','').isalnum():
        raise ValueError('Unsafe run prefix')
    evidence = root/'verification/fusion_full_20261102'
    extraction = read_json(evidence/'extraction_preflight.json')
    cache = root/'fusion_full_cache'/prefix; base = root/'pipeline'/prefix
    inventory = gpu_inventory(); jobs = []
    def command(script, *args, gpu=None):
        result = [PY, '-B', str(ROOT/script), *map(str,args)]
        return ['/usr/bin/env',f"CUDA_VISIBLE_DEVICES={inventory[gpu]['uuid']}",*result] if gpu is not None else result
    def add(name,kind,gpu,requires,commands,progress=None,fresh=None,run=None,checks=()):
        job = dict(name=name,job_id=f'{prefix}_{name}',kind=kind,device=f'cuda:{gpu}' if gpu is not None else None,
            physical_gpu=inventory[gpu] if gpu is not None else None,process_device='cuda:0' if gpu is not None else 'cpu',
            requires=requires,after=[],commands=commands,session=f'v5full_{prefix}_{name}',
            log_path=str(root/'logs'/f'{prefix}_{name}.log'),status_path=str(base/'jobs'/f'{name}.json'),
            run_dir=str(run) if run else None,progress_path=str(progress) if progress else None,
            fresh_output=str(fresh) if fresh else None,checks=list(checks))
        jobs.append(job);return job
    add('cache','cache',0,[],[command('prepare_fusion_full_cache.py','--scope','full','--allow-full-cache',
        '--checkpoint',SOURCE,'--manifest',MANIFEST,'--flow',FLOW,'--rgb-frames',root/'rgb_frames/v3_uint8_train_dev_v1',
        '--reference-cache',spec['reference_cache'],'--output',cache,'--preflight-receipt',evidence/'extraction_preflight.json',
        '--batch-size',extraction['selected_batch_size'],'--shard-size',4096,'--threads',4,'--device','cuda:0',gpu=0)],
        progress=cache/'status.json',fresh=cache,checks=[dict(path=str(cache/'status.json'),field='status',equals='completed')])
    training_preflight = evidence/'full_cache_training_receipt.json'
    add('full_cache_check','verification',None,['cache'],[command('verify_fusion_full_training.py','--cache',cache,
        '--output',training_preflight,'--cache-check-only')],checks=[dict(path=str(training_preflight),field='status',equals='passed')])
    add('publish_train','publication',None,['full_cache_check'],[command('fusion_full_pipeline.py','--plan',base/'plan.json','--publish-ready','train')])
    for group,gpu in [('S23',0),('J23',1),('F23',2)]:
        run=root/'runs'/f'{prefix}_{group}';config=ROOT/f'configs/next/FUSION_FULL_{group}.json'
        job=add(group,'training',gpu,['publish_train'],[command('train_fusion_grid.py','--config',config,
            '--cache',cache,'--output',run,'--device','cuda:0',gpu=gpu)],progress=run/'status.json',fresh=run,run=run,
            checks=[dict(path=str(run/'status.json'),field=k,equals=v) for k,v in
                    [('status','completed'),('completed_epochs',20),('optimizer_updates',41560)]])
        job.update(group=group,seed=42,variant=group,config_path=str(config))
    report=root/'reports'/f'fusion_full_{prefix}'
    plan=dict(schema='v5_pipeline_v1',stage=STAGE,prefix=prefix,root=str(root),code_root=str(ROOT),python=PY,code=version,
        jobs=jobs,features=None,features_status=None,gpu_inventory=inventory,cachefull=str(cache),reference_cache=spec['reference_cache'],
        baseline_plan=str(root/'pipeline/20261102_0052_fusion_grid_s42/plan.json'),
        full_scope=str(evidence/'full_scope.json'),history_manifest=str(evidence/'history_manifest.json'),
        extraction_preflight=str(evidence/'extraction_preflight.json'),training_implementation_preflight=str(evidence/'training_receipt.json'),
        training_preflight=str(training_preflight),report_root=str(report),completion_report=str(report/'summary.json'))
    locked=[Path(spec_path),SOURCE,Path(MANIFEST),Path(spec['reference_cache'])/'index.json',
        *[Path(plan[k]) for k in ('baseline_plan','full_scope','history_manifest','extraction_preflight','training_implementation_preflight')],
        *[Path(j['config_path']) for j in jobs if j['kind']=='training']]
    plan['locked_inputs']={str(p):sha256(p) for p in locked}
    plan['identity']=object_hash(plan);validate_inputs(plan);return plan


def publish(plan, phase=None):
    from publish_results import publish as existing_publish
    validate_inputs(plan)
    extras={}
    if phase == 'train':
        index=read_json(Path(plan['cachefull'])/'index.json')
        proof=read_json(plan['training_preflight'])
        if (read_json(Path(plan['cachefull'])/'status.json')['status']!='completed' or index['scope']!='full'
                or index['source']['checkpoint_sha256']!=SOURCE_SHA or index['code']['commit']!=plan['code']['commit']
                or index['identity']!=object_hash({k:v for k,v in index.items() if k!='identity'})
                or {p:v['count'] for p,v in index['partitions'].items()}!=COUNTS or proof['status']!='passed'
                or proof.get('cache_identity')!=index['identity']
                or proof.get('cache_index_sha256')!=sha256(Path(plan['cachefull'])/'index.json')
                or proof.get('cache_scope')!='full' or proof.get('formal_full_coverage_checked') is not True
                or proof.get('counts')!=COUNTS or proof.get('expected_formal_updates')!=41560
                or proof.get('scene_counts')!={'train_core':582,'dev':103}):
            raise ValueError('Complete cache and full-scope evidence must pass before training')
        extras=dict(cache_index_sha256=sha256(Path(plan['cachefull'])/'index.json'),
                    training_preflight_sha256=sha256(plan['training_preflight']))
    # The isolated publisher serializes writes and verifies the actual remote ref.
    result=existing_publish(plan['root'],Path(plan['root'])/'report-worktree',[plan_path(plan)],
        push_url='git@github.com:reingorcement/SpeedChallenge_nuScenes.git')
    if phase:
        validate_inputs(plan)
        write_json(gate_path(plan,phase),dict(status='published',phase=phase,plan_identity=plan['identity'],
            plan_sha256=sha256(plan_path(plan)),verified_report_commit=result['verified_remote_commit'],time=now(),**extras))
    return result


class Tmux(pipeline.Tmux):
    def launch(self,path,plan,job):
        if self.exists(job['session']):raise ValueError('Existing worker session must not be replaced')
        cmd=['/usr/bin/env',f'GIT_SSH_COMMAND={SSH_COMMAND}',PY,'-B',str(ROOT/'fusion_full_pipeline.py'),
             '--worker','--plan',str(path),'--job',job['name']]
        Path(job['log_path']).parent.mkdir(parents=True,exist_ok=True)
        shell='exec '+shlex.join(cmd)+' >> '+shlex.quote(job['log_path'])+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),shell],check=True)


def worker(path,name):
    plan=read_json(path);job=next(j for j in plan['jobs'] if j['name']==name)
    pipeline.check_planned_inputs=validate_inputs
    result=pipeline.worker(path,name)
    state=read_json(job['status_path'])
    contention=(state.get('error')=='RuntimeError: GPU became occupied by another compute process before launch'
                or state.get('error','').startswith('BlockingIOError: [Errno 11]'))
    if (state.get('status')=='failed' and contention and not state.get('child_pid')
            and (not job.get('fresh_output') or not Path(job['fresh_output']).exists())):
        for key in ('error','exit_code','finished_at','pid','pid_start_ticks'):state.pop(key,None)
        state.update(status='queued',reason='Prelaunch GPU contention; external work preserved',updated_at=now())
        write_json(job['status_path'],state);pipeline.register(plan,job,state);return 0
    return result


def report_event(plan):
    path=Path(plan['root'])/'logs'/f"{plan['prefix']}_comparison.log"
    with path.open('a') as stream:
        result=subprocess.run([PY,'-B',str(ROOT/'fusion_full_comparison.py'),'--plan',str(plan_path(plan)),
            '--output',plan['report_root']],stdout=stream,stderr=subprocess.STDOUT)
    return result.returncode==0


def run(plan):
    validate_inputs(plan)
    if code_version(require_pushed=True)['commit']!=plan['code']['commit']:raise ValueError('Fixed implementation changed')
    publisher=EventPublisher(plan);observed={}
    with pipeline.lock(Path(plan['root'])/'pipeline'/f"coordinator_{plan['prefix']}.lock"):
        while True:
            done,launched=pipeline.tick(plan,plan_path(plan),tmux=Tmux())
            if launched:print(json.dumps(dict(time=now(),launched=launched)),flush=True)
            current={}
            for job in plan['jobs']:
                if job['kind']=='training':
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
                if publisher.last_returncode:raise RuntimeError('Training/local report finished; retry final publication')
                from monitor import publication_allows_stop,modify_cron
                if publication_allows_stop(plan['root']):
                    modify_cron(plan['root']);modify_cron(plan['root'],publisher_only=True)
                    write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_completion.json",
                        dict(status='completed',time=now(),plan_identity=plan['identity'],schedules_removed=True))
                return
            time.sleep(3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-spec',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--job')
    p.add_argument('--materialize',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--publish-ready',choices=['cache','train']);p.add_argument('--publish',action='store_true')
    p.add_argument('--install',action='store_true');a=p.parse_args()
    if a.materialize:
        plan=build_plan(a.run_spec,code_version(require_pushed=True));path=pipeline.initialize(plan)
        print(json.dumps(dict(plan=str(path),identity=plan['identity'])));return
    if a.worker:raise SystemExit(worker(a.plan,a.job))
    plan=read_json(a.plan)
    if a.publish_ready or a.publish:print(json.dumps(publish(plan,a.publish_ready)));return
    if a.install:print(json.dumps(install(plan)));return
    run(plan)


if __name__=='__main__':main()
