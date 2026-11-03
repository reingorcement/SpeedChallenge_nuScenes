"""One pilot A/B, independent full A, and a real conditional full B queue."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import pipeline
from fusion_abc_launch import PY
from fusion_abc_pipeline import gpu_inventory
from fusion_confirmation_pipeline import runtime_view as allocate
from purevision_v5 import fusion_seed_pipeline as reuse
from purevision_v5.common import ROOT,read_json,write_json,sha256,object_hash,code_version,now
from fusion_geometry_comparison import collect,pilot_decision

STAGE='s23_geometry_ab_v1'
ORDER=(('PILOT_A','pilot','A'),('PILOT_B','pilot','B'),('FULL_A','full','A'),('FULL_B','full','B'))


def validate_inputs(plan,require_ready=False):
    pipeline.validate_plan(plan)
    if plan['stage']!=STAGE or [(j['name'],j['scope'],j['variant']) for j in plan['jobs']]!=list(ORDER):
        raise ValueError('Only one seed42 pilot/full matched A/B authorized')
    for path,digest in plan['locked_inputs'].items():
        if sha256(path)!=digest:raise ValueError('Locked input changed: '+path)
    for key in ('geometry_preflight','trainer_preflight','control_preflight','lambda_proof'):
        if read_json(plan[key])['status']!='passed':raise ValueError('Required short verification failed')
    if require_ready:
        ready=read_json(reuse.ready_path(plan))
        if (ready.get('status')!='published' or ready['plan_identity']!=plan['identity']
                or ready['plan_sha256']!=sha256(reuse.plan_path(plan))
                or ready['code_commit']!=plan['code']['commit'] or ready['locked_inputs_hash']!=object_hash(plan['locked_inputs'])):
            raise ValueError('Implementation/configuration/evidence must be published first')


def build_plan(spec_path,version):
    spec=read_json(spec_path);root=Path(spec['root']);prefix=spec['prefix'];geo=Path(spec['geometry_root']);base=root/'pipeline'/prefix
    jobs=[]
    for name,scope,variant in ORDER:
        config=ROOT/'configs/next'/f'S23_GEOMETRY_{scope.upper()}_{variant}.json';run=root/'runs'/f'{prefix}_{name}'
        cache=spec[f'{scope}_features'];command=[PY,'-B',str(ROOT/'train_fusion_geometry.py'),'--config',str(config),
            '--cache',cache,'--lambda-proof',str(geo/'lambda.json'),'--output',str(run),'--device','cuda:0']
        if variant=='B':command+=['--geometry',str(geo/scope)]
        jobs.append(dict(name=name,job_id=run.name,kind='training',group='S23',seed=42,scope=scope,variant=variant,
            config_path=str(config),cache_root=cache,run_dir=str(run),device=None,physical_gpu=None,process_device='cuda:0',
            allowed_gpu_indices=[0,1,2],commands=[command],requires=['PILOT_A','PILOT_B','GEOMETRY_FULL'] if name=='FULL_B' else [],after=[],
            session=f'v5geo_{run.name}',log_path=str(root/'logs'/f'{run.name}.log'),status_path=str(base/'jobs'/f'{name}.json'),
            progress_path=str(run/'status.json'),fresh_output=str(run),checks=[dict(path=str(run/'status.json'),field=k,equals=v)
                for k,v in (('status','completed'),('completed_epochs',20),('optimizer_updates',2300 if scope=='pilot' else 41560))]))
    prepared=read_json(geo/'preparation.json');cachejob=dict(name='GEOMETRY_FULL',job_id=f'{prefix}_GEOMETRY_FULL',kind='geometry_cache',
        device='cuda:0',run_dir=None,session=f'geo_cache_external_{prefix}',log_path=prepared['log_path'],
        status_path=str(base/'jobs/GEOMETRY_FULL.json'),progress_path=str(base/'jobs/GEOMETRY_FULL.json'),commands=[],requires=[],after=[],checks=[],fresh_output=None)
    report=root/'reports'/f'fusion_geometry_{prefix}'
    plan=dict(schema='v5_pipeline_v1',stage=STAGE,prefix=prefix,root=str(root),code_root=str(ROOT),python=PY,code=version,
        code_commit=version['commit'],jobs=jobs,external_preparation=[cachejob],gpu_inventory=gpu_inventory(),
        features=None,features_status=None,geometry_root=str(geo),geometry_preflight=str(geo/'preflight/preflight.json'),
        trainer_preflight=str(geo/'trainer_preflight.json'),control_preflight=str(geo/'control_preflight.json'),lambda_proof=str(geo/'lambda.json'),
        pilot_gate=str(base/'pilot_gate.json'),full_geometry_ready=str(base/'full_geometry_ready.json'),
        full_B_ready=str(base/'full_B_ready.json'),preparation=str(geo/'preparation.json'),
        report_root=str(report),completion_report=str(report/'summary.json'),
        automation='FULL_B requires completed pilotA/B, inclusive1% gate, nonzero valid geometry, full geometry cache and verified GitHub stage publication',
        stopping_rule='No extra seed, lambda, epoch, structure or encoder training')
    locked=[Path(spec_path),Path(plan['geometry_preflight']),Path(plan['trainer_preflight']),Path(plan['lambda_proof']),
        Path(plan['preparation']),Path(plan['control_preflight']),geo/'pilot/index.json',geo/'pilot/geometry.pt',geo/'geometry_contract.json',
        *[Path(spec[f'{scope}_features'])/'index.json' for scope in ('pilot','full')],*[Path(j['config_path']) for j in jobs]]
    plan['locked_inputs']={str(p):sha256(p) for p in locked};plan['identity']=object_hash(plan);validate_inputs(plan);return plan


def cache_summary(plan,scope):
    geo=Path(plan['geometry_root']);path=geo/scope/'index.json'
    if not path.exists() or not (geo/scope/'coverage.json').exists():return None
    index=read_json(path);coverage=read_json(geo/scope/'coverage.json')
    if index['identity']!=object_hash({k:v for k,v in index.items() if k!='identity'}):raise ValueError('Geometry cache changed')
    value=dict(scope=scope,index_sha256=sha256(path),identity=index['identity'],geometry_sha256=index['geometry_sha256'],
        total=coverage['total'],valid=coverage['valid'],coverage=coverage['valid']/coverage['total'],reasons=coverage['reasons'],
        feature_cache_identity=index['feature_cache_identity'],source_flow_identity=index['source_flow_identity'],source_model=index['source_model'],
        policy=index['policy'],sample_keys_sha256=object_hash(index['keys']),source_file_sha256=index['source_file_sha256'])
    write_json(geo/f'{scope}_summary.json',value);return value


def observe_cache(plan):
    job=plan['external_preparation'][0];state=pipeline.read_json(job['status_path'],{})
    if state.get('status') in pipeline.TERMINAL:return state
    prepared=read_json(plan['preparation']);proc=pipeline.process_observation(prepared['pid'],prepared['pid_start_ticks'])
    if ((Path(plan['geometry_root'])/'full/index.json').exists()
            and (Path(plan['geometry_root'])/'full/coverage.json').exists()):
        summary=cache_summary(plan,'full')
        state.update(status='completed',reason='Immutable train geometry index committed',total=summary['total'],valid=summary['valid'])
    elif proc['verdict']=='dead':state.update(status='failed',reason='Geometry preparation process ended without an index; partial artifacts retained')
    else:
        lines=Path(job['log_path']).read_text()[-4000:].splitlines();last=next((x for x in reversed(lines) if x.startswith('road_frames')),None)
        state.update(status='running',phase='train_geometry_preparation',last_progress=last)
    state.update(updated_at=now(),pid=prepared['pid'],pid_start_ticks=prepared['pid_start_ticks'])
    write_json(job['status_path'],state);pipeline.register(plan,job,state);return state


def cancel_own_cache(plan):
    job=plan['external_preparation'][0];state=pipeline.read_json(job['status_path'],{})
    if state.get('status') in pipeline.TERMINAL:return
    prepared=read_json(plan['preparation']);pid=prepared['pid']
    if pipeline.process_observation(pid,prepared['pid_start_ticks'])['alive'] is True:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(Path(plan['geometry_root'])/'full').encode() not in command or b'--mode' not in command:
            raise ValueError('Refusing to stop an unrelated process')
        os.kill(pid,signal.SIGTERM)
    state.update(status='cancelled',updated_at=now(),reason='Own speculative full cache cancelled after pilot rejection; all partial artifacts preserved')
    write_json(job['status_path'],state);pipeline.register(plan,job,state)


def validate_full_B(plan):
    ready=read_json(plan['full_B_ready'])
    if (ready.get('status')!='published' or ready['plan_identity']!=plan['identity']
            or ready['pilot_gate_sha256']!=sha256(plan['pilot_gate'])
            or ready['full_geometry_ready_sha256']!=sha256(plan['full_geometry_ready'])
            or read_json(plan['pilot_gate'])['status']!='passed'
            or ready['full_cache_index_sha256']!=sha256(Path(plan['geometry_root'])/'full/index.json')):
        raise ValueError('Conditional full B is not authorized by verified stage evidence')


class Tmux(pipeline.Tmux):
    def launch(self,path,view,job):
        plan=read_json(path);validate_inputs(plan,True)
        if job['name']=='FULL_B':validate_full_B(plan)
        canonical=next(j for j in plan['jobs'] if j['name']==job['name'])
        if job!=reuse.apply_assignment(canonical,{'physical_gpu':job['physical_gpu']}):raise ValueError('Invalid GPU assignment')
        assignment=dict(plan_identity=plan['identity'],job_id=job['job_id'],physical_gpu=job['physical_gpu'],assigned_at=now(),
            attempt=pipeline.read_json(job['status_path'],{}).get('launch_count',0));assignment['identity']=object_hash(assignment)
        target=reuse.assignment_path(plan,canonical);immutable=target.with_name(f"{job['name']}_attempt{assignment['attempt']}.json")
        if immutable.exists() or self.exists(job['session']):raise ValueError('Existing attempt/session cannot be replaced')
        write_json(immutable,assignment);write_json(target,assignment)
        command=['/usr/bin/env','GIT_SSH_COMMAND='+reuse.SSH_COMMAND,PY,'-B',str(ROOT/'fusion_geometry_pipeline.py'),'--worker','--plan',str(path),'--job',job['name']]
        shell='exec '+shlex.join(command)+' >> '+shlex.quote(job['log_path'])+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),shell],check=True)


def check_published_stage(plan):
    from publish_results import REPORT_DIRECTORY
    directory=Path(plan['root'])/'report-worktree'/REPORT_DIRECTORY/'fusion_geometry'/plan['prefix']/'events'
    receipt=pipeline.read_json(Path(plan['root'])/'monitoring/publication.json',{})
    if receipt.get('status')!='published' or plan['identity'] not in receipt.get('plan_identities',[]):return False
    return all((directory/name).exists() and read_json(directory/name)==read_json(plan[key])
        for name,key in (('pilot_gate.json','pilot_gate'),('full_geometry_ready.json','full_geometry_ready')))


def run(plan):
    validate_inputs(plan,True)
    if code_version(require_pushed=True)['commit']!=plan['code']['commit']:raise ValueError('Wrong fixed implementation')
    publisher=reuse.EventPublisher(plan);observed=None;retry_at=0;final_requested=False
    with pipeline.lock(Path(plan['root'])/'pipeline'/f"coordinator_{plan['prefix']}.lock"):
        while True:
            cache=observe_cache(plan)
            results={j['name']:Path(j['run_dir'])/'result.json' for j in plan['jobs']}
            if not Path(plan['pilot_gate']).exists() and all(results[n].exists() for n in ('PILOT_A','PILOT_B')):
                report=collect(reuse.plan_path(plan));gate=pilot_decision(report['runs']['PILOT_A'],report['runs']['PILOT_B'],
                    read_json_lines(Path(next(j for j in plan['jobs'] if j['name']=='PILOT_A')['run_dir'])/'metrics.jsonl'),
                    read_json_lines(Path(next(j for j in plan['jobs'] if j['name']=='PILOT_B')['run_dir'])/'metrics.jsonl'),read_json(plan['geometry_preflight']))
                gate.update(plan_identity=plan['identity'],observed_at=now());write_json(plan['pilot_gate'],gate)
            gate=pipeline.read_json(plan['pilot_gate'])
            if gate and gate['status']=='rejected':
                job=next(j for j in plan['jobs'] if j['name']=='FULL_B');state=read_json(job['status_path'])
                if state['status']=='queued':state.update(status='blocked',reason='One pilot failed inclusive1% expansion gate; no retries/tuning',updated_at=now());write_json(job['status_path'],state)
                cancel_own_cache(plan)
            if cache['status']=='completed' and not Path(plan['full_geometry_ready']).exists():
                ready=dict(status='ready',plan_identity=plan['identity'],cache=cache_summary(plan,'full'),time=now())
                write_json(plan['full_geometry_ready'],ready)
            evidence=[results[n] for n in results]+[Path(plan['pilot_gate']),Path(plan['full_geometry_ready'])]
            for j in plan['jobs']:evidence.append(Path(j['run_dir'])/'dev_ready.json')
            current=object_hash({str(p):sha256(p) for p in evidence if p.exists()})
            if current!=observed:
                collect(reuse.plan_path(plan));publisher.request();observed=current
            publisher.poll()
            if publisher.last_returncode and time.monotonic()>=retry_at:
                publisher.request();retry_at=time.monotonic()+300
            if (gate and gate['status']=='passed' and Path(plan['full_geometry_ready']).exists()
                    and not Path(plan['full_B_ready']).exists() and check_published_stage(plan)):
                receipt=read_json(Path(plan['root'])/'monitoring/publication.json')
                write_json(plan['full_B_ready'],dict(status='published',plan_identity=plan['identity'],time=now(),
                    pilot_gate_sha256=sha256(plan['pilot_gate']),full_geometry_ready_sha256=sha256(plan['full_geometry_ready']),
                    full_cache_index_sha256=sha256(Path(plan['geometry_root'])/'full/index.json'),verified_report_commit=receipt['verified_remote_commit']))
            busy=set(pipeline.busy_gpus());view=allocate(plan,busy)
            if not Path(plan['full_B_ready']).exists():
                for j in view['jobs']:
                    if j['name']=='FULL_B' and read_json(j['status_path'])['status']=='queued':j['device']=reuse.WAITING_DEVICE
            done,launched=pipeline.tick(view,reuse.plan_path(plan),tmux=Tmux(),gpu_probe=lambda:busy|{reuse.WAITING_DEVICE})
            if launched:print(json.dumps(dict(time=now(),launched=launched)),flush=True)
            if done and pipeline.read_json(plan['external_preparation'][0]['status_path'],{}).get('status') in pipeline.TERMINAL:
                report=collect(reuse.plan_path(plan))
                if not final_requested:
                    publisher.request();final_requested=True
                from monitor import publication_allows_stop,modify_cron
                if publication_allows_stop(plan['root']) and report['status']=='completed':
                    modify_cron(plan['root']);modify_cron(plan['root'],publisher_only=True)
                    write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_completion.json",
                        dict(status='completed',time=now(),plan_identity=plan['identity'],schedules_removed=True))
                    return
            time.sleep(3)


def read_json_lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path);p.add_argument('--spec',type=Path)
    p.add_argument('--materialize',action='store_true');p.add_argument('--publish-ready',action='store_true');p.add_argument('--install',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--job');a=p.parse_args();reuse.validate_inputs=validate_inputs
    if a.materialize:
        plan=build_plan(a.spec,code_version(require_pushed=True));print(pipeline.initialize(plan));observe_cache(plan);return
    plan=read_json(a.plan)
    if a.worker:
        if a.job=='FULL_B':validate_full_B(plan)
        raise SystemExit(reuse.worker(a.plan,a.job))
    if a.publish_ready:cache_summary(plan,'pilot');print(json.dumps(reuse.publish_ready(plan)));return
    if a.install:
        reuse.install(plan)
        # Gate rejection is an authorized terminal outcome; still require its
        # own final report to be remotely verified before any schedule cleanup.
        guard=Path(plan['root'])/'monitoring/lifecycle_guard.json';v=read_json(guard);v['completion_requires_success']=False;write_json(guard,v)
        # Cron needs the same scoped SSH443 environment as event publications.
        from monitor import modify_cron,cron_marker
        current=subprocess.check_output(['crontab','-l'],text=True);marker=cron_marker(plan['root'])+'_PUBLISHER'
        line=next(x for x in current.splitlines() if marker in x)
        injected='GIT_SSH_COMMAND='+shlex.quote(reuse.SSH_COMMAND).replace('%',r'\%')
        modify_cron(plan['root'],[line.replace('/usr/bin/env PATH=','/usr/bin/env '+injected+' PATH=',1)],publisher_only=True)
        print('Installed5min server/30min Astra/hourly publisher with scoped SSH443');return
    run(plan)


if __name__=='__main__':main()
