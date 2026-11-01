"""Eight authorized seed confirmations using the existing worker, locks and patrols.

The published plan fixes every scientific input. GPU assignments are separate
receipts: runtime views may change only device identity and its CUDA environment.
The original worker still owns process lifetime, GPU/job locks and completion.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import pipeline
from fusion_abc_launch import PY
from fusion_abc_pipeline import gpu_inventory
from .common import ROOT, code_version, now, object_hash, read_json, sha256, write_json

STAGE = 'frozen_fusion_seed_review'
ORDER = tuple((group, seed) for seed in (43, 44) for group in ('S23', 'J23', 'F23')) + (('S46', 43), ('S46', 44))
WAITING_DEVICE = 'waiting_for_free_gpu'
SSH_COMMAND = ("ssh -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 "
               "-o StrictHostKeyChecking=yes -o HostName=ssh.github.com -p 443 "
               "-o HostKeyAlias=github.com -o 'ProxyCommand=nc -X connect -x 127.0.0.1:7900 %h %p'")


def plan_path(plan):
    return Path(plan['root'])/'pipeline'/plan['prefix']/'plan.json'


def ready_path(plan):
    return Path(plan['root'])/'evidence'/f"{plan['prefix']}_ready.json"


def validate_inputs(plan, require_ready=False):
    pipeline.validate_plan(plan)
    if plan.get('stage') != STAGE or [(j.get('group'), j.get('seed')) for j in plan['jobs']] != list(ORDER):
        raise ValueError('Only J23/S23/F23/S46 seed43/44 are authorized')
    if plan.get('code_commit') != plan['code']['commit']:
        raise ValueError('Code provenance differs inside plan')
    for path, digest in plan['locked_inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'Locked confirmation input changed: {path}')
    if read_json(plan['verification']).get('status') != 'passed':
        raise ValueError('Candidate preflight must pass')
    for job in plan['jobs']:
        config = read_json(job['config_path'])
        if (config.get('group'), config.get('seed')) != (job['group'], job['seed']):
            raise ValueError('A job configuration does not match its declared group/seed')
    if require_ready:
        ready = read_json(ready_path(plan))
        if (ready.get('status') != 'published' or ready.get('plan_identity') != plan['identity']
                or ready.get('plan_sha256') != sha256(plan_path(plan))
                or ready.get('code_commit') != plan['code']['commit']
                or ready.get('locked_inputs_hash') != object_hash(plan['locked_inputs'])
                or not ready.get('verified_report_commit')):
            raise ValueError('All eight run IDs, configurations and evidence must be published before launch')


def build_plan(spec_path, version):
    from fusion_grid_pipeline import validate_cache_readiness
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    root = Path(spec['root']).resolve()
    prefix = spec['prefix']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', prefix):
        raise ValueError('Unsafe run prefix')
    baseline_path = Path(spec['baseline_plan']).resolve()
    baseline = read_json(baseline_path)
    pipeline.validate_plan(baseline)
    if baseline['stage'] != 'frozen_fusion_grid_pilot':
        raise ValueError('Expected the completed seed42 grid baseline')
    for grid in ('23', '46'):
        validate_cache_readiness(baseline, grid)
    inventory = gpu_inventory()
    base = root/'pipeline'/prefix
    jobs = []
    for group, seed in ORDER:
        name = f'{group}_s{seed}'
        job_id = f'{prefix}_{name}'
        run = root/'runs'/job_id
        config = ROOT/f'configs/next/FUSION_GRID_{group}_s{seed}.json'
        cache = baseline['cache46' if group == 'S46' else 'cache23']
        jobs.append(dict(name=name, job_id=job_id, kind='training', group=group, seed=seed,
            device=None, physical_gpu=None, allowed_gpu_indices=[0,1,2], process_device='cuda:0',
            requires=[], after=[], commands=[[PY, '-B', str(ROOT/'train_fusion_grid.py'),
                '--config', str(config), '--cache', cache, '--output', str(run), '--device', 'cuda:0']],
            config_path=str(config), cache_root=cache, session=f'v5seed_{job_id}',
            log_path=str(root/'logs'/f'{job_id}.log'), status_path=str(base/'jobs'/f'{name}.json'),
            run_dir=str(run), progress_path=str(run/'status.json'), fresh_output=str(run),
            checks=[dict(path=str(run/'status.json'), field='status', equals='completed'),
                    dict(path=str(run/'status.json'), field='completed_epochs', equals=20),
                    dict(path=str(run/'status.json'), field='optimizer_updates', equals=2300)]))
    report = root/'reports'/f'fusion_grid_{prefix}'
    locked = [spec_path, baseline_path, Path(spec['verification']).resolve(),
              Path(baseline['cache23'])/'index.json', Path(baseline['cache46'])/'index.json',
              *[Path(j['config_path']) for j in jobs]]
    plan = dict(schema='v5_pipeline_v1', stage=STAGE, prefix=prefix, root=str(root), code_root=str(ROOT),
        python=PY, code=version, code_commit=version['commit'], jobs=jobs, baseline_plan=str(baseline_path),
        cache23=baseline['cache23'], cache46=baseline['cache46'], verification=str(Path(spec['verification']).resolve()),
        locked_inputs={str(p):sha256(p) for p in locked}, features=None, features_status=None,
        gpu_inventory=inventory, report_root=str(report), completion_report=str(report/'summary.json'),
        scheduling='one training process per free physical GPU; S23/J23/F23 first; S46 prefers GPU0 without waiting',
        publication_gate=str(root/'evidence'/f'{prefix}_ready.json'))
    plan['identity'] = object_hash(plan)
    validate_inputs(plan)
    return plan


def assignment_path(plan, job):
    return plan_path(plan).parent/'assignments'/f"{job['name']}.json"


def apply_assignment(job, assignment):
    result = copy.deepcopy(job)
    gpu = assignment['physical_gpu']
    result.update(device=f"cuda:{gpu['index']}", physical_gpu=gpu)
    result['commands'] = [['/usr/bin/env', f"CUDA_VISIBLE_DEVICES={gpu['uuid']}", *command]
                          for command in job['commands']]
    return result


def verified_assignment(plan, job):
    value = read_json(assignment_path(plan, job))
    expected_gpu = plan['gpu_inventory'].get(str(value['physical_gpu']['index']),
                                            plan['gpu_inventory'].get(value['physical_gpu']['index']))
    if (value.get('identity') != object_hash({k:v for k,v in value.items() if k != 'identity'})
            or value.get('plan_identity') != plan['identity'] or value.get('job_id') != job['job_id']
            or value['physical_gpu'] != expected_gpu):
        raise ValueError('Runtime GPU assignment does not belong to this published plan')
    return value


def runtime_view(plan, busy):
    """Assign only queued jobs, preserving every active assignment across restarts."""
    view = copy.deepcopy(plan)
    free = {0,1,2} - {int(d.split(':')[1]) for d in busy if d.startswith('cuda:')}
    states = {j['name']:pipeline.read_json(j['status_path'], {'status':'unknown'}) for j in plan['jobs']}
    for job in plan['jobs']:
        state = states[job['name']]
        if state['status'] in pipeline.ACTIVE | {'unknown'} and assignment_path(plan, job).exists():
            free.discard(verified_assignment(plan, job)['physical_gpu']['index'])
    resolved = []
    for job in plan['jobs']:
        state = states[job['name']]
        if state['status'] == 'queued' and free:
            preference = (0,1,2) if job['group'] == 'S46' else (1,2,0)
            gpu = next(i for i in preference if i in free)
            free.remove(gpu)
            inventory = plan['gpu_inventory'].get(str(gpu), plan['gpu_inventory'].get(gpu))
            resolved.append(apply_assignment(job, {'physical_gpu':inventory}))
        elif assignment_path(plan, job).exists() and state['status'] != 'queued':
            resolved.append(apply_assignment(job, verified_assignment(plan, job)))
        else:
            resolved.append({**job, 'device':WAITING_DEVICE})
    view['jobs'] = resolved
    return view


class Tmux(pipeline.Tmux):
    def launch(self, path, view, job):
        canonical = read_json(path)
        validate_inputs(canonical, require_ready=True)
        original = next(j for j in canonical['jobs'] if j['name'] == job['name'])
        if self.exists(job['session']):
            raise RuntimeError('Existing worker session must not be replaced')
        if job != apply_assignment(original, {'physical_gpu':job['physical_gpu']}):
            raise ValueError('Runtime view changed more than its GPU assignment')
        assignment = dict(plan_identity=canonical['identity'], job_id=job['job_id'],
                          physical_gpu=job['physical_gpu'], assigned_at=now(),
                          attempt=pipeline.read_json(job['status_path'], {}).get('launch_count', 0))
        assignment['identity'] = object_hash(assignment)
        receipt = assignment_path(canonical, original)
        immutable = receipt.with_name(f"{job['name']}_attempt{assignment['attempt']}.json")
        if immutable.exists():
            raise ValueError('Assignment attempt already exists; refusing duplicate launch')
        write_json(immutable, assignment)
        write_json(receipt, assignment)
        command = ['/usr/bin/env', f'GIT_SSH_COMMAND={SSH_COMMAND}', PY, '-B',
                   str(ROOT/'run_fusion_seed_pipeline.py'), '--worker', '--plan', str(path), '--job', job['name']]
        Path(job['log_path']).parent.mkdir(parents=True, exist_ok=True)
        shell = 'exec '+shlex.join(command)+' >> '+shlex.quote(job['log_path'])+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),shell],check=True)


def worker(path, name):
    canonical = read_json(path)
    job = next(j for j in canonical['jobs'] if j['name'] == name)
    job_lock = Path(canonical['root'])/'pipeline/locks'/f"job_{job['job_id']}.lock"
    # Own the same established job lock before reading or rewriting launch state.
    # A second worker cannot demote an active/completed run back to queued.
    with pipeline.lock(job_lock):
        state = read_json(job['status_path'])
        if state.get('status') in pipeline.TERMINAL:
            return 0
        if state.get('status') != 'launching':
            raise ValueError('Only the current coordinator launch may enter this worker')
        return _worker_locked(path, name, job_lock)


def _worker_locked(path, name, job_lock):
    """Adapt only the plan read; the established worker owns the full lifecycle."""
    canonical = read_json(path)
    validate_inputs(canonical, require_ready=True)
    job = next(j for j in canonical['jobs'] if j['name'] == name)
    assignment = verified_assignment(canonical, job)
    resolved = apply_assignment(job, assignment)
    if gpu_inventory()[assignment['physical_gpu']['index']] != assignment['physical_gpu']:
        raise ValueError('Assigned GPU UUID/model changed')
    if resolved['device'] in pipeline.busy_gpus():
        state = read_json(job['status_path'])
        state.update(status='queued', reason='GPU occupied before launch; no external process changed', updated_at=now())
        write_json(job['status_path'],state);pipeline.register(canonical,resolved,state)
        return 0
    view = copy.deepcopy(canonical)
    view['jobs'] = [resolved if j['name'] == name else j for j in canonical['jobs']]
    original_read, original_validate, original_check, original_lock = (
        pipeline.read_json, pipeline.validate_plan, pipeline.check_planned_inputs, pipeline.lock)
    def adapted_read(candidate, default=None):
        return view if Path(candidate).resolve() == Path(path).resolve() else original_read(candidate, default)
    def adapted_validate(value):
        original_validate(canonical)
        if value != view:
            raise ValueError('Unexpected runtime plan mutation')
    def check(_view, _job):
        # Validate the canonical published plan with its original identity function.
        pipeline.validate_plan = original_validate
        try:
            validate_inputs(canonical, require_ready=True)
            if verified_assignment(canonical,job) != assignment:
                raise ValueError('Assignment changed while a worker was active')
            if Path(job['run_dir']).exists():
                raise ValueError('Fresh training output appeared before child launch; refusing overwrite')
        finally:
            pipeline.validate_plan = adapted_validate
    def adapted_lock(candidate, *, blocking=False):
        # The outer adapter already holds exactly this lock; all GPU/registry
        # locks continue through the unchanged implementation.
        return nullcontext() if Path(candidate).resolve() == job_lock.resolve() else original_lock(candidate, blocking=blocking)
    pipeline.read_json, pipeline.validate_plan, pipeline.check_planned_inputs = adapted_read, adapted_validate, check
    pipeline.lock = adapted_lock
    try:
        result = pipeline.worker(path, name)
    finally:
        pipeline.read_json, pipeline.validate_plan, pipeline.check_planned_inputs = original_read, original_validate, original_check
        pipeline.lock = original_lock
    state = read_json(job['status_path'])
    contention = (state.get('error') == 'RuntimeError: GPU became occupied by another compute process before launch'
                  or state.get('error','').startswith('BlockingIOError: [Errno 11]'))
    if state.get('status') == 'failed' and contention and not state.get('child_pid') and not Path(job['run_dir']).exists():
        for key in ('error','exit_code','finished_at','pid','pid_start_ticks'):
            state.pop(key,None)
        state.update(status='queued', reason='GPU contention before child launch', updated_at=now())
        write_json(job['status_path'],state);pipeline.register(canonical,resolved,state)
        return 0
    return result


def publisher_command(plan):
    return [PY,'-B',str(ROOT/'publish_results.py'),'--root',plan['root'],'--worktree',str(Path(plan['root'])/'report-worktree'),
            '--plan',str(plan_path(plan)),'--push-url','git@github.com:reingorcement/SpeedChallenge_nuScenes.git']


def publisher_environment():
    env = dict(os.environ)
    env.setdefault('GIT_SSH_COMMAND',SSH_COMMAND)
    return env


def publish_ready(plan):
    validate_inputs(plan)
    version = code_version(require_pushed=True)
    if version['commit'] != plan['code']['commit']:
        raise ValueError('Readiness publication must use the fixed pushed implementation')
    subprocess.run(publisher_command(plan), env=publisher_environment(), check=True)
    receipt = read_json(Path(plan['root'])/'monitoring/publication.json')
    if (receipt.get('status') != 'published' or receipt.get('report_commit') != receipt.get('verified_remote_commit')
            or plan['identity'] not in receipt.get('plan_identities', [])):
        raise ValueError('Remote publication did not verify this complete candidate plan')
    validate_inputs(plan)
    value = dict(status='published', plan_identity=plan['identity'], plan_sha256=sha256(plan_path(plan)),
                 code_commit=plan['code']['commit'], locked_inputs_hash=object_hash(plan['locked_inputs']),
                 verified_report_commit=receipt['verified_remote_commit'], created_at=now())
    write_json(ready_path(plan),value)
    return value


class EventPublisher:
    """Serialize asynchronous publications while coalescing completion events."""
    def __init__(self, plan):
        self.plan, self.process, self.log, self.pending = plan, None, None, False
        self.attempts, self.last_returncode = 0, None
    def request(self):
        self.pending = True
    def poll(self):
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return
            self.last_returncode = code
            self.log.close();self.log=None;self.process=None
        if self.pending and self.process is None:
            path = Path(self.plan['root'])/'logs'/f"{self.plan['prefix']}_publication.log"
            path.parent.mkdir(parents=True,exist_ok=True)
            self.log = path.open('a')
            self.process = subprocess.Popen(publisher_command(self.plan), env=publisher_environment(),
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            self.pending=False;self.attempts+=1


def report_event(plan):
    command = [PY,'-B',str(ROOT/'fusion_multiseed_comparison.py'),'--plan',str(plan_path(plan)),
               '--output',plan['report_root']]
    path = Path(plan['root'])/'logs'/f"{plan['prefix']}_comparison.log"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    value = dict(time=now(), returncode=result.returncode, command=command)
    write_json(plan_path(plan).parent/'report_event.json',value)
    return result.returncode == 0


def install(plan):
    from monitor import cron_marker, install_cron, install_publisher
    root=Path(plan['root'])
    current=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if current.returncode not in (0,1):raise RuntimeError('Cannot inspect existing schedules')
    marker=cron_marker(root)
    existing=[line for line in current.stdout.splitlines() if marker in line]
    if existing:
        # Reuse observed schedules without replacing other tasks; root can reconcile stale code separately.
        return dict(status='existing_schedules_reused',lines=existing,requires_root_reconciliation=True)
    write_json(root/'monitoring/lifecycle_guard.json',dict(plan=str(plan_path(plan)),plan_identity=plan['identity'],completion_requires_success=True))
    command=root/'monitoring'/f"publisher_command_{plan['prefix']}.json"
    write_json(command,dict(command=publisher_command(plan)));command.chmod(0o600)
    result=dict(server_astra=install_cron(root,ROOT),publisher=install_publisher(root,ROOT,command),time=now())
    write_json(root/'evidence'/f"{plan['prefix']}_schedule_install.json",result)
    return result


def run(plan, *, tmux=None, gpu_probe=None):
    validate_inputs(plan,require_ready=True)
    version=code_version(require_pushed=True)
    if version['commit'] != plan['code']['commit'] or str(ROOT) != plan['code_root']:
        raise ValueError('Coordinator must use the fixed pushed checkout')
    path=plan_path(plan); publisher=EventPublisher(plan); tmux=tmux or Tmux(); gpu_probe=gpu_probe or pipeline.busy_gpus
    event_path=path.parent/'completion_events.json'
    delivered=set(pipeline.read_json(event_path,{}).get('reported_terminal_jobs',[]))
    with pipeline.lock(Path(plan['root'])/'pipeline'/f"coordinator_{plan['prefix']}.lock"):
        while True:
            try:busy=set(gpu_probe())
            except (OSError,subprocess.SubprocessError):busy={'cuda:0','cuda:1','cuda:2'}
            view=runtime_view(plan,busy)
            done,launched=pipeline.tick(view,path,tmux=tmux,gpu_probe=lambda:busy|{WAITING_DEVICE})
            if launched:print(json.dumps(dict(time=now(),launched=launched)),flush=True)
            states={j['name']:read_json(j['status_path']) for j in plan['jobs']}
            terminal={name for name,state in states.items() if state['status'] in pipeline.TERMINAL}
            if terminal-delivered:
                # Local core results are written first; network work starts independently.
                success=report_event(plan)
                publisher.request()
                if success:
                    delivered|=terminal
                    write_json(event_path,dict(reported_terminal_jobs=sorted(delivered),updated_at=now()))
            publisher.poll()
            if done:
                if terminal != delivered and not report_event(plan):
                    raise RuntimeError('Final local report failed; training artifacts remain intact')
                # A fresh final snapshot must run after every earlier asynchronous publication.
                publisher.request()
                while publisher.pending or publisher.process is not None:
                    publisher.poll();time.sleep(1)
                if publisher.last_returncode:
                    raise RuntimeError('Local results complete; final GitHub publication failed (see publication log)')
                receipt=read_json(Path(plan['root'])/'monitoring/publication.json')
                if (receipt.get('status') != 'published' or plan['identity'] not in receipt.get('plan_identities',[])
                        or receipt.get('report_commit') != receipt.get('verified_remote_commit')):
                    raise RuntimeError('Final publication identity is not verified')
                from monitor import publication_allows_stop, modify_cron
                stopped=False
                if publication_allows_stop(plan['root']):
                    modify_cron(plan['root']);modify_cron(plan['root'],publisher_only=True);stopped=True
                write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_completion.json",
                    dict(status='completed' if all(s['status']=='completed' for s in states.values()) else 'finished_with_failures',
                         time=now(),plan_identity=plan['identity'],schedules_removed=stopped,
                         final_report_commit=receipt['verified_remote_commit']))
                return
            time.sleep(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-spec',type=Path);p.add_argument('--plan',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--materialize',action='store_true');mode.add_argument('--publish-ready',action='store_true')
    mode.add_argument('--install',action='store_true');mode.add_argument('--worker',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--job')
    args=p.parse_args()
    if args.materialize:
        if args.run_spec is None:p.error('--materialize requires --run-spec')
        plan=build_plan(args.run_spec,code_version(require_pushed=True));path=pipeline.initialize(plan)
        print(json.dumps(dict(plan=str(path),identity=plan['identity'])));return
    if args.plan is None:p.error('--plan is required')
    if args.worker:
        if not args.job:p.error('--worker requires --job')
        raise SystemExit(worker(args.plan,args.job))
    plan=read_json(args.plan)
    if args.publish_ready:print(json.dumps(publish_ready(plan)));return
    if args.install:print(json.dumps(install(plan)));return
    run(plan)
