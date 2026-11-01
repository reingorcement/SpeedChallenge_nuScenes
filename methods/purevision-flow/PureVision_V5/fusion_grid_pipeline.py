"""Six authorized grid/fusion runs, using the established worker and GPU locks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import pipeline
from fusion_abc_launch import ART, FLOW, GPU, MANIFEST, PY, SOURCE
from fusion_abc_pipeline import gpu_inventory
from purevision_v5.common import ROOT, code_version, now, object_hash, read_json, sha256, write_json

STAGE = 'frozen_fusion_grid_pilot'
GROUPS = ('J23', 'S23', 'S46', 'J46', 'F23', 'R23')

def validate_inputs(plan, job=None):
    pipeline.validate_plan(plan)
    if plan['stage'] != STAGE:
        raise ValueError('Wrong experiment stage')
    if [j['name'] for j in plan['jobs'] if j['kind'] == 'training'] != list(GROUPS):
        raise ValueError('Only the six authorized groups are permitted')
    for name, digest in plan['locked_inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'Locked source changed: {name}')
    if read_json(plan['verification'])['status'] != 'passed':
        raise ValueError('Preflight not passed')
    if job and job.get('physical_gpu'):
        if gpu_inventory()[job['physical_gpu']['index']] != job['physical_gpu']:
            raise ValueError('GPU identity changed')
    if job and job['kind'] == 'training':
        grid = '46' if job['name'].endswith('46') else '23'
        ready = read_json(Path(plan['root']) / 'evidence' / f"{plan['prefix']}_ready{grid}.json")
        if ready.get('status') != 'published' or ready['plan_identity'] != plan['identity']:
            raise ValueError('This grid has not been published before training')
        if ready['cache_index_sha256'] != sha256(Path(plan[f'cache{grid}']) / 'index.json'):
            raise ValueError('Published cache index changed')


def build_plan(spec_path, version):
    spec = read_json(spec_path)
    root, code = Path(spec['root']), ROOT
    prefix = spec['prefix']; base = root / 'pipeline' / prefix
    cache23, cache46 = Path(spec['cache23']), Path(spec['cache46'])
    report = root / 'reports' / f'fusion_grid_{prefix}'
    inventory = gpu_inventory(); jobs = []
    def cmd(script, *args, gpu=None):
        c = [PY, '-B', str(code / script), *map(str, args)]
        return ['/usr/bin/env', f'CUDA_VISIBLE_DEVICES={GPU[gpu]}', *c] if gpu is not None else c
    def add(name, kind, gpu, deps, commands, run=None, progress=None, fresh=None, checks=()):
        jobs.append(dict(name=name, job_id=f'{prefix}_{name}', kind=kind,
            device=f'cuda:{gpu}' if gpu is not None else None, physical_gpu=inventory[gpu] if gpu is not None else None,
            process_device='cuda:0' if gpu is not None else 'cpu', requires=deps, after=[], commands=commands,
            session=f'v5grid_{prefix}_{name}', log_path=str(root/'logs'/f'{prefix}_{name}.log'),
            status_path=str(base/'jobs'/f'{name}.json'), run_dir=str(run) if run else None,
            progress_path=str(progress) if progress else None, fresh_output=str(fresh) if fresh else None, checks=list(checks)))
    add('cache46', 'cache', 0, [], [cmd('prepare_fusion_grid.py', '--grid', 4, 6,
        '--reference-cache', cache23, '--checkpoint', SOURCE, '--manifest', MANIFEST,
        '--selection', root/'plans/resolution_pilot_s42_32t8v.json', '--flow', FLOW,
        '--rgb-frames', root/'rgb_frames/v3_uint8_train_dev_v1', '--scope', 'pilot', '--allow-pilot-cache',
        '--batch-size', 2, '--device', 'cuda:0', '--output', cache46, gpu=0)],
        progress=cache46/'status.json', fresh=cache46,
        checks=[dict(path=str(cache46/'status.json'), field='status', equals='completed')])
    add('publish46', 'publication', None, ['cache46'], [cmd('fusion_grid_pipeline.py', '--plan', base/'plan.json', '--publish-grid', '46')])
    # Eligible 4x6 dual paths precede the diagnostic jobs in each free-GPU scan.
    for group, gpu in [('J23', 1), ('S23', 2), ('S46', 0), ('J46', 1), ('F23', 1), ('R23', 2)]:
        run = root/'runs'/f'{prefix}_{group}'; cache = cache46 if group.endswith('46') else cache23
        add(group, 'training', gpu, ['publish46'] if group.endswith('46') else [],
            [cmd('train_fusion_grid.py', '--config', code/f'configs/next/FUSION_GRID_{group}.json',
                 '--cache', cache, '--output', run, '--device', 'cuda:0', gpu=gpu)],
            run=run, progress=run/'status.json', fresh=run,
            checks=[dict(path=str(run/'status.json'), field='status', equals='completed'),
                    dict(path=str(run/'status.json'), field='completed_epochs', equals=20),
                    dict(path=str(run/'status.json'), field='optimizer_updates', equals=2300)])
        jobs[-1].update(variant=group, config_path=str(code/f'configs/next/FUSION_GRID_{group}.json'))
    add('dual_summary', 'comparison', None, ['J23','S23','J46','S46'],
        [cmd('fusion_grid_comparison.py', '--plan', base/'plan.json', '--output', report, '--dual-only'),
         cmd('fusion_grid_pipeline.py', '--plan', base/'plan.json', '--publish')])
    add('compare', 'comparison', None, list(GROUPS)+['dual_summary'],
        [cmd('fusion_grid_comparison.py', '--plan', base/'plan.json', '--output', report)])
    locked = [str(spec_path), str(SOURCE), MANIFEST, str(root/'plans/resolution_pilot_s42_32t8v.json'),
              spec['verification'], str(cache23/'index.json'),
              *[str(code/f'configs/next/FUSION_GRID_{g}.json') for g in GROUPS]]
    plan = dict(schema='v5_pipeline_v1', stage=STAGE, prefix=prefix, root=str(root), code_root=str(code), python=PY,
        code=version, jobs=jobs, features=None, features_status=None, cache23=str(cache23), cache46=str(cache46),
        verification=spec['verification'], locked_inputs={p:sha256(p) for p in locked},
        completion_report=str(report/'summary.json'), report_root=str(report), gpu_inventory=inventory,
        worker_network_file=str(root/'monitoring/network.json'))
    plan['identity'] = object_hash(plan)
    validate_inputs(plan)
    return plan


def validate_cache_readiness(plan, grid):
    """Validate small immutable manifests before publishing training readiness."""
    from purevision_v5.fusion_abc import SOURCE_SHA
    from purevision_v5.fusion_abc_cache import COUNTS, SCHEMA as LEGACY_SCHEMA
    from purevision_v5.fusion_grid_cache import SCHEMA as GRID_SCHEMA
    root = Path(plan[f'cache{grid}'])
    index, state = read_json(root/'index.json'), read_json(root/'status.json')
    if state.get('status') != 'completed' or index.get('scope') != 'pilot':
        raise ValueError('Only a completed pilot cache may be published as ready')
    if index.get('identity') != object_hash({k:v for k,v in index.items() if k != 'identity'}):
        raise ValueError('Cache manifest identity changed before publication')
    if index.get('source', {}).get('checkpoint_sha256') != SOURCE_SHA:
        raise ValueError('Cache source differs from the frozen checkpoint')
    if set(index.get('partitions', {})) != set(COUNTS) or any(
            index['partitions'][p].get('count') != n for p,n in COUNTS.items()):
        raise ValueError('Readiness requires all 9150 authorized pilot windows')
    if grid == '23':
        if index.get('schema') != LEGACY_SCHEMA:
            raise ValueError('2x3 readiness must refer to the verified historical cache')
    elif grid == '46':
        if index.get('schema') != GRID_SCHEMA or index.get('grid') != [4,6]:
            raise ValueError('4x6 readiness has the wrong grid/schema')
        if index.get('code', {}).get('commit') != plan['code']['commit']:
            raise ValueError('4x6 extraction did not use this fixed implementation')
        legacy_root = Path(plan['cache23'])
        legacy = read_json(legacy_root/'index.json')
        reference = index.get('reference_cache', {})
        if (reference.get('root') != str(legacy_root.resolve())
                or reference.get('identity') != legacy.get('identity')
                or reference.get('index_sha256') != sha256(legacy_root/'index.json')):
            raise ValueError('4x6 source-cache identity/hash differs from the published 2x3 cache')
        evidence = read_json(root/'cross_grid_evidence.json')
        if (evidence.get('reference_identity') != legacy['identity']
                or evidence.get('grid_identity') != index['identity']):
            raise ValueError('Cross-grid evidence refers to different cache manifests')
        for part, count in COUNTS.items():
            row = evidence.get('partitions', {}).get(part, {})
            if (row.get('count') != count or not all(row.get(k) is True for k in
                    ('metadata_exact_equal', 'labels_exact_equal', 'dt_exact_equal'))
                    or row.get('ordered_sample_digest') != index['partitions'][part].get('ordered_sample_digest')):
                raise ValueError('Complete ordered cross-grid sample verification is missing')
    else:
        raise ValueError('Unknown readiness grid')
    return sha256(root/'index.json')


def publish(plan, grid=None):
    from publish_results import publish as existing_publish
    if grid:
        index_digest = validate_cache_readiness(plan, grid)
    result = existing_publish(plan['root'], Path(plan['root'])/'report-worktree',
        [Path(plan['root'])/'pipeline'/plan['prefix']/'plan.json'],
        push_url='git@github.com:reingorcement/SpeedChallenge_nuScenes.git')
    if grid:
        if validate_cache_readiness(plan, grid) != index_digest:
            raise ValueError('Cache manifest changed while readiness was being published')
        write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_ready{grid}.json",
            dict(status='published', plan_identity=plan['identity'], code=plan['code'],
                 cache_index_sha256=index_digest, report_commit=result['report_commit'], time=now()))
    return result


class Tmux(pipeline.Tmux):
    def launch(self, plan_path, plan, job):
        if self.exists(job['session']):
            raise RuntimeError('Existing session must not be replaced')
        command = [PY, '-B', str(ROOT/'scripts/secure_exec.py'), '--network', plan['worker_network_file'], '--',
                   PY, '-B', str(ROOT/'fusion_grid_pipeline.py'), '--worker', '--plan', str(plan_path), '--job', job['name']]
        Path(job['log_path']).parent.mkdir(parents=True, exist_ok=True)
        shell = 'exec '+shlex.join(command)+' >> '+shlex.quote(job['log_path'])+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),shell],check=True)


def worker(plan_path, name):
    plan=read_json(plan_path)
    job=next(j for j in plan['jobs'] if j['name']==name)
    if job['device'] and job['device'] in pipeline.busy_gpus():
        state=read_json(job['status_path']);state.update(status='queued', reason='GPU occupied; preserving external work', updated_at=now())
        write_json(job['status_path'],state);pipeline.register(plan,job,state);return 0
    pipeline.check_planned_inputs=validate_inputs
    result = pipeline.worker(plan_path,name)
    state = read_json(job['status_path'])
    # A competing process can acquire the GPU after the coordinator's probe.
    # Only prelaunch contention is retryable; genuine training errors stay failed.
    contention = (state.get('error') == 'RuntimeError: GPU became occupied by another compute process before launch'
                  or state.get('error', '').startswith('BlockingIOError: [Errno 11]'))
    if (job['device'] and state.get('status') == 'failed' and contention
            and not state.get('child_pid') and (not job.get('fresh_output') or not Path(job['fresh_output']).exists())):
        for key in ('error','exit_code','finished_at','pid','pid_start_ticks'):
            state.pop(key, None)
        state.update(status='queued', reason='GPU contention before launch; preserving external work', updated_at=now())
        write_json(job['status_path'],state);pipeline.register(plan,job,state)
        return 0
    return result


def install(plan, path):
    from monitor import cron_marker, install_cron, install_publisher
    current=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if current.returncode not in (0,1): raise RuntimeError('Cannot inspect existing schedules')
    if any(cron_marker(plan['root']) in line for line in current.stdout.splitlines()):
        raise RuntimeError('Existing task schedule requires reconciliation; no duplicate installed')
    root=Path(plan['root'])
    write_json(root/'monitoring/lifecycle_guard.json',dict(plan=str(path),plan_identity=plan['identity'],completion_requires_success=True))
    command=root/'monitoring'/f"publisher_command_{plan['prefix']}.json"
    write_json(command,dict(command=[PY,'-B',str(ROOT/'publish_results.py'),'--root',str(root),'--worktree',str(root/'report-worktree'),
                                   '--plan',str(path),'--push-url','git@github.com:reingorcement/SpeedChallenge_nuScenes.git']))
    command.chmod(0o600)
    receipt=dict(server_astra=install_cron(root,ROOT),publisher=install_publisher(root,ROOT,command),
                 plan_identity=plan['identity'],previous_crontab=current.stdout,time=now())
    write_json(root/'evidence'/f"{plan['prefix']}_schedule_install.json",receipt)
    return receipt


def run(plan, path):
    validate_inputs(plan)
    version=code_version(require_pushed=True)
    if version['commit']!=plan['code']['commit']:raise ValueError('Code commit changed')
    with pipeline.lock(Path(plan['root'])/'pipeline'/f"coordinator_{plan['prefix']}.lock"):
        while True:
            done, launched=pipeline.tick(plan,path,tmux=Tmux())
            if launched:print(json.dumps(dict(time=now(),launched=launched)),flush=True)
            if done:
                # Queue completion itself triggers publication, independent of patrol timing.
                publish(plan)
                from monitor import publication_allows_stop, modify_cron
                if publication_allows_stop(plan['root']):
                    modify_cron(plan['root'])
                    write_json(Path(plan['root'])/'evidence'/f"{plan['prefix']}_completion.json",
                        dict(status='completed',time=now(),plan_identity=plan['identity'],schedules_removed=True))
                    publish(plan)
                return
            time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path);p.add_argument('--run-spec',type=Path)
    p.add_argument('--materialize',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--job')
    p.add_argument('--publish-grid',choices=['23','46']);p.add_argument('--publish',action='store_true');p.add_argument('--install',action='store_true')
    a=p.parse_args()
    if a.materialize:
        plan=build_plan(a.run_spec,code_version(require_pushed=True));path=pipeline.initialize(plan)
        print(json.dumps(dict(plan=str(path),identity=plan['identity'])));return
    if a.worker:raise SystemExit(worker(a.plan,a.job))
    plan=read_json(a.plan)
    if a.publish_grid or a.publish:print(json.dumps(publish(plan,a.publish_grid)));return
    if a.install:print(json.dumps(install(plan,a.plan)));return
    run(plan,a.plan)

if __name__=='__main__':main()
