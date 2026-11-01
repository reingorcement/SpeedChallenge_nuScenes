"""Bounded metadata export for frozen-grid pilots and their seed review."""
from pathlib import Path
import hashlib
import json
from publish_results import MAX_BYTES, check_public, encoded, read_source, source_path


def collect_public(root, plans):
    root=Path(root).resolve();files={};rows=[];sources=[]
    def export(path,target,lines=False):
        value,origin=read_source(root,path,lines=lines)
        if value is not None:
            files[target]=(b''.join((json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n').encode() for x in value) if lines else encoded(value))
            sources.append(origin)
        return value
    def text(path,target):
        path=source_path(root,path)
        if not path.is_file():return
        with path.open('rb') as f:blob=f.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES:raise ValueError('Public report exceeds bounded size')
        check_public(blob.decode('utf-8'));files[target]=blob
        sources.append(dict(path=str(path),source_sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),representation='original_text'))
    for path in plans:
        plan=json.loads(Path(path).read_text())
        if plan.get('stage') not in {'frozen_fusion_grid_pilot', 'frozen_fusion_seed_review'}:continue
        base=f"fusion_grid/{plan['prefix']}"
        export(plan['verification'],f'{base}/preflight.json')
        if plan.get('stage') == 'frozen_fusion_seed_review':
            export(plan['baseline_plan'], f'{base}/baseline_plan.json')
            for name, path in plan.get('preflight_evidence', {}).items():
                if not name.replace('_', '').isalnum():
                    raise ValueError('Unsafe seed evidence name')
                export(path, f'{base}/evidence/{name}.json')
        for grid in ('23','46'):
            cache=Path(plan[f'cache{grid}'])
            for name in ('index.json','status.json','verification.json','cross_grid_evidence.json'):
                export(cache/name,f'{base}/cache{grid}/{name}')
        for job in plan['jobs']:
            if job['kind']!='training':continue
            run=Path(job['run_dir']);target=f"experiments/{job['job_id']}"
            assignment = None
            # Include the exact planned config before admission, even when the
            # training output directory deliberately does not exist yet.
            if plan.get('stage') == 'frozen_fusion_seed_review':
                export(job['config_path'], f'{target}/planned_config.json')
                assignment = export(root/'pipeline'/plan['prefix']/'assignments'/f"{job['name']}.json",
                                    f'{target}/gpu_assignment.json')
            state=export(run/'status.json',f'{target}/status.json') or {}
            result=export(run/'result.json',f'{target}/result.json') or {}
            for name in ('run.json','config.json','recovery.json','epoch_orders.json','cache_identity.json'):
                export(run/name,f'{target}/{name}')
            export(run/'metrics.jsonl',f'{target}/metrics.jsonl',lines=True)
            for part in ('dev','train_core'):
                for name in ('metrics.json','evaluation.json'):
                    export(run/f'eval_{part}'/name,f'{target}/eval_{part}/{name}')
                text(run/f'eval_{part}'/'predictions.csv',f'{target}/eval_{part}/predictions.csv')
            worker=read_source(root,job['status_path'])[0] or {}
            row=dict(run_id=job['job_id'],name=job['name'],experiment=f"FUSION_GRID_{job.get('group', job['name'])}",
                kind='training',status=worker.get('status','queued'),training_status=state.get('status','not_started'),
                training_commit=plan['code']['commit'],plan=plan['prefix'],device=job.get('device'),physical_gpu=job.get('physical_gpu'),
                progress=state,dev_metrics=result.get('dev',{}).get('overall',{}),result=result)
            if 'seed' in job:
                row.update(seed=job['seed'], group=job['group'])
            if assignment:
                row.update(physical_gpu=assignment['physical_gpu'],
                           device=f"cuda:{assignment['physical_gpu']['index']}")
            if result.get('gpu'):
                row['actual_gpu'] = result['gpu']
            rows.append(row)
        report=source_path(root,plan['report_root'])
        if report.is_dir():
            # Only immediate bounded report formats produced by the comparison.
            for file in sorted(report.iterdir()):
                if file.suffix=='.json':export(file,f'{base}/reports/{file.name}')
                elif file.suffix in ('.md','.csv','.svg'):text(file,f'{base}/reports/{file.name}')
    return files,rows,sources
