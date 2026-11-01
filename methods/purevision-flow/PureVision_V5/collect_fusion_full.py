"""Publish bounded full-scope evidence, with oversized prediction files hashed."""
from pathlib import Path
import hashlib
import json
from publish_results import MAX_BYTES, check_public, encoded, read_source, source_path


def collect_public(root, plans):
    root = Path(root).resolve()
    files, rows, sources = {}, [], []
    def export(path, target, lines=False):
        value, origin = read_source(root, path, lines=lines)
        if value is not None:
            files[target] = (b''.join((json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n').encode()
                                     for x in value) if lines else encoded(value))
            sources.append(origin)
        return value
    def text(path, target):
        path = source_path(root, path)
        if not path.is_file():
            return
        if path.stat().st_size > MAX_BYTES:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            value = dict(path=str(path), sha256=digest, bytes=path.stat().st_size,
                         representation='hash_only_size_limit')
            files[target+'.reference.json'] = encoded(value)
            sources.append(dict(path=str(path), source_sha256=digest, bytes=value['bytes'],
                                representation=value['representation']))
            return
        blob = path.read_bytes()
        check_public(blob.decode('utf-8'))
        files[target] = blob
        sources.append(dict(path=str(path), source_sha256=hashlib.sha256(blob).hexdigest(),
                            bytes=len(blob), representation='original_text'))
    for path in plans:
        plan = json.loads(Path(path).read_text())
        if plan.get('stage') != 'frozen_fusion_full_v1':
            continue
        base = f"fusion_full/{plan['prefix']}"
        for key in ('full_scope', 'history_manifest', 'extraction_preflight', 'training_implementation_preflight', 'training_preflight'):
            export(plan[key], f'{base}/{key}.json')
        for name in ('index.json', 'status.json', 'verification.json', 'overlap_verification.json'):
            export(Path(plan['cachefull'])/name, f'{base}/cache/{name}')
        for job in plan['jobs']:
            if job['kind'] != 'training':
                continue
            directory = Path(job['run_dir']); target = f"experiments/{job['job_id']}"
            export(job['config_path'], f'{target}/planned_config.json')
            state = export(directory/'status.json', f'{target}/status.json') or {}
            result = export(directory/'result.json', f'{target}/result.json') or {}
            core = export(directory/'dev_ready.json', f'{target}/dev_ready.json') or {}
            for name in ('run.json', 'config.json', 'recovery.json', 'epoch_orders.json', 'cache_identity.json'):
                export(directory/name, f'{target}/{name}')
            export(directory/'metrics.jsonl', f'{target}/metrics.jsonl', lines=True)
            for part in ('dev', 'train_core'):
                for name in ('metrics.json', 'evaluation.json'):
                    export(directory/f'eval_{part}'/name, f'{target}/eval_{part}/{name}')
                text(directory/f'eval_{part}'/'predictions.csv', f'{target}/eval_{part}/predictions.csv')
            worker = read_source(root, job['status_path'])[0] or {}
            measured = result or core
            rows.append(dict(run_id=job['job_id'], name=job['name'], experiment=f"FUSION_FULL_{job['group']}",
                kind='training', status=worker.get('status', 'queued'), training_status=state.get('status', 'not_started'),
                training_commit=plan['code']['commit'], plan=plan['prefix'], device=job['device'],
                physical_gpu=job['physical_gpu'], group=job['group'], seed=42, progress=state,
                dev_metrics=measured.get('dev', {}).get('overall', {}),
                best_dev_epe=measured.get('best_dev_epe'), best_epoch=measured.get('best_epoch'),
                result_path=str(directory/'result.json')))
        report = source_path(root, plan['report_root'])
        if report.is_dir():
            for file in sorted(report.iterdir()):
                if file.suffix == '.json':
                    export(file, f'{base}/reports/{file.name}')
                elif file.suffix in ('.md', '.csv', '.svg'):
                    text(file, f'{base}/reports/{file.name}')
    return files, rows, sources
