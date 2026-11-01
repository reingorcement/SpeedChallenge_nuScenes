"""Bounded public A/B/C artifacts for publish_results.collect.

No scheduler, model execution, subprocess, cache arrays, weights or credential
files are accessed here. The existing isolated publisher owns commit/push.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from fusion_abc_comparison import cache_path, evaluation_path, jobs_by_variant
from publish_results import (MAX_BYTES, TERMINAL, check_public, encoded, read_source,
                             source_path)

STAGE = 'frozen_fusion_abc_pilot'
REPORT_FILES = {'summary.json', 'summary.md', 'learning_curves.csv', 'learning_curves.svg',
                'paired_dev_predictions.csv', 'paired_train_core_predictions.csv', 'paired_dev_scenes.csv'}


def collect_public(root, plans):
    """Return extra files, replacement training rows, and exact source receipts."""
    root = Path(root).resolve()
    files, rows, sources = {}, [], []

    def export_json(path, target):
        value, origin = read_source(root, path)
        if value is not None:
            files[target] = encoded(value); sources.append(origin)
        return value

    def export_text(path, target):
        path = source_path(root, path)
        if not path.is_file(): return
        with path.open('rb') as stream:
            blob = stream.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES:
            raise ValueError('ABC public artifact exceeds bounded16MiB size')
        check_public(blob.decode('utf-8'))
        files[target] = blob
        sources.append(dict(path=str(path), source_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), representation='original_text'))

    for plan_path in plans:
        plan, _ = read_source(root, plan_path)
        if not plan or plan.get('stage') != STAGE: continue
        prefix = plan['prefix']
        cache = source_path(root, cache_path(plan))
        cache_base = f'fusion_abc/{prefix}/cache'
        for name in ('index.json', 'status.json', 'verification.json'):
            export_json(cache / name, f'{cache_base}/{name}')
        for key in ('verification', 'preflight', 'verification_path', 'launch_receipt'):
            if plan.get(key):
                export_json(plan[key], f'fusion_abc/{prefix}/receipts/{key}.json')
        for variant, job in jobs_by_variant(plan).items():
            directory = source_path(root, job['run_dir']); base = f'experiments/{job["job_id"]}'
            run = export_json(directory / 'run.json', f'{base}/run.json') or {}
            state = export_json(directory / 'status.json', f'{base}/status.json') or {}
            worker = export_json(job['status_path'], f'{base}/job_status.json') or {}
            for name in ('config.json', 'epoch_orders.json', 'cache_identity.json'):
                export_json(directory / name, f'{base}/{name}')
            history, origin = read_source(root, directory / 'metrics.jsonl', lines=True)
            if history is not None:
                files[f'{base}/metrics.jsonl'] = b''.join((json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n').encode() for v in history)
                sources.append(origin)
            row = dict(run_id=job['job_id'], name=job['name'], experiment=f'FUSION_ABC_{variant}', variant=variant,
                       kind='training', status=worker.get('status', 'queued'), training_status=state.get('status', 'not_started'),
                       training_commit=plan['code']['commit'], plan=prefix, device=job['device'], physical_gpu=job.get('physical_gpu'),
                       log_path=job.get('log_path'), progress={k: state[k] for k in ('completed_epochs', 'epoch', 'next_batch', 'processed_batches',
                           'optimizer_updates', 'amp_skips', 'best_dev_epe', 'best_epoch', 'elapsed_seconds', 'cuda_peak_allocated_bytes') if k in state},
                       cache_identity=run.get('contract', {}).get('cache_identity'), evaluations={})
            for part in ('dev', 'train_core'):
                directory_eval = source_path(root, evaluation_path(plan, variant, part, job))
                evaluation_base = f'fusion_abc/{prefix}/evaluations/{variant}/{part}'
                receipt = export_json(directory_eval / 'evaluation.json', f'{evaluation_base}/evaluation.json') or {}
                row['evaluations'][part] = dict(status=receipt.get('status', 'not_started'))
                if receipt.get('status') == 'completed':
                    metric = export_json(directory_eval / 'metrics.json', f'{evaluation_base}/metrics.json')
                    export_text(directory_eval / 'predictions.csv', f'{evaluation_base}/predictions.csv')
                    row['evaluations'][part]['metrics'] = metric
                    if part == 'dev':
                        row['dev_metrics'] = metric['overall']; row['evaluation_status'] = 'completed'
            if worker.get('status') == 'completed':
                row['status'] = ('completed' if state.get('status') == 'completed' and state.get('completed_epochs') == 20
                                 and all(x['status'] == 'completed' for x in row['evaluations'].values()) else 'evaluation_pending')
            # Full checkpoints remain private; preserve exact stable-inode hashes.
            for name in ('last.pt', 'best.pt'):
                checkpoint = directory / name
                if checkpoint.is_file():
                    with checkpoint.open('rb') as stream:
                        import os
                        info = os.fstat(stream.fileno())
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    row.setdefault('checkpoint_hashes', {})[name] = dict(path=str(checkpoint), sha256=digest, size_bytes=info.st_size, representation='hash_only')
            # A few parsed update/validation messages explain activity without
            # uploading full logs or unstructured environment/credential output.
            log = source_path(root, job['log_path'])
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 32768)); tail = stream.read().splitlines()
                events = []
                for raw in tail:
                    try:
                        value = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(value, dict) and value.get('event') in {'update', 'validation'}:
                        check_public(value); events.append(value)
                files[f'{base}/log_summary.json'] = encoded(dict(source=str(log), complete_log_uploaded=False, last_events=events[-4:]))
            rows.append(row)
        report = source_path(root, plan['completion_report']).parent
        for name in sorted(REPORT_FILES):
            path = report / name
            target = f'comparisons/{report.name}/{name}'
            if path.suffix == '.json': export_json(path, target)
            else: export_text(path, target)
        # Fresh stage-specific scheduler and startup receipts. Do not rely on
        # historical global schedule files to claim this pilot is monitored.
        monitoring = root / 'monitoring' / f'fusion_abc_{prefix}'
        for name in ('schedule_install.json', 'schedule_trigger.json', 'lifecycle.json', 'cleanup_receipt.json'):
            export_json(monitoring / name, f'fusion_abc/{prefix}/monitoring/{name}')
        for name in ('cache', 'A', 'B', 'C', 'compare'):
            export_json(monitoring / 'startup' / f'{name}.json',
                        f'fusion_abc/{prefix}/monitoring/startup/{name}.json')
        # Current scope only. Global private network controls and raw AI logs
        # remain excluded; ordinary publisher exports safe patrol receipts.
        for suffix in ('launch', 'schedule_install', 'schedule_trigger', 'lightweight_startup', 'completion'):
            path = root / 'evidence' / f'{prefix}_{suffix}.json'
            export_json(path, f'fusion_abc/{prefix}/receipts/{suffix}.json')
    return files, rows, sources
