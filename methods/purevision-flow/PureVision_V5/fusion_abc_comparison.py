"""Read-only final aggregation for the frozen A/B/C pilot; never train models."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from collect_results import check_summary, require, target_identity
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import lateral_acceptance, summarize, validate_lateral_rule
from purevision_v5.fusion_abc import SOURCE_SHA
from purevision_v5.fusion_abc_cache import validate_selection
from purevision_v5.fusion_abc_training import validate_checkpoint, validate_config

VARIANTS = ('A', 'B', 'C')
PARAMETERS = dict(A=124354, B=100738, C=101250)
METRICS = ('epe', 'mae_vx', 'mae_vy', 'rmse_vx', 'rmse_vy', 'bias_vx', 'bias_vy')
LIMITS = [
    'Single seed42, 32 training scenes and 8 development scenes: exploratory development screening only.',
    'Encoders were already trained on the complete original training partition; the source checkpoint was selected on the existing complete dev. Only the new readouts use 32 training scenes.',
    'A/B change both fusion and readout. B/C are not a strict attention-only ablation (their feedforward widths and block structures also differ).',
    'PRO6000 and RTX5090 wall times are not a same-hardware efficiency comparison.',
    'Nonzero-vy inference requires at least 100 independent windows and 5 scenes; otherwise insufficient_data. Repeated training does not add independent samples.',
    'PLANAR_PROXY rear-axle labels, dev-selected checkpoints, and a split that is not log-disjoint; no held-out or full-data stable advantage claim.',
    'The experiment assesses these frozen features and new readouts, not the accuracy limit of visual velocity estimation.',
]


def jobs_by_variant(plan):
    jobs = {}
    for job in plan['jobs']:
        if job['kind'] != 'training':
            continue
        variant = job.get('variant', job.get('experiment', job['name']))
        if variant not in VARIANTS:
            variant = job['name'].rsplit('_', 1)[-1]
        require(variant in VARIANTS and variant not in jobs, 'Expected exactly one A/B/C job')
        jobs[variant] = job
    require(set(jobs) == set(VARIANTS), 'Missing A/B/C training job')
    return jobs


def cache_path(plan):
    return Path(plan.get('cache', plan.get('cache_root', '')))


def evaluation_path(plan, variant, part, job):
    if variant in plan.get('evaluation_paths', {}):
        return Path(plan['evaluation_paths'][variant][part])
    return Path(job['run_dir']) / f'eval_{part}'


def read_optional(path):
    return read_json(path) if Path(path).is_file() else {}


def read_history(directory):
    path = Path(directory) / 'metrics.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def difference(candidate, baseline):
    return {key: dict(delta=candidate[key] - baseline[key],
                      absolute_magnitude=abs(candidate[key] - baseline[key]),
                      relative_percent=100 * (candidate[key] - baseline[key]) / baseline[key]
                      if baseline[key] != 0 else None)
            for key in METRICS}


def expected_annotations(root, index, part):
    entry = index['partitions'][part]
    path = (root / entry['annotations']).resolve()
    require(path.is_relative_to(root.resolve()) and sha256(path) == entry['annotations_sha256'], 'Annotation path/hash mismatch')
    packet = torch.load(path, map_location='cpu', weights_only=True)
    require(object_hash(packet['metadata']) == entry['sample_manifest_sha256'], 'Annotation metadata changed')
    return [dict(scene=m['scene'], token=m['token'], time_s=m['time_s'], target_vx=float(y[0]), target_vy=float(y[1]))
            for m, y in zip(packet['metadata'], packet['targets'])]


def inspect(plan, variant, job, index, annotations):
    directory = Path(job['run_dir'])
    run, status = read_json(directory / 'run.json'), read_json(directory / 'status.json')
    config = run['contract']['config']; validate_config(config)
    require(config['variant'] == variant, 'Wrong variant')
    require(run['run_id'] == directory.name == job['job_id'], 'Run identity mismatch')
    require(run['contract_hash'] == object_hash(run['contract']), 'Invalid contract')
    require(run['contract']['scope'] == 'frozen_fusion_pilot' and run['contract']['smoke_steps'] == 0, 'Preflight excluded')
    require(run['contract']['cache_identity'] == index['identity'], 'Different common cache')
    require(run['encoder_training'] is False and run['arithmetic'] == dict(dtype='float32', amp=False, tf32=False, gradient_accumulation=1), 'Wrong arithmetic/encoder policy')
    require(run['code']['commit'] == plan['code']['commit'] == run['code']['verified_remote_commit'] and run['code']['dirty'] is False, 'Training was not clean/fixed/pushed')
    require(run['train_count'] == 7324 and run['dev_count'] == 1826, 'Wrong sample budget')
    require(run['trainable_parameters'] == PARAMETERS[variant], 'Wrong parameter count')
    config_path = Path(job['config_path'])
    require(read_json(config_path) == config and read_json(directory / 'config.json') == config, 'Config differs')
    config_hash = sha256(config_path)
    expected_hash = plan.get('config_hashes', {}).get(variant, plan.get('config_hashes', {}).get(job['name']))
    require(expected_hash == config_hash, 'Config checksum differs from plan')
    history = read_history(directory)
    require(len(history) == 20 and [r['epoch'] for r in history] == list(range(1, 21)), 'Incomplete epochs')
    for epoch, row in enumerate(history, 1):
        require(row['processed_batches'] == row['optimizer_updates'] == epoch * 115 and row['amp_skips'] == 0, 'Incomplete FP32 epoch updates')
        expected_lr = .001 * (1 + math.cos(math.pi * (epoch - 1) / 20)) / 2
        require(np.allclose(row['learning_rates'], [expected_lr], rtol=1e-12, atol=1e-15), 'Different LR schedule')
    require(run['status'] == status['status'] == 'completed' and run['completed_epochs'] == status['completed_epochs'] == 20, 'Run incomplete')
    require(run['processed_batches'] == run['optimizer_updates'] == status['processed_batches'] == status['optimizer_updates'] == 2300, 'Run update budget differs')
    best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
    for packet in (best, last):
        validate_checkpoint(packet, index)
        require(packet['run_id'] == run['run_id'] and packet['contract_hash'] == run['contract_hash'] and packet['code'] == run['code'], 'Checkpoint identity differs')
    require(last['completed_epochs'] == 20 and last['next_batch'] == 0 and last['processed_batches'] == last['optimizer_updates'] == 2300, 'Last checkpoint is not completed20')
    require(last['scheduler']['T_max'] == 20 and last['scheduler']['eta_min'] == 0 and last['scheduler']['last_epoch'] == 20, 'Wrong completed scheduler')
    best_i = min(range(20), key=lambda i: history[i]['dev']['overall']['epe'])
    require(best['best_epoch'] == best_i + 1 and best['best_epe'] == history[best_i]['dev']['overall']['epe'], 'Best violates minimum-EPE/earlier-tie rule')
    orders = read_json(directory / 'epoch_orders.json')
    require(len(orders) == 20 and [r['order_sha256'] for r in orders] == [r['sample_order_sha256'] for r in history], 'Order receipts differ')
    rule = validate_lateral_rule(config.get('lateral_rule'))
    row = dict(variant=variant, status='completed', run_id=run['run_id'], training_commit=run['code']['commit'],
               config_sha256=config_hash, contract_hash=run['contract_hash'], cache_identity=index['identity'],
               source_checkpoint_sha256=index['source']['checkpoint_sha256'], trainable_parameters=run['trainable_parameters'],
               completed_epochs=20, processed_batches=2300, optimizer_updates=2300, amp_skips=0,
               best_epoch=best_i + 1, best_dev_epe=best['best_epe'], last_dev_epe=history[-1]['dev']['overall']['epe'],
               best_checkpoint=str(directory / 'best.pt'), best_sha256=sha256(directory / 'best.pt'),
               last_checkpoint=str(directory / 'last.pt'), last_sha256=sha256(directory / 'last.pt'),
               initialization=run['initialization'], epoch_orders=orders,
               gpu=run.get('runtime_gpu', run.get('physical_gpu', job.get('physical_gpu'))),
               training_elapsed_seconds=run.get('elapsed_seconds'),
               peak_cuda_allocated_bytes=run.get('cuda_peak_allocated_bytes', status.get('cuda_peak_allocated_bytes')),
               cost_scope='trainer loop including validation/checkpoint writes; startup/cache loading recorded separately when available',
               evaluations={})
    require(row['best_sha256'] == run['best_checkpoint_sha256'] and row['last_sha256'] == run['last_checkpoint_sha256'], 'Run checkpoint hashes differ')
    predictions = {}
    for part in ('train_core', 'dev'):
        out = evaluation_path(plan, variant, part, job)
        receipt = read_json(out / 'evaluation.json')
        require(receipt['status'] == 'completed' and receipt['partition'] == part, 'Evaluation not complete')
        require(receipt['checkpoint_sha256'] == row['best_sha256'] and receipt['training_run'] == run['run_id'], 'Evaluation checkpoint mismatch')
        require(receipt['training_code'] == run['code'] and receipt['evaluation_code']['commit'] == plan['code']['commit'] and not receipt['evaluation_code']['dirty'], 'Evaluation version differs')
        require(receipt['cache_identity'] == index['identity'] and receipt['contract_hash'] == run['contract_hash'], 'Evaluation contract differs')
        with (out / 'predictions.csv').open(newline='') as stream:
            pred = list(csv.DictReader(stream))
        require(len(pred) == receipt['count'] == len(annotations[part]) and target_identity(pred) == target_identity(annotations[part]), 'Evaluation sample/target mismatch')
        actual = summarize([[float(p['vx']), float(p['vy'])] for p in pred], [[float(p['target_vx']), float(p['target_vy'])] for p in pred], [p['scene'] for p in pred], rule)
        actual['nonzero_vy']['status'] = 'sufficient_data' if actual['nonzero_vy']['adequate'] else 'insufficient_data'
        actual['task'] = 'PLANAR_PROXY'
        check_summary(read_json(out / 'metrics.json'), actual)
        if part == 'dev':
            check_summary(history[best_i]['dev'], actual)
        row['evaluations'][part] = dict(metrics=actual, count=len(pred), predictions=str(out / 'predictions.csv'), predictions_sha256=sha256(out / 'predictions.csv'))
        predictions[part] = pred
    row['generalization_gap_dev_minus_train_epe'] = row['evaluations']['dev']['metrics']['overall']['epe'] - row['evaluations']['train_core']['metrics']['overall']['epe']
    return row, history, predictions


def collect(plan_path):
    plan = read_json(plan_path)
    require(plan.get('identity') == object_hash({k: v for k, v in plan.items() if k != 'identity'}), 'Invalid plan identity')
    jobs = jobs_by_variant(plan)
    states = {}
    for variant, job in jobs.items():
        run = read_optional(Path(job['run_dir']) / 'run.json')
        states[variant] = dict(status=run.get('status', 'not_started'), completed_epochs=run.get('completed_epochs', 0), run_id=job['job_id'],
                               evaluations={p: read_optional(evaluation_path(plan, variant, p, job) / 'evaluation.json').get('status', 'not_started') for p in ('train_core', 'dev')})
    complete = all(s['status'] == 'completed' and s['completed_epochs'] == 20 and all(v == 'completed' for v in s['evaluations'].values()) for s in states.values())
    base = dict(schema='v5_fusion_abc_pilot_comparison_v1', status='completed' if complete else 'incomplete',
                observed_at=now(), plan_identity=plan['identity'], experiment_id=plan['prefix'], training_commit=plan['code']['commit'], limits=LIMITS)
    if not complete:
        return dict(base, rows=states, comparisons={}, note='Final comparison withheld until all three20-epoch runs and independent train/dev evaluations complete.'), {}, {}
    root = cache_path(plan).resolve(); index = read_json(root / 'index.json')
    require(index['scope'] == 'pilot' and index['identity'] == object_hash({k: v for k, v in index.items() if k != 'identity'}), 'Invalid pilot cache identity')
    validate_selection(index['selection'])
    require(index['source']['checkpoint_sha256'] == SOURCE_SHA and index['code']['commit'] == plan['code']['commit'], 'Cache source/code differs')
    require(index['frozen_state_before_sha256'] == index['frozen_state_after_sha256'], 'Encoder mutated during extraction')
    annotations = {p: expected_annotations(root, index, p) for p in ('train_core', 'dev')}
    rows, histories, predictions = {}, {}, {}
    for variant, job in jobs.items():
        rows[variant], histories[variant], predictions[variant] = inspect(plan, variant, job, index, annotations)
    for variant in ('B', 'C'):
        require(rows[variant]['epoch_orders'] == rows['A']['epoch_orders'], 'Variant sample orders differ')
    require(rows['B']['initialization']['common_modules_sha256'] == rows['C']['initialization']['common_modules_sha256'], 'B/C common initialization differs')
    comparisons = {}
    for candidate, baseline in (('B', 'A'), ('C', 'B'), ('C', 'A')):
        a = rows[candidate]['evaluations']['dev']['metrics']; b = rows[baseline]['evaluations']['dev']['metrics']
        scenes = {scene: difference(a['per_scene'][scene], b['per_scene'][scene]) for scene in b['per_scene']}
        comparisons[f'{candidate}_minus_{baseline}'] = dict(candidate=candidate, baseline=baseline,
            delta_convention='candidate minus baseline; negative EPE/MAE/RMSE means improvement; relative_percent denominator is baseline',
            metrics=difference(a['overall'], b['overall']), per_scene=scenes,
            improved_scenes=sum(s['epe']['delta'] < 0 for s in scenes.values()), degraded_scenes=sum(s['epe']['delta'] > 0 for s in scenes.values()),
            tied_scenes=sum(s['epe']['delta'] == 0 for s in scenes.values()), lateral=lateral_acceptance(a, b))
    winner = min(VARIANTS, key=lambda v: rows[v]['best_dev_epe'])
    report = dict(base, rows=rows, comparisons=comparisons, cache=dict(path=str(root), identity=index['identity'], index_sha256=sha256(root / 'index.json'),
        source=index['source'], selection=index['selection'], partitions=index['partitions'],
        elapsed_seconds=index.get('elapsed_seconds', index.get('runtime', {}).get('elapsed_seconds')),
        gpu=index.get('runtime_gpu', index.get('physical_gpu')), peak_cuda_allocated_bytes=index.get('cuda_peak_allocated_bytes')),
        sample_counts={p: len(v) for p, v in annotations.items()}, scene_counts={p: len({x['scene'] for x in v}) for p, v in annotations.items()},
        recommendation=f'{winner} has the lowest dev EPE in this completed single-seed frozen-feature pilot. Treat it as a candidate for a user-approved next comparison, not established superiority.',
        follow_up='Review paired scene and lateral outcomes first. Any extra seeds, full-data training, or encoder changes require a separate user decision; none are launched by this report.')
    return report, histories, predictions


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def export_tables(out, report, histories, predictions):
    curves = []
    for variant, history in histories.items():
        for row in history:
            curves.append(dict(variant=variant, epoch=row['epoch'], training_loss=row['training_loss'], dev_epe=row['dev']['overall']['epe'], learning_rate=row['learning_rates'][0],
                               next_learning_rate=row['next_learning_rates'][0], processed_batches=row['processed_batches'], optimizer_updates=row['optimizer_updates']))
    write_csv(out / 'learning_curves.csv', curves)
    for part in ('dev', 'train_core'):
        indexed = {v: {(p['scene'], p['token']): p for p in predictions[v][part]} for v in VARIANTS}
        require(all(set(indexed[v]) == set(indexed['A']) for v in VARIANTS), 'Paired prediction identities differ')
        pairs = []
        for key in sorted(indexed['A']):
            source = indexed['A'][key]
            row = {k: source[k] for k in ('scene', 'token', 'time_s', 'target_vx', 'target_vy')}
            for v in VARIANTS:
                p = indexed[v][key]
                require(all(p[k] == source[k] for k in ('time_s', 'target_vx', 'target_vy')), 'Paired targets/timestamp differ')
                row.update({f'{v}_{axis}': float(p[axis]) for axis in ('vx', 'vy')})
                row[f'{v}_epe'] = math.hypot(float(p['vx']) - float(p['target_vx']), float(p['vy']) - float(p['target_vy']))
            for candidate, baseline in (('B', 'A'), ('C', 'B'), ('C', 'A')):
                row[f'{candidate}_minus_{baseline}_epe'] = row[f'{candidate}_epe'] - row[f'{baseline}_epe']
            pairs.append(row)
        write_csv(out / f'paired_{part}_predictions.csv', pairs)
    scene_rows = []
    for name, comparison in report['comparisons'].items():
        for scene, metrics in comparison['per_scene'].items():
            row = dict(comparison=name, scene=scene)
            for metric, values in metrics.items():
                for kind, value in values.items(): row[f'{metric}_{kind}'] = value
            scene_rows.append(row)
    write_csv(out / 'paired_dev_scenes.csv', scene_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for variant, history in histories.items():
        epochs = [r['epoch'] for r in history]
        for axis, values in zip(axes, ([r['training_loss'] for r in history], [r['dev']['overall']['epe'] for r in history], [r['learning_rates'][0] for r in history])):
            axis.plot(epochs, values, label=variant)
    for axis, title, ylabel in zip(axes, ('Training loss', 'Development EPE', 'Learning rate'), ('Scaled SmoothL1', 'm/s', 'LR')):
        axis.set(title=title, xlabel='Complete epoch', ylabel=ylabel); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out / 'learning_curves.svg'); plt.close(fig)


def markdown(report):
    lines = ['# Frozen-encoder A/B/C fusion pilot', '', f"Status: **{report['status']}**. Experiment: `{report['experiment_id']}`.", f"Actual training commit: `{report['training_commit']}`.", '']
    if report['status'] == 'completed':
        lines += [report['recommendation'], '', '| Model | Best epoch | Best dev EPE | Last dev EPE | Best train EPE | vx MAE | vy MAE | Parameters | Updates | Seconds |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for v in VARIANTS:
            r = report['rows'][v]; d = r['evaluations']['dev']['metrics']['overall']; t = r['evaluations']['train_core']['metrics']['overall']
            lines.append(f"| {v} | {r['best_epoch']} | {d['epe']:.6f} | {r['last_dev_epe']:.6f} | {t['epe']:.6f} | {d['mae_vx']:.6f} | {d['mae_vy']:.6f} | {r['trainable_parameters']} | {r['optimizer_updates']} | {r['training_elapsed_seconds']:.1f} |")
        lines += ['', '| Comparison | Dev EPE delta (m/s) | Relative delta | Improved / degraded / tied scenes | Nonzero-vy |', '|---|---:|---:|---:|---|']
        for name, c in report['comparisons'].items():
            d = c['metrics']['epe']; lat = c['lateral']['checks']['nonzero_vy']['status']
            relative = 'undefined (zero baseline)' if d['relative_percent'] is None else f"{d['relative_percent']:+.2f}%"
            lines.append(f"| {name} | {d['delta']:+.6f} | {relative} | {c['improved_scenes']} / {c['degraded_scenes']} / {c['tied_scenes']} | {lat} |")
        lines += ['', 'All three groups completed20 epochs/2300 batches/2300 optimizer updates with AMP disabled. Full component metrics, GPU identities, cache/source hashes and paired comparisons are in summary.json.', '', '![Learning curves](learning_curves.svg)', '', report['follow_up']]
    else:
        lines += [report['note'], '', '```json', json.dumps(report['rows'], indent=2), '```']
    return '\n'.join(lines + ['', *['- ' + v for v in report['limits']], ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); report, histories, predictions = collect(args.plan)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    if report['status'] == 'completed': export_tables(out, report, histories, predictions)
    report['artifacts'] = {p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / 'summary.json', report); (out / 'summary.md').write_text(markdown(report))
    print(json.dumps(dict(status=report['status'], output=str(out))), flush=True)
    if report['status'] != 'completed': raise SystemExit(2)


if __name__ == '__main__': main()
