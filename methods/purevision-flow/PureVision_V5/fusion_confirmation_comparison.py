"""Saved-prediction full three-seed confirmation and matched S23 stage20 report.

Original-protocol seeds and parent-initialized stages are separate estimands.
No model is executed here. A partial report never declares three-seed stability.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from collect_results import check_summary, require, target_identity
from fusion_full_comparison import (COUNTS, SCENES, GROUPS, PAIRS, checked_index,
                                    compare as full_compare, inspect_full)
from fusion_full_history import identity_map, load_manifest
from fusion_grid_comparison import METRICS, expected_rows, read_history, read_predictions, summarize_rows
from fusion_multiseed_comparison import atomic_csv, atomic_text, mean_sd, verify_plan
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import DEFAULT_LATERAL_RULE
from purevision_v5.fusion_abc import SOURCE_SHA
from purevision_v5.fusion_tail import READOUT_SHA, PARENT_EPE

SEEDS = (42, 43, 44)
STAGES = ('S23_FROZEN_CONT', 'S23_TAIL_FT')
BASELINE_COMMIT = 'bc9a3d8addbdb5413589e0b2121d87ee85e5b026'
BASELINE_PREFIX = '20261102_0154_fusion_full_s42'
STAGE_EPOCHS, STAGE_UPDATES = 20, 41560
NEAR_ZERO_THRESHOLD, FALSE_ALARM_THRESHOLD = .05, .2


def compare(candidate, baseline, a, b, category):
    value = full_compare(candidate, baseline, a, b, category)
    contributions = value['scene_contribution']['weighted_gain_m_s']
    top = sorted((s for s in contributions if contributions[s] > 0), key=contributions.get, reverse=True)[:3]
    net = value['scene_contribution']['total_gain_m_s']
    value['scene_contribution'].update(top_three_positive_scenes=top,
        top_three_weighted_gain_m_s=sum(contributions[s] for s in top),
        top_three_share_of_net_gain=sum(contributions[s] for s in top) / net if net > 0 else None)
    return value


def lateral_diagnostics(rows, rule=None):
    """Fixed reporting masks; repeated predictions never multiply label counts."""
    rule = rule or DEFAULT_LATERAL_RULE
    target = np.asarray([float(r['target_vy']) for r in rows], dtype=np.float64)
    prediction = np.asarray([float(r['vy']) for r in rows], dtype=np.float64)
    scenes = np.asarray([r['scene'] for r in rows])
    require(len(target) > 0 and np.isfinite(target).all() and np.isfinite(prediction).all(),
            'Nonfinite/missing lateral predictions')
    threshold = rule['nonzero_threshold_m_s']
    masks = dict(overall=np.ones(len(rows), dtype=bool), nonzero=np.abs(target) >= threshold,
        positive=target >= threshold, negative=target <= -threshold,
        near_zero=np.abs(target) < NEAR_ZERO_THRESHOLD, below_nonzero=np.abs(target) < threshold)
    result = dict(nonzero_threshold_m_s=threshold, near_zero_threshold_m_s=NEAR_ZERO_THRESHOLD,
        false_alarm_threshold_m_s=FALSE_ALARM_THRESHOLD,
        near_zero_definition='abs(target_vy)<0.05; reporting diagnostic only; labels unchanged', subsets={})
    for name, mask in masks.items():
        count, scene_count = int(mask.sum()), len(set(scenes[mask]))
        value = dict(count=count, scene_count=scene_count, adequate=bool(count >= rule['min_subset_windows']
            and scene_count >= rule['min_subset_scenes']))
        value['status'] = 'adequate' if value['adequate'] else 'insufficient_data'
        if count:
            error, zero_error = prediction[mask] - target[mask], -target[mask]
            value.update(mae_vy=float(np.abs(error).mean()), rmse_vy=float(np.sqrt(np.square(error).mean())),
                bias_vy=float(error.mean()), zero_vy_mae=float(np.abs(zero_error).mean()),
                zero_vy_rmse=float(np.sqrt(np.square(zero_error).mean())), zero_vy_bias=float(zero_error.mean()))
            if name in ('near_zero', 'below_nonzero'):
                alarms = np.abs(prediction[mask]) >= FALSE_ALARM_THRESHOLD
                value.update(false_alarm_count=int(alarms.sum()), false_alarm_rate=float(alarms.mean()))
        result['subsets'][name] = value
    result['acceptance_status'] = result['subsets']['nonzero']['status']
    return result


def sign_summary(values):
    values = list(values)
    n = len(values)
    improved, degraded = sum(v < 0 for v in values), sum(v > 0 for v in values)
    return dict(paired_seed_count=n, improved_seeds=improved, degraded_seeds=degraded,
        tied_seeds=sum(v == 0 for v in values),
        stability='pending_all_three_seeds' if n < 3 else 'improved_all_three' if improved == 3
        else 'degraded_all_three' if degraded == 3 else 'tied_all_three' if all(v == 0 for v in values)
        else 'mixed_across_seeds')


def aggregate_confirmation(runs):
    aggregate, pairs = {}, {}
    for group in GROUPS:
        values = [runs[str(seed)][group] for seed in SEEDS if group in runs[str(seed)]]
        metrics = {key: mean_sd([r['dev']['overall'][key] for r in values]) for key in METRICS}
        metrics.update({key: mean_sd([r[key] for r in values if r.get(key) is not None]) for key in
            ('best_dev_epe', 'last_dev_epe', 'best_epoch', 'best_train_epe', 'generalization_gap_dev_minus_train_epe')})
        speeds = {}
        if values:
            for label, first in values[0]['dev']['speed_bins'].items():
                require(all(v['dev']['speed_bins'][label]['count'] == first['count'] for v in values),
                        'Speed-bin independent labels changed')
                speeds[label] = dict(independent_window_count=first['count'],
                    metrics={key: mean_sd([v['dev']['speed_bins'][label]['metrics'][key] for v in values])
                             for key in METRICS} if first['count'] else {})
        aggregate[group] = dict(seeds=[v['seed'] for v in values], metrics=metrics, speed_bins=speeds,
                                conclusion_ready=len(values) == 3)
    for candidate, baseline in PAIRS:
        per_seed = {str(seed): compare(candidate, baseline, runs[str(seed)][candidate]['dev'],
            runs[str(seed)][baseline]['dev'], 'A_original_full_protocol') for seed in SEEDS
            if candidate in runs[str(seed)] and baseline in runs[str(seed)]}
        values = list(per_seed.values())
        scenes, speeds = {}, {}
        if values:
            for scene in values[0]['per_scene']:
                scenes[scene] = dict(metrics={key: mean_sd([v['per_scene'][scene][key]['delta'] for v in values])
                    for key in METRICS}, weighted_gain_m_s=mean_sd([v['scene_contribution']['weighted_gain_m_s'][scene]
                    for v in values]), **sign_summary(v['per_scene'][scene]['epe']['delta'] for v in values))
            for label, first in values[0]['speed_bins'].items():
                speeds[label] = dict(independent_window_count=first['count'],
                    metrics={key: mean_sd([v['speed_bins'][label]['metrics'][key]['delta'] for v in values]) for key in METRICS},
                    **sign_summary(v['speed_bins'][label]['metrics']['epe']['delta'] for v in values))
        pairs[f'{candidate}_minus_{baseline}'] = dict(candidate=candidate, baseline=baseline, per_seed=per_seed,
            metrics={key: mean_sd([v['metrics'][key]['delta'] for v in values]) for key in METRICS},
            relative_epe_percent=mean_sd([v['metrics']['epe']['relative_percent'] for v in values]),
            per_scene=scenes, speed_bins=speeds, **sign_summary(v['metrics']['epe']['delta'] for v in values))
    return dict(aggregate=aggregate, paired_comparisons=pairs,
        dev_complete=all(len(runs[str(seed)]) == 3 for seed in SEEDS),
        uncertainty='sample SD ddof=1 across three training seeds, not a confidence interval or independent dev samples')


def stage_decision(stage, parent):
    """An epoch0 retention decision never replaces stage1..20 best metadata."""
    improved = stage['best_dev_epe'] < parent['best_dev_epe']
    return dict(improved_over_parent=improved, preferred='stage_best' if improved else 'parent_epoch0',
        tie_rule='parent_on_tie', best_including_parent_epoch=stage['best_epoch'] if improved else 0,
        stage_best_epoch=stage['best_epoch'], stage_best_dev_epe=stage['best_dev_epe'],
        parent_epoch0_dev_epe=parent['best_dev_epe'],
        selected_dev_epe=min(stage['best_dev_epe'], parent['best_dev_epe']))


def fitting_comparison(candidate, baseline):
    a, b = candidate.get('best_train_epe'), baseline.get('best_train_epe')
    return dict(dev_delta_epe=candidate['best_dev_epe'] - baseline['best_dev_epe'],
        train_delta_epe=a - b if a is not None and b is not None else None,
        train_evaluation_pending=a is None or b is None,
        gap_delta_epe=(candidate['best_dev_epe'] - a) - (baseline['best_dev_epe'] - b)
            if a is not None and b is not None else None)


def compact_row(row):
    # Full per-scene dev is useful; 582 train scene records remain in each run.
    value = dict(row)
    raw = read_json(row['result_path'])
    for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'peak_memory_allocated_bytes',
                'peak_memory_reserved_bytes', 'selected_best_train_evaluation_seconds'):
        if key in raw:
            value[key] = raw[key]
    if value.get('train'):
        value['train'] = {k: v for k, v in value['train'].items() if k != 'per_scene'}
    return value


def jobs_by_key(plan, baseline=False):
    jobs = {}
    for original in plan['jobs']:
        if original.get('kind') != 'training':
            continue
        group = original.get('group', original.get('variant'))
        seed = 42 if baseline else original.get('seed')
        key = (seed, group)
        require(type(seed) is int and key not in jobs, 'Duplicate/noninteger group seed')
        job = dict(original, seed=seed)
        job.setdefault('job_id', Path(job['run_dir']).name)
        if not baseline:
            require(job['protocol'] == ('s23_stage20' if group in STAGES else 'full_confirmation'),
                    'Only revised stage20 or original full protocol is authorized')
        jobs[key] = job
    expected = {(42, group) for group in GROUPS} if baseline else (
        {(seed, group) for seed in (43, 44) for group in GROUPS} | {(42, group) for group in STAGES})
    require(set(jobs) == expected, 'Unexpected/missing confirmation or stage20 jobs')
    return jobs


def inspect_stage(plan, group, job, index, expected, manifest, parent):
    directory = Path(job['run_dir'])
    result_path = directory / 'result.json'
    if not result_path.is_file():
        result_path = directory / 'dev_ready.json'
    if not result_path.is_file():
        return None, None
    result = read_json(result_path)
    if result.get('status') not in {'dev_ready', 'completed'}:
        return None, None
    from purevision_v5.fusion_tail_training import CONTRACT_SCOPE, load_recovery, validate_config
    run = read_json(directory / 'run.json')
    contract, config = run['contract'], run['contract']['config']
    validate_config(config)
    require(config['group'] == group and config['seed'] == job['seed'] == 42 and config['epochs'] == STAGE_EPOCHS,
            'Wrong stage group/seed/budget')
    require(contract['scope'] == CONTRACT_SCOPE and contract['partition_counts'] == COUNTS
            and contract['scene_counts'] == SCENES and contract['stage_budget_epochs'] == STAGE_EPOCHS
            and contract['stage_optimizer_updates'] == STAGE_UPDATES, 'Wrong formal stage scope/budget')
    require(result['train_count'] == COUNTS['train_core'] and result['dev_count'] == COUNTS['dev']
            and result['data_scope'] == 'full', 'Stage result lacks complete data')
    require(result['group'] == group and result['run_id'] == run['run_id'] == job['job_id'], 'Wrong stage run identity')
    require(run['contract_hash'] == result['contract_hash'] == object_hash(contract), 'Stage contract hash differs')
    require(result['code'] == run['code'] and result['code']['commit'] == plan['code']['commit']
            == result['code']['verified_remote_commit'] and result['code']['dirty'] is False, 'Stage code not fixed/pushed')
    require(result['cache_identity'] == contract['cache_identity'] == index['identity'], 'Stage cache differs')
    require(run['encoder_training'] == (group == 'S23_TAIL_FT')
            and all(run['arithmetic'][k] == v for k, v in dict(dtype='float32', amp=False, tf32=False, effective_batch=64).items()),
            'Stage encoder/arithmetic policy differs')
    require(result['completed_epochs'] == STAGE_EPOCHS and result['optimizer_updates'] == STAGE_UPDATES,
            'Stage budget incomplete')
    require(read_json(job['config_path']) == config == read_json(directory / 'config.json'), 'Stage config changed')
    trajectory = read_history(directory / 'metrics.jsonl')
    require([v['epoch'] for v in trajectory] == list(range(1, STAGE_EPOCHS + 1)), 'Stage history incomplete')
    rates = [1e-4, 1e-5] if group == 'S23_TAIL_FT' else [1e-4]
    for epoch, item in enumerate(trajectory, 1):
        require(item['optimizer_updates'] == item['processed_batches'] == epoch * 2078 and item['amp_skips'] == 0,
                'Stage effective update count differs')
        require(item['dev']['overall']['count'] == 23480 and item['dev']['scene_count'] == 103,
                'Stage epoch dev coverage differs')
        expected_rates = [rate * (1 + math.cos(math.pi * (epoch - 1) / STAGE_EPOCHS)) / 2 for rate in rates]
        require(np.allclose(item['learning_rates'], expected_rates, rtol=1e-10, atol=1e-14), 'Stage cosine20 differs')
        for total, x, y in (('training_loss', 'train_online_loss_x', 'train_online_loss_y'),
                            ('dev_loss', 'dev_loss_x', 'dev_loss_y')):
            require(math.isclose(item[total], (item[x] + item[y]) / 2, rel_tol=2e-6, abs_tol=2e-7),
                    'Stage axis losses differ')
    recovery, loaded = load_recovery(directory / 'recovery.json', run['contract_hash'])
    require(recovery['completed_epochs'] == STAGE_EPOCHS and loaded['current'][1]['optimizer_updates'] == STAGE_UPDATES,
            'Stage atomic recovery is incomplete')
    best = min(trajectory, key=lambda v: v['dev']['overall']['epe'])
    require(best['epoch'] == result['best_epoch'] == loaded['best'][1]['completed_epochs']
            and result['best_dev_epe'] == best['dev']['overall']['epe']
            and result['last_dev_epe'] == trajectory[-1]['dev']['overall']['epe'], 'Stage-only best/last selection differs')
    for name, label in (('best', 'best'), ('last', 'current')):
        require(sha256(directory / f'{name}.pt') == result[f'{name}_checkpoint_sha256'] == recovery[label]['sha256'],
                'Full stage checkpoint export differs')
    checkpoint = loaded['best'][1]
    require(result['parents'] == run['parents'] == checkpoint['provenance'], 'Stage parent provenance differs')
    require(checkpoint['provenance']['encoder']['checkpoint_sha256'] == SOURCE_SHA
            and checkpoint['provenance']['readout']['checkpoint_sha256'] == READOUT_SHA
            and checkpoint['provenance']['readout']['selected_epoch'] == 19, 'Stage parent weights differ')
    freeze = checkpoint['freeze_manifest']
    require(freeze == result['freeze_manifest'] and result['trainable_parameters'] == freeze['trainable_parameter_count'],
            'Stage freeze manifest/parameter count differs')
    epoch0 = result['parent_epoch0']
    require(epoch0['run_id'] == parent['run_id'] and epoch0['selected_parent_epoch'] == 19
            and epoch0['checkpoint_sha256'] == READOUT_SHA and epoch0['dev_epe'] == PARENT_EPE,
            'Stage epoch0 reference differs')
    check_summary(epoch0['dev'], parent['dev'])
    check_summary(epoch0['train'], parent['train'])
    decision = stage_decision(result, parent)
    require(result['best_including_parent_epoch'] == decision['best_including_parent_epoch']
            and all(result['replacement_decision'][k] == decision[k]
                    for k in ('improved_over_parent', 'preferred', 'tie_rule')),
            'Stage epoch0 replacement decision differs')
    orders = read_json(directory / 'epoch_orders.json')
    require(orders == result['epoch_orders'] == loaded['current'][1]['order_records'] and len(orders) == STAGE_EPOCHS
            and all(v['batch_sizes'] == [64] * 2077 + [50] and len(v['actual_batch_identity_sha256']) == 2078
                    for v in orders), 'Stage actual effective-batch receipts differ')
    require(result['actual_sample_identity'] == parent['actual_sample_identity'], 'Stage input/sample identity differs')
    parts = ('dev', 'train_core') if result['status'] == 'completed' else ('dev',)
    lateral = None
    for part in parts:
        base = directory / f'eval_{part}'
        receipt, rows = read_json(base / 'evaluation.json'), read_predictions(base / 'predictions.csv')
        require(receipt['status'] == 'completed' and receipt['checkpoint_sha256'] == result['best_checkpoint_sha256']
                and receipt['contract_hash'] == run['contract_hash'] and receipt['cache_identity'] == index['identity']
                and sha256(base / 'predictions.csv') == receipt['predictions_sha256'], 'Stage evaluation receipt differs')
        require(receipt['feature_source'] == ('current_online_encoders' if group == 'S23_TAIL_FT' else 'verified_frozen_cache'),
                'Stage evaluation used wrong encoder feature path')
        expected_map = {(s, t): (time, vx, vy) for s, t, time, vx, vy in expected[part]}
        require(len(rows) == COUNTS[part] and identity_map(rows) == expected_map
                and target_identity(rows) == manifest['full_labels'][part]['target_identity'], 'Stage prediction coverage differs')
        summary = summarize_rows(rows, config)
        check_summary(result['dev' if part == 'dev' else 'train'], summary)
        check_summary(read_json(base / 'metrics.json'), summary)
        if part == 'dev':
            check_summary(best['dev'], summary)
            lateral = lateral_diagnostics(rows, config.get('lateral_rule'))
    require(result['best_dev_predictions_reused'] and result['best_train_full_evaluation_count'] ==
            (1 if result['status'] == 'completed' else 0), 'Stage inference-pass policy differs')
    kept = ('schema', 'status', 'run_id', 'group', 'best_epoch', 'best_dev_epe', 'last_dev_epe', 'best_train_epe',
        'dev', 'train', 'trainable_parameters', 'optimizer_updates', 'completed_epochs', 'gpu', 'elapsed_seconds',
        'training_elapsed_seconds', 'code', 'cache_identity', 'actual_sample_identity', 'initialization',
        'best_checkpoint_sha256', 'last_checkpoint_sha256', 'contract_hash', 'generalization_gap_dev_minus_train_epe',
        'best_train_full_evaluation_count', 'best_dev_predictions_reused', 'parents', 'discarded_attempts')
    row = {k: result.get(k) for k in kept}
    row.update(seed=42, run_dir=str(directory), source_checkpoint_sha256=SOURCE_SHA,
        order_records_sha256=object_hash(orders), order_receipt_path=str(directory / 'epoch_orders.json'),
        order_receipt_file_sha256=sha256(directory / 'epoch_orders.json'), result_path=str(result_path),
        result_sha256=sha256(result_path), config_sha256=sha256(job['config_path']),
        recovery_manifest_sha256=sha256(directory / 'recovery.json'), freeze_manifest_identity=freeze['identity'],
        freeze_manifest_path=str(directory / 'freeze_manifest.json'), lateral_diagnostics=lateral,
        replacement_decision=decision, numerical_or_provenance_anomaly=False)
    return row, trajectory


def collect(plan_path):
    plan = verify_plan(plan_path)
    require(plan['stage'] == 'fusion_confirmation_tail_v1', 'Wrong confirmation stage')
    baseline = verify_plan(plan['baseline_plan'])
    require(baseline['prefix'] == BASELINE_PREFIX and baseline['code']['commit'] == BASELINE_COMMIT,
            'Original full seed42 provenance changed')
    cache = Path(plan['cache_full'])
    index, manifest = checked_index(cache), load_manifest(plan['history_manifest'])
    require(index['scope'] == 'full' and index['schema'] == 'v5_fusion_full_cache_v1'
            and tuple(index['grid']) == (2, 3)
            and checked_index(baseline.get('cache_full', baseline.get('cachefull')))['identity'] == index['identity'],
            'Original complete 2x3 cache changed')
    expected = {part: expected_rows(cache, index, part) for part in COUNTS}
    require({part: len(rows) for part, rows in expected.items()} == COUNTS, 'Wrong full sample coverage')
    old_jobs, new_jobs = jobs_by_key(baseline, True), jobs_by_key(plan)
    runs, stages, histories, progress = {str(seed): {} for seed in SEEDS}, {}, {}, {}
    for (seed, group), job in {**old_jobs, **new_jobs}.items():
        key = f'{group}_s{seed}'
        path = Path(job['run_dir']) / 'status.json'
        state = read_json(path) if path.is_file() else {}
        progress[key] = {k: state.get(k) for k in ('status', 'completed_epochs', 'optimizer_updates')}
        if group in STAGES:
            continue
        row, trajectory, predictions = inspect_full(baseline if seed == 42 else plan, group, job,
                                                    index, expected, manifest)
        if row is None:
            continue
        row['lateral_diagnostics'] = lateral_diagnostics(predictions['dev'], row['dev']['effective_lateral_rule'])
        runs[str(seed)][group], histories[key] = compact_row(row), trajectory
        del predictions
    require(len(runs['42']) == 3, 'Fixed full seed42 baseline must remain complete')
    parent = runs['42']['S23']
    require(parent['best_checkpoint_sha256'] == READOUT_SHA and parent['best_epoch'] == 19
            and parent['best_dev_epe'] == PARENT_EPE, 'Parent S23 best changed')
    for seed in SEEDS:
        values = list(runs[str(seed)].values())
        for row in values[1:]:
            require(row['order_records_sha256'] == values[0]['order_records_sha256'], 'Same-seed actual full batches differ')
            require(row['initialization']['common_modules_sha256'] == values[0]['initialization']['common_modules_sha256'],
                    'Same-seed shared Transformer/head initialization differs')
        for row in values:
            require(row['actual_sample_identity'] == parent['actual_sample_identity'], 'Full input/sample identities differ')
    for group in STAGES:
        row, trajectory = inspect_stage(plan, group, new_jobs[(42, group)], index, expected, manifest, parent)
        if row is not None:
            stages[group], histories[f'{group}_s42'] = compact_row(row), trajectory
    if len(stages) == 2:
        a, b = (stages[g] for g in STAGES)
        require(a['order_records_sha256'] == b['order_records_sha256'], 'Matched stage effective batches differ')
        require(a['actual_sample_identity'] == b['actual_sample_identity'], 'Matched stage samples differ')
        require(a['initialization']['state_sha256'] == b['initialization']['state_sha256']
                and a['initialization']['readout_state_sha256'] == b['initialization']['readout_state_sha256'],
                'Matched stage initial complete/readout weights differ')
    comparisons = {}
    for candidate, baseline_name in ((STAGES[0], 'parent_epoch0'), (STAGES[1], 'parent_epoch0'), (STAGES[1], STAGES[0])):
        if candidate not in stages or (baseline_name != 'parent_epoch0' and baseline_name not in stages):
            continue
        a, b = stages[candidate], parent if baseline_name == 'parent_epoch0' else stages[baseline_name]
        value = compare(candidate, baseline_name, a['dev'], b['dev'], 'B_matched_S23_stage20')
        value['fitting'] = fitting_comparison(a, b)
        comparisons[f'{candidate}_minus_{baseline_name}'] = value
    confirmation = aggregate_confirmation(runs)
    new_values = [runs[str(seed)][group] for seed in (43, 44) for group in GROUPS if group in runs[str(seed)]] + list(stages.values())
    core_complete = len(new_values) == 8
    complete = core_complete and all(v['status'] == 'completed' for v in new_values)
    report = dict(schema='v5_fusion_confirmation_comparison_v1',
        status='completed' if complete else 'dev_ready' if core_complete else 'partial', observed_at=now(),
        experiment_id=plan['prefix'], plan_identity=plan['identity'], training_commit=plan['code']['commit'],
        baseline_plan=dict(path=plan['baseline_plan'], identity=baseline['identity'], code_commit=BASELINE_COMMIT),
        progress=progress, runs=runs, stages=stages, A_confirmation=confirmation,
        B_stage20=dict(parent_epoch0=parent, comparisons=comparisons,
            replacement_decisions={g: stage_decision(v, parent) for g, v in stages.items()},
            stage_budget_epochs=STAGE_EPOCHS, stage_updates=STAGE_UPDATES, seed=42,
            initialization='parent best19; new AdamW and cosine20; not old-last exact resume'),
        dev_core_complete=core_complete, train_evaluation_complete=complete,
        confirmation_dev_complete=confirmation['dev_complete'],
        frozen_dev_ready=STAGES[0] in stages, tail_dev_ready=STAGES[1] in stages,
        cache=dict(path=str(cache), identity=index['identity'], counts=COUNTS, scene_counts=SCENES,
                   source_checkpoint_sha256=SOURCE_SHA),
        independent_lateral_labels=manifest['full_labels'], numerical_or_provenance_anomaly=False,
        limitations=[
            'A contains only original full 20-epoch protocol seeds42/43/44. Parent-initialized stage20 runs are never pooled into A.',
            'Sample SD uses ddof=1 across training seeds. Three seeds are descriptive; repeated dev windows are not new independent labels.',
            'Order policy is seed+zero_based_epoch: same numbered epochs differ across seeds, but offset epochs can reuse permutations.',
            'B is seed42 only; no cross-seed stability claim and no attribution to an individual encoder branch from joint tail unfreezing.',
            'Stage best is selected only among epochs1..20. Parent epoch0 is separately eligible for replacement, with parent retained on ties.',
            'All 23480 dev windows remain the official metric. Focus-scene exclusions are diagnostic only; no holdout/test selection.',
            'Online train loss is not best-checkpoint train EPE; each best full train evaluation is performed once by its trainer.',
            'Dev nonzero lateral has 31 windows/5 scenes: insufficient_data remains, regardless of EPE or repeated seeds. PLANAR_PROXY is not true sideslip.',
            'Near-zero diagnostic is abs(true vy)<0.05 m/s, alarm abs(pred vy)>=0.2 m/s; no label or selection rule changes.',
            'High-speed bins above15m/s retain their small independent sample counts; signed bias closer to zero and accuracy are reported separately.',
            'No additional run, seed, budget, architecture or freeze change is automatically authorized by this report.',
        ])
    return report, histories


def all_rows(report):
    return {**{f'{group}_s{seed}': row for seed, values in report['runs'].items() for group, row in values.items()},
            **{f'{group}_s42': row for group, row in report['stages'].items()}}


def export_tables(output, report, histories):
    core, scenes, speeds, lateral, curves = [], [], [], [], []
    for key, row in all_rows(report).items():
        core.append(dict(model=key, protocol='s23_stage20' if row['group'] in STAGES else 'full_original20',
            group=row['group'], seed=row['seed'], status=row['status'], run_id=row['run_id'],
            best_epoch=row['best_epoch'], best_dev_epe=row['best_dev_epe'], last_dev_epe=row['last_dev_epe'],
            best_train_epe=row.get('best_train_epe'), train_dev_gap=row.get('generalization_gap_dev_minus_train_epe'),
            **{k: row['dev']['overall'][k] for k in METRICS}, trainable_parameters=row['trainable_parameters'],
            updates=row['optimizer_updates'], elapsed_seconds=row.get('elapsed_seconds'),
            training_elapsed_seconds=row.get('training_elapsed_seconds'),
            peak_memory_allocated_bytes=row.get('cuda_peak_allocated_bytes', row.get('peak_memory_allocated_bytes')),
            peak_memory_reserved_bytes=row.get('cuda_peak_reserved_bytes', row.get('peak_memory_reserved_bytes')),
            best_checkpoint=str(Path(row['run_dir']) / 'best.pt'), best_sha256=row['best_checkpoint_sha256'],
            last_checkpoint=str(Path(row['run_dir']) / 'last.pt'), last_sha256=row['last_checkpoint_sha256'],
            result_path=row['result_path'], result_sha256=row['result_sha256']))
        scenes.extend(dict(model=key, scene=scene, **metrics) for scene, metrics in row['dev']['per_scene'].items())
        for label, values in row['dev']['speed_bins'].items():
            speeds.append(dict(model=key, speed_bin=label, count=values['count'],
                **{k: values['metrics'][k] if values['metrics'] else None for k in METRICS}))
        for subset, values in row['lateral_diagnostics']['subsets'].items():
            lateral.append(dict(model=key, subset=subset, **{k: values.get(k) for k in
                ('count', 'scene_count', 'status', 'mae_vy', 'rmse_vy', 'bias_vy', 'zero_vy_mae',
                 'zero_vy_rmse', 'zero_vy_bias', 'false_alarm_count', 'false_alarm_rate')}))
        for epoch in histories[key]:
            curves.append(dict(model=key, epoch=epoch['epoch'], train_online_loss=epoch['training_loss'],
                train_online_loss_x=epoch['train_online_loss_x'], train_online_loss_y=epoch['train_online_loss_y'],
                dev_loss=epoch['dev_loss'], dev_loss_x=epoch['dev_loss_x'], dev_loss_y=epoch['dev_loss_y'],
                dev_epe=epoch['dev']['overall']['epe'], readout_lr=epoch['learning_rates'][0],
                encoder_lr=epoch['learning_rates'][1] if len(epoch['learning_rates']) > 1 else None))
    aggregate, paired_aggregate, consistency = [], [], []
    a = report['A_confirmation']
    for group, values in a['aggregate'].items():
        for metric, stats in values['metrics'].items():
            aggregate.append(dict(group=group, metric=metric, **stats))
    comparisons = dict(report['B_stage20']['comparisons'])
    for name, value in a['paired_comparisons'].items():
        for metric, stats in value['metrics'].items():
            paired_aggregate.append(dict(comparison=name, metric=metric, **stats,
                improved_seeds=value['improved_seeds'], degraded_seeds=value['degraded_seeds'], stability=value['stability']))
        for kind, items in (('scene', value['per_scene']), ('speed_bin', value['speed_bins'])):
            for label, item in items.items():
                for metric, stats in item['metrics'].items():
                    consistency.append(dict(comparison=name, kind=kind, label=label, metric=metric, **stats,
                        independent_window_count=item.get('independent_window_count'),
                        improved_seeds=item['improved_seeds'], degraded_seeds=item['degraded_seeds'], stability=item['stability']))
        comparisons.update({f'{name}_s{seed}': v for seed, v in value['per_seed'].items()})
    pairs, paired_scenes, paired_speeds, sensitivities = [], [], [], []
    for name, value in comparisons.items():
        pairs.append(dict(comparison=name, category=value['category'],
            **{k: value['metrics'][k]['delta'] for k in METRICS},
            relative_epe_percent=value['metrics']['epe']['relative_percent'], improved_scenes=value['improved_scenes'],
            degraded_scenes=value['degraded_scenes'], tied_scenes=value['tied_scenes']))
        for scene, metrics in value['per_scene'].items():
            paired_scenes.append(dict(comparison=name, scene=scene, **{k: metrics[k]['delta'] for k in METRICS},
                weighted_gain_m_s=value['scene_contribution']['weighted_gain_m_s'][scene]))
        for label, values in value['speed_bins'].items():
            paired_speeds.append(dict(comparison=name, speed_bin=label, count=values['count'],
                **{k: values['metrics'][k]['delta'] for k in METRICS}))
        for label, values in value['focus_scene_sensitivity'].items():
            sensitivities.append(dict(comparison=name, diagnostic=label, **values))
    for name, rows in (('core_results', core), ('per_scene_results', scenes), ('speed_bins', speeds),
        ('lateral_diagnostics', lateral), ('learning_curves', curves), ('A_seed_statistics', aggregate),
        ('A_paired_statistics', paired_aggregate), ('A_scene_speed_consistency', consistency),
        ('paired_comparisons', pairs), ('paired_scene_differences', paired_scenes),
        ('paired_speed_differences', paired_speeds), ('focus_scene_sensitivity', sensitivities)):
        atomic_csv(output / f'{name}.csv', rows)


def fmt(value):
    return 'pending' if value is None else f'{value:.6f}'


def markdown(report):
    a, b = report['A_confirmation'], report['B_stage20']
    lines = ['# 完整数据三种子复核与 S23 新阶段20轮', '',
        f"状态：**{report['status']}**；实验 `{report['experiment_id']}`；新代码 `{report['training_commit']}`。", '',
        'A 仅统计原完整数据协议 seed42/43/44；B 是从 seed42 父 best19 开始的新20轮阶段，二者分别报告。', '',
        '## A 原协议三种子复核', '',
        '| 组别 | seed | best轮 | best dev EPE | last dev EPE | best train EPE | dev−train |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for group in GROUPS:
            row = report['runs'][str(seed)].get(group)
            if row:
                lines.append(f"| {group} | {seed} | {row['best_epoch']} | {fmt(row['best_dev_epe'])} | "
                    f"{fmt(row['last_dev_epe'])} | {fmt(row.get('best_train_epe'))} | {fmt(row.get('generalization_gap_dev_minus_train_epe'))} |")
            else:
                lines.append(f'| {group} | {seed} | pending | pending | pending | pending | pending |')
    lines += ['', '| 组别 | 已有seed数 | best dev EPE均值 ± 样本SD |', '|---|---:|---:|']
    for group, value in a['aggregate'].items():
        stat = value['metrics']['best_dev_epe']
        lines.append(f"| {group} | {stat['count']} | {fmt(stat['mean'])} ± {fmt(stat['sample_sd'])} |")
    lines += ['', '样本SD按 ddof=1；尚不足三seed时仅描述进度，不作稳定性结论。', '',
              '| 配对（前者−后者） | ΔEPE均值 ± SD | 改善seed/已有seed | 方向 |', '|---|---:|---:|---|']
    for name, value in a['paired_comparisons'].items():
        stat = value['metrics']['epe']
        lines.append(f"| {name} | {fmt(stat['mean'])} ± {fmt(stat['sample_sd'])} | "
            f"{value['improved_seeds']}/{value['paired_seed_count']} | {value['stability']} |")
        if value['paired_seed_count'] == 3:
            count = sum(s['improved_seeds'] == 3 for s in value['per_scene'].values())
            lines.append(f"\n{name} 在 {count}/{len(value['per_scene'])} 个场景中三个seed均改善。净收益场景贡献及逐seed方向保留于JSON/CSV，不据场景胜数推断均匀收益。\n")
    lines += ['', '## B 两组 S23 新20轮阶段', '',
        f"父模型 `{b['parent_epoch0']['run_id']}` best19，epoch0 dev EPE={fmt(b['parent_epoch0']['best_dev_epe'])}；SHA `{READOUT_SHA}`。",
        f'编码器固定起点 SHA `{SOURCE_SHA}`。两阶段均新建 AdamW 与 cosine20；每组41560次更新。', '',
        '| 组别 | 阶段best轮 | 阶段best dev | 阶段last dev | best train | 是否替换父模型 | epoch0含入后的选中轮 |',
        '|---|---:|---:|---:|---:|---|---:|']
    for group in STAGES:
        row, decision = report['stages'].get(group), b['replacement_decisions'].get(group)
        if row:
            lines.append(f"| {group} | {row['best_epoch']} | {fmt(row['best_dev_epe'])} | {fmt(row['last_dev_epe'])} | "
                f"{fmt(row.get('best_train_epe'))} | {decision['preferred']} | {decision['best_including_parent_epoch']} |")
        else:
            lines.append(f'| {group} | pending | pending | pending | pending | pending | pending |')
    lines += ['', '| 对照（前者−后者） | Δdev EPE | 相对% | Δbest train EPE | 改善/退化场景 |', '|---|---:|---:|---:|---:|']
    for name, value in b['comparisons'].items():
        lines.append(f"| {name} | {fmt(value['metrics']['epe']['delta'])} | {fmt(value['metrics']['epe']['relative_percent'])} | "
            f"{fmt(value['fitting']['train_delta_epe'])} | {value['improved_scenes']}/{value['degraded_scenes']} |")
        contribution = value['scene_contribution']
        share = contribution['top_three_share_of_net_gain']
        if share is not None:
            lines.append(f"\n{name} 的前三贡献场景 {', '.join(contribution['top_three_positive_scenes'])} 占净收益 {100 * share:.1f}%；占比受其他场景退化抵消影响，可能超过100%。\n")
        bins = value['speed_bins']
        if '10..15' in bins:
            speed = bins['10..15']
            lines.append(f"\n{name}：10–15m/s 共 {speed['count']} 个窗口，ΔEPE={fmt(speed['metrics']['epe']['delta'])}，"
                f"Δvx偏差={fmt(speed['metrics']['bias_vx']['delta'])}。偏差变化需与绝对偏差共同查看，正变化本身不等于精度提升。\n")
    lines += ['', '## 两轴、侧向与完整证据', '',
        '| 运行 | vx MAE/RMSE | vy MAE/RMSE | vx/vy偏差 | 非零vy MAE/RMSE | 近零误报率 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in all_rows(report).items():
        m = row['dev']['overall']; sub = row['lateral_diagnostics']['subsets']
        nz, near = sub['nonzero'], sub['near_zero']
        lines.append(f"| {name} | {fmt(m['mae_vx'])}/{fmt(m['rmse_vx'])} | {fmt(m['mae_vy'])}/{fmt(m['rmse_vy'])} | "
            f"{fmt(m['bias_vx'])}/{fmt(m['bias_vy'])} | {fmt(nz.get('mae_vy'))}/{fmt(nz.get('rmse_vy'))} | {fmt(near.get('false_alarm_rate'))} |")
    lines += ['', '侧向正/负方向偏差、恒零vy参照及各子集独立样本数见 lateral_diagnostics.csv。dev非零侧向仍为31窗口/5场景，**insufficient_data**。',
        '曲线数据见 learning_curves.csv；online train loss不是best-checkpoint完整train EPE。逐场景、速度段绝对值与配对差异均已导出；原始预测与checkpoint只引用路径及SHA，不重复推理。', '',
        '## 限制与后续', '', *[f'- {line}' for line in report['limitations']], '',
        '下一步由实际三seed配对方向、速度段取舍和B的验证收益共同审阅；本报告不自动追加训练。B仅seed42，任何稳定性复核须另行授权。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, histories = collect(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    export_tables(args.output, report, histories)
    atomic_text(args.output / 'summary.md', markdown(report))
    write_json(args.output / 'summary.json', report)
    print(f"{report['status']}: {args.output / 'summary.json'}")


if __name__ == '__main__':
    main()
