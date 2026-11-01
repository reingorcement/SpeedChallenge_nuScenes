"""Immediate full-data dev comparisons A/B/C, followed by best train metrics.

All calculations use saved predictions. The full-dev-selected checkpoint stays
fixed when its predictions are restricted by identity to the original pilot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from collect_results import check_summary, require, target_identity
from fusion_full_history import identity_map, load_manifest
from fusion_grid_comparison import (METRICS, difference, expected_rows, inspect as inspect_pilot,
    paired_comparison, read_history, read_predictions, summarize_rows)
from fusion_multiseed_comparison import atomic_csv, atomic_text, exclude_scene_epe, verify_plan
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc import SOURCE_SHA
from purevision_v5.fusion_grid import EXPECTED_PARAMETERS
from purevision_v5.fusion_grid_training import FULL_CONTRACT_SCOPE, load_recovery, validate_config


GROUPS = ('J23', 'S23', 'F23')
PAIRS = (('S23', 'J23'), ('S23', 'F23'), ('F23', 'J23'))
PILOT_COMMIT = '6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d'
COUNTS = {'train_core': 132978, 'dev': 23480}
SCENES = {'train_core': 582, 'dev': 103}
FOCUS = ('scene-0402', 'scene-1013')


def compare(candidate, baseline, a, b, category):
    value = paired_comparison(candidate, baseline, {candidate: {'dev': a}, baseline: {'dev': b}})
    value['category'] = category
    value['speed_bins'] = {key: dict(count=item['count'], metrics=difference(a['speed_bins'][key]['metrics'], item['metrics']))
                           for key, item in b['speed_bins'].items() if item['count']}
    value['focus_scene_sensitivity'] = {}
    for label, excluded in (('without_scene0402', ('scene-0402',)), ('without_scene1013', ('scene-1013',)),
                            ('without_both', FOCUS)):
        ca, ba = exclude_scene_epe(a, excluded), exclude_scene_epe(b, excluded)
        value['focus_scene_sensitivity'][label] = dict(delta_epe=ca['epe'] - ba['epe'],
            candidate_epe=ca['epe'], baseline_epe=ba['epe'], remaining_windows=ca['count'],
            diagnostic_only=True, official_metric_includes_all_scenes=True)
    return value


def checked_index(path):
    index = read_json(Path(path) / 'index.json')
    require(index.get('identity') == object_hash({k: v for k, v in index.items() if k != 'identity'}),
            'Cache index identity differs')
    require(index['source']['checkpoint_sha256'] == SOURCE_SHA and
            index['frozen_state_before_sha256'] == index['frozen_state_after_sha256'], 'Frozen source differs or mutated')
    return index


def inspect_full(plan, group, job, cache_index, expected, history_manifest):
    directory = Path(job['run_dir'])
    result_path = directory / 'result.json'
    if not result_path.is_file():
        result_path = directory / 'dev_ready.json'
    if not result_path.is_file():
        return None, None, None
    result, run = read_json(result_path), read_json(directory / 'run.json')
    if result.get('status') not in {'dev_ready', 'completed'}:
        return None, None, None
    config, contract = run['contract']['config'], run['contract']
    validate_config(config)
    require(config['group'] == group and config['seed'] == 42 and config['scope'] == 'full', 'Wrong full model/config')
    require(contract['scope'] == FULL_CONTRACT_SCOPE and contract['data_scope'] == 'full' and
            contract['partition_counts'] == COUNTS and contract['scene_counts'] == SCENES, 'Full scope/counts lost')
    require(result['data_scope'] == 'full' and result['train_count'] == 132978 and result['dev_count'] == 23480,
            'Result sample budget differs')
    require(run['contract_hash'] == object_hash(contract) == result['contract_hash'], 'Full contract hash differs')
    require(result['group'] == group and run['run_id'] == result['run_id'] == job['job_id'], 'Full run identity differs')
    require(result['completed_epochs'] == 20 and result['optimizer_updates'] == 41560, 'Incomplete full training budget')
    require(run['code'] == result['code'] and result['code']['commit'] == plan['code']['commit'] ==
            result['code']['verified_remote_commit'] and result['code']['dirty'] is False, 'Unfixed/unpublished training code')
    require(result['cache_identity'] == contract['cache_identity'] == cache_index['identity'], 'Full cache differs')
    require(run['encoder_training'] is False and run['arithmetic'] ==
            dict(dtype='float32', amp=False, tf32=False, gradient_accumulation=1), 'Arithmetic/freeze protocol differs')
    require(result['trainable_parameters'] == EXPECTED_PARAMETERS[group], 'Parameter count differs')
    require(read_json(job['config_path']) == config == read_json(directory / 'config.json'), 'Tracked full configuration differs')
    history = read_history(directory / 'metrics.jsonl')
    require([row['epoch'] for row in history] == list(range(1, 21)), 'Full epoch history incomplete')
    for epoch, row in enumerate(history, 1):
        require(row['processed_batches'] == row['optimizer_updates'] == epoch * 2078 and row['amp_skips'] == 0,
                'FP32 full update count differs')
        require(row['dev']['overall']['count'] == 23480 and row['dev']['scene_count'] == 103, 'Incomplete epoch dev evaluation')
        expected_lr = .001 * (1 + math.cos(math.pi * (epoch - 1) / 20)) / 2
        require(np.allclose(row['learning_rates'], [expected_lr], rtol=1e-12, atol=1e-15), 'Full LR schedule differs')
        for total, x, y in (('training_loss', 'train_online_loss_x', 'train_online_loss_y'),
                            ('dev_loss', 'dev_loss_x', 'dev_loss_y')):
            require(math.isclose(row[total], (row[x] + row[y]) / 2, rel_tol=2e-6, abs_tol=2e-7), 'Axis loss accounting differs')
    recovery, loaded = load_recovery(directory / 'recovery.json', run['contract_hash'])
    require(recovery['completed_epochs'] == 20 and loaded['current'][1]['optimizer_updates'] == 41560,
            'Current atomic recovery is not complete20')
    best = min(history, key=lambda row: row['dev']['overall']['epe'])
    require(result['best_epoch'] == best['epoch'] == loaded['best'][1]['completed_epochs'] and
            result['best_dev_epe'] == best['dev']['overall']['epe'] and
            result['last_dev_epe'] == history[-1]['dev']['overall']['epe'], 'Best/last selection or earlier tie rule differs')
    for name, label in (('best', 'best'), ('last', 'current')):
        require(sha256(directory / f'{name}.pt') == result[f'{name}_checkpoint_sha256'] == recovery[label]['sha256'],
                'Checkpoint export does not match atomic recovery binding')
    orders = read_json(directory / 'epoch_orders.json')
    require(orders == result['epoch_orders'] and len(orders) == 20, 'Full epoch order receipts differ')
    require(all(row['batch_sizes'] == [64] * 2077 + [50] and len(row['actual_batch_identity_sha256']) == 2078
                for row in orders), 'Full actual batch sizes/identities differ')
    predictions = {}
    parts = ('dev', 'train_core') if result['status'] == 'completed' else ('dev',)
    for part in parts:
        destination = directory / f'eval_{part}'
        receipt = read_json(destination / 'evaluation.json')
        rows = read_predictions(destination / 'predictions.csv')
        require(receipt['status'] == 'completed' and receipt['checkpoint_sha256'] == result['best_checkpoint_sha256'] and
                receipt['contract_hash'] == run['contract_hash'] and receipt['cache_identity'] == cache_index['identity'],
                'Full evaluation receipt differs')
        require(sha256(destination / 'predictions.csv') == receipt['predictions_sha256'], 'Prediction hash differs')
        expected_rows_map = {(scene, token): (time, vx, vy) for scene, token, time, vx, vy in expected[part]}
        require(identity_map(rows) == expected_rows_map and len(rows) == COUNTS[part],
                'Full prediction identity/labels/times/coverage differs')
        require(target_identity(rows) == history_manifest['full_labels'][part]['target_identity'],
                'Full predictions differ from original manifest labels')
        summary = summarize_rows(rows, config)
        check_summary(result['dev' if part == 'dev' else 'train'], summary)
        check_summary(read_json(destination / 'metrics.json'), summary)
        if part == 'dev':
            check_summary(best['dev'], summary)
        predictions[part] = rows
    require(result['best_dev_predictions_reused'] and result['best_train_full_evaluation_count'] ==
            (1 if result['status'] == 'completed' else 0), 'Full inference-pass policy differs')
    # Do not replicate 124680 batch hashes in the public summary. Their complete
    # immutable file and result hashes retain the evidence.
    kept = ('schema', 'status', 'run_id', 'group', 'best_epoch', 'best_dev_epe', 'last_dev_epe', 'best_train_epe',
            'dev', 'train', 'trainable_parameters', 'optimizer_updates', 'completed_epochs', 'gpu', 'elapsed_seconds',
            'training_elapsed_seconds', 'code', 'cache_identity', 'actual_sample_identity', 'initialization',
            'best_checkpoint_sha256', 'last_checkpoint_sha256', 'contract_hash', 'generalization_gap_dev_minus_train_epe',
            'best_train_full_evaluation_count', 'best_dev_predictions_reused', 'discarded_attempts')
    row = {key: result.get(key) for key in kept}
    row.update(seed=42, run_dir=str(directory), source_checkpoint_sha256=SOURCE_SHA,
        order_records_sha256=object_hash(orders), order_receipt_path=str(directory / 'epoch_orders.json'),
        order_receipt_file_sha256=sha256(directory / 'epoch_orders.json'), result_path=str(result_path),
        result_sha256=sha256(result_path), config_sha256=sha256(job['config_path']),
        recovery_manifest_sha256=sha256(directory / 'recovery.json'), numerical_or_provenance_anomaly=False,
        curve_diagnostic=dict(best_epoch=best['epoch'], last_minus_best_dev_epe=result['last_dev_epe'] - result['best_dev_epe'],
            final_five_dev_epe=[value['dev']['overall']['epe'] for value in history[-5:]],
            final_five_train_online_loss=[value['training_loss'] for value in history[-5:]],
            final_five_dev_loss=[value['dev_loss'] for value in history[-5:]],
            terminal_learning_rate=history[-1]['next_learning_rates'][0],
            automatic_budget_extension=False, continuation_warning='cosine ends at zero; do not extend unchanged schedule'))
    return row, history, predictions


def pilot_comparisons(plan, rows, predictions):
    pilot_plan = verify_plan(plan['baseline_plan'])
    require(pilot_plan['code']['commit'] == PILOT_COMMIT, 'Pilot seed42 implementation reference changed')
    index = checked_index(pilot_plan['cache23'])
    expected = {part: expected_rows(pilot_plan['cache23'], index, part) for part in ('dev', 'train_core')}
    jobs = {job.get('group', job.get('variant', job['name'])): job for job in pilot_plan['jobs'] if job['kind'] == 'training'}
    result = {}
    for group in rows:
        pilot, _, pilot_predictions = inspect_pilot(pilot_plan, group, jobs[group], index, expected)
        lookup = {(row['scene'], row['token']): row for row in predictions[group]['dev']}
        selected = []
        for pilot_row in pilot_predictions['dev']:
            key = (pilot_row['scene'], pilot_row['token'])
            require(key in lookup, f'Pilot dev identity absent from full dev: {key}')
            actual = lookup[key]
            require(identity_map([actual]) == identity_map([pilot_row]), 'Pilot/full common labels/timestamps differ')
            selected.append(actual)
        require(len(selected) == 1826 and len({row['scene'] for row in selected}) == 8, 'Pilot restriction coverage differs')
        restricted = summarize_rows(selected, {'loss_scales_m_s': [5., 1.], 'lateral_rule': rows[group]['dev']['effective_lateral_rule']})
        comparison = compare(f'full_{group}_on_pilot', f'pilot_{group}', restricted, pilot['dev'], 'B_training_coverage_change')
        result[group] = dict(full_restricted_metrics=restricted, pilot_metrics=pilot['dev'], comparison=comparison,
            full_best_epoch=rows[group]['best_epoch'], full_best_checkpoint_sha256=rows[group]['best_checkpoint_sha256'],
            full_checkpoint_selection='unchanged best selected on all 23480 dev windows; no pilot reselection',
            pilot_best_epoch=pilot['best_epoch'], pilot_checkpoint_sha256=pilot['best_checkpoint_sha256'],
            pilot_run_id=pilot['run_id'], pilot_training_commit=PILOT_COMMIT, sample_identity=target_identity(selected),
            scope_caveat='training windows, updates and dev selection coverage all expanded; not isolated data-volume causality')
    return result


def collect(plan_path):
    plan = verify_plan(plan_path)
    require(plan['stage'] == 'frozen_fusion_full_v1', 'Wrong full pipeline stage')
    history = load_manifest(plan['history_manifest'])
    cache = Path(plan.get('cache_full', plan.get('cachefull', plan.get('cache', ''))))
    index = checked_index(cache)
    require(index['schema'] == 'v5_fusion_full_cache_v1' and index['scope'] == 'full' and tuple(index['grid']) == (2, 3),
            'Formal full comparison requires the complete shared 2x3 cache')
    expected = {part: expected_rows(cache, index, part) for part in ('dev', 'train_core')}
    require({part: len(value) for part, value in expected.items()} == COUNTS, 'Full cache annotation counts differ')
    jobs = {job.get('group', job.get('variant', job['name'])): job for job in plan['jobs'] if job['kind'] == 'training'}
    require(set(jobs) == set(GROUPS), 'Only full J23/S23/F23 are allowed')
    rows, histories, predictions, progress = {}, {}, {}, {}
    for group in GROUPS:
        status_path = Path(jobs[group]['run_dir']) / 'status.json'
        state = read_json(status_path) if status_path.is_file() else {}
        progress[group] = {key: state.get(key) for key in ('status', 'completed_epochs', 'optimizer_updates')}
        row, trajectory, prediction = inspect_full(plan, group, jobs[group], index, expected, history)
        if row is not None:
            rows[group], histories[group], predictions[group] = row, trajectory, prediction
    values = list(rows.values())
    for row in values:
        require(row['order_records_sha256'] == values[0]['order_records_sha256'], 'Actual full batches differ across groups')
        require(row['actual_sample_identity'] == values[0]['actual_sample_identity'], 'Full sample/input identities differ')
        require(row['initialization']['common_modules_sha256'] == values[0]['initialization']['common_modules_sha256'],
                'Shared initial Transformer/head differ')
    a = {f'{candidate}_minus_{baseline}': compare(candidate, baseline, rows[candidate]['dev'], rows[baseline]['dev'],
         'A_full_matched_23480dev') for candidate, baseline in PAIRS if candidate in rows and baseline in rows}
    b = pilot_comparisons(plan, rows, predictions)
    historical_predictions = {name: read_predictions(entry['predictions']) for name, entry in history['references'].items()}
    c = {}
    for group, row in rows.items():
        for name, reference in history['references'].items():
            require(identity_map(predictions[group]['dev']) == identity_map(historical_predictions[name]),
                    'Historical/full dev labels/times/coverage differ')
            c[f'{group}_minus_{name}'] = compare(group, name, row['dev'], reference['metrics'], 'C_cross_budget_historical_accuracy')
    dev_complete = len(rows) == 3
    complete = dev_complete and all(row['status'] == 'completed' for row in rows.values())
    status = 'completed' if complete else 'dev_ready' if dev_complete else 'partial'
    report = dict(schema='v5_fusion_full_comparison_v1', status=status, observed_at=now(),
        experiment_id=plan['prefix'], plan_identity=plan['identity'], training_commit=plan['code']['commit'],
        dev_core_complete=dev_complete, train_evaluation_complete=complete, seed=42, progress=progress, rows=rows,
        A_full_matched=a, B_pilot_identity_matched=b, C_fixed_historical=c,
        historical_manifest=dict(path=plan['history_manifest'], sha256=sha256(plan['history_manifest']), identity=history['identity']),
        historical_references=history['references'], independent_lateral_labels=history['full_labels'],
        cache=dict(path=str(cache), identity=index['identity'], index_sha256=sha256(cache / 'index.json'),
            source_checkpoint_sha256=SOURCE_SHA, scope='full', grid=[2, 3], counts=COUNTS, scene_counts=SCENES),
        numerical_or_provenance_anomaly=False,
        interpretation=dict(single_full_seed_only=True, stable_superiority_claim=False, follow_up_authorized=False,
            S23_beats_J23_this_full_seed=a.get('S23_minus_J23', {}).get('metrics', {}).get('epe', {}).get('delta', 0) < 0 if dev_complete else None,
            S23_beats_F23_this_full_seed=a.get('S23_minus_F23', {}).get('metrics', {}).get('epe', {}).get('delta', 0) < 0 if dev_complete else None,
            S23_beats_historical={name: c[f'S23_minus_{name}']['metrics']['epe']['delta'] < 0 for name in history['references']}
                if 'S23' in rows else {},
            budget_extension='No automatic extension. If suggested, use a new matched budget/schedule for all three; current cosine ends at zero.'),
        limitations=[
            'Only this full-data seed42 is compared; pilot three-seed evidence cannot establish stable full-data superiority.',
            'A is matched new-reader training; C is historical accuracy context with different architectures, encoder training, precision and stage budgets.',
            'B restricts the already full-dev-selected best predictions by (scene,token) to the original 1826 pilot windows; no pilot-specific checkpoint selection.',
            'B jointly changes training coverage, update count and dev selection coverage, so it is not isolated data-volume causality.',
            'Official metrics include all scenes. scene-0402/1013 exclusion is diagnostic only and never changes selection or sampling.',
            'Online training loss is not best-checkpoint train EPE. Missing train EPE stays pending until its single full inference completes.',
            'PLANAR_PROXY and dev-selected frozen source; no held-out/test conclusion. Lateral adequacy follows the original independent-window rule.',
        ])
    return report, histories


def export_tables(output, report, histories):
    core, scenes, bins, curves, paired_scenes = [], [], [], [], []
    metric_sources = {group: row['dev'] for group, row in report['rows'].items()}
    metric_sources.update({name: row['metrics'] for name, row in report['historical_references'].items()})
    for name, metric in metric_sources.items():
        for scene, values in metric['per_scene'].items():
            scenes.append(dict(model=name, scene=scene, focus_scene=scene in FOCUS, **values))
        for label, values in metric['speed_bins'].items():
            bins.append(dict(model=name, speed_bin=label, count=values['count'],
                **{key: values['metrics'][key] if values['metrics'] else None for key in METRICS}))
    for group, row in report['rows'].items():
        core.append(dict(group=group, status=row['status'], best_epoch=row['best_epoch'],
            best_dev_epe=row['best_dev_epe'], last_dev_epe=row['last_dev_epe'], best_train_epe=row['best_train_epe'],
            **{key: row['dev']['overall'][key] for key in METRICS if key != 'epe'},
            train_dev_gap=row['generalization_gap_dev_minus_train_epe'], parameters=row['trainable_parameters'],
            updates=row['optimizer_updates'], training_seconds=row['training_elapsed_seconds'],
            gpu=json.dumps(row['gpu']), best_sha256=row['best_checkpoint_sha256']))
        for value in histories[group]:
            curves.append(dict(group=group, epoch=value['epoch'], train_online_loss=value['training_loss'],
                train_online_loss_x=value['train_online_loss_x'], train_online_loss_y=value['train_online_loss_y'],
                dev_loss=value['dev_loss'], dev_loss_x=value['dev_loss_x'], dev_loss_y=value['dev_loss_y'],
                dev_epe=value['dev']['overall']['epe'], learning_rate=value['learning_rates'][0]))
    pairs = {**report['A_full_matched'], **report['C_fixed_historical'],
             **{f'full_minus_pilot_{group}': value['comparison'] for group, value in report['B_pilot_identity_matched'].items()}}
    comparison_rows, sensitivities, paired_bins = [], [], []
    for name, value in pairs.items():
        comparison_rows.append(dict(comparison=name, category=value['category'],
            **{key: metric['delta'] for key, metric in value['metrics'].items()},
            relative_epe_percent=value['metrics']['epe']['relative_percent'], improved_scenes=value['improved_scenes'],
            degraded_scenes=value['degraded_scenes'], tied_scenes=value['tied_scenes']))
        for scene, metrics in value['per_scene'].items():
            paired_scenes.append(dict(comparison=name, category=value['category'], scene=scene,
                **{key: metric['delta'] for key, metric in metrics.items()},
                weighted_gain_m_s=value['scene_contribution']['weighted_gain_m_s'][scene]))
        for label, item in value['focus_scene_sensitivity'].items():
            sensitivities.append(dict(comparison=name, category=value['category'], diagnostic=label, **item))
        for label, item in value['speed_bins'].items():
            paired_bins.append(dict(comparison=name, category=value['category'], speed_bin=label, count=item['count'],
                                    **{key: metric['delta'] for key, metric in item['metrics'].items()}))
    for filename, values in (('core_results.csv', core), ('per_scene_results.csv', scenes), ('speed_bins.csv', bins),
        ('learning_curves.csv', curves), ('comparisons_ABC.csv', comparison_rows), ('paired_scene_differences.csv', paired_scenes),
        ('paired_speed_differences.csv', paired_bins), ('focus_scene_sensitivity.csv', sensitivities)):
        atomic_csv(output / filename, values)


def markdown(report):
    lines = ['# 完整数据 J23/S23/F23 seed42 结果', '',
        f"状态：**{report['status']}**；实验 `{report['experiment_id']}`；代码 `{report['training_commit']}`。", '',
        '结论仅适用于本次完整数据seed42；完整train评估未完成时先交付dev核心结果。', '',
        '| 组别 | Best轮 | Best dev EPE | Last dev EPE | Best train EPE | vx MAE/RMSE | vy MAE/RMSE | vx/vy偏差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, row in report['rows'].items():
        value = row['dev']['overall']
        train = f"{row['best_train_epe']:.6f}" if row['best_train_epe'] is not None else '评估待完成'
        lines.append(f"| {group} | {row['best_epoch']} | {row['best_dev_epe']:.6f} | {row['last_dev_epe']:.6f} | {train} | "
            f"{value['mae_vx']:.6f}/{value['rmse_vx']:.6f} | {value['mae_vy']:.6f}/{value['rmse_vy']:.6f} | "
            f"{value['bias_vx']:.6f}/{value['bias_vy']:.6f} |")
    for title, comparisons in (('A：相同完整23480窗口的匹配比较', report['A_full_matched']),
        ('B：按身份选取原pilot1826窗口，完整best保持不变',
         {group: value['comparison'] for group, value in report['B_pilot_identity_matched'].items()}),
        ('C：固定历史参照，跨训练预算的实际精度比较', report['C_fixed_historical'])):
        lines += ['', title, '', '| 比较（候选−基线） | ΔEPE | 相对变化 | 改善/退化/持平场景 |', '|---|---:|---:|---:|']
        for name, value in comparisons.items():
            metric = value['metrics']['epe']
            lines.append(f"| {name} | {metric['delta']:.6f} | {metric['relative_percent']:.2f}% | "
                f"{value['improved_scenes']}/{value['degraded_scenes']}/{value['tied_scenes']} |")
    lines += ['', '固定历史checkpoint（阶段预算单独列出，不等于全部从零训练）：', '',
        '| 历史参照 | 阶段预算 | 选中轮 | 完整dev EPE |', '|---|---:|---:|---:|']
    for name, row in report['historical_references'].items():
        lines.append(f"| {name} | {row['stage_budget_epochs']} | {row['selected_epoch_one_based']} | {row['metrics']['overall']['epe']:.6f} |")
    lines += ['', '重点场景和高速段（正式指标始终保留全部场景）：', '',
        '| 组别 | scene-0402 EPE | scene-1013 EPE | 10–15m/s EPE | 10–15m/s vx偏差 |', '|---|---:|---:|---:|---:|']
    for group, row in report['rows'].items():
        metric = row['dev']
        speed = metric['speed_bins']['10..15']['metrics']
        lines.append(f"| {group} | {metric['per_scene']['scene-0402']['epe']:.6f} | "
            f"{metric['per_scene']['scene-1013']['epe']:.6f} | {speed['epe']:.6f} | {speed['bias_vx']:.6f} |")
    lines += ['', '| 分区 | abs(vy)≥0.2独立窗口 | 场景 | 侧向充分性 |', '|---|---:|---:|---|']
    for part, values in report['independent_lateral_labels'].items():
        value = values['lateral']
        lines.append(f"| {part} | {value['count']} | {value['scene_count']} | {value['status']} |")
    if report['dev_core_complete']:
        outcome = report['interpretation']
        verb = lambda condition: '优于' if condition else '未优于'
        lines += ['', f"本次完整数据seed42下，S23 **{verb(outcome['S23_beats_J23_this_full_seed'])} J23**，"
            f"**{verb(outcome['S23_beats_F23_this_full_seed'])} F23**。",
            '相对固定历史模型：' + '；'.join(f"{name}：{verb(value)}" for name, value in outcome['S23_beats_historical'].items()) + '。']
        if outcome['S23_beats_J23_this_full_seed'] and (not outcome['S23_beats_F23_this_full_seed'] or
                not all(outcome['S23_beats_historical'].values())):
            lines.append('融合组织相对J23有改善，但尚未形成相对F23或全部固定旧模型的相应实际精度升级。')
    lines += ['', '逐轮两轴损失、全部103场景、速度分段、配对差异、0402/1013敏感性及来源见CSV/JSON。',
        '曲线只支持提出下一步建议，不自动延长预算；cosine末端学习率为0，不能原样续跑视为有效扩训。',
        '本轮未授权额外seed、模型、网格、训练轮数或编码器微调。', '', '解释边界：', '']
    lines.extend('- ' + value for value in report['limitations'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report, histories = collect(args.plan)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', report)
    atomic_text(output / 'summary.md', markdown(report))
    export_tables(output, report, histories)
    print(json.dumps(dict(status=report['status'], dev_core_complete=report['dev_core_complete'],
                          summary=str(output / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
