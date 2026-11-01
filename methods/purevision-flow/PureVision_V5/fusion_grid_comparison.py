"""Immediate read-only paired comparisons; never starts a follow-up experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from collect_results import check_summary, require
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import lateral_acceptance
from purevision_v5.fusion_abc import SOURCE_SHA
from purevision_v5.fusion_grid import EXPECTED_PARAMETERS, GRID_VARIANTS
from purevision_v5.fusion_grid_training import load_recovery, summarize_prediction, validate_config


GROUPS = tuple(GRID_VARIANTS)
DUAL = ('J23', 'S23', 'J46', 'S46')
PAIRS = (('S23', 'J23'), ('S46', 'J46'), ('J46', 'J23'), ('S46', 'S23'),
         ('S46', 'J23'), ('F23', 'J23'), ('F23', 'S23'), ('R23', 'J23'), ('R23', 'S23'))
METRICS = ('epe', 'mae_vx', 'mae_vy', 'rmse_vx', 'rmse_vy', 'bias_vx', 'bias_vy')
LIMITS = [
    '单个 seed42、32/8 场景 pilot，只用于筛选；best 使用 dev 整体 EPE 选择，未使用 holdout/test。',
    '编码器源模型曾使用原完整 train 训练并由原完整 dev 选型；本轮只训练新读出模块。',
    '网格对照包含更细池化区域及配套整数位置编码的整体变化，不是纯 token 数量效应。',
    'S46 的收益不能全部归因于分离 token 或网格中的单一因素；两路仅粗粒度对应。',
    '单路只与相同 2x3 双路直接比较，作为诊断，研究主线仍为光流与 RGB 互补。',
    '在线 train loss 不是 best checkpoint 的 train EPE；后者仅在训练结束时完整计算一次。',
    '不同 GPU 的耗时不是同硬件性能对照；PLANAR_PROXY 不等于真实三维车体速度。',
]


def read_history(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def difference(candidate, baseline):
    return {key: dict(delta=candidate[key] - baseline[key],
                     relative_percent=100 * (candidate[key] - baseline[key]) / baseline[key]
                     if baseline[key] else None) for key in METRICS}


def read_predictions(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def prediction_identity(rows):
    return [(row['scene'], row['token'], float(row['time_s']),
             float(row['target_vx']), float(row['target_vy'])) for row in rows]


def summarize_rows(rows, config):
    prediction = torch.tensor([[float(row['vx']), float(row['vy'])] for row in rows], dtype=torch.float32)
    target = torch.tensor([[float(row['target_vx']), float(row['target_vy'])] for row in rows], dtype=torch.float32)
    return summarize_prediction(prediction, dict(targets=target, metadata=rows), config)


def expected_rows(cache, index, partition):
    entry = index['partitions'][partition]
    path = (Path(cache) / entry['annotations']).resolve()
    require(path.is_relative_to(Path(cache).resolve()) and sha256(path) == entry['annotations_sha256'],
            'Cache annotation path/hash differs')
    packet = torch.load(path, map_location='cpu', weights_only=True)
    require(object_hash(packet['metadata']) == entry['sample_manifest_sha256'], 'Cache metadata differs')
    return [(row['scene'], row['token'], float(row['time_s']), float(target[0]), float(target[1]))
            for row, target in zip(packet['metadata'], packet['targets'])]


def inspect(plan, group, job, index, expected):
    directory = Path(job['run_dir'])
    result, run = read_json(directory / 'result.json'), read_json(directory / 'run.json')
    require(result['status'] == run['status'] == 'completed', f'{group} is incomplete')
    config = run['contract']['config']
    validate_config(config)
    require(config['group'] == group == result['group'], 'Wrong group')
    require(run['contract_hash'] == object_hash(run['contract']) == result['contract_hash'], 'Invalid contract')
    require(run['contract']['scope'] == 'frozen_fusion_grid_pilot', 'Preflight cannot be collected as formal')
    require(run['train_count'] == 7324 and run['dev_count'] == 1826, 'Wrong pilot sample count')
    require(run['encoder_training'] is False and run['arithmetic'] ==
            dict(dtype='float32', amp=False, tf32=False, gradient_accumulation=1), 'Wrong arithmetic/freeze policy')
    require(result['trainable_parameters'] == EXPECTED_PARAMETERS[group], 'Parameter count differs')
    require(result['completed_epochs'] == 20 and result['optimizer_updates'] == 2300, 'Incomplete training budget')
    require(run['code'] == result['code'] and run['code']['commit'] == plan['code']['commit'] ==
            run['code']['verified_remote_commit'] and run['code']['dirty'] is False, 'Code was not fixed/clean/pushed')
    require(result['run_id'] == run['run_id'] == job['job_id'], 'Run ID differs')
    require(result['cache_identity'] == index['identity'] == run['contract']['cache_identity'], 'Cache identity differs')
    require(read_json(directory / 'config.json') == config == read_json(job['config_path']), 'Config differs')
    history = read_history(directory / 'metrics.jsonl')
    require([row['epoch'] for row in history] == list(range(1, 21)), 'Wrong epoch history')
    for epoch, row in enumerate(history, 1):
        require(row['optimizer_updates'] == row['processed_batches'] == epoch * 115 and row['amp_skips'] == 0,
                'Wrong effective update count')
        expected_lr = .001 * (1 + math.cos(math.pi * (epoch - 1) / 20)) / 2
        require(np.allclose(row['learning_rates'], [expected_lr], rtol=1e-12, atol=1e-15), 'LR schedule differs')
        require(math.isclose(row['training_loss'], (row['train_online_loss_x'] + row['train_online_loss_y']) / 2,
                             rel_tol=2e-6, abs_tol=2e-7), 'Online axis loss accounting differs')
        require(math.isclose(row['dev_loss'], (row['dev_loss_x'] + row['dev_loss_y']) / 2,
                             rel_tol=2e-6, abs_tol=2e-7), 'Dev axis loss accounting differs')
    recovery, checkpoints = load_recovery(directory / 'recovery.json', run['contract_hash'])
    require(recovery['completed_epochs'] == 20 and checkpoints['current'][1]['optimizer_updates'] == 2300,
            'Current recovery point is incomplete')
    best_row = min(history, key=lambda row: row['dev']['overall']['epe'])
    require(result['best_epoch'] == best_row['epoch'] == checkpoints['best'][1]['completed_epochs'] and
            result['best_dev_epe'] == best_row['dev']['overall']['epe'], 'Best minimum/earlier-tie rule differs')
    for name in ('best', 'last'):
        label = 'current' if name == 'last' else 'best'
        require(sha256(directory / f'{name}.pt') == recovery[label]['sha256'] == result[f'{name}_checkpoint_sha256'],
                f'{name} export differs from atomic recovery manifest')
    orders = read_json(directory / 'epoch_orders.json')
    require(orders == result['epoch_orders'] and len(orders) == 20, 'Order receipts differ')
    require(all(len(row['actual_batch_identity_sha256']) == 115 and row['batch_sizes'] == [64] * 114 + [28]
                for row in orders), 'Actual batch identities or sizes differ')
    predictions = {}
    for part, key in (('dev', 'dev'), ('train_core', 'train')):
        destination = directory / f'eval_{part}'
        receipt = read_json(destination / 'evaluation.json')
        rows = read_predictions(destination / 'predictions.csv')
        require(receipt['status'] == 'completed' and receipt['checkpoint_sha256'] == result['best_checkpoint_sha256']
                and receipt['cache_identity'] == index['identity'] and receipt['contract_hash'] == run['contract_hash'],
                'Evaluation provenance differs')
        require(receipt['predictions_sha256'] == sha256(destination / 'predictions.csv'), 'Prediction checksum differs')
        require(prediction_identity(rows) == expected[part], 'Actual ordered prediction samples/targets differ')
        recomputed = summarize_rows(rows, config)
        check_summary(result[key], recomputed)
        check_summary(read_json(destination / 'metrics.json'), recomputed)
        if part == 'dev':
            check_summary(best_row['dev'], recomputed)
        predictions[part] = rows
    require(result['best_train_full_evaluation_count'] == 1 and result['best_dev_predictions_reused'],
            'Final evaluation pass policy differs')
    result.update(source_checkpoint_sha256=index['source']['checkpoint_sha256'],
                  config_sha256=sha256(job['config_path']), run_dir=str(directory),
                  numerical_or_provenance_anomaly=False)
    return result, history, predictions


def paired_comparison(candidate, baseline, rows):
    a, b = rows[candidate]['dev'], rows[baseline]['dev']
    scenes = {scene: difference(a['per_scene'][scene], b['per_scene'][scene]) for scene in b['per_scene']}
    total = b['overall']['count']
    contributions = {scene: -values['epe']['delta'] * b['per_scene'][scene]['count'] / total
                     for scene, values in scenes.items()}
    total_gain = b['overall']['epe'] - a['overall']['epe']
    positive_gain = sum(max(0., value) for value in contributions.values())
    largest = max(contributions, key=contributions.get)
    focus = 'scene-1013'
    focus_count = b['per_scene'].get(focus, {}).get('count', 0)
    excluded_delta = None
    if focus_count and focus_count < total:
        excluded_delta = ((a['overall']['epe'] * total - a['per_scene'][focus]['epe'] * focus_count) -
                          (b['overall']['epe'] * total - b['per_scene'][focus]['epe'] * focus_count)) / (total - focus_count)
    return dict(candidate=candidate, baseline=baseline,
        delta_convention='candidate minus baseline; negative EPE/MAE/RMSE improves; baseline denominator',
        metrics=difference(a['overall'], b['overall']), per_scene=scenes,
        improved_scenes=sum(row['epe']['delta'] < 0 for row in scenes.values()),
        degraded_scenes=sum(row['epe']['delta'] > 0 for row in scenes.values()),
        tied_scenes=sum(row['epe']['delta'] == 0 for row in scenes.values()),
        lateral=lateral_acceptance(a, b, a['effective_lateral_rule']),
        scene_contribution=dict(weighted_gain_m_s=contributions, total_gain_m_s=total_gain,
            largest_positive_scene=largest, largest_share_of_positive_gain=contributions[largest] / positive_gain
                if positive_gain > 0 else None,
            scene1013_count=focus_count, scene1013_weighted_gain_m_s=contributions.get(focus),
            scene1013_share_of_net_gain=contributions.get(focus, 0.) / total_gain if total_gain else None,
            excluding_scene1013_diagnostic_delta_epe=excluded_delta,
            official_metric_includes_all_scenes=True))


def collect(plan_path, dual_only=False):
    plan = read_json(plan_path)
    require(plan.get('identity') == object_hash({key: value for key, value in plan.items() if key != 'identity'}),
            'Plan identity differs')
    require(plan['stage'] == 'frozen_fusion_grid_pilot', 'Wrong experiment stage')
    selected = DUAL if dual_only else GROUPS
    jobs = {job.get('group', job.get('variant', job['name'])): job
            for job in plan['jobs'] if job['kind'] == 'training'}
    require(set(jobs) == set(GROUPS), 'Plan must contain exactly the six fixed training groups')
    rows, histories, predictions, caches, expected_by_grid = {}, {}, {}, {}, {}
    for suffix in ('23', '46'):
        root = Path(plan[f'cache{suffix}'])
        index = read_json(root / 'index.json')
        require(index['scope'] == 'pilot' and index['identity'] == object_hash({k: v for k, v in index.items() if k != 'identity'}),
                'Invalid complete pilot cache')
        require(index['source']['checkpoint_sha256'] == SOURCE_SHA and
                index['frozen_state_before_sha256'] == index['frozen_state_after_sha256'], 'Frozen source differs or mutated')
        expected_by_grid[suffix] = {part: expected_rows(root, index, part) for part in ('dev', 'train_core')}
        caches[suffix] = dict(path=str(root), identity=index['identity'], index_sha256=sha256(root / 'index.json'),
            source=index['source'], sample_counts={p: len(v) for p, v in expected_by_grid[suffix].items()})
        caches[suffix]['index'] = index
    require(expected_by_grid['23'] == expected_by_grid['46'], 'Actual cross-grid ordered samples/labels differ')
    for group in selected:
        suffix = '46' if group.endswith('46') else '23'
        rows[group], histories[group], predictions[group] = inspect(
            plan, group, jobs[group], caches[suffix]['index'], expected_by_grid[suffix])
    for group in selected:
        require(rows[group]['epoch_orders'] == rows['J23']['epoch_orders'], 'Actual training batches differ across groups')
        require(rows[group]['actual_sample_identity'] == rows['J23']['actual_sample_identity'],
                'Ordered metadata/input frames/times/labels/dt differ across groups')
        require(rows[group]['initialization']['common_modules_sha256'] ==
                rows['J23']['initialization']['common_modules_sha256'], 'Shared initial Transformer/head differ')
    comparisons = {f'{a}_minus_{b}': paired_comparison(a, b, rows) for a, b in PAIRS if a in rows and b in rows}
    winner = min(selected, key=lambda name: rows[name]['best_dev_epe'])
    dual_winner = min(DUAL, key=lambda name: rows[name]['best_dev_epe'])
    gain = rows['J23']['best_dev_epe'] - rows[winner]['best_dev_epe']
    candidates = sorted((name for name in DUAL if name != 'J23'), key=lambda name: rows[name]['best_dev_epe'])[:2]
    report = dict(schema='v5_fusion_grid_comparison_v1', status='completed',
        scope='four_dual_preliminary' if dual_only else 'six_group_first_round', observed_at=now(),
        experiment_id=plan['prefix'], plan_identity=plan['identity'], training_commit=plan['code']['commit'],
        rows=rows, comparisons=comparisons, winner=winner, dual_winner=dual_winner,
        winner_vs_J23=dict(absolute_improvement_m_s=gain, relative_improvement_percent=100 * gain / rows['J23']['best_dev_epe']),
        review_candidates=candidates, numerical_or_provenance_anomaly=False, limits=LIMITS,
        interpretation=dict(fusion_organization=['S23_minus_J23', 'S46_minus_J46'],
            grid_with_position_encoding=['J46_minus_J23', 'S46_minus_S23'],
            interaction_delta_epe=(rows['S46']['best_dev_epe'] - rows['J46']['best_dev_epe']) -
                                  (rows['S23']['best_dev_epe'] - rows['J23']['best_dev_epe'])),
        caches={key: {k: v for k, v in value.items() if k != 'index'} for key, value in caches.items()},
        follow_up_authorized=False,
        follow_up_prompt=('请仅复核 J23 与候选 ' + '、'.join(candidates) + '，补 seed43、44，每组沿用本轮已冻结来源、'
            '32/8 场景 pilot、相应直接池化缓存、20 epoch/batch64/FP32/TF32关闭、AdamW及loss协议。'
            '先核实本轮证据与候选，按每个seed的历史build_triplet来源建立共同初始化，复制Transformer/head及对应分支，'
            '保持实际样本批次一致与独立存储。用同一固定已推送代码，在空闲GPU队列执行并及时汇总三个seed的'
            '配对dev EPE、两轴误差、场景差异、侧向充分性和训练/验证差距。只执行以上候选复核，'
            '不提取完整集缓存、不启动582/103场景训练、不微调编码器、不增加组别或改损失；异常时停止受影响组并报告。'))
    return report, histories, predictions


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(destination, report, histories, predictions, prefix=''):
    rows = report['rows']
    core = []
    for group, result in rows.items():
        core.append(dict(group=group, best_epoch=result['best_epoch'], best_dev_epe=result['best_dev_epe'],
            last_dev_epe=result['last_dev_epe'], best_train_epe=result['best_train_epe'],
            **{key: result['dev']['overall'][key] for key in METRICS if key != 'epe'},
            generalization_gap=result['generalization_gap_dev_minus_train_epe'], parameters=result['trainable_parameters'],
            updates=result['optimizer_updates'], training_seconds=result['training_elapsed_seconds'],
            gpu=json.dumps(result['gpu'], ensure_ascii=False), source_sha256=result['source_checkpoint_sha256']))
    write_csv(destination / f'{prefix}core_results.csv', core)
    curves = []
    for group, history in histories.items():
        for row in history:
            curves.append(dict(group=group, epoch=row['epoch'], training_loss=row['training_loss'],
                train_online_loss_x=row['train_online_loss_x'], train_online_loss_y=row['train_online_loss_y'],
                dev_loss=row['dev_loss'], dev_loss_x=row['dev_loss_x'], dev_loss_y=row['dev_loss_y'],
                dev_epe=row['dev']['overall']['epe'], learning_rate=row['learning_rates'][0], updates=row['optimizer_updates']))
    write_csv(destination / f'{prefix}learning_curves.csv', curves)
    scenes = []
    for name, comparison in report['comparisons'].items():
        for scene, values in comparison['per_scene'].items():
            scenes.append(dict(comparison=name, scene=scene,
                **{key + '_delta': value['delta'] for key, value in values.items()},
                weighted_gain_m_s=comparison['scene_contribution']['weighted_gain_m_s'][scene]))
    write_csv(destination / f'{prefix}paired_dev_scenes.csv', scenes)
    bins = []
    for group, result in rows.items():
        for interval, value in result['dev']['speed_bins'].items():
            bins.append(dict(group=group, speed_bin=interval, count=value['count'],
                **{key: value['metrics'][key] if value['metrics'] else None for key in METRICS}))
    write_csv(destination / f'{prefix}dev_speed_bins.csv', bins)
    paired = []
    for i, reference in enumerate(predictions['J23']['dev']):
        row = {key: reference[key] for key in ('scene', 'token', 'time_s', 'target_vx', 'target_vy')}
        for group in rows:
            prediction = predictions[group]['dev'][i]
            row.update({f'{group}_{axis}': float(prediction[axis]) for axis in ('vx', 'vy')})
            row[f'{group}_epe'] = math.hypot(float(prediction['vx']) - float(prediction['target_vx']),
                                             float(prediction['vy']) - float(prediction['target_vy']))
        paired.append(row)
    write_csv(destination / f'{prefix}paired_dev_predictions.csv', paired)


def markdown(report):
    gain = report['winner_vs_J23']
    lines = ['# 六组融合与网格首轮结果' if report['scope'] == 'six_group_first_round' else '# 四个双路组即时初步结果', '',
        f"实验 `{report['experiment_id']}`；固定实现 `{report['training_commit']}`；单种子 seed42。", '',
        f"当前最低 dev EPE：**{report['winner']}**，相对 J23 改善 **{gain['absolute_improvement_m_s']:.6f} m/s"
        f"（{gain['relative_improvement_percent']:.2f}%）**；双路最低为 **{report['dual_winner']}**。", '',
        '| 组别 | Best轮 | Best dev EPE | Last dev EPE | Best train EPE | vx MAE/RMSE | vy MAE/RMSE | 参数 | 更新 | 训练秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in report['rows'].items():
        overall = result['dev']['overall']
        lines.append(f"| {name} | {result['best_epoch']} | {result['best_dev_epe']:.6f} | {result['last_dev_epe']:.6f} | "
            f"{result['best_train_epe']:.6f} | {overall['mae_vx']:.6f}/{overall['rmse_vx']:.6f} | "
            f"{overall['mae_vy']:.6f}/{overall['rmse_vy']:.6f} | {result['trainable_parameters']} | "
            f"{result['optimizer_updates']} | {result['training_elapsed_seconds']:.1f} |")
    lines += ['', '下表差值为候选减基线，负 EPE 表示改善。', '',
        '| 配对 | Δdev EPE | 相对变化 | 场景改善/退化/持平 | 最大正收益场景及贡献比例 | 排除scene-1013诊断ΔEPE |',
        '|---|---:|---:|---:|---|---:|']
    for name, comparison in report['comparisons'].items():
        delta, scene = comparison['metrics']['epe'], comparison['scene_contribution']
        share = scene['largest_share_of_positive_gain']
        exclusion = scene['excluding_scene1013_diagnostic_delta_epe']
        share_text = f'{100 * share:.1f}%' if share is not None else '无正收益'
        exclusion_text = f'{exclusion:.6f}' if exclusion is not None else '不适用'
        lines.append(f"| {name} | {delta['delta']:.6f} | {delta['relative_percent']:.2f}% | "
            f"{comparison['improved_scenes']}/{comparison['degraded_scenes']}/{comparison['tied_scenes']} | "
            f"{scene['largest_positive_scene']} / {share_text} | {exclusion_text} |")
    interaction = report['interpretation']['interaction_delta_epe']
    lines += ['', '融合组织证据：S23−J23、S46−J46；细网格及位置编码证据：J46−J23、S46−S23。',
        f'描述性交互差值 (S46−J46)−(S23−J23) = {interaction:.6f} m/s，单种子不作显著性结论。', '',
        '| 组别 | dev−train EPE | vx有符号偏差 | abs(vy)≥0.2窗口/场景 | 侧向充分性 |', '|---|---:|---:|---:|---|']
    for name, result in report['rows'].items():
        lateral = result['dev']['nonzero_vy']
        lines.append(f"| {name} | {result['generalization_gap_dev_minus_train_epe']:.6f} | "
            f"{result['dev']['overall']['bias_vx']:.6f} | {lateral['count']}/{lateral['scene_count']} | {lateral['status']} |")
    lines += ['', '所有正式指标保留全部场景。scene-1013剔除值只用于诊断集中程度；完整场景贡献、速度分段和两轴损失见CSV。',
        '实现、来源、完整epoch恢复、实际样本顺序、共同初始化及数值校验均通过；未发现影响本轮筛选的异常。', '',
        '建议与 J23 补 seed43、44 的候选：**' + '、'.join(report['review_candidates']) + '**，按当前双路dev EPE排序；'
        '仍需结合上表场景集中、侧向充分性和泛化差距审阅。', '', '后续可审核执行提示词（本次未执行）：', '',
        '> ' + report['follow_up_prompt'], '', '解释边界：', '']
    lines.extend('- ' + item for item in report['limits'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dual-only', action='store_true')
    args = parser.parse_args()
    report, histories, predictions = collect(args.plan, args.dual_only)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = 'dual_' if args.dual_only else ''
    # Core JSON and text are committed before optional presentation exports.
    write_json(destination / f'{prefix}summary.json', report)
    (destination / f'{prefix}summary.md').write_text(markdown(report))
    export_tables(destination, report, histories, predictions, prefix)
    print(json.dumps(dict(status='completed', scope=report['scope'], winner=report['winner'],
                          summary=str(destination / f'{prefix}summary.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
