"""Read-only three-seed candidate review, reusing saved best predictions.

Seed42 retains its original implementation provenance. Seeds43/44 must use
the review plan's common fixed implementation. Partial reports show progress
and descriptive observations only; candidate conclusions require all 12 runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from collect_results import require
from fusion_grid_comparison import METRICS, expected_rows, inspect, paired_comparison
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc import SOURCE_SHA


GROUPS = ('J23', 'S23', 'F23', 'S46')
SEEDS = (42, 43, 44)
PAIRS = (('S23', 'J23'), ('S23', 'F23'), ('S46', 'S23'))
BASELINE_COMMIT = '6d8d6feb2c47713d6fdacf98cf1b4ec3fe2d909d'
BASELINE_PREFIX = '20261102_0052_fusion_grid_s42'
FOCUS_SCENES = ('scene-1013', 'scene-0402')


def mean_sd(values):
    values = [float(value) for value in values]
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None, ddof=1,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def verify_plan(path):
    plan = read_json(path)
    require(plan.get('identity') == object_hash({k: v for k, v in plan.items() if k != 'identity'}),
            f'Plan identity differs: {path}')
    return plan


def plan_commit(plan):
    commit = plan.get('code', {}).get('commit', plan.get('code_commit'))
    require(isinstance(commit, str) and len(commit) == 40, 'Missing full fixed implementation commit')
    if 'code_commit' in plan:
        require(plan['code_commit'] == commit, 'Conflicting plan code commit fields')
    return commit


def training_jobs(plan, baseline=False):
    jobs = {}
    for original in plan['jobs']:
        if original.get('kind', 'training') != 'training':
            continue
        group = original.get('group', original.get('variant', original.get('name')))
        if baseline and group not in GROUPS:
            continue
        seed = 42 if baseline else original.get('seed')
        require(group in GROUPS and type(seed) is int and seed in (SEEDS if baseline else (43, 44)),
                'Unexpected review group/seed')
        job = dict(original, group=group, seed=seed)
        job.setdefault('job_id', Path(job['run_dir']).name)
        require((seed, group) not in jobs, 'Duplicate group/seed job')
        jobs[(seed, group)] = job
    expected = {(seed, group) for seed in ((42,) if baseline else (43, 44)) for group in GROUPS}
    require(set(jobs) == expected, 'Review must contain exactly four original and eight new group/seed jobs')
    return jobs


def exclude_scene_epe(summary, excluded):
    removed_count = sum(summary['per_scene'][scene]['count'] for scene in excluded if scene in summary['per_scene'])
    remaining = summary['overall']['count'] - removed_count
    require(remaining > 0, 'Diagnostic exclusion removed all validation samples')
    removed_error = sum(summary['per_scene'][scene]['count'] * summary['per_scene'][scene]['epe']
                        for scene in excluded if scene in summary['per_scene'])
    return dict(count=remaining, excluded_count=removed_count,
                epe=(summary['overall']['epe'] * summary['overall']['count'] - removed_error) / remaining,
                diagnostic_only=True, official_metric_includes_all_scenes=True)


def aggregate_runs(runs):
    output = {}
    for group in GROUPS:
        available = [runs[str(seed)][group] for seed in SEEDS if group in runs[str(seed)]]
        metrics = {key: mean_sd([row['dev']['overall'][key] for row in available]) for key in METRICS}
        metrics.update({key: mean_sd([row[key] for row in available]) for key in
            ('best_dev_epe', 'last_dev_epe', 'best_train_epe', 'best_epoch', 'generalization_gap_dev_minus_train_epe')})
        output[group] = dict(completed_seeds=[row['seed'] for row in available], metrics=metrics,
                             conclusion_ready=len(available) == 3)
    return output


def aggregate_pairs(runs):
    output = {}
    for candidate, baseline in PAIRS:
        values = {}
        for seed in SEEDS:
            rows = runs[str(seed)]
            if candidate in rows and baseline in rows:
                comparison = paired_comparison(candidate, baseline, rows)
                comparison['exclusion_sensitivity'] = {}
                for label, excluded in (('without_scene1013', ('scene-1013',)),
                                        ('without_scene0402', ('scene-0402',)),
                                        ('without_both_focus_scenes', FOCUS_SCENES)):
                    a = exclude_scene_epe(rows[candidate]['dev'], excluded)
                    b = exclude_scene_epe(rows[baseline]['dev'], excluded)
                    comparison['exclusion_sensitivity'][label] = dict(candidate=a, baseline=b,
                        delta_epe=a['epe'] - b['epe'], excluded_scenes=list(excluded), diagnostic_only=True)
                values[str(seed)] = comparison
        deltas = [value['metrics']['epe']['delta'] for value in values.values()]
        count = len(deltas)
        improvements, degradations, ties = (sum(value < 0 for value in deltas), sum(value > 0 for value in deltas),
                                             sum(value == 0 for value in deltas))
        stability = ('pending_all_three_seeds' if count < 3 else 'improved_all_three' if improvements == 3
                     else 'degraded_all_three' if degradations == 3 else 'tied_all_three' if ties == 3 else 'mixed_across_seeds')
        scene_names = list(next(iter(values.values()))['per_scene']) if values else []
        scenes = {scene: dict(metrics={metric: mean_sd([value['per_scene'][scene][metric]['delta']
                    for value in values.values()]) for metric in METRICS},
                improved_seed_count=sum(value['per_scene'][scene]['epe']['delta'] < 0 for value in values.values()),
                degraded_seed_count=sum(value['per_scene'][scene]['epe']['delta'] > 0 for value in values.values()),
                weighted_gain_m_s=mean_sd([value['scene_contribution']['weighted_gain_m_s'][scene]
                                          for value in values.values()])) for scene in scene_names}
        output[f'{candidate}_minus_{baseline}'] = dict(candidate=candidate, baseline=baseline,
            delta_convention='candidate minus baseline; negative EPE/MAE/RMSE improves', per_seed=values,
            metrics={metric: mean_sd([value['metrics'][metric]['delta'] for value in values.values()]) for metric in METRICS},
            relative_epe_percent=mean_sd([value['metrics']['epe']['relative_percent'] for value in values.values()]),
            improved_seeds=improvements, degraded_seeds=degradations, tied_seeds=ties,
            stability=stability, paired_seed_count=count, per_scene=scenes,
            exclusion_sensitivity={label: mean_sd([value['exclusion_sensitivity'][label]['delta_epe']
                for value in values.values()]) for label in
                ('without_scene1013', 'without_scene0402', 'without_both_focus_scenes')})
    return output


def speed_aggregates(runs):
    output = {}
    for group in GROUPS:
        available = [runs[str(seed)][group] for seed in SEEDS if group in runs[str(seed)]]
        output[group] = {}
        if not available:
            continue
        for label, first in available[0]['dev']['speed_bins'].items():
            require(all(row['dev']['speed_bins'][label]['count'] == first['count'] for row in available),
                    'Independent speed-bin window counts changed across seeds')
            output[group][label] = dict(independent_window_count=first['count'],
                lower_m_s=first['lower_m_s'], upper_m_s=first['upper_m_s'],
                metrics={metric: mean_sd([row['dev']['speed_bins'][label]['metrics'][metric] for row in available])
                         for metric in METRICS} if first['count'] else {})
    return output


def recommend(report):
    if report['status'] != 'completed':
        return dict(status='withheld_until_all_three_seeds', next_stage_prompt=None, automatic_action=False)
    aggregate, pairs = report['aggregate'], report['paired_comparisons']
    mean = lambda group: aggregate[group]['metrics']['best_dev_epe']['mean']
    winner = min(GROUPS, key=mean)
    dual_winner = min(('J23', 'S23', 'S46'), key=mean)
    preferred = min(('S23', 'S46'), key=mean)
    selected = ['J23', preferred]
    # The single path remains a matched diagnostic when it is competitive.
    diagnostic = mean('F23') <= mean('S23')
    if diagnostic:
        selected.append('F23')
    needed_grids = ['2x3'] + (['4x6'] if 'S46' in selected else [])
    warnings = []
    for name, comparison in pairs.items():
        if comparison['stability'] == 'mixed_across_seeds':
            warnings.append(f'{name} has mixed seed outcomes')
        official = comparison['metrics']['epe']['mean']
        for label, value in comparison['exclusion_sensitivity'].items():
            if official * value['mean'] < 0:
                warnings.append(f'{name}: {label} reverses the mean EPE difference (diagnostic only)')
    group_text = '、'.join(selected)
    grid_text = '、'.join(needed_grids)
    prompt = (f'请执行经本轮三seed结果审阅后的完整数据对照，仅保留 {group_text}。冻结来源为 '
        '20261101_rgb_path_s42_R03_APPEAR_FT_s42_18ep/best.pt，'
        f'SHA256={SOURCE_SHA}。先核对该来源、'
        f'原始划分及 PLANAR_PROXY 定义；完整 train/dev 为 582/103 场景、132978/23480 窗口，只准备保留组需要的 {grid_text} '
        '网格缓存，各保留网格从原汇总前特征独立池化，禁止跨网格插值或复制生成。保持同一已训练外观+光流编码器冻结、'
        'eval/no_grad、BN不变；不使用holdout/test、不重算RAFT。首次完整数据对照统一seed42，每组共同20个epoch、'
        'batch64/drop_last=False（每轮2078批，末批50，完整41560次更新）、FP32/AMP关闭/TF32关闭，'
        '保留AdamW lr1e-3、weight_decay1e-4、betas(0.9,0.999)、eps1e-8、clip2、20轮cosine eta_min0、'
        '完整[B,2] SmoothL1(pred/[5,1],target/[5,1],beta1,mean)，不调整各组预算。'
        '初始化沿用历史build_triplet并显式复制共同Transformer/head及对应分支，参数独立存储，同seed实际样本批次一致。'
        '先完成必要实现、短预检及固定代码/来源/缓存清单GitHub同步，再按空闲GPU排队；采用完整epoch原子恢复清单。'
        '每轮完整dev验证，结束时仅完整评估best的train一次并复用best dev预测，按全部窗口dev EPE选best和早轮次平局规则。'
        '报告配对误差、两轴损失、场景/速度分段、scene-1013/0402敏感性、侧向独立样本充分性、耗时及来源。'
        '不自动增加seed、组别、缓存网格、损失、归一化或编码器微调；此提示词须用户另行授权后才执行。')
    return dict(status='review_recommendation_only', mean_epe_winner=winner, dual_mean_epe_winner=dual_winner,
        preferred_dual_candidate=preferred, proposed_full_data_groups=selected, required_cache_grids=needed_grids,
        flow_only_diagnostic_retained=diagnostic, risks=warnings, automatic_action=False,
        S46_retention=dict(proposed='S46' in selected,
            reason='S46 has the lower three-seed mean dev EPE among S23/S46.' if preferred == 'S46'
                else 'S23 has no higher three-seed mean dev EPE; keep the proposed full-data comparison at 2x3.',
            paired_vs_S23_stability=pairs['S46_minus_S23']['stability'],
            paired_vs_S23_delta= pairs['S46_minus_S23']['metrics']['epe'],
            requires_human_review_of_stability_and_scene_sensitivity=True),
        automatic_ranking_is_provisional=True,
        rationale='Select the strongest mean-EPE dual candidate against J23; retain F23 only as a matched 2x3 diagnostic when competitive.',
        next_stage_prompt=prompt)


def collect(plan_path):
    plan = verify_plan(plan_path)
    require(plan.get('stage') == 'frozen_fusion_seed_review', 'Wrong review stage')
    baseline = verify_plan(plan['baseline_plan'])
    require(baseline['prefix'] == BASELINE_PREFIX and plan_commit(baseline) == BASELINE_COMMIT,
            'Original seed42 plan/code provenance differs')
    commit = plan_commit(plan)
    jobs = {**training_jobs(baseline, baseline=True), **training_jobs(plan)}
    caches, expected = {}, {}
    for suffix in ('23', '46'):
        root = Path(baseline[f'cache{suffix}']).resolve()
        if f'cache{suffix}' in plan:
            require(Path(plan[f'cache{suffix}']).resolve() == root, 'Review must reuse the original pilot caches')
        index = read_json(root / 'index.json')
        require(index.get('identity') == object_hash({k: v for k, v in index.items() if k != 'identity'}) and
                index['scope'] == 'pilot' and index['source']['checkpoint_sha256'] == SOURCE_SHA,
                'Frozen cache identity/source differs')
        require(index['frozen_state_before_sha256'] == index['frozen_state_after_sha256'], 'Frozen encoder mutated')
        if suffix == '46':
            require(index['code']['commit'] == BASELINE_COMMIT, '4x6 cache must retain original extraction commit')
        caches[suffix] = (root, index)
        expected[suffix] = {part: expected_rows(root, index, part) for part in ('dev', 'train_core')}
    require(expected['23'] == expected['46'], 'Ordered cache samples/labels differ between grids')
    runs, progress = {str(seed): {} for seed in SEEDS}, {}
    for (seed, group), job in jobs.items():
        directory = Path(job['run_dir'])
        result_path, run_path = directory / 'result.json', directory / 'run.json'
        state = read_json(directory / 'status.json') if (directory / 'status.json').is_file() else {}
        progress[f'{group}_s{seed}'] = dict(status=state.get('status', 'not_started'), run_dir=str(directory),
                                         completed_epochs=state.get('completed_epochs', 0))
        if not (result_path.is_file() and run_path.is_file()):
            continue
        result, run = read_json(result_path), read_json(run_path)
        if result.get('status') != 'completed' or run.get('status') != 'completed':
            continue
        require(run['contract']['config']['seed'] == seed and run['initialization']['seed'] == seed,
                'Recorded initialization/config seed differs from job')
        if 'config_path' not in job:
            job['config_path'] = run['command']['config']
        suffix = '46' if group.endswith('46') else '23'
        checked, _, _ = inspect(dict(code=dict(commit=BASELINE_COMMIT if seed == 42 else commit)),
                                group, job, caches[suffix][1], expected[suffix])
        checked.update(seed=seed, evidence_scope='original_seed42' if seed == 42 else 'candidate_seed_review')
        runs[str(seed)][group] = checked
        progress[f'{group}_s{seed}'].update(status='completed', completed_epochs=20, optimizer_updates=2300)
    all_rows = [row for seed_runs in runs.values() for row in seed_runs.values()]
    reference = all_rows[0] if all_rows else None
    for seed, seed_runs in runs.items():
        rows = list(seed_runs.values())
        for row in rows:
            require(row['actual_sample_identity'] == reference['actual_sample_identity'],
                    'Underlying sample/input frame/time/label/dt identity differs across runs')
            require(row['epoch_orders'] == rows[0]['epoch_orders'], 'Actual batches differ within one seed')
            require(row['initialization']['common_modules_sha256'] == rows[0]['initialization']['common_modules_sha256'],
                    'Common initialization differs within one seed')
    count = len(all_rows)
    lateral = reference['dev']['nonzero_vy'] if reference else None
    if lateral:
        require(all((row['dev']['nonzero_vy']['count'], row['dev']['nonzero_vy']['scene_count']) ==
                    (lateral['count'], lateral['scene_count']) for row in all_rows), 'Lateral independent sample set differs')
    report = dict(schema='v5_fusion_multiseed_review_v1', status='completed' if count == 12 else 'partial',
        observed_at=now(), experiment_id=plan['prefix'], plan_identity=plan['identity'],
        baseline_plan_identity=baseline['identity'], baseline_commit=BASELINE_COMMIT, review_commit=commit,
        completed_runs=count, expected_runs=12, new_completed_runs=sum(len(runs[str(seed)]) for seed in (43, 44)),
        expected_new_runs=8, progress=progress, runs=runs, aggregate=aggregate_runs(runs),
        paired_comparisons=aggregate_pairs(runs), speed_bins=speed_aggregates(runs),
        lateral_independent_evidence=dict(count=lateral['count'], scene_count=lateral['scene_count'],
            status=lateral['status'], threshold_m_s=lateral['threshold_m_s'],
            rule=reference['dev']['effective_lateral_rule'], repeated_seeds_do_not_increase_independent_windows=True) if lateral else None,
        caches={suffix: dict(path=str(root), identity=index['identity'], index_sha256=sha256(root / 'index.json'),
            source_checkpoint_sha256=index['source']['checkpoint_sha256'], extraction_commit=index['code']['commit'])
            for suffix, (root, index) in caches.items()},
        numerical_or_provenance_anomaly=False,
        validation=dict(full_epoch_and_2300_updates=True, best_earlier_tie_rule=True, atomic_current_best_pair=True,
            saved_predictions_and_metrics=True, shared_initialization_checked_within_seed_only=True,
            actual_batches_checked_within_seed_only=True, no_model_inference=True),
        limitations=[
            'Three seeds reuse the same 32/8 scenes and 7324/1826 windows; seed variation does not create new independent data.',
            'Mean and sample SD use ddof=1 across the three run metrics, not pooled predictions or per-window pseudo-replication.',
            'All-scene window-weighted dev EPE remains the formal selection metric. Excluding scene-1013/0402 is diagnostic only.',
            'Online train loss is not best-checkpoint train EPE. Train/dev gap uses the selected checkpoint.',
            'Cross-grid effects include integer position encoding; S46 effects cannot be attributed to one factor alone.',
            'Historical epoch order remains seed + zero-based epoch; different seeds may share shifted epoch permutations.',
            'F23 is a matched 2x3 diagnostic. Repeated seeds do not establish lateral adequacy or authorize a change of research direction.',
            'Source encoders were trained on original full train and selected on original full dev; this pilot is not held-out evidence.',
        ])
    report['recommendation'] = recommend(report)
    return report


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(text)
    temporary.replace(path)


def atomic_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_tables(output, report):
    per_seed, curves, absolute_scenes = [], [], []
    for seed, groups in report['runs'].items():
        for group, row in groups.items():
            per_seed.append(dict(group=group, seed=int(seed), best_epoch=row['best_epoch'],
                best_dev_epe=row['best_dev_epe'], last_dev_epe=row['last_dev_epe'], last_epoch=row['completed_epochs'], best_train_epe=row['best_train_epe'],
                **{metric: row['dev']['overall'][metric] for metric in METRICS if metric != 'epe'},
                train_dev_gap=row['generalization_gap_dev_minus_train_epe'], updates=row['optimizer_updates'],
                parameters=row['trainable_parameters'], training_seconds=row['training_elapsed_seconds'],
                code_commit=row['code']['commit'], run_id=row['run_id']))
            for scene, values in row['dev']['per_scene'].items():
                absolute_scenes.append(dict(group=group, seed=int(seed), scene=scene,
                    focus_scene=scene in FOCUS_SCENES, count=values['count'],
                    **{metric: values[metric] for metric in METRICS}))
            for line in (Path(row['run_dir']) / 'metrics.jsonl').read_text().splitlines():
                value = json.loads(line)
                curves.append(dict(group=group, seed=int(seed), epoch=value['epoch'],
                    train_online_loss=value['training_loss'], train_online_loss_x=value['train_online_loss_x'],
                    train_online_loss_y=value['train_online_loss_y'], dev_loss=value['dev_loss'],
                    dev_loss_x=value['dev_loss_x'], dev_loss_y=value['dev_loss_y'], dev_epe=value['dev']['overall']['epe']))
    atomic_csv(output / 'per_seed_results.csv', per_seed)
    atomic_csv(output / 'per_scene_results.csv', absolute_scenes)
    atomic_csv(output / 'learning_curves.csv', curves)
    aggregate = [dict(group=group, metric=metric, **values) for group, item in report['aggregate'].items()
                 for metric, values in item['metrics'].items()]
    atomic_csv(output / 'aggregate_mean_sd.csv', aggregate)
    paired, scenes, sensitivity, speed = [], [], [], []
    for name, item in report['paired_comparisons'].items():
        for seed, value in item['per_seed'].items():
            paired.append(dict(comparison=name, seed=int(seed),
                **{key: metric['delta'] for key, metric in value['metrics'].items()},
                relative_epe_percent=value['metrics']['epe']['relative_percent'],
                improved_scenes=value['improved_scenes'], degraded_scenes=value['degraded_scenes'], tied_scenes=value['tied_scenes']))
            for scene, metrics in value['per_scene'].items():
                scenes.append(dict(comparison=name, seed=int(seed), scene=scene,
                    focus_scene=scene in FOCUS_SCENES, **{key: metric['delta'] for key, metric in metrics.items()},
                    weighted_gain_m_s=value['scene_contribution']['weighted_gain_m_s'][scene]))
            for label, values in value['exclusion_sensitivity'].items():
                sensitivity.append(dict(comparison=name, seed=int(seed), diagnostic=label,
                    allscene_delta_epe=value['metrics']['epe']['delta'], diagnostic_delta_epe=values['delta_epe'],
                    remaining_windows=values['candidate']['count'], official_metric_includes_all_scenes=True))
    for group, bins in report['speed_bins'].items():
        for label, item in bins.items():
            for metric, values in item['metrics'].items():
                speed.append(dict(group=group, speed_bin=label, independent_window_count=item['independent_window_count'],
                                  metric=metric, **values))
    atomic_csv(output / 'paired_seed_differences.csv', paired)
    atomic_csv(output / 'paired_scene_differences.csv', scenes)
    atomic_csv(output / 'focus_scene_sensitivity.csv', sensitivity)
    atomic_csv(output / 'speed_bin_mean_sd.csv', speed)


def formatted(values):
    if values['mean'] is None:
        return '—'
    return f"{values['mean']:.6f} ± {values['sample_sd']:.6f}" if values['sample_sd'] is not None else f"{values['mean']:.6f} (n=1)"


def markdown(report):
    lines = ['# 候选三seed复核', '',
        f"状态：**{report['status']}**；已核验 {report['completed_runs']}/12 份结果（本轮新增 {report['new_completed_runs']}/8）。", '',
        f"原 seed42 固定提交 `{report['baseline_commit']}`；seed43/44 固定提交 `{report['review_commit']}`。", '',
        '以下均值±样本标准差按seed计算（ddof=1），正式指标包含全部8个dev场景。', '',
        '| 组别 | 已完成seed | Best dev EPE | Last dev EPE | Best train EPE | dev−train EPE |',
        '|---|---|---:|---:|---:|---:|']
    for group, item in report['aggregate'].items():
        metric = item['metrics']
        lines.append(f"| {group} | {','.join(map(str, item['completed_seeds']))} | {formatted(metric['best_dev_epe'])} | "
            f"{formatted(metric['last_dev_epe'])} | {formatted(metric['best_train_epe'])} | "
            f"{formatted(metric['generalization_gap_dev_minus_train_epe'])} |")
    lines += ['', '| 组别 | Seed | Best轮 | Best dev EPE | Last dev EPE（轮20） | Best train EPE |',
              '|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for group in GROUPS:
            row = report['runs'][str(seed)].get(group)
            if row:
                lines.append(f"| {group} | {seed} | {row['best_epoch']} | {row['best_dev_epe']:.6f} | "
                             f"{row['last_dev_epe']:.6f} | {row['best_train_epe']:.6f} |")
            else:
                lines.append(f'| {group} | {seed} | 待完成 | — | — | — |')
    lines += ['', '| 配对（候选−基线） | ΔEPE均值±SD | 改善/退化/持平seed | 稳定性 |', '|---|---:|---:|---|']
    for name, item in report['paired_comparisons'].items():
        lines.append(f"| {name} | {formatted(item['metrics']['epe'])} | "
            f"{item['improved_seeds']}/{item['degraded_seeds']}/{item['tied_seeds']} | {item['stability']} |")
    lines += ['', '| 组别 | vx MAE | vx RMSE | vx有符号偏差 | vy MAE | vy RMSE |', '|---|---:|---:|---:|---:|---:|']
    for group, item in report['aggregate'].items():
        lines.append('| ' + group + ' | ' + ' | '.join(formatted(item['metrics'][key]) for key in
                     ('mae_vx', 'rmse_vx', 'bias_vx', 'mae_vy', 'rmse_vy')) + ' |')
    lines += ['', 'scene-1013/0402敏感性仅诊断，不改变正式全场景结果：', '',
        '| 配对 | 全场景ΔEPE | 去1013诊断 | 去0402诊断 | 去两者诊断 |', '|---|---:|---:|---:|---:|']
    for name, item in report['paired_comparisons'].items():
        values = item['exclusion_sensitivity']
        lines.append(f"| {name} | {formatted(item['metrics']['epe'])} | " + ' | '.join(formatted(values[label]) for label in
                     ('without_scene1013', 'without_scene0402', 'without_both_focus_scenes')) + ' |')
    lateral = report['lateral_independent_evidence']
    if lateral:
        lines += ['', f"侧向 abs(vy)≥{lateral['threshold_m_s']} m/s：独立窗口 **{lateral['count']}**、场景 **{lateral['scene_count']}**，"
            f"状态 **{lateral['status']}**。重复seed不增加独立样本数，不能将该子集数量乘3。"]
    recommendation = report['recommendation']
    if report['status'] != 'completed':
        lines += ['', '**12份结果未齐全；当前仅报告已完成记录，不给候选胜负或下一阶段执行结论。**']
    else:
        lines += ['', f"三seed均值最低：**{recommendation['mean_epe_winner']}**；双路最低：**{recommendation['dual_mean_epe_winner']}**。",
            f"建议下一阶段保留：**{'、'.join(recommendation['proposed_full_data_groups'])}**；需要缓存网格："
            f"**{'、'.join(recommendation['required_cache_grids'])}**。F23若保留仅作为2x3诊断，主线仍检验光流＋RGB互补。",
            '该建议按实际三seed均值与配对稳定性给出，需结合场景敏感性、侧向充分性和训练/验证差距审阅。']
        lines += [''] + ['- ' + risk for risk in recommendation['risks']] if recommendation['risks'] else []
        lines += ['', '下一阶段可审核执行提示词（本次未执行）：', '', '> ' + recommendation['next_stage_prompt']]
    lines += ['', '逐seed最佳轮次、best/last、两轴误差、曲线、全部场景配对及速度分段见CSV；来源与恢复一致性校验见JSON。',
              '已完成记录未检出来源、完整预算、最佳选择、恢复绑定、预测指标、同seed初始化或实际批次异常。', '', '解释边界：', '']
    lines.extend('- ' + item for item in report['limitations'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = collect(args.plan)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Core report first; presentation tables cannot delay the completion event.
    write_json(output / 'summary.json', report)
    atomic_text(output / 'summary.md', markdown(report))
    export_tables(output, report)
    print(json.dumps(dict(status=report['status'], completed_runs=report['completed_runs'],
                          summary=str(output / 'summary.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
