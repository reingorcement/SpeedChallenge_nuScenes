import pytest

from fusion_confirmation_comparison import (GROUPS, SEEDS, STAGES, aggregate_confirmation,
    jobs_by_key, lateral_diagnostics, stage_decision, fitting_comparison)
from fusion_grid_comparison import summarize_rows


def result(group, seed, error):
    predictions = [dict(scene=scene, token=str(i), time_s=float(i), target_vx=speed,
        target_vy=.01, vx=speed + error, vy=.01) for i, (scene, speed) in enumerate(
            (('scene-0402', 12.), ('scene-1013', 8.), ('other', 3.)))]
    summary = summarize_rows(predictions, dict(loss_scales_m_s=[5., 1.]))
    return dict(group=group, seed=seed, dev=summary, best_dev_epe=summary['overall']['epe'],
        last_dev_epe=error + .1, best_epoch=19, best_train_epe=error / 2,
        generalization_gap_dev_minus_train_epe=error / 2)


def test_paired_seed_sample_sd_and_partial_stability():
    runs = {str(seed): {group: result(group, seed, error) for group, error in
        zip(GROUPS, (2., 1. + (seed - 42) * .25, 2.5))} for seed in SEEDS}
    a = aggregate_confirmation(runs)
    pair = a['paired_comparisons']['S23_minus_J23']
    assert a['dev_complete']
    assert pair['metrics']['epe']['mean'] == pytest.approx(-.75)
    assert pair['metrics']['epe']['sample_sd'] == pytest.approx(.25)
    assert pair['improved_seeds'] == 3 and pair['stability'] == 'improved_all_three'
    assert pair['speed_bins']['10..15']['independent_window_count'] == 1
    assert pair['per_scene']['scene-1013']['improved_seeds'] == 3
    runs['44'] = {}
    partial = aggregate_confirmation(runs)
    assert not partial['dev_complete']
    assert partial['paired_comparisons']['S23_minus_J23']['stability'] == 'pending_all_three_seeds'


def test_stage_best_is_not_overwritten_by_parent_and_ties_keep_parent():
    parent = dict(best_dev_epe=.18, best_train_epe=.1)
    for epe in (.19, .18):
        stage = dict(best_epoch=7, best_dev_epe=epe, best_train_epe=.08)
        decision = stage_decision(stage, parent)
        assert decision['best_including_parent_epoch'] == 0
        assert decision['stage_best_epoch'] == 7 and decision['stage_best_dev_epe'] == epe
        assert decision['preferred'] == 'parent_epoch0'
    stage = dict(best_epoch=20, best_dev_epe=.17, best_train_epe=None)
    assert stage_decision(stage, parent)['best_including_parent_epoch'] == 20
    assert fitting_comparison(stage, parent)['train_evaluation_pending']
    assert fitting_comparison(stage, parent)['train_delta_epe'] is None


def test_lateral_masks_directions_zero_reference_and_independent_adequacy():
    rows = [dict(scene=f'scene-{i % 5}', target_vy=.3 if i % 2 else -.3, vy=0.) for i in range(31)]
    rows += [dict(scene='scene-zero', target_vy=0., vy=.25),
             dict(scene='scene-zero', target_vy=.04, vy=0.),
             dict(scene='scene-zero', target_vy=.05, vy=.3)]
    diagnostic = lateral_diagnostics(rows)
    subsets = diagnostic['subsets']
    assert subsets['nonzero']['count'] == 31 and subsets['nonzero']['scene_count'] == 5
    assert diagnostic['acceptance_status'] == 'insufficient_data'
    assert subsets['positive']['bias_vy'] < 0 < subsets['negative']['bias_vy']
    assert subsets['nonzero']['mae_vy'] == subsets['nonzero']['zero_vy_mae']
    assert subsets['near_zero']['count'] == 2 and subsets['near_zero']['false_alarm_rate'] == .5


def test_exact_eight_jobs_and_revised_stage20_protocol():
    jobs = [dict(kind='training', seed=seed, group=group, run_dir=f'/tmp/{group}_{seed}',
                 protocol='full_confirmation') for seed in (43, 44) for group in GROUPS]
    jobs += [dict(kind='training', seed=42, group=group, run_dir=f'/tmp/{group}',
                  protocol='s23_stage20') for group in STAGES]
    assert len(jobs_by_key(dict(jobs=jobs))) == 8
    jobs[-1]['protocol'] = 's23_stage10'
    with pytest.raises(ValueError, match='revised stage20'):
        jobs_by_key(dict(jobs=jobs))
