import pytest

from fusion_multiseed_comparison import exclude_scene_epe, mean_sd, recommend


def test_sample_sd_and_single_seed_do_not_claim_three_seed_precision():
    values = mean_sd([1., 2., 3.])
    assert values['mean'] == 2. and values['sample_sd'] == 1. and values['ddof'] == 1
    assert mean_sd([1.])['sample_sd'] is None
    assert mean_sd([])['mean'] is None


def test_scene_exclusion_is_window_weighted_and_only_diagnostic():
    summary = dict(overall=dict(count=10, epe=2.), per_scene={
        'scene-1013': dict(count=2, epe=5.), 'scene-0402': dict(count=3, epe=2.),
        'other': dict(count=5, epe=.8)})
    output = exclude_scene_epe(summary, ['scene-1013', 'scene-0402'])
    assert output['epe'] == pytest.approx(.8)
    assert output['count'] == 5 and output['excluded_count'] == 5
    assert output['diagnostic_only'] and output['official_metric_includes_all_scenes']
    assert summary['overall']['epe'] == 2.


def test_partial_report_withholds_candidate_decisions():
    output = recommend(dict(status='partial'))
    assert output['status'] == 'withheld_until_all_three_seeds'
    assert output['next_stage_prompt'] is None and not output['automatic_action']
