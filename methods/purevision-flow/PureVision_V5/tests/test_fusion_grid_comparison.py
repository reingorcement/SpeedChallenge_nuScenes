import pytest

from fusion_grid_comparison import paired_comparison, summarize_rows


def test_scene_contribution_uses_window_weights_and_lateral_stays_insufficient():
    config = {'loss_scales_m_s': [5., 1.]}
    baseline = [dict(scene='scene-1013' if i < 3 else 'scene-other', token=str(i),
                     vx=2., vy=0., target_vx=0., target_vy=0.) for i in range(4)]
    candidate = [dict(row, vx=1. if i < 3 else 3.) for i, row in enumerate(baseline)]
    rows = {'J23': {'dev': summarize_rows(baseline, config)},
            'S23': {'dev': summarize_rows(candidate, config)}}
    result = paired_comparison('S23', 'J23', rows)
    assert result['metrics']['epe']['delta'] == pytest.approx(-.5)
    assert result['improved_scenes'] == result['degraded_scenes'] == 1
    contribution = result['scene_contribution']
    assert sum(contribution['weighted_gain_m_s'].values()) == pytest.approx(.5)
    assert contribution['scene1013_weighted_gain_m_s'] == pytest.approx(.75)
    assert contribution['excluding_scene1013_diagnostic_delta_epe'] == pytest.approx(1.)
    assert contribution['official_metric_includes_all_scenes']
    assert result['lateral']['checks']['nonzero_vy']['status'] == 'insufficient_data'
