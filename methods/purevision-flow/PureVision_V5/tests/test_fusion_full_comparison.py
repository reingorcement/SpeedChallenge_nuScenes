import pytest

from fusion_full_history import identity_map, lateral_labels
from fusion_full_comparison import compare
from fusion_grid_comparison import summarize_rows
from purevision_v5.evaluation import DEFAULT_LATERAL_RULE


def test_identity_mapping_rejects_duplicates_and_is_order_independent():
    a = dict(scene='scene-0402', token='a', time_s=1., target_vx=12., target_vy=.01)
    b = dict(scene='scene-1013', token='b', time_s=2., target_vx=8., target_vy=.02)
    assert identity_map([a, b]) == identity_map([b, a])
    with pytest.raises(ValueError, match='Duplicate'):
        identity_map([a, a])
    assert identity_map([a]) != identity_map([dict(a, time_s=2.)])
    assert identity_map([a]) != identity_map([dict(a, target_vx=13.)])


def test_full_comparison_retains_all_scenes_and_separates_focus_diagnostics():
    labels = [dict(scene=scene, token=str(i), time_s=float(i), target_vx=speed, target_vy=.01)
              for i, (scene, speed) in enumerate((('scene-0402', 12.), ('scene-1013', 8.), ('other', 3.)))]
    a = [dict(row, vx=row['target_vx'] + error, vy=.01) for row, error in zip(labels, (3., 1., 1.))]
    b = [dict(row, vx=row['target_vx'] + 2., vy=.01) for row in labels]
    config = {'loss_scales_m_s': [5., 1.]}
    value = compare('S23', 'J23', summarize_rows(a, config), summarize_rows(b, config), 'A_full_matched_23480dev')
    assert value['metrics']['epe']['delta'] == pytest.approx(-1/3)
    assert value['improved_scenes'] == 2 and value['degraded_scenes'] == 1
    assert value['speed_bins']['10..15']['metrics']['epe']['delta'] == pytest.approx(1.)
    diagnostic = value['focus_scene_sensitivity']['without_both']
    assert diagnostic['delta_epe'] == pytest.approx(-1.) and diagnostic['remaining_windows'] == 1
    assert diagnostic['diagnostic_only'] and diagnostic['official_metric_includes_all_scenes']


def test_lateral_window_minimum_remains_required_when_scene_count_is_sufficient():
    rows = [dict(scene=f'scene-{i % 5}', target_vy=.3) for i in range(31)]
    evidence = lateral_labels(rows, DEFAULT_LATERAL_RULE)
    assert evidence['count'] == 31 and evidence['scene_count'] == 5
    assert evidence['status'] == 'insufficient_data'
