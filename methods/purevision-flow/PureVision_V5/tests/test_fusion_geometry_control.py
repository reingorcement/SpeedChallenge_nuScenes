"""Critical one-pilot expansion guard and protocol isolation."""
import copy
from pathlib import Path
import pytest
from fusion_geometry_comparison import pilot_decision,inspect
from fusion_geometry_pipeline import build_plan,validate_inputs
from purevision_v5.common import code_version
from purevision_v5.fusion_geometry_training import validate_config


def pair(b=1.01,coverage=.5):
    a=dict(best_dev_epe=1.,initialization=dict(state_sha256='same'))
    candidate=dict(best_dev_epe=b,initialization=dict(state_sha256='same'))
    ha=[dict(training_loss=.1,geometry_loss=0.)]
    hb=[dict(training_loss=.1,geometry_loss=.01,geometry_valid_window_fraction=coverage,geometry_nonpositive_predicted_depth=0)]
    return a,candidate,ha,hb,dict(status='passed')


def test_inclusive_gate_and_actual_coverage():
    assert pilot_decision(*pair())['status']=='passed'
    assert pilot_decision(*pair(1.01001))['status']=='rejected'
    assert pilot_decision(*pair(1.,0.))['status']=='rejected'
    values=pair();values[1]['initialization']['state_sha256']='other'
    assert pilot_decision(*values)['status']=='rejected'


def test_only_one_fixed_lambda_seed_and_budget():
    import json
    root=Path(__file__).resolve().parents[1]
    for scope in ('PILOT','FULL'):
        for variant in ('A','B'):
            config=json.loads((root/f'configs/next/S23_GEOMETRY_{scope}_{variant}.json').read_text());validate_config(config)
            for key,value in (('seed',43),('epochs',10),('amp',True),('group','J23')):
                bad=dict(config,**{key:value})
                with pytest.raises(ValueError):validate_config(bad)


def test_actual_partial_preflight_never_formal_result():
    root=Path('/home/zhangyu/.local/share/speedest-v5-next/geometry/20261104_0025_s23_geometry')
    with pytest.raises(ValueError,match='Preflight'):
        inspect({},dict(run_dir=str(root/'trainer_B_preflight')))


def test_real_conditional_job_dependencies_and_fixed_plan():
    path=Path('/home/zhangyu/.local/share/speedest-v5-next/pipeline/20261104_0025_s23_geometry/spec.json')
    plan=build_plan(path,code_version());validate_inputs(plan)
    assert len(plan['jobs'])==4
    jobs={j['name']:j for j in plan['jobs']}
    assert jobs['FULL_A']['requires']==[]
    assert jobs['FULL_B']['requires']==['PILOT_A','PILOT_B','GEOMETRY_FULL']
    assert all(j['seed']==42 for j in jobs.values())
