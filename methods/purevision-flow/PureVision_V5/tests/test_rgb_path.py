import copy
import math
import pytest
import torch
from purevision_v5.rgb import AppearanceFlexiNetBackbone
from purevision_v5._rgb_reference.flexinet_backbone import FlexiNetBackbone
from purevision_v5.model import build_model
from purevision_v5.optimization import make_scheduler


def test_zero_alpha_keeps_common_forward_and_raw_static_information():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    old=FlexiNetBackbone(3).eval();new=AppearanceFlexiNetBackbone().eval()
    result=new.load_state_dict(old.state_dict(),strict=False)
    assert set(result.missing_keys)=={'appearance_alpha','appearance_stem.0.weight','appearance_stem.0.bias'}
    images=torch.rand(2,5,3,8,8)
    with torch.no_grad():
        torch.testing.assert_close(new.forward_feature_map(images),old.forward_feature_map(images),rtol=0,atol=0)
        static=images[:,:1].expand(-1,5,-1,-1,-1).permute(0,2,1,3,4)
        zeros=torch.zeros_like(static)
        torch.testing.assert_close(new.cma_block(static),new.cma_block(zeros),rtol=0,atol=0)
        assert not torch.equal(new.appearance_stem(static),new.appearance_stem(zeros))
    new.forward_feature_map(images).square().mean().backward()
    assert torch.isfinite(new.appearance_alpha.grad) and new.appearance_alpha.grad.abs()>0
    assert new.appearance_stem[0].weight.grad.count_nonzero()==0


def test_stop_budget_separate_from_cosine_period_and_restore():
    p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([{'params':[p],'lr':1e-3},{'params':[torch.nn.Parameter(torch.ones(1))],'lr':1e-4}])
    cfg={'epochs':18,'scheduler_epochs':40,'scheduler_policy':'cosine_each_group_to_0.1','learning_rate':1e-3}
    s=make_scheduler(o,cfg)
    for e in range(18):
        factor=.1+.9*(1+math.cos(math.pi*e/40))/2
        assert [g['lr'] for g in o.param_groups]==[1e-3*factor,1e-4*factor]
        o.step();s.step()
    saved=copy.deepcopy(s.state_dict());s2=make_scheduler(o,cfg);s2.load_state_dict(saved)
    assert s2.T_max==40 and s2.last_epoch==18
    with pytest.raises(ValueError):make_scheduler(o,{**cfg,'scheduler_epochs':17})


def test_appearance_scope_is_explicit():
    with pytest.raises(ValueError):build_model({'experiment':'R02','encoder_variant':'flex','rgb_variant':'appearance_difference'})
    model=build_model({'experiment':'R03','encoder_variant':'flex','rgb_policy':'finetune','rgb_variant':'appearance_difference'})
    assert model.rgb_encoder.backbone.appearance_alpha.requires_grad


def test_reference_history_excludes_partial_epoch19(tmp_path):
    import json
    from rgb_path_comparison import old_history
    rows=[{'epoch':e,'global_step':2078*(e+1)} for e in range(18)]
    rows.append({'epoch':18,'global_step':37445})
    p=tmp_path/'metrics.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert len(old_history(tmp_path))==18 and old_history(tmp_path)[-1]['global_step']==37404
    rows[17]['global_step']=37445;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):old_history(tmp_path)


def test_new_stage_patrol_cannot_end_before_comparison_and_publication(tmp_path):
    from monitor import publication_allows_stop
    from purevision_v5.common import write_json
    plan_path=tmp_path/'plan.json';job=tmp_path/'job.json';report=tmp_path/'comparison.json'
    write_json(plan_path,{'identity':'new_rgb','jobs':[{'status_path':str(job)}],'completion_report':str(report)})
    write_json(tmp_path/'monitoring/lifecycle_guard.json',{'plan_identity':'new_rgb','plan':str(plan_path),'completion_requires_success':True})
    write_json(job,{'status':'interrupted'})
    assert not publication_allows_stop(tmp_path)
    write_json(job,{'status':'completed'});write_json(report,{'status':'completed'})
    write_json(tmp_path/'monitoring/publication.json',{'status':'published','all_terminal':True,'plan_identities':['old'],'report_commit':'x','verified_remote_commit':'x'})
    assert not publication_allows_stop(tmp_path)
    write_json(tmp_path/'monitoring/publication.json',{'status':'published','all_terminal':True,'plan_identities':['new_rgb'],'report_commit':'x','verified_remote_commit':'x'})
    assert publication_allows_stop(tmp_path)
