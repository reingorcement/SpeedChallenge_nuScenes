import pytest
import torch
from purevision_v5.model import build_model
from purevision_v5.training import model_state_sha256


@pytest.mark.parametrize("seed",[42,43,44])
def test_explicit_head_initialization_and_signed_temporal_paths(seed):
    torch.set_num_threads(2)
    models=[]
    for variant in ("base","flex"):
        torch.manual_seed(seed)
        models.append(build_model(dict(experiment="R02",encoder_variant=variant,head_seed=seed)))
    assert model_state_sha256(models[0].head)==model_state_sha256(models[1].head)
    x=torch.randn(2,4,2,72,128)
    inputs=dict(flows=x,valid=torch.ones(2,4,1,72,128,dtype=torch.bool),pair_dt=torch.full((2,4),1/12))
    for model in models:
        model.eval()
        x.requires_grad_()
        result=model(inputs)
        result.sum().backward()
        assert result.shape==(2,2)
        assert all(x.grad[:,t].abs().sum()>0 for t in range(4))
        assert not torch.equal(result,model(dict(inputs,flows=-x)))
        assert not torch.equal(result,model(dict(inputs,flows=x.flip(1))))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    flex=models[1].flow_encoder
    value=flex.temporal(torch.randn(2,32,4,9,16))
    assert value.shape==(2,32,4,1,1)
    with pytest.raises(ValueError): build_model(dict(experiment="R03",encoder_variant="flex"))


@pytest.mark.parametrize("variant",["base","flex"])
def test_encoder_complete_resume_state_matches_uninterrupted(tmp_path,variant):
    from test_training import make_case, assert_tree_equal, load_state
    from purevision_v5 import training
    from purevision_v5.common import read_json, write_json
    args=make_case(tmp_path/"case",experiment="R02",flows=True)
    conf=read_json(args.config);conf.update(encoder_variant=variant,head_seed=42)
    write_json(args.config,conf)
    args.output=str(tmp_path/"full");full=load_state(training.train(args))
    args.output=str(tmp_path/"interrupted");args.stop_after_steps=2
    interrupted=training.train(args)
    args.output=str(tmp_path/"resumed");args.resume=str(interrupted/"last.pt");args.stop_after_steps=0
    resumed=load_state(training.train(args))
    for key in ("model","optimizer","scheduler","scaler","rng","epoch","next_batch","global_step",
                "processed_batches","optimizer_updates","amp_skips","best_epe","initialization"):
        assert_tree_equal(full[key],resumed[key])


def test_terminal_patrol_requires_this_plan_remote_publication(tmp_path):
    from monitor import publication_allows_stop
    from purevision_v5.common import write_json
    write_json(tmp_path/"monitoring/lifecycle_guard.json",dict(plan_identity="new"))
    receipt=dict(status="published",all_terminal=True,plan_identities=["old"],report_commit="a",verified_remote_commit="a")
    write_json(tmp_path/"monitoring/publication.json",receipt)
    assert not publication_allows_stop(tmp_path)
    receipt["plan_identities"]=["new"];write_json(tmp_path/"monitoring/publication.json",receipt)
    assert publication_allows_stop(tmp_path)
    receipt["status"]="pending";write_json(tmp_path/"monitoring/publication.json",receipt)
    assert not publication_allows_stop(tmp_path)
