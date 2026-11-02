"""Bounded CPU evidence for actual S23 stage update/recovery code (never formal training)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from purevision_v5.common import object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_full_cache import load_full_cache
from purevision_v5 import fusion_tail_training as training
from purevision_v5.fusion_tail import S23TailModel
from purevision_v5.training import epoch_order, model_state_sha256, restore_rng


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left, dict):
        assert left.keys()==right.keys()
        for key in left:
            assert_tree_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):
            assert_tree_equal(a,b)
    else:
        assert left==right,(left,right)


def last_batch_evidence(config):
    generator=torch.Generator().manual_seed(420)
    x=torch.randn(50,7,generator=generator)
    y=torch.randn(50,2,generator=generator)
    reference=nn.Linear(7,2)
    split=copy.deepcopy(reference)
    left=torch.optim.SGD(reference.parameters(),lr=.01)
    right=torch.optim.SGD(split.parameters(),lr=.01)
    reference_totals,_=training.effective_update(reference,left,[(x,y)],50,config)
    split_totals,_=training.effective_update(split,right,
        [(x[i:i+8],y[i:i+8]) for i in range(0,50,8)],50,config)
    max_error=0.
    for a,b in zip(reference.parameters(),split.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
        max_error=max(max_error,float((a.grad-b.grad).abs().max()))
    torch.testing.assert_close(reference_totals,split_totals,atol=1e-5,rtol=1e-6)
    # Count real clip/optimizer calls through the production helper.
    counted=copy.deepcopy(split)
    optimizer=torch.optim.SGD(counted.parameters(),lr=.01)
    with patch.object(optimizer,'step',wraps=optimizer.step) as step, patch.object(
            torch.nn.utils,'clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip:
        training.effective_update(counted,optimizer,
            [(x[i:i+8],y[i:i+8]) for i in range(0,50,8)],50,config)
        assert step.call_count==clip.call_count==1
    return dict(count=50,microbatch_sizes=[8,8,8,8,8,8,2],normalization_denominator=50,
        production_helper='fusion_tail_training.effective_update',max_abs_gradient_error=max_error,
        clip_calls_per_effective_batch=1,optimizer_steps_per_effective_batch=1,
        full_vs_accumulated_loss_gradient_update_close=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','encoder-checkpoint','readout-checkpoint','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('New isolated verification directory required')
    root.mkdir(parents=True,exist_ok=True)
    configure_fp32(1)
    config_path=Path(__file__).parent/'configs/next/FUSION_TAIL_S23_FROZEN_CONT.json'
    config=read_json(config_path)
    training.validate_config(config)
    normalization=last_batch_evidence(config)
    packets,indices={},{}
    for part,n in (('train_core',114),('dev',4)):
        packet,index=load_full_cache(args.cache,part)
        packets[part]=dict(features={k:v[:n].clone() for k,v in packet['features'].items()},
            targets=packet['targets'][:n].clone(),metadata=packet['metadata'][:n])
        indices[part]=index
    del packet
    def small_cache(cache,part,allow_preflight=False):
        assert allow_preflight
        return packets[part],indices[part]
    command=dict(config=str(config_path),cache=args.cache,manifest='unused-in-frozen-cpu-proof',
        flow='unused-in-frozen-cpu-proof',rgb_frames='unused-in-frozen-cpu-proof',
        encoder_checkpoint=args.encoder_checkpoint,readout_checkpoint=args.readout_checkpoint,
        device='cpu',microbatch=8,workers=0,threads=1,verification=None,preflight=True,
        preflight_samples=114,resume=None,output=str(root/'epoch_one'))
    with patch.object(training,'load_full_cache',side_effect=small_cache):
        training.train(SimpleNamespace(**command))
    recovery,loaded=training.load_recovery(root/'epoch_one')
    checkpoint=loaded['current'][1]
    assert checkpoint['completed_epochs']==1 and checkpoint['optimizer_updates']==2
    assert checkpoint['order_records'][0]['batch_sizes']==[64,50]
    assert checkpoint['scheduler']['T_max']==20 and checkpoint['scheduler']['last_epoch']==1
    # Parent files may not be opened when reconstructing a complete model.
    with patch.object(S23TailModel,'__init__',side_effect=AssertionError('parent constructor must not execute')):
        model=S23TailModel.from_state(checkpoint['model'],config['group'],checkpoint['provenance'])
    assert_tree_equal(model.state_dict(),checkpoint['model'])
    optimizer=training.make_optimizer(model,config)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=0.)
    optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng'])
    order,_=epoch_order(114,42,1)
    runtime,_=training.resident_packet(packets['train_core'],torch.device('cpu'))
    model.train()
    for start in (0,64):
        chosen=order[start:start+64]
        training.effective_update(model,optimizer,[training.batch(runtime,chosen,torch.device('cpu'))],
                                  len(chosen),config,cached=True)
    training.evaluate_packet(model,packets['dev'],
        training.resident_packet(packets['dev'],torch.device('cpu'))[0],None,torch.device('cpu'),config)
    scheduler.step()
    expected_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    expected_optimizer=copy.deepcopy(optimizer.state_dict())
    expected_scheduler=copy.deepcopy(scheduler.state_dict())
    command.update(resume=str(root/'epoch_one'),output=str(root/'resumed_epoch_two'))
    with patch.object(training,'load_full_cache',side_effect=small_cache), patch.object(
            S23TailModel,'__init__',side_effect=AssertionError('resume must not reopen parents')):
        training.train(SimpleNamespace(**command))
    manifest,resumed=training.load_recovery(root/'resumed_epoch_two')
    current=resumed['current'][1]
    assert current['completed_epochs']==2 and current['optimizer_updates']==4
    assert_tree_equal(current['model'],expected_state)
    assert_tree_equal(current['optimizer'],expected_optimizer)
    assert_tree_equal(current['scheduler'],expected_scheduler)
    assert not list((root/'resumed_epoch_two').glob('*.partial'))
    # Re-export unchanged best/current cannot leave .partial hardlink aliases.
    training.export_recovery(root/'resumed_epoch_two',resumed['current'][0],resumed['best'][0],current)
    assert not list((root/'resumed_epoch_two').glob('*.partial'))
    rejected=[]
    for field,value in [('next_batch',1),('optimizer_updates',5)]:
        bad=dict(current);bad[field]=value
        try:training.validate_checkpoint(bad)
        except ValueError:rejected.append(field)
        else:raise AssertionError('partial/inconsistent checkpoint accepted')
    bad=dict(current);bad['model']={k:v for k,v in current['model'].items() if k.startswith('readout.')}
    try:training.validate_checkpoint(bad)
    except ValueError:rejected.append('readout_only')
    else:raise AssertionError('readout-only deployment accepted')
    for group in training.GROUPS:
        value=read_json(Path(__file__).parent/f'configs/next/FUSION_TAIL_{group}.json')
        training.validate_config(value)
        for key,wrong in [('epochs',10),('scheduler_epochs',10)]:
            bad=dict(value);bad[key]=wrong
            try:training.validate_config(bad)
            except ValueError:pass
            else:raise AssertionError('superseded stage budget accepted')
    result=read_json(root/'resumed_epoch_two'/'result.json')
    ready=read_json(root/'resumed_epoch_two'/'dev_ready.json')
    assert ready['train'] is None and ready['best_train_full_evaluation_count']==0
    assert result['best_train_full_evaluation_count']==1 and result['best_dev_predictions_reused']
    sampler=training.EffectiveBatchSampler(114,16)
    assert [len(ids) for ids in sampler]==[16,16,16,16,16,16,16,2]
    report=dict(schema='v5_s23_tail_trainer_preflight_v1',status='passed',device='cpu',
        production_stage=dict(epochs=20,scheduler_T_max=20,effective_batch=64,updates_per_epoch=2078,
            final_effective_batch=50,total_updates=41560),
        bounded_execution=dict(train_count=114,dev_count=4,completed_epochs_per_attempt=1,
            optimizer_updates_per_attempt=2,formal_training=False),
        last_batch=normalization,complete_model_independent_restore=True,
        complete_state_prefixes=['flow_encoder.','rgb_encoder.','legacy_head.','readout.'],
        complete_epoch_boundary_restores_optimizer_scheduler_rng=True,
        resumed_next_complete_epoch_matches_cpu_reference_exactly=True,
        frozen_and_BN_digest_preserved=current['initialization']['frozen_state_sha256'],
        rejected_invalid_recovery=rejected,superseded_budget_rejected=True,
        dev_ready_precedes_selected_train=True,best_dev_prediction_reused=True,selected_best_train_evaluations=1,
        canonical_effective_batch_sizes=[64,50],online_microbatch_sampler_preserves_effective_boundaries=True,
        recovery_manifest_sha256=sha256(root/'resumed_epoch_two'/'recovery.json'),
        cache_identity=indices['train_core']['identity'],source_sha256=training.source_digest(),
        verification_script_sha256=sha256(__file__),evidence_root=str(root))
    report['identity']=object_hash(report)
    write_json(root.parent/'trainer_preflight.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
