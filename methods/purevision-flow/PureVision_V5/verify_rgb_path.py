"""Bounded real-data checks for the single appearance bypass candidate."""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from purevision_v5.common import ROOT, read_json, write_json, now, sha256
from purevision_v5.datasets import VelocityDataset
from purevision_v5.model import build_model
from purevision_v5.initialization import initialize_continuation
from purevision_v5.optimization import make_optimizer, make_scheduler
from purevision_v5.training import seed_everything, loss_fn, optimizer_update, model_state_sha256


def verify(a):
    torch.set_num_threads(4); seed_everything(42)
    conf=read_json(ROOT/'configs/next/R03_APPEAR_FT_s42_18ep.json')
    oldconf=read_json(ROOT/'configs/next/R03_FT.json')
    data=VelocityDataset(a.manifest,'train_core','R03',None,a.flow,rgb_policy='finetune',frame_root=a.rgb_frames)
    dev=VelocityDataset(a.manifest,'dev','R03',None,a.flow,rgb_policy='finetune',frame_root=a.rgb_frames)
    assert (len(data),len(dev),len(data.scene_names),len(dev.scene_names))==(132978,23480,582,103)
    assert not data.source_tokens & dev.source_tokens and not data.scene_names & dev.scene_names
    ids=np.linspace(0,len(data)-1,64,dtype=int).tolist()
    batch=default_collate([data[i] for i in ids]);x={k:v.to(a.device) for k,v in batch['inputs'].items()};target=batch['target'].to(a.device)
    assert x['images'].shape==(64,5,3,72,128) and x['flows'].shape==(64,4,2,72,128)
    assert x['pair_dt'].shape==(64,4) and (x['pair_dt']>0).all()
    # The same dataset's per-window token/pose/flow contracts were previously verified;
    # recheck cached RGB pixels against direct original-file preprocessing in 3 scenes.
    direct=VelocityDataset(a.manifest,'train_core','R03',None,a.flow,rgb_policy='finetune')
    for i in ids[:3]:
        c,d=data[i],direct[i]
        assert c['token']==d['token'] and c['scene']==d['scene']
        for key in c['inputs']:torch.testing.assert_close(c['inputs'][key],d['inputs'][key],rtol=0,atol=0)
        torch.testing.assert_close(c['target'],d['target'],rtol=0,atol=0)
    model=build_model(conf);initial=initialize_continuation(model,conf)
    old=build_model(oldconf);oldinit=initialize_continuation(old,oldconf)
    common=old.state_dict();assert all(torch.equal(model.state_dict()[k],v) for k,v in common.items())
    assert initial['head_state_sha256']==oldinit['head_state_sha256']
    historical=read_json(Path(a.old_run)/'run.json')['initialization']
    for key in ['head_state_sha256','encoder_state_sha256','rgb_state_sha256']:
        assert oldinit[key]==historical[key],key
    opt=make_optimizer(model,conf);sched=make_scheduler(opt,conf)
    history=[__import__('json').loads(s) for s in (Path(a.old_run)/'metrics.jsonl').read_text().splitlines()][:18]
    lr=[]
    for entry in history:
        rates={g['name']:g['lr'] for g in opt.param_groups};assert rates==entry['learning_rates']
        lr.append(rates);opt.step();sched.step()
    assert len(lr)==18 and sched.T_max==40
    model=model.to(a.device).eval();old=old.to(a.device).eval();small={k:v[:4] for k,v in x.items()}
    with torch.inference_mode():
        p=model(small);q=old(small)
        rd=model.rgb_encoder(small['images']);ro=old.rgb_encoder(small['images'])
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-5);torch.testing.assert_close(rd,ro,rtol=0,atol=0)
        static=small['images'][:1,:1].expand(-1,5,-1,-1,-1).permute(0,2,1,3,4)
        other=torch.zeros_like(static)
        bb=model.rgb_encoder.backbone
        torch.testing.assert_close(bb.cma_block(static),bb.cma_block(other),rtol=0,atol=0)
        assert (bb.appearance_stem(static)-bb.appearance_stem(other)).abs().max()>0
    initial_prediction_error=float((p-q).abs().max());del old,rd,ro,p,q
    # Hook confirms the new stem sees the normalized raw pixel tensor directly.
    seen=[]
    hook=model.rgb_encoder.backbone.appearance_stem.register_forward_pre_hook(lambda m,inputs: seen.append(torch.equal(inputs[0],x['images'].permute(0,2,1,3,4))))
    model.train();opt=make_optimizer(model,conf);sched=make_scheduler(opt,conf);scaler=torch.amp.GradScaler('cuda')
    rgb_ids={id(p) for p in model.rgb_encoder.parameters()};assert rgb_ids=={id(p) for g in opt.param_groups if g['name']=='rgb' for p in g['params']}
    prefixes=['flow_encoder','head','rgb_encoder.backbone.cma_block','rgb_encoder.backbone.aft_block','rgb_encoder.backbone.sfe_module','rgb_encoder.backbone.mfe_module','rgb_encoder.backbone.dig','rgb_encoder.region_projection','rgb_encoder.backbone.appearance_stem','rgb_encoder.backbone.appearance_alpha']
    checks={k:False for k in prefixes};steps=[];accepted=0
    torch.cuda.reset_peak_memory_stats(a.device)
    for attempt in range(12):
        before={n:p.detach().clone() for n,p in model.named_parameters()};opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(a.device);t=time.monotonic()
        with torch.autocast('cuda',dtype=torch.float16):pred=model(x);loss=loss_fn(pred.float(),target,conf)
        assert torch.isfinite(loss);scaler.scale(loss).backward()
        grads={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0) for n,p in model.named_parameters()}
        update=optimizer_update(model,opt,scaler,conf['grad_clip']);accepted+=int(update['updated'])
        for prefix in prefixes:
            checks[prefix]|=bool(update['updated'] and any(grads[n] and not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n==prefix or n.startswith(prefix+'.')))
        torch.cuda.synchronize(a.device)
        steps.append(dict(attempt=attempt,updated=update['updated'],loss=float(loss.detach()),alpha=float(model.rgb_encoder.backbone.appearance_alpha.detach()),stem_nonzero_gradient=any(v for n,v in grads.items() if 'appearance_stem.' in n),seconds=time.monotonic()-t))
        if accepted>=4 and all(checks.values()):break
    hook.remove();assert seen and all(seen);assert all(checks.values()),checks
    output=Path(a.output);state_path=output.with_suffix('.pt')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict()),state_path)
    saved=torch.load(state_path,map_location='cpu',weights_only=True);restored=build_model(conf).to(a.device);restored.load_state_dict(saved['model'],strict=True)
    opt2=make_optimizer(restored,conf);opt2.load_state_dict(saved['optimizer']);sched2=make_scheduler(opt2,conf);sched2.load_state_dict(saved['scheduler']);scaler2=torch.amp.GradScaler('cuda');scaler2.load_state_dict(saved['scaler'])
    model.eval();restored.eval()
    with torch.inference_mode():torch.testing.assert_close(model(small),restored(small),rtol=0,atol=0)
    assert model_state_sha256(model)==model_state_sha256(restored)
    report=dict(status='passed',schema='v5_rgb_path_preflight_v1',finished_at=now(),scope='bounded_validation_not_formal_training',initialization=initial,
        old_initialization=oldinit,counts=dict(train_core=132978,dev=23480,train_scenes=582,dev_scenes=103),windows=64,
        raw_pixels_verified=3,raw_stem_input_verified=True,static_appearance_preserved=True,common_parameters_exact=True,
        rgb_alpha_zero_output_exact=True,initial_prediction_max_abs_error=initial_prediction_error,initial_prediction_tolerance=dict(atol=1e-5,rtol=1e-5),
        learning_rates_exact_old_first18=lr,scheduler_epochs=40,stop_epochs=18,module_checks=checks,steps=steps,
        processed_batches=len(steps),optimizer_updates=accepted,amp_skips=len(steps)-accepted,
        parameters=sum(p.numel() for p in model.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
        batch=64,full_state_roundtrip_exact=True,checkpoint_sha256=sha256(state_path),old_best_sha256=sha256(Path(a.old_run)/'best.pt'))
    write_json(output,report);print({k:report[k] for k in ['status','steps','module_checks','peak_allocated_bytes']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['manifest','flow','rgb-frames','old-run','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda:0');verify(p.parse_args())
