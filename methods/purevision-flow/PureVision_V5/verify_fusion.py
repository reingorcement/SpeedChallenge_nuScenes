"""Bounded real-window checks for common-start FLEX fusion, not training results."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from purevision_v5.common import ROOT, now, read_json, write_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.initialization import initialize_continuation
from purevision_v5.model import build_model
from purevision_v5.optimization import make_optimizer, make_scheduler
from purevision_v5.training import loss_fn, model_state_sha256, optimizer_update, seed_everything


def verify(args):
    torch.set_num_threads(4);seed_everything(42);device=args.device
    # Full manifest contracts are checked, but pixels/flow content only for 64 representative windows.
    frozen=VelocityDataset(args.manifest,"train_core","R03",args.features,args.flow,rgb_policy="frozen")
    online=VelocityDataset(args.manifest,"train_core","R03",None,args.flow,rgb_policy="finetune")
    dev=VelocityDataset(args.manifest,"dev","R02",None,args.flow)
    assert (len(frozen),len(dev),len(frozen.scene_names),len(dev.scene_names))==(132978,23480,582,103)
    assert not frozen.source_tokens & dev.source_tokens and not frozen.scene_names & dev.scene_names
    assert frozen.selection_hash==online.selection_hash
    ids=np.linspace(0,len(frozen)-1,64,dtype=int).tolist()
    a=default_collate([frozen[i] for i in ids]);b=default_collate([online[i] for i in ids])
    assert a['token']==b['token'] and a['scene']==b['scene']
    torch.testing.assert_close(a['target'],b['target'],rtol=0,atol=0)
    for k in ('flows','valid','pair_dt'):torch.testing.assert_close(a['inputs'][k],b['inputs'][k],rtol=0,atol=0)
    assert set(b['inputs'])=={'images','flows','valid','pair_dt'}
    assert tuple(b['inputs']['images'].shape)==(64,5,3,72,128)
    flow=a['inputs']['flows'];dt=a['inputs']['pair_dt']
    assert torch.isfinite(flow).all() and flow.min()<0<flow.max() and (dt>0).all()
    report=dict(schema="v5_fusion_verification_v1",status="running",time=now(),scope="bounded_real_preflight_not_formal_results",
        counts=dict(train_windows=len(frozen),dev_windows=len(dev),train_scenes=len(frozen.scene_names),dev_scenes=len(dev.scene_names)),
        windows_checked=len(ids),scenes_checked=len(set(a['scene'])),data=frozen.contract,
        alignment="identical tokens, scenes, targets, four signed flows and dt; causal five frames from same manifest window",
        flow_range=[float(flow.min()),float(flow.max())],dt_range_s=[float(dt.min()),float(dt.max())],models={})
    predictions={};heads={};rgb_inits={}
    for variant in ('R02_FLEX_CONT','R03_Frozen','R03_FT'):
        seed_everything(42);config=read_json(ROOT/'configs/next'/f'{variant}.json')
        model=build_model(config);initial=initialize_continuation(model,config)
        model=model.to(device);model.eval()
        inputs=b['inputs'] if variant=='R03_FT' else a['inputs']
        if variant=='R02_FLEX_CONT':inputs={k:v for k,v in inputs.items() if k!='descriptor'}
        small={k:v[:8].to(device) for k,v in inputs.items()}
        with torch.inference_mode():predictions[variant]=model(small).cpu()
        row=dict(initialization=initial,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()))
        if model.rgb_encoder is not None:
            rgb_inits[variant]=model_state_sha256(model.rgb_encoder);heads[variant]=model_state_sha256(model.head)
            with torch.inference_mode():descriptor=model.rgb_encoder(b['inputs']['images'][:8].to(device)).cpu()
            diff=float((descriptor-a['inputs']['descriptor'][:8]).abs().max())
            torch.testing.assert_close(descriptor,a['inputs']['descriptor'][:8],atol=2e-4,rtol=2e-5)
            row['frozen_descriptor_max_abs_error']=diff
        optimizer=make_optimizer(model,config);scheduler=make_scheduler(optimizer,config)
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        model.train();updated=0;skipped=0;steps=[]
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        rgb_before=model_state_sha256(model.rgb_encoder) if model.rgb_encoder else None
        # Three accepted real updates: first zero-RGB-input-column step, then verify upstream training.
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device);started=time.monotonic()
            x={k:v.to(device) for k,v in inputs.items()};target=a['target'].to(device)
            with torch.autocast('cuda',dtype=torch.float16):
                prediction=model(x);loss=loss_fn(prediction.float(),target,config)
            assert torch.isfinite(loss);scaler.scale(loss).backward()
            gradients={name:dict(finite=bool(torch.isfinite(p.grad).all()),nonzero=bool(p.grad.abs().max()>0))
                       for name,p in model.named_parameters() if p.grad is not None}
            if variant=='R03_FT' and updated==0:
                assert not any(v['nonzero'] for n,v in gradients.items() if n.startswith('rgb_encoder.'))
            result=optimizer_update(model,optimizer,scaler,config['grad_clip'])
            torch.cuda.synchronize(device)
            steps.append(dict(attempt=attempt,updated=result['updated'],loss=float(loss.detach()),
                seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
            updated+=int(result['updated']);skipped+=int(not result['updated'])
            if updated>=3:break
        assert updated>=3
        modules=['flow_encoder','head']
        if variant=='R03_FT':modules+=['rgb_encoder.backbone.'+n for n in ('cma_block','aft_block','sfe_module','mfe_module','dig')]+['rgb_encoder.region_projection']
        module_checks={}
        for module in modules:
            params={n:p for n,p in model.named_parameters() if n.startswith(module+'.')}
            grad=[gradients[n] for n in params if n in gradients]
            changed=sum(not torch.equal(before[n],p.detach()) for n,p in params.items())
            assert grad and all(g['finite'] for g in grad) and any(g['nonzero'] for g in grad) and changed>0,module
            module_checks[module]=dict(finite_gradients=True,nonzero_gradients=True,changed_parameter_tensors=changed)
        if variant=='R03_Frozen':
            assert rgb_before==model_state_sha256(model.rgb_encoder) and not model.rgb_encoder.training
            assert all(p.grad is None for p in model.rgb_encoder.parameters())
            row['frozen_rgb_parameters_and_bn_unchanged']=True
        row.update(steps=steps,processed_batches=len(steps),optimizer_updates=updated,amp_skips=skipped,module_checks=module_checks)
        # Full RGB state, optimizer groups, scheduler and scaler round trip. No initial RGB reloading.
        path=Path(args.output).parent/f'{variant}_preflight_state.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict()),path)
        packet=torch.load(path,map_location='cpu',weights_only=True)
        restored=build_model(config).to(device);restored.load_state_dict(packet['model'],strict=True)
        opt2=make_optimizer(restored,config);opt2.load_state_dict(packet['optimizer'])
        sched2=make_scheduler(opt2,config);sched2.load_state_dict(packet['scheduler'])
        scaler2=torch.amp.GradScaler('cuda');scaler2.load_state_dict(packet['scaler'])
        model.eval();restored.eval()
        with torch.inference_mode():
            first=model(small);second=restored(small)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        assert model_state_sha256(model)==model_state_sha256(restored)
        row['full_rgb_state_restore_exact']=True;row['checkpoint']=str(path)
        row['micro_batch_measured']=64;row['evaluation_batch_measured']=8
        report['models'][variant]=row;write_json(args.output,report)
        del model,restored,optimizer,opt2,packet,x,target,prediction,loss,before
        torch.cuda.empty_cache()
    assert len(set(heads.values()))==len(set(rgb_inits.values()))==1
    base=predictions['R02_FLEX_CONT'];deltas={}
    for variant,value in predictions.items():
        torch.testing.assert_close(value,base,atol=1e-5,rtol=1e-5)
        deltas[variant]=float((value-base).abs().max())
    report.update(status='passed',finished_at=now(),initial_prediction_tolerance=dict(atol=1e-5,rtol=1e-5),
        initial_prediction_max_abs_error=deltas,common_fusion_head_sha256=heads['R03_FT'],common_rgb_state_sha256=rgb_inits['R03_FT'])
    write_json(args.output,report);print(dict(status=report['status'],initial_prediction_max_abs_error=deltas),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','flow','features','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');verify(p.parse_args())
