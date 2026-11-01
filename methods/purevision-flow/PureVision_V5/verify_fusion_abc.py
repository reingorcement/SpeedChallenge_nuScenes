"""Bounded real-data checks. Disposable heads only; never formal pilot training."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from purevision_v5.common import code_version, environment, now, sha256, write_json
from purevision_v5.fusion_abc import FrozenFusionExtractor, build_triplet
from purevision_v5.fusion_abc_cache import datasets, extract_cache, load_cache, fp32_policy
from purevision_v5.training import model_state_sha256, loss_fn


def verify(args):
    fp32_policy();torch.set_num_threads(args.threads)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Preflight output must be new')
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'tiny_cache'
    extract_args=argparse.Namespace(**vars(args));extract_args.output=str(cache)
    extract_args.scope='preflight';extract_args.allow_pilot_cache=False
    index=extract_cache(extract_args)
    data=datasets(args.manifest,args.selection,args.flow,args.rgb_frames)
    frozen=FrozenFusionExtractor(args.checkpoint).to(args.device)
    state_before=model_state_sha256(frozen)
    frozen.train(True)
    assert not any(m.training for m in frozen.modules())
    assert not any(p.requires_grad for p in frozen.parameters())
    tapped={};hooks=[]
    def tap(name):
        def hook(module, inputs, output):tapped[name]=output.detach()
        return hook
    def input_tap(name):
        def hook(module, inputs):tapped[name]=inputs[0].detach()
        return hook
    hooks.append(frozen.source.flow_encoder.spatial.register_forward_hook(tap('stem')))
    hooks.append(frozen.source.flow_encoder.sfe.register_forward_pre_hook(input_tap('sfe_input')))
    hooks.append(frozen.source.flow_encoder.temporal.register_forward_pre_hook(input_tap('temporal_input')))
    hooks.append(frozen.source.flow_encoder.sfe.register_forward_hook(tap('sfe')))
    hooks.append(frozen.source.flow_encoder.temporal.register_forward_hook(tap('temporal')))
    hooks.append(frozen.source.rgb_encoder.backbone.dig.register_forward_hook(tap('rgb')))
    checks={}; shapes={}
    for part in ['train_core','dev']:
        packet,_=load_cache(cache,part,allow_preflight=True)
        ids=[m['dataset_index'] for m in packet['metadata'][:args.batch_size]]
        batch=default_collate([data[part][i] for i in ids]);inputs={k:v.to(args.device) for k,v in batch['inputs'].items()}
        online=frozen(inputs);n=len(ids)
        errors={}
        for k,v in online.items():
            assert not v.requires_grad and v.dtype==torch.float32
            cached=packet['features'][k][:n].to(args.device)
            torch.testing.assert_close(v,cached,rtol=1e-5,atol=1e-6)
            errors[k]=float((v-cached).abs().max());shapes[k]=list(v.shape)
        assert torch.equal(tapped['sfe_input'],tapped['temporal_input'])
        stem=tapped['stem'].reshape(n,4,32,9,16).transpose(1,2)
        torch.testing.assert_close(tapped['sfe_input'],stem,rtol=0,atol=0)
        # Explicit slice-by-slice spatial pooling proves no time interpolation/averaging.
        def manual_regions(x):
            return torch.stack([F.adaptive_avg_pool2d(x[:,:,t],(2,3)).permute(0,2,3,1).reshape(n,6,x.shape[1])
                                for t in range(x.shape[2])],dim=1)
        torch.testing.assert_close(online['F_sp'],manual_regions(tapped['sfe']),rtol=0,atol=0)
        rgb5=manual_regions(tapped['rgb']);assert rgb5.shape==(n,5,6,256)
        torch.testing.assert_close(online['R'],rgb5[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(online['F_tm'],tapped['temporal'].squeeze(-1).squeeze(-1).transpose(1,2),rtol=0,atol=0)
        for j,i in enumerate(ids):
            item=data[part].items[i]
            assert batch['token'][j]==packet['metadata'][j]['token']==item['token']
            assert batch['scene'][j]==packet['metadata'][j]['scene']==item['scene']
            torch.testing.assert_close(batch['target'][j],packet['targets'][j],rtol=0,atol=0)
            for t,(source,target,t0,t1) in enumerate(item['pairs']):
                assert [source,target]==[f[0] for f in item['image_frames'][t:t+2]]
                assert abs(float(online['dt'][j,t])-(t1-t0))<1e-7
        checks[part]=dict(sample_ids=ids,online_cache_max_abs_error=errors,
                          pair_to_rgb_positions=[0,1,2,3],spatial_pool_only=True,
                          flow_sfe_temporal_parallel_exact=True)
    for hook in hooks:hook.remove()
    models=build_triplet(seed=42)
    packet,_=load_cache(cache,'train_core',allow_preflight=True)
    features={k:v.to(args.device) for k,v in packet['features'].items()}
    target=packet['targets'].to(args.device)
    initial={k:model_state_sha256(m) for k,m in models.items()}
    # Compare every same-named same-shaped shared B/C module except blocks.
    common={k:v for k,v in models['B'].state_dict().items() if k in models['C'].state_dict()
            and v.shape==models['C'].state_dict()[k].shape and not k.startswith(('blocks.','layers.'))}
    assert common and all(torch.equal(v,models['C'].state_dict()[k]) for k,v in common.items())
    interaction={}; updates={}
    for name,model in models.items():
        model=model.to(args.device);model.train()
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        if name in {'B','C'}:
            model.eval()
            with torch.no_grad():
                first=model.tokens(features)
                perturbed={k:v.clone() for k,v in features.items()};perturbed['F_sp'][:,0,0,0]+=1
                second=model.tokens(perturbed);d=(first-second).abs().amax(dim=(0,2))
                if name=='B':assert float(d[0])>0 and torch.equal(d[1:],torch.zeros_like(d[1:]))
                else:assert float(d[1:].max())>1e-8
                interaction[name]=dict(changed_token_max=float(d[0]),other_tokens_max=float(d[1:].max()))
            model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
        losses=[]
        for _ in range(3):
            opt.zero_grad(set_to_none=True);pred=model(features);assert pred.shape==(len(target),2)
            loss=loss_fn(pred,target,dict(loss_scales_m_s=[5,1],loss_beta=1))
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        changed=[k for k,p in model.named_parameters() if not torch.equal(before[k],p)]
        assert changed
        path=out/f'disposable_{name}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path)
        restored=build_triplet(42)[name].to(args.device)
        restored.load_state_dict(torch.load(path,weights_only=True,map_location=args.device)['model'],strict=True)
        model.eval();restored.eval()
        with torch.no_grad():torch.testing.assert_close(model(features),restored(features),rtol=0,atol=0)
        updates[name]=dict(parameters=sum(p.numel() for p in model.parameters()),losses=losses,
            processed_batches=3,optimizer_updates=3,amp_skips=0,changed_parameter_tensors=len(changed),
            finite_gradient_norm_last=float(norm),roundtrip_exact=True,initial_state_sha256=initial[name])
    assert model_state_sha256(frozen)==state_before
    assert not any(m.training for m in frozen.modules()) and all(p.grad is None for p in frozen.parameters())
    report=dict(schema='v5_fusion_abc_preflight_v1',status='passed',finished_at=now(),
        scope='12 real windows; disposable three-update heads; NOT formal pilot training',
        code=code_version(),environment=environment(),source=index['source'],cache_identity=index['identity'],
        counts={k:len(v) for k,v in data.items()},cached_counts={k:v['count'] for k,v in index['partitions'].items()},
        actual_shapes=shapes,checks=checks,frozen_state_before=state_before,frozen_state_after=model_state_sha256(frozen),
        all_parameters_buffers_exact=True,eval_override_verified=True,common_bc_initial_keys=sorted(common),
        bc_common_initial_exact=True,token_interaction=interaction,models=updates,
        cache_path=str(cache),checkpoint_sha256=sha256(args.checkpoint),
        formal_training_started=False,full_cache_generated=False,schedules_changed=False)
    write_json(out/'receipt.json',report);print({k:report[k] for k in ['status','cached_counts','token_interaction','models']},flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['checkpoint','manifest','selection','flow','rgb-frames','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--train-samples',type=int,default=8);p.add_argument('--dev-samples',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda:0');verify(p.parse_args())
