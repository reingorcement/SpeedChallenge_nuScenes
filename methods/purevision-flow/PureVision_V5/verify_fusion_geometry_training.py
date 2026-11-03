"""Check only new objective, matched initialization and complete recovery."""
import argparse
from pathlib import Path
import copy
import math
import subprocess
import torch
import torch.nn.functional as F
from purevision_v5.common import ROOT,read_json,write_json,sha256,now
from purevision_v5.fusion_abc_cache import load_cache
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_geometry_training import load_recovery,validate_checkpoint,new_model
from purevision_v5.training import epoch_order


def main(a):
    root=Path(a.root);configure_fp32(4)
    packet,index=load_cache(a.cache,'train_core')
    pa=root/'trainer_A_preflight';pb=root/'trainer_B_preflight'
    ra,sa=load_recovery(pa);rb,sb=load_recovery(pb)
    ca=sa['current'][1];cb=sb['current'][1]
    assert ca['contract']['scope']==cb['contract']['scope']=='preflight'
    assert ca['completed_epochs']==cb['completed_epochs']==1
    assert ca['optimizer_updates']==cb['optimizer_updates']==1
    assert ca['initialization']['state_sha256']==cb['initialization']['state_sha256']
    assert ca['order_records']==cb['order_records']
    # A's actual first update must remain exactly the historical velocity step.
    model=new_model(ca['config']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    order,_=epoch_order(64,42,0);features={k:v[:64][order] for k,v in packet['features'].items()};target=packet['targets'][:64][order]
    prediction=model(features);scale=target.new_tensor([5.,1.])
    loss=F.smooth_l1_loss(prediction/scale,target/scale,beta=1.,reduction='mean');loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
    assert all(torch.equal(x,ca['model'][k]) for k,x in model.state_dict().items())
    assert any(not torch.equal(ca['model'][k],cb['model'][k]) for k in ca['model'])
    assert cb['history'][0]['geometry_valid_window_fraction']>0 and cb['history'][0]['geometry_loss']>0
    for field in ('next_batch','optimizer_updates'):
        changed=copy.deepcopy(cb);changed[field]+=1
        try:validate_checkpoint(changed)
        except ValueError:pass
        else:raise AssertionError('Incomplete epoch accepted')
    # Actual trainer resume uses a new directory and restores the whole state.
    command=[a.python,'-B',str(ROOT/'train_fusion_geometry.py'),'--config',str(ROOT/'configs/next/S23_GEOMETRY_PILOT_B.json'),
        '--cache',a.cache,'--geometry',str(root/'pilot'),'--lambda-proof',str(root/'lambda.json'),
        '--output',str(root/'trainer_B_resume_preflight'),'--resume',str(pb),'--preflight','--preflight-samples','64','--device','cpu']
    with (root/'trainer_resume.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    _,loaded=load_recovery(root/'trainer_B_resume_preflight');resumed=loaded['current'][1]
    assert resumed['completed_epochs']==2 and resumed['optimizer_updates']==2 and resumed['scheduler']['last_epoch']==2
    assert resumed['initialization']==cb['initialization'] and resumed['parent_run']['run_id']==pb.name
    sources={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'train_fusion_geometry.py',
        ROOT/'purevision_v5/fusion_geometry_training.py',ROOT/'purevision_v5/fusion_geometry_cache.py',ROOT/'purevision_v5/fusion_geometry.py')}
    value=dict(schema='v5_s23_geometry_trainer_preflight_v1',status='passed',time=now(),formal_training=False,
        actual_cached_train_windows=64,actual_updates_per_attempt=1,
        A_first_update_exact_historical_velocity_step=True,B_finite_nonzero_geometry_changes_update=True,
        same_initialization_and_actual_batches=True,initialization_sha256=ca['initialization']['state_sha256'],
        full_optimizer_scheduler_rng_boundary_restored=True,resumed_completed_epochs=2,resumed_optimizer_updates=2,
        partial_boundaries_rejected=True,stage_epochs=20,pilot_updates=2300,full_updates=41560,
        source_sha256=sources,script_sha256=sha256(__file__))
    write_json(root/'trainer_preflight.json',value);print('PASSED exact historical A update, actual B update and complete epoch resume')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--python',default='/home/zhangyu/.conda/envs/speedest/bin/python');main(p.parse_args())
