"""One fixed pilot-train calibration, four first epoch0 batches, never dev."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from purevision_v5.common import write_json,read_json,sha256,object_hash,now
from purevision_v5.fusion_abc_cache import load_cache
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_geometry_cache import load_geometry
from purevision_v5.fusion_geometry import geometry_loss,select,to_device
from purevision_v5.fusion_grid import build_six
from purevision_v5.training import epoch_order,model_state_sha256


def main(a):
    output=Path(a.output)
    if output.exists():raise ValueError('Lambda may be calibrated only once; preserve and reuse prior proof')
    proof=read_json(a.verification)
    if proof.get('status')!='passed':raise ValueError('Geometric short verification not passed')
    configure_fp32(4)
    packet,index=load_cache(a.cache,'train_core')
    geometry,provenance=load_geometry(a.geometry,packet,'pilot')
    model=build_six(42)['S23'].float().to(a.device).train();initial=model_state_sha256(model)
    order,digest=epoch_order(7324,42,0);measurements=[];ratios=[]
    for b in range(4):
        ids=order[b*64:(b+1)*64]
        features={k:v[ids].to(a.device) for k,v in packet['features'].items()}
        target=packet['targets'][ids].to(a.device);g=to_device(select(geometry,ids),a.device)
        prediction=model(features)
        vel=F.smooth_l1_loss(prediction/prediction.new_tensor([5.,1.]),target/target.new_tensor([5.,1.]),beta=1.,reduction='mean')
        geo=geometry_loss(prediction,g)
        gv=torch.autograd.grad(vel,prediction,retain_graph=True)[0]
        gg=torch.autograd.grad(geo,prediction)[0]
        vn,gn=float(gv.norm()),float(gg.norm());finite=np.isfinite(vn) and np.isfinite(gn)
        if not finite or gn<=0 or vn<=0:raise ValueError('No finite nonzero geometric gradient in prescribed calibration batches')
        ratio=.1*vn/gn;ratios.append(ratio)
        measurements.append(dict(batch=b,keys=[(packet['metadata'][i]['scene'],packet['metadata'][i]['token']) for i in ids],
            velocity_gradient_l2=vn,geometry_gradient_l2=gn,lambda_candidate=ratio,
            velocity_loss=float(vel.detach()),geometry_loss=float(geo.detach()),valid_windows=int(g['valid'].sum())))
    weight=float(np.median(ratios))
    assert initial==model_state_sha256(model)
    value=dict(schema='v5_s23_geometry_lambda_v1',status='passed',created_at=now(),scope='pilot_train_only',seed=42,
        target_gradient_ratio=.1,lambda_geo=weight,rule='median(0.1*L2(dL_vel/dprediction)/L2(dL_geo/dprediction)) of fixed first4 epoch0 batches',
        calibration_batches=measurements,initialization_sha256=initial,epoch_order_sha256=digest,
        feature_cache_identity=index['identity'],geometry=provenance,short_verification_sha256=sha256(a.verification),
        source_sha256=sha256(__file__),model_updated=False,dev_used=False,
        reset='Formal processes independently rebuild build_six(42)[S23], new optimizer/cosine20/RNG')
    value['identity']=object_hash(value);write_json(output,value);print('Fixed lambda_geo',weight,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','geometry','verification','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
