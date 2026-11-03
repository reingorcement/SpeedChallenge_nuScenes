"""Finalize the one bounded128-window check; no new pixel or parameter search."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from purevision_v5.common import read_json,write_json,sha256,now
from purevision_v5.fusion_abc_cache import load_cache
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_geometry import geometry_loss,to_device,validate_packet
from purevision_v5.fusion_grid import build_six
from purevision_v5.training import model_state_sha256


def main(a):
    root=Path(a.preflight);proof=read_json(root/'preflight.json');index=read_json(root/'index.json')
    if proof['status']!='passed_pending_visual_and_synthetic' or proof['total_windows']!=128:
        raise ValueError('Cannot silently rerun or change a failed bounded check')
    source=Path(__file__).parent
    for name,digest in index['source_file_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Initial geometry source changed')
    subprocess.run([a.python,'-B','-m','pytest','-q','tests/test_fusion_geometry.py'],cwd=source,check=True)
    configure_fp32(4)
    geometry=torch.load(root/'geometry.pt',map_location='cpu',weights_only=True);validate_packet(geometry)
    cache,cache_index=load_cache(a.cache,'train_core')
    if cache_index['identity']!=index['feature_cache_identity']:raise ValueError('Pilot source differs')
    ids=read_json(root/'selection.json')['ids'];g=to_device(geometry,a.device)
    model=build_six(42)['S23'].to(a.device).train();initial=model_state_sha256(model)
    features={k:v[ids].to(a.device) for k,v in cache['features'].items()}
    output=model(features);loss=geometry_loss(output,g)
    output_gradient=torch.autograd.grad(loss,output,retain_graph=True)[0]
    loss.backward();norms={}
    for prefix in ('flow_proj','rgb_proj','layers','head'):
        selected=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if selected:norms[prefix]=float(torch.sqrt(sum(p.grad.double().square().sum() for p in selected)))
    all_gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if (not all_gradients or not all(torch.isfinite(x).all() for x in all_gradients)
            or not float(output_gradient.norm())>0 or not norms.get('head',0)>0):
        raise ValueError('Geometry gradient does not reach actual S23 speed head')
    assert initial==model_state_sha256(model)
    truth=cache['targets'][ids].to(a.device)
    delta=torch.tensor(Rotation.from_euler('y',.2,degrees=True).as_matrix(),dtype=torch.float32,device=a.device)
    changed=dict(g,R21=delta[None]@g['R21'])
    rotation_loss=float(geometry_loss(truth,changed))
    if not np.isfinite(rotation_loss) or rotation_loss<=proof['geometry_loss_at_endpoint_target']:
        raise ValueError('Fixed rotation perturbation does not increase observed residual')
    proof.update(status='passed',finalized_at=now(),synthetic_tests=dict(passed=10,
        source_sha256=sha256(source/'tests/test_fusion_geometry.py')),
        visual_review=dict(contact_sheet_sha256=sha256(root/'support_contact_sheet.jpg'),
            views=16,observation='points lie on road interiors in representative day/rain scenes; cars/pedestrians largely excluded; low-observability/night/temporal failures retained invalid',
            limitation='pretrained semantics and homography are imperfect priors, not certified static-road ground truth'),
        real_S23_gradient=dict(initial_state_sha256=initial,output_gradient_l2=float(output_gradient.norm()),
            module_gradient_l2=norms,all_finite=True,model_updated=False),
        rotation_perturbation=dict(camera_y_degrees=.2,loss=rotation_loss),
        supported_scope='fixed v1 thresholds only; valid geometry optional per training window, velocity supervision always retained',
        reference_sources=['https://github.com/nutonomy/nuscenes-devkit/issues/161#issuecomment-508015958',
            'https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/tree/21b3847fae21ddee674abd31129307b6a1235bd9'],
        finalizer_source_sha256=sha256(__file__))
    write_json(root/'preflight.json',proof);print('PASSED',proof['valid_windows'],'/128; S23 speed-head gradient',norms['head'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--python',default='/home/zhangyu/.conda/envs/speedest/bin/python')
    main(p.parse_args())
