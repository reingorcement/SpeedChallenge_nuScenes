"""Bounded train geometry checks and immutable train-only point caches."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from purevision_v5.common import read_json,write_json,sha256,object_hash,now,atomic_save
from purevision_v5.fusion_abc_cache import MANIFEST_SHA
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_geometry import validate_packet,geometry_loss,theoretical_flow,to_device
from purevision_v5.fusion_geometry_prepare import (POLICY,RoadSegmenter,load_metadata,pair_constants,construct_support)
from purevision_v5.packed_flow import PackedFlow


def read_annotations(root):
    root=Path(root);index=read_json(root/'index.json');part=index['partitions']['train_core'];p=root/part['annotations']
    if sha256(p)!=part['annotations_sha256']:raise ValueError('Feature annotations changed')
    packet=torch.load(p,map_location='cpu',weights_only=True)
    if len(packet['metadata'])!=part['count']:raise ValueError('Incomplete train annotations')
    return packet,index


def fixed_preflight_indices(packet,constants):
    scenes=defaultdict(list)
    for i,row in enumerate(packet['metadata']):scenes[row['scene']].append(i)
    if len(scenes)!=32:raise ValueError('Preflight selection must use original32 pilot train scenes')
    ids=[]
    for scene,values in sorted(scenes.items()):
        speed=lambda i:float(packet['targets'][i].norm())
        picks=[min(values,key=speed),max(values,key=speed),
               max(values,key=lambda i:constants[i]['yaw_rate_deg_s']),
               max(values,key=lambda i:constants[i]['endpoint_interval_error'])]
        distinct=list(dict.fromkeys(picks))
        for i in values:
            if len(distinct)==4:break
            if i not in distinct:distinct.append(i)
        ids.extend(distinct)
    if len(ids)!=128 or len(set(ids))!=128:raise ValueError('Bounded selection changed')
    return ids


def packet_for(constants,support):
    n=len(constants);p=256
    # Padding is explicitly absent support, never a substitute depth measurement.
    points=torch.zeros(n,p,3);points[:,:,2]=1.
    result=dict(points=points,flow=torch.zeros(n,p,2),mask=torch.zeros(n,p,dtype=torch.bool),
        K=torch.tensor(np.stack([x['K'] for x in constants]),dtype=torch.float32),
        R21=torch.tensor(np.stack([x['R21'] for x in constants]),dtype=torch.float32),
        base=torch.tensor(np.stack([x['base'] for x in constants]),dtype=torch.float32),
        velocity_map=torch.tensor(np.stack([x['velocity_map'] for x in constants]),dtype=torch.float32),
        dt=torch.tensor([x['dt'] for x in constants],dtype=torch.float32),valid=torch.zeros(n,dtype=torch.bool))
    for i,value in enumerate(support):
        if value is None:continue
        m=len(value['points']);result['points'][i,:m]=torch.from_numpy(value['points'])
        result['flow'][i,:m]=torch.from_numpy(value['flow']);result['mask'][i,:m]=True;result['valid'][i]=True
    validate_packet(result);return result


def main(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Geometry output must be new')
    output.mkdir(parents=True)
    configure_fp32(4);torch.manual_seed(42);started=time.monotonic()
    packet,index=read_annotations(args.cache)
    if args.mode=='preflight' and index['scope']!='pilot':raise ValueError('Bounded check must use pilot train')
    if args.mode!='preflight':
        proof=read_json(args.verification)
        if proof.get('status')!='passed' or proof['policy']!=POLICY:raise ValueError('Passed fixed geometry preflight required')
    metadata=load_metadata(args.dataroot,packet['metadata'],output/'train_metadata.json')
    constants=[pair_constants(row,target.tolist(),metadata) for row,target in zip(packet['metadata'],packet['targets'])]
    ids=fixed_preflight_indices(packet,constants) if args.mode=='preflight' else list(range(len(constants)))
    rows=[packet['metadata'][i] for i in ids];constants=[constants[i] for i in ids];targets=packet['targets'][ids]
    manifest=dict(rule='four per scene: slowest, fastest, greatest yaw rate, largest endpoint/interval difference; fill duplicate picks in source order',
        ids=ids,keys=[(x['scene'],x['token']) for x in rows],mode=args.mode,feature_cache_identity=index['identity'],policy=POLICY)
    manifest['identity']=object_hash(manifest);write_json(output/'selection.json',manifest)
    flows=PackedFlow(args.flow,dataset_sha256=MANIFEST_SHA)
    model=RoadSegmenter(args.model,args.device)
    frames={x['token']:x for c in constants for x in (c['source'],c['target'])};masks={}
    frame_list=list(frames.values());dataroot=Path(args.dataroot)
    for start in range(0,len(frame_list),8):
        part=frame_list[start:start+8];computed=model([dataroot/x['filename'] for x in part])
        masks.update((x['token'],mask) for x,mask in zip(part,computed))
        if start%256==0:print('road_frames',start+len(part),'/',len(frame_list),flush=True)
    values=[];records=[]
    for row,c in zip(rows,constants):
        flow,valid,_=flows.load_pairs([tuple(row['pairs'][-1])])
        value,record=construct_support(c,flow[0].numpy(),valid[0,0].numpy(),masks[c['source']['token']],masks[c['target']['token']])
        values.append(value);records.append(dict(scene=row['scene'],token=row['token'],**record))
    geometry=packet_for(constants,values);atomic_save(output/'geometry.pt',geometry)
    manifest.update(geometry_file='geometry.pt',geometry_sha256=sha256(output/'geometry.pt'),
        metadata_identity=metadata['identity'],source_model=read_json(Path(args.model)/'source.json'),
        source_flow_identity=flows.identity,source_file_sha256={name:sha256(Path(__file__).parent/name) for name in
            ('prepare_fusion_geometry.py','purevision_v5/fusion_geometry_prepare.py','purevision_v5/fusion_geometry.py')})
    manifest['identity']=object_hash({k:v for k,v in manifest.items() if k!='identity'});write_json(output/'index.json',manifest)
    write_json(output/'coverage.json',dict(total=len(rows),valid=sum(x['valid'] for x in records),reasons=dict(Counter(x['reason'] for x in records)),records=records))
    if args.mode=='preflight':
        # Fixed physical/model approximations are diagnostics, not label changes.
        g=to_device(geometry,args.device);truth=targets.to(args.device)
        pred=truth.clone().requires_grad_(True);loss=geometry_loss(pred,g);grad=torch.autograd.grad(loss,pred)[0]
        diagnostics=[]
        for name,delta in (('vx_plus1',[1.,0.]),('vx_minus1',[-1.,0.]),('vy_plus1',[0.,1.]),('vy_minus1',[0.,-1.])):
            diagnostics.append(dict(name=name,loss=float(geometry_loss(truth+truth.new_tensor(delta),g))))
        for name,factor in (('depth_minus5percent',.95),('depth_plus5percent',1.05)):
            changed=dict(g,points=g['points']*factor);diagnostics.append(dict(name=name,loss=float(geometry_loss(truth,changed))))
        changed=dict(g,dt=g['dt']*1.1,velocity_map=g['velocity_map']*1.1)
        diagnostics.append(dict(name='interval_plus10percent',loss=float(geometry_loss(truth,changed))))
        # Actual interval velocity is reported only for temporal approximation.
        interval=torch.tensor(np.stack([c['interval_velocity'] for c in constants]),device=args.device,dtype=torch.float32)
        interval_loss=float(geometry_loss(interval,g))
        valid_count=int(geometry['valid'].sum());finite=bool(torch.isfinite(grad).all());norm=float(grad.norm())
        # Show16 deterministic source views, including invalid cases, without a
        # prediction-dependent choice or support adaptation.
        montage=Image.new('RGB',(4*320,4*200));draw=ImageDraw.Draw(montage)
        for j,i in enumerate(np.linspace(0,len(rows)-1,16).astype(int)):
            with Image.open(dataroot/constants[i]['source']['filename']) as im:im=im.convert('RGB').resize((320,180))
            painter=ImageDraw.Draw(im)
            for x,y in records[i].get('selected_source_pixels',[]):painter.ellipse((x*2.5-1,y*2.5-1,x*2.5+1,y*2.5+1),fill=(0,255,0))
            x,y=(j%4)*320,(j//4)*200;montage.paste(im,(x,y));draw.text((x,y+181),f"{rows[i]['scene']} {records[i]['reason'][:30]}",fill='white')
        montage.save(output/'support_contact_sheet.jpg')
        passed=valid_count>0 and finite and norm>0
        receipt=dict(schema='v5_s23_geometry_preflight_v1',status='passed_pending_visual_and_synthetic' if passed else 'failed',
            observed_at=now(),formal_training=False,policy=POLICY,selection_sha256=sha256(output/'selection.json'),
            valid_windows=valid_count,total_windows=len(rows),coverage=valid_count/len(rows),
            failure_reasons=dict(Counter(x['reason'] for x in records)),
            geometry_loss_at_endpoint_target=float(loss),geometry_loss_at_interval_velocity=interval_loss,
            endpoint_prediction_gradient_l2=norm,gradient_finite=finite,perturbations=diagnostics,
            elapsed_seconds=time.monotonic()-started,source_sha256=manifest['source_file_sha256'],
            caveat='Offline orientations/semantic model/geometric priors are extra training supervision and uncertain, never inference inputs')
        write_json(output/'preflight.json',receipt);print(receipt,flush=True)
    else:print('Geometry cache complete',len(rows),'windows',sum(x['valid'] for x in records),'valid',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','dataroot','flow','model','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('preflight','pilot','full'),required=True);p.add_argument('--verification');p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
