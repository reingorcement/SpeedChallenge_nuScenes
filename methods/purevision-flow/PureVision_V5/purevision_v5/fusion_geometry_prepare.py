"""Train-only, frozen Cityscapes road support and calibrated source planes."""
from __future__ import annotations
import math
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from .common import read_json, sha256, object_hash, write_json

MODEL_REPOSITORY = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
MODEL_REVISION = '21b3847fae21ddee674abd31129307b6a1235bd9'
MODEL_SHA = '027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c'
POLICY = dict(version='v5_geometry_support_v1', max_points=256, min_points=32,
    segmentation_input_hw=[576,1024], road_class=0, road_probability=.9,
    erosion_pixels=1, homography_ransac_pixels=.5, min_homography_points=60,
    max_homography_rotation_error_deg=.5, max_normal_tilt_deg=10.,
    min_translation_signal_px=.25, min_depth_m=3., max_depth_m=40.,
    max_endpoint_interval_velocity_difference_m_s=.2, max_roll_pitch_change_deg=.5,
    min_pair_dt_s=.04, max_pair_dt_s=.15, spatial_strata_hw=[16,16],
    support='source+warped-target frozen road probability, eroded boundaries; finite valid RAFT and independent planar coherence',
    plane='visual homography decomposition selected by offline rotation and nominal upward sign; passes source ego ground projection',
    scale='actual calibrated_sensor translation; ego origin ground projection per official devkit issue161',
    no_prediction_based_selection=True, missing_geometry='invalid reason; all windows retained; padded points always masked')


def matrix(q):
    q=np.asarray(q,float)
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def yaw_matrix(r):
    angle=math.atan2(r[1,0],r[0,0]);c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def resize_intrinsic(k, width, height):
    k=np.asarray(k,float).copy();sx,sy=128/width,72/height
    k[0]*=sx;k[1]*=sy
    k[0,2]+=(sx-1)/2;k[1,2]+=(sy-1)/2
    return k


def load_metadata(dataroot, rows, output=None):
    """Stream only requested last-pair tokens, never load test labels."""
    import ijson
    root=Path(dataroot)/'v1.0-trainval'
    wanted={token for row in rows for token in row['image_tokens'][-2:]}
    sd={}
    with (root/'sample_data.json').open('rb') as f:
        for value in ijson.items(f,'item',use_float=True):
            if value['token'] in wanted:sd[value['token']]=value
    if set(sd)!=wanted:raise ValueError('Requested training camera tokens absent')
    pose_tokens={x['ego_pose_token'] for x in sd.values()};poses={}
    with (root/'ego_pose.json').open('rb') as f:
        for value in ijson.items(f,'item',use_float=True):
            if value['token'] in pose_tokens:poses[value['token']]=value
    if set(poses)!=pose_tokens:raise ValueError('Missing actual pair ego poses')
    sensors={x['token']:x for x in read_json(root/'calibrated_sensor.json')}
    result=dict(schema='v5_geometry_train_metadata_v1',camera=sd,poses=poses,
        sensors={key:sensors[key] for key in {x['calibrated_sensor_token'] for x in sd.values()}},
        source_files={name:sha256(root/name) for name in ('sample_data.json','ego_pose.json','calibrated_sensor.json')})
    result['identity']=object_hash(result)
    if output:write_json(output,result)
    return result


def pair_constants(row, target, metadata):
    s,t=[metadata['camera'][token] for token in row['image_tokens'][-2:]]
    if [s['token'],t['token']]!=row['pairs'][-1][:2]:raise ValueError('Not the last forward pair')
    dt=(t['timestamp']-s['timestamp'])/1e6
    if dt<=0 or abs(row['pairs'][-1][2]-s['timestamp']/1e6)>1e-6 or abs(row['pairs'][-1][3]-t['timestamp']/1e6)>1e-6:
        raise ValueError('Frame/flow/timestamp identity differs')
    ps,pt=[metadata['poses'][x['ego_pose_token']] for x in (s,t)]
    if any(abs(x['timestamp']-p['timestamp'])>1000 for x,p in ((s,ps),(t,pt))):
        raise ValueError('Camera/pose time alignment requires interpolation')
    cs,ct=[metadata['sensors'][x['calibrated_sensor_token']] for x in (s,t)]
    rs,rt=matrix(ps['rotation']),matrix(pt['rotation']);c0,c1=matrix(cs['rotation']),matrix(ct['rotation'])
    l0,l1=np.asarray(cs['translation']),np.asarray(ct['translation'])
    r21=c1.T@rt.T@rs@c0;base=c1.T@(rt.T@rs@l0-l1)
    velocity_map=-dt*c1.T@rt.T@yaw_matrix(rt)[:,:2]
    interval=yaw_matrix(rt).T@(np.asarray(pt['translation'])-np.asarray(ps['translation']))/dt
    endpoint_error=float(np.linalg.norm(np.asarray(target)-interval[:2]))
    angle=Rotation.from_matrix(rt.T@rs).as_rotvec()
    return dict(K=resize_intrinsic(cs['camera_intrinsic'],s['width'],s['height']),R21=r21,base=base,
        velocity_map=velocity_map,dt=dt,source=s,target=t,lever_camera=c0.T@l0,nominal=c0.T@np.array([0.,0.,1.]),
        interval_velocity=interval[:2],endpoint_interval_error=endpoint_error,
        roll_pitch_change_deg=float(np.degrees(np.linalg.norm(angle[:2]))),
        yaw_rate_deg_s=float(np.degrees(abs(math.atan2((rt.T@rs)[1,0],(rt.T@rs)[0,0])))/dt),
        camera_height_source_m=float(l0[2]))


class RoadSegmenter:
    def __init__(self, source, device):
        from transformers import SegformerForSemanticSegmentation
        source=Path(source);origin=read_json(source/'source.json')
        if origin['repository']!=MODEL_REPOSITORY or origin['revision']!=MODEL_REVISION or sha256(source/'pytorch_model.bin')!=MODEL_SHA:
            raise ValueError('Road support model source differs')
        self.model=SegformerForSemanticSegmentation.from_pretrained(str(source),local_files_only=True).float().eval().to(device)
        self.model.requires_grad_(False);self.device=device
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]

    @torch.no_grad()
    def __call__(self, paths):
        values=[]
        for path in paths:
            with Image.open(path) as im:
                value=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.BILINEAR)).copy()
            values.append(torch.from_numpy(value).permute(2,0,1))
        inputs=torch.stack(values).to(self.device,dtype=torch.float32)/255
        logits=self.model(pixel_values=(inputs-self.mean)/self.std).logits
        # Upsample logits to the cached flow lattice with the same half-pixel rule.
        logits=F.interpolate(logits,size=(72,128),mode='bilinear',align_corners=False)
        road=logits.softmax(1)[:,0].cpu().numpy()
        return [cv2.erode((p>=.9).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool) for p in road]


def stratified_points(mask):
    chosen=[]
    for ys in np.array_split(np.arange(72),16):
        for xs in np.array_split(np.arange(128),16):
            yy,xx=np.nonzero(mask[np.ix_(ys,xs)])
            if len(xx):
                distance=(ys[yy]-ys.mean())**2+(xs[xx]-xs.mean())**2;i=int(distance.argmin())
                chosen.append((int(xs[xx[i]]),int(ys[yy[i]])))
    return np.asarray(chosen,dtype=int).reshape(-1,2)


def construct_support(constants, flow, valid, road0, road1):
    """Masks/plane never depend on a learned speed or its residual."""
    c=constants;k=c['K'];up=c['nominal']
    record=dict(valid=False,reason=None,dt=c['dt'],endpoint_interval_error_m_s=c['endpoint_interval_error'],
        yaw_rate_deg_s=c['yaw_rate_deg_s'],camera_height_source_m=c['camera_height_source_m'])
    def reject(reason):record['reason']=reason;return None,record
    if not POLICY['min_pair_dt_s']<=c['dt']<=POLICY['max_pair_dt_s']:return reject('interval_out_of_range')
    if c['roll_pitch_change_deg']>POLICY['max_roll_pitch_change_deg']:return reject('nonplanar_orientation_change')
    if c['endpoint_interval_error']>POLICY['max_endpoint_interval_velocity_difference_m_s']:return reject('endpoint_interval_approximation')
    yy,xx=np.mgrid[:72,:128].astype(np.float32);pixels=np.stack((xx,yy),-1)
    q=pixels+flow.transpose(1,2,0)
    target=cv2.remap(road1.astype(np.float32),q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.999
    support=road0&target&valid.astype(bool)&np.isfinite(flow).all(0)&(q[...,0]>1)&(q[...,0]<126)&(q[...,1]>1)&(q[...,1]<70)
    p=pixels[support];observed=q[support];record['candidate_points']=len(p)
    if len(p)<60:return reject('insufficient_joint_semantic_road')
    cv2.setRNGSeed(42)
    h,inliers=cv2.findHomography(p,observed,cv2.RANSAC,.5,maxIters=2000,confidence=.995)
    if h is None or not np.isfinite(h).all():return reject('no_coherent_road_plane')
    rays=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
    rotation=rays@c['R21'].T@k.T;rotxy=rotation[:,:2]/rotation[:,2:]
    signal=float(np.median(np.linalg.norm(observed-rotxy,axis=1)));record['translation_signal_px']=signal
    if signal<.25:return reject('normal_unobservable_near_pure_rotation')
    count,rotations,translations,normals=cv2.decomposeHomographyMat(h,k)
    candidates=[]
    for r,n in zip(rotations,normals):
        err=float(np.degrees(Rotation.from_matrix(r@c['R21'].T).magnitude()))
        n=n.ravel();n=n if n@up>=0 else -n
        tilt=float(np.degrees(np.arccos(np.clip(n@up,-1,1))))
        candidates.append((err,tilt,n))
    if not candidates:return reject('plane_decomposition_failed')
    error,tilt,normal=min(candidates,key=lambda x:(x[0],x[1]))
    record.update(rotation_error_deg=error,normal_tilt_deg=tilt,normal_source_camera=normal.tolist())
    if error>.5:return reject('plane_rotation_disagrees_with_pose')
    if tilt>10:return reject('plane_normal_outside_local_road_range')
    allrays=np.stack((xx,yy,np.ones_like(xx)),-1)@np.linalg.inv(k).T
    distance=float(normal@c['lever_camera']);denom=allrays@normal
    depth=np.divide(-distance,denom,out=np.full_like(denom,-1.),where=np.abs(denom)>1e-8)
    homogeneous=np.stack((xx,yy,np.ones_like(xx)),-1)@h.T;fitted=homogeneous[...,:2]/homogeneous[...,2:]
    support &= (np.linalg.norm(fitted-q,axis=-1)<=.5)&(depth>=3)&(depth<=40)
    selected=stratified_points(support);record['support_points']=len(selected)
    if len(selected)<32:return reject('insufficient_reliable_metric_plane_points')
    x,y=selected.T;points=allrays[y,x]*depth[y,x,None]
    record.update(valid=True,reason='valid',plane_camera_distance_m=distance,
        selected_source_pixels=selected.tolist(),depth_min_m=float(depth[y,x].min()),depth_max_m=float(depth[y,x].max()))
    return dict(points=points.astype(np.float32),flow=flow[:,y,x].T.astype(np.float32)),record
