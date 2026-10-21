"""Frozen V3 interface adapter; no learned parameter modification."""
from __future__ import annotations
import numpy as np
import torch
from .finite_geometry import solve
from model.camera_geometry import road_plane_depth,pixel_grid
from model.v3.geometry import RobustVisualTwistSolver,motion_design_matrix
from model.v3.types import PairTwistEstimate
class Adapter(torch.nn.Module):
    def __init__(self,config)->None:
        super().__init__();self.original=RobustVisualTwistSolver();self.config=dict(config);self.backend='finite';self.planes=[];self.index=0;self.records=[];self.support_variant='original'
    def forward(self,flow:torch.Tensor,confidence:torch.Tensor,intrinsics:torch.Tensor,dt:torch.Tensor,plane_normal_camera:torch.Tensor,plane_offset:torch.Tensor,reliability:torch.Tensor|None=None,support_mask:torch.Tensor|None=None)->PairTwistEstimate:
        assert len(flow)==1
        if self.planes:
            n,h=self.planes[self.index];plane_normal_camera=flow.new_tensor(n)[None];plane_offset=flow.new_tensor([h])
        self.index+=1;_,_,height,width=flow.shape
        depth,valid=road_plane_depth(intrinsics,plane_normal_camera,plane_offset,height,width)
        candidate=valid&torch.isfinite(flow).all(1)
        if support_mask is not None:candidate &=support_mask[:,0].bool() if support_mask.ndim==4 else support_mask.bool()
        if confidence.ndim==3:confidence=confidence[:,None]
        if reliability is None:reliability=torch.ones_like(confidence)
        weight=torch.nan_to_num(confidence[:,0]).clamp(0,1)*torch.nan_to_num(reliability[:,0]).clamp(0,1)*candidate
        if self.support_variant=='confident':weight=weight*(confidence[:,0]>.3)
        positive_ids=np.flatnonzero((weight[0]>0).cpu().numpy().ravel());maxp=self.config['max_points']
        ids=positive_ids
        if len(ids)>maxp:
            if self.support_variant=='confident':
                # Balanced interleaving over occupied spatial cells, no fabricated support.
                yy,xx=np.divmod(ids,width);cells=(yy*3//height)*4+xx*4//width
                buckets=[ids[cells==cell] for cell in np.unique(cells)]
                allocation=np.zeros(len(buckets),dtype=int)
                while allocation.sum()<maxp:
                    for cell,b in enumerate(buckets):
                        if allocation.sum()==maxp:break
                        if allocation[cell]<len(b):allocation[cell]+=1
                ids=np.concatenate([b[np.linspace(0,len(b)-1,count).astype(int)] for b,count in zip(buckets,allocation) if count])
            else:ids=ids[np.linspace(0,len(ids)-1,maxp).astype(int)]
        ys,xs=np.divmod(ids,width);p=np.c_[xs,ys].astype(float)
        if len(ids):
            support_x_span=float(np.ptp(xs)/max(width-1,1))
            support_y_span=float(np.ptp(ys)/max(height-1,1))
            support_cells=int(len(set((ys*3//height*4+xs*4//width).tolist())))
        else:
            support_x_span=0.0;support_y_span=0.0;support_cells=0
        f=flow[0,:,ys,xs].T.cpu().numpy();w=weight[0,ys,xs].cpu().numpy()
        k=intrinsics[0].cpu().numpy().astype(float);n=plane_normal_camera[0].cpu().numpy().astype(float);h=float(plane_offset[0]);dtime=float(dt[0])
        if self.backend=='V3_matched':
            design=motion_design_matrix(intrinsics,depth,dt)[0,ys,xs].reshape(1,-1,2,6)
            old=self.original._solve_one(flow[0,:,ys,xs].reshape(2,1,-1),design,weight[0,ys,xs].reshape(1,-1),torch.ones((1,len(ids)),device=flow.device,dtype=torch.bool))
            twist,cov,condition,residual,isvalid,count=old
            record=dict(valid=bool(isvalid),reason='ok' if bool(isvalid) else 'V3_invalid',count=int(count),condition=float(condition),residual=float(residual),rank=6 if bool(isvalid) else 0,iterations=4)
            from model.v3.geometry import motion_flow_from_twist
            predicted=motion_flow_from_twist(twist[None],intrinsics,depth,dt)
        else:
            record=solve(p,p+f,k,n,h,dtime,weights=w,**self.config)
            twist=flow.new_tensor(record['twist']);cov=flow.new_tensor(record['covariance']);condition=flow.new_tensor(record['condition']);residual=flow.new_tensor(record['residual'] if np.isfinite(record['residual']) else 0);isvalid=torch.tensor(record['valid'],device=flow.device);count=torch.tensor(record['count'],device=flow.device)
            u,v=pixel_grid(height,width,flow);rays=torch.stack(((u-intrinsics[0,0,2])/intrinsics[0,0,0],(v-intrinsics[0,1,2])/intrinsics[0,1,1],torch.ones_like(u)),-1)
            points=rays*depth[0,:,:,None];target=points@flow.new_tensor(record['r21']).T+flow.new_tensor(record['t21']);pix=target@intrinsics[0].T
            xy=pix[:,:,:2]/target[:,:,2:].clamp_min(1e-6);predicted=(xy-torch.stack((u,v),-1)).permute(2,0,1)[None]
        self.records.append({key:value for key,value in record.items() if key not in ['twist','covariance','r21','t21']})
        support_fraction=weight.flatten(1).sum(-1)/valid.flatten(1).sum(-1).clamp_min(1)
        self.records[-1].update(
            plane_normal_camera_x=float(n[0]),
            plane_normal_camera_y=float(n[1]),
            plane_normal_camera_z=float(n[2]),
            plane_height_m=h,
            pair_dt_s=dtime,
            candidate_point_count=int(candidate[0].sum().item()),
            positive_support_point_count=int(len(positive_ids)),
            support_point_count=int(len(ids)),
            support_point_fraction=float(len(ids)/max(len(positive_ids),1)),
            support_x_span_fraction=support_x_span,
            support_y_span_fraction=support_y_span,
            support_occupied_cells=support_cells,
        )
        return PairTwistEstimate(twist[None],cov[None],condition[None],residual[None],isvalid[None],count[None],support_fraction,predicted,candidate[:,None])
