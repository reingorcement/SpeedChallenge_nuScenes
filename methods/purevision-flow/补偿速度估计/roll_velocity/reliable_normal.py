"""Bounded causal ground-normal state machine; only image correspondences."""
import cv2
import numpy as np


def angle(a: np.ndarray,b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))


def candidates(h: np.ndarray,k: np.ndarray,nominal: np.ndarray) -> list:
    _,rotations,translations,normals=cv2.decomposeHomographyMat(h,k)
    result=[]
    for r,t,n in zip(rotations,translations,normals):
        n=n.ravel()
        if np.linalg.norm(n)<.5 or np.linalg.norm(t)<.0005:
            continue
        n=n/np.linalg.norm(n)
        if n@nominal<0:n=-n
        if not any(angle(n,item[0])<.01 for item in result):result.append((n,r))
    return result


class ReliableNormal:
    def __init__(self,k: np.ndarray,nominal: np.ndarray,config: dict):
        self.k=k;self.nominal=nominal/np.linalg.norm(nominal);self.previous=self.nominal.copy()
        self.config=config;self.stale=0;self.confirmations=0;self.pending=None;self.ever_valid=False

    def step(self,flow: np.ndarray,confidence: np.ndarray) -> dict:
        c=self.config;h,w=confidence.shape
        yy,xx=np.mgrid[0:h:4,0:w:4]
        vectors=flow[:,yy.ravel(),xx.ravel()].T
        xy=np.stack([xx.ravel(),yy.ravel()],-1).astype(np.float32)
        keep=(confidence[yy.ravel(),xx.ravel()]>.3)&np.isfinite(vectors).all(1)&(np.linalg.norm(vectors,axis=1)>.2)
        p=xy[keep];q=(xy+vectors)[keep]
        record={'points':len(p),'inlier_fraction':0.,'hull_fraction':0.,'x_span_fraction':0.,'y_span_fraction':0.,
                'occupied_cells':0,'residual_median_px':None,'residual_p90_px':None,'ambiguity_gap_deg':None,
                'split_difference_deg':None,'innovation_deg':None,'candidate_nominal_deg':None}
        reason='too_few_points';selected=None;selected_r=None
        if len(p)>=c['min_points']:
            hmat,inliers=cv2.findHomography(p,q,cv2.RANSAC,.8,maxIters=2000,confidence=.995)
            if hmat is not None and inliers is not None:
                inliers=inliers.ravel().astype(bool);points=p[inliers];dest=q[inliers]
                residual=np.linalg.norm(cv2.perspectiveTransform(points[None],hmat)[0]-dest,axis=1)
                record.update(inlier_fraction=float(inliers.mean()),hull_fraction=float(cv2.contourArea(cv2.convexHull(points))/(h*w)),
                    x_span_fraction=float(np.ptp(points[:,0])/w),y_span_fraction=float(np.ptp(points[:,1])/h),
                    occupied_cells=len(set(map(tuple,(points/np.array([w/4,h/3])).astype(int)))),
                    residual_median_px=float(np.median(residual)),residual_p90_px=float(np.quantile(residual,.9)))
                checks=[(record['inlier_fraction']>=c['min_inlier_fraction'],'low_inlier_fraction'),
                    (record['hull_fraction']>=c['min_hull_fraction'] and record['x_span_fraction']>=c['min_x_span_fraction'] and record['y_span_fraction']>=c['min_y_span_fraction'] and record['occupied_cells']>=c['min_cells'],'poor_spatial_coverage'),
                    (record['residual_median_px']<=c['median_residual_px'] and record['residual_p90_px']<=c['p90_residual_px'],'large_residual')]
                reason=next((name for ok,name in checks if not ok),'decomposition_rejected')
                if all(ok for ok,_ in checks):
                    options=[]
                    for n,r in candidates(hmat,self.k,self.nominal):
                        deviation=angle(n,self.nominal);innovation=angle(n,self.previous)
                        if deviation<=c['max_nominal_angle_deg'] and innovation<=c['max_innovation_deg']:
                            options.append((innovation+.15*deviation,n,r))
                    options.sort(key=lambda item:item[0])
                    if options:
                        gap=options[1][0]-options[0][0] if len(options)>1 else None
                        record['ambiguity_gap_deg']=gap
                        if gap is None or gap>=c['ambiguity_gap_deg']:
                            n,r=options[0][1:];split=[]
                            fold=((points[:,0]//16+points[:,1]//16).astype(int)%2)==0
                            for group in [fold,~fold]:
                                if group.sum()<40:continue
                                hs,_=cv2.findHomography(points[group],dest[group],0)
                                if hs is None:continue
                                choices=candidates(hs,self.k,self.nominal)
                                if choices:split.append(min(choices,key=lambda item:angle(item[0],n))[0])
                            difference=max([angle(a,n) for a in split]+([angle(split[0],split[1])] if len(split)==2 else [180.]))
                            record.update(split_difference_deg=difference,innovation_deg=angle(n,self.previous),candidate_nominal_deg=angle(n,self.nominal))
                            if difference<=c['split_consistency_deg']:
                                selected,selected_r=n,r;reason='candidate_accepted'
                            else:reason='spatial_split_disagreement'
                        else:reason='ambiguous_solution'
        if selected is not None:
            self.confirmations=self.confirmations+1 if self.pending is not None and angle(selected,self.pending)<=c['max_innovation_deg'] else 1
            self.pending=selected.copy()
            if self.confirmations>=c['confirmation_pairs']:
                n=(1-c['innovation_weight'])*self.previous+c['innovation_weight']*selected;n/=np.linalg.norm(n)
                self.previous=selected_r@n;self.previous/=np.linalg.norm(self.previous)
                self.stale=0;self.ever_valid=True;status='update';reason='all_gates_pass'
            else:selected=None;reason='await_temporal_confirmation'
        else:self.confirmations=0;self.pending=None
        if selected is None:
            self.stale+=1
            if self.ever_valid and self.stale<=c['max_hold_pairs']:
                n=self.previous.copy();status='hold'
            else:
                n=self.nominal.copy();self.previous=n.copy();status='fixed_fallback'
        record.update(normal=n.tolist(),status=status,reason=reason,stale_pairs=self.stale,
                      confirmations=self.confirmations,nominal_deviation_deg=angle(n,self.nominal))
        return record
