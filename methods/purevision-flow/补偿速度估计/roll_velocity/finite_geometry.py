"""Metric finite-frame planar PnP. No state truth, empirical scaling, or training."""
from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def skew(v:np.ndarray)->np.ndarray:
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def left_jacobian(w:np.ndarray)->np.ndarray:
    t=np.linalg.norm(w);a=skew(w)
    if t<1e-7:return np.eye(3)+a/2+a@a/6
    return np.eye(3)+(1-np.cos(t))/t**2*a+(t-np.sin(t))/t**3*a@a
def camera_motion(r21:np.ndarray,t21:np.ndarray,dt:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    q=r21.T;d=-q@t21;w=Rotation.from_matrix(q).as_rotvec()
    v=np.linalg.solve(left_jacobian(w),d)/dt
    return np.r_[v,w/dt],q,d
def project(points:np.ndarray,r:np.ndarray,t:np.ndarray,k:np.ndarray)->np.ndarray:
    x=points@r.T+t;z=x[:,2:3];p=x@k.T
    return p[:,:2]/np.where(abs(z)>1e-10,z,1e-10)
def solve(p:np.ndarray,q:np.ndarray,k:np.ndarray,n:np.ndarray,height:float,dt:float,weights:np.ndarray|None=None,max_points:int=1024,min_points:int=48,huber_delta_px:float=1.5,max_condition:float=1e8,max_nfev:int=35)->dict:
    """p/q source/target pixel points. Returns source-camera constant-twist equivalent."""
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float);weights=np.ones(len(p)) if weights is None else np.asarray(weights,float)
    rays=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
    den=rays@n;depth=-height/np.where(abs(den)>1e-12,den,1e-12)
    valid=np.isfinite(q).all(1)&np.isfinite(p).all(1)&np.isfinite(weights)&(weights>0)&(depth>=.5)&(depth<=150)
    ids=np.flatnonzero(valid)
    if len(ids)>max_points:ids=ids[np.linspace(0,len(ids)-1,max_points).astype(int)]
    out={'valid':False,'reason':'too_few_points','count':len(ids),'condition':float('inf'),'residual':float('nan'),'ambiguity_gap_px':float('nan'),'twist':np.zeros(6),'covariance':np.eye(6)*1e6,'r21':np.eye(3),'t21':np.zeros(3),'candidate_count':0,'rank':0,'iterations':0}
    if not np.isfinite(dt) or dt<=1e-4:
        out['reason']='invalid_interval';return out
    if len(ids)<min_points:return out
    x=rays[ids]*depth[ids,None];obs=q[ids];wt=np.sqrt(weights[ids].clip(0,1))
    if np.linalg.matrix_rank(x-x.mean(0),tol=1e-8)<2:out['reason']='collinear_support';return out
    result=cv2.solvePnPGeneric(np.ascontiguousarray(x),np.ascontiguousarray(obs),k,None,flags=cv2.SOLVEPNP_IPPE)
    guesses=[]
    if result[0]:
        for rv,tv in zip(result[1],result[2]):
            r=cv2.Rodrigues(rv)[0];t=tv.ravel()
            if np.mean((x@r.T+t)[:,2]>.1)<.99:continue
            rms=np.sqrt(np.mean(((project(x,r,t,k)-obs)*wt[:,None])**2))
            if np.isfinite(rms):guesses.append((rms,np.r_[rv.ravel(),t]))
    # Independent planar initialization: IPPE can be numerically unstable for
    # exactly fronto-parallel/noiseless configurations. Select by reprojection,
    # never by ground-truth motion or speed.
    ok,rv,tv=cv2.solvePnP(np.ascontiguousarray(x),np.ascontiguousarray(obs),k,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        r=cv2.Rodrigues(rv)[0];t=tv.ravel()
        if np.mean((x@r.T+t)[:,2]>.1)>.99:
            score=np.sqrt(np.mean(((project(x,r,t,k)-obs)*wt[:,None])**2))
            if np.isfinite(score):guesses.append((score,np.r_[rv.ravel(),t]))
    guesses.sort(key=lambda a:a[0]);out['candidate_count']=len(guesses)
    if not guesses:out['reason']='no_cheiral_pose';return out
    out['ambiguity_gap_px']=guesses[1][0]-guesses[0][0] if len(guesses)>1 else float('inf')
    def residual(a:np.ndarray)->np.ndarray:
        return ((project(x,cv2.Rodrigues(a[:3])[0],a[3:],k)-obs)*wt[:,None]).ravel()
    fit=least_squares(residual,guesses[0][1],loss='huber',f_scale=huber_delta_px,max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    r=cv2.Rodrigues(fit.x[:3])[0];t=fit.x[3:];jac=fit.jac;sv=np.linalg.svd(jac,compute_uv=False);rank=np.linalg.matrix_rank(jac)
    pose_condition=(sv[0]/max(sv[-1],1e-15))**2;err=residual(fit.x);rms=float(np.sqrt((err@err)/max(2*np.sum(wt**2),1e-12)))
    twist,_,_=camera_motion(r,t,dt)
    # Delta-method covariance from metric pose fit into source-camera twist.
    variance=float(err@err/max(len(err)-6,1));posecov=np.linalg.pinv(jac.T@jac)*variance
    deriv=np.zeros((6,6))
    for j in range(6):
        eps=1e-6;z=fit.x.copy();z[j]+=eps
        deriv[:,j]=(camera_motion(cv2.Rodrigues(z[:3])[0],z[3:],dt)[0]-twist)/eps
    covariance=deriv@posecov@deriv.T
    jtwist=jac@np.linalg.pinv(deriv);svt=np.linalg.svd(jtwist,compute_uv=False)
    condition=(svt[0]/max(svt[-1],1e-15))**2
    ok=bool(fit.success and rank==6 and condition<=max_condition and np.isfinite(twist).all() and np.mean((x@r.T+t)[:,2]>.1)>.99)
    out.update(valid=ok,reason='ok' if ok else ('iteration_limit' if not fit.success else 'ill_conditioned'),twist=twist if ok else np.zeros(6),covariance=covariance if ok else np.eye(6)*1e6,r21=r,t21=t,condition=float(condition),pose_condition=float(pose_condition),residual=rms,rank=int(rank),iterations=fit.nfev)
    return out
