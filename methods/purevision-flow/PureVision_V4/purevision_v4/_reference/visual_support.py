"""RGB-only road proposal and optical-flow consistency; no simulation truth."""
import cv2
import numpy as np


def road_mask(rgb: np.ndarray, k: np.ndarray, nominal: np.ndarray, config: dict) -> np.ndarray:
    h,w=rgb.shape[:2]
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    seed=lab[int(.7*h):int(.9*h),int(.4*w):int(.6*w)]
    centre=np.median(seed.reshape(-1,3),axis=0)
    scatter=np.median(np.linalg.norm(seed[...,1:]-centre[1:],axis=-1))
    threshold=np.clip(3*scatter,config['lab_chroma_min'],config['lab_chroma_max'])
    mask=(np.linalg.norm(lab[...,1:]-centre[1:],axis=-1)<=threshold)&(abs(lab[...,0]-centre[0])<=config['luma_tolerance'])
    y,x=np.mgrid[:h,:w]
    rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(k).T
    mask &= (rays@nominal < -.05)&(y>.35*h)
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    seed_labels=labels[int(.7*h):int(.9*h),int(.4*w):int(.6*w)]
    ids,counts=np.unique(seed_labels[seed_labels>0],return_counts=True)
    if not len(ids):
        return np.zeros((h,w),bool)
    chosen=ids[np.argmax(counts)]
    return labels==chosen if stats[chosen,cv2.CC_STAT_AREA]>=config['min_component_fraction']*h*w else np.zeros((h,w),bool)


def supported_flow(flow: np.ndarray, confidence: np.ndarray, source: np.ndarray, target: np.ndarray, config: dict) -> tuple[np.ndarray,dict]:
    h,w=source.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    qx,qy=x+flow[0],y+flow[1]
    target_valid=cv2.remap(target.astype(np.float32),qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.999
    valid=source&target_valid&np.isfinite(flow).all(0)&(qx>=1)&(qx<w-2)&(qy>=1)&(qy<h-2)
    yy,xx=np.mgrid[0:h:4,0:w:4]
    selected=valid[yy,xx]&(confidence[yy,xx]>config['confidence_fit_threshold'])
    p=np.stack([xx[selected],yy[selected]],-1).astype(np.float32)
    q=p+flow[:,yy[selected],xx[selected]].T
    homography=None
    residual=np.full((h,w),np.inf,np.float32)
    if len(p)>=60:
        homography,inliers=cv2.findHomography(p,q,cv2.RANSAC,config['homography_threshold_px'],maxIters=2000,confidence=.995)
        if homography is not None and np.isfinite(homography).all():
            pixels=np.stack([x,y,np.ones_like(x)],-1)
            projected=pixels@homography.T
            projected=projected[...,:2]/projected[...,2:]
            residual=np.linalg.norm(projected-np.stack([qx,qy],-1),axis=-1)
            valid &= np.isfinite(residual)&(residual<=config['flow_rejection_px'])
        else:
            homography=None
    # Conservative reject when no coherent road motion can be established.
    if homography is None:
        valid[:]=False
    result=confidence*valid.astype(np.float32)
    return result,{'rgb_road_fraction':float(source.mean()),'joint_motion_fraction':float(valid.mean()),
        'weighted_coverage':float(result.mean()),'high_conf_points':int((result>.6).sum()),
        'fit_points':len(p),'region_status':'planar_support' if homography is not None else 'no_coherent_support'}
