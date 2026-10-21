"""Independent geometry-only aggregation in fixed calibrated vehicle-body axes.

No learning correction/fusion, no dynamic state. Fixed mount yields body reference.
"""
import numpy as np
def body_reference(cal):
    optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]],float)
    cbody=np.array(cal['body_mount_rotation'])@optical
    cnominal=np.array(cal['rotation_ego_from_camera'])
    return cbody@cnominal.T
def aggregate(pair_twists,valid,cal):
    good=np.asarray(valid,bool)
    if not good.any():return {'valid':False,'reason':'all_three_pairs_invalid','vx':np.nan,'vy':np.nan,'vz':np.nan}
    value=np.asarray(pair_twists)[good,:3].mean(0);body=body_reference(cal)@value
    return {'valid':True,'reason':'all_pairs_valid' if good.all() else 'partial_valid_pairs','vx':float(body[0]),'vy':float(body[1]),'vz':float(body[2])}
