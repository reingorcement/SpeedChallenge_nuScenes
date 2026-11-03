"""Bind train-only geometric points by(scene,token), never shared row indices."""
from pathlib import Path
import torch
from .common import read_json,sha256,object_hash
from .fusion_geometry import validate_packet
from .fusion_geometry_prepare import POLICY


def load_geometry(root,packet,scope):
    root=Path(root);index=read_json(root/'index.json')
    if index.get('identity')!=object_hash({k:v for k,v in index.items() if k!='identity'}):
        raise ValueError('Geometry index changed')
    if index['mode']!=scope or index['policy']!=POLICY:
        raise ValueError('Geometry scope or applicability policy differs')
    path=root/index['geometry_file']
    if sha256(path)!=index['geometry_sha256']:raise ValueError('Geometry content changed')
    value=torch.load(path,map_location='cpu',weights_only=True);validate_packet(value)
    count={'pilot':7324,'full':132978}[scope]
    keys=[tuple(x) for x in index['keys']]
    if len(keys)!=count or len(set(keys))!=count or len(value['valid'])!=count:
        raise ValueError('Incomplete geometric sample scope')
    mapping={key:i for i,key in enumerate(keys)}
    ids=torch.tensor([mapping[(m['scene'],m['token'])] for m in packet['metadata']],dtype=torch.long)
    aligned={key:t[ids].contiguous() for key,t in value.items()}
    times=[m['pairs'][-1][2:] for m in packet['metadata']]
    # The historical model input subtracts large float64 epoch-second stamps
    # before FP32 conversion. Geometry uses exact integer microsecond deltas.
    # Validate both exact constructions, rather than relax a tolerance or alter
    # either the frozen model input or the actual geometric interval.
    historical=torch.tensor([t-s for s,t in times],dtype=torch.float32)
    actual=torch.tensor([(round(t*1e6)-round(s*1e6))/1e6 for s,t in times],dtype=torch.float32)
    if not torch.equal(historical,packet['features']['dt'][:,-1]) or not torch.equal(actual,aligned['dt']):
        raise ValueError('Historical input dt or exact actual geometry interval disagrees with pair identity')
    coverage=read_json(root/'coverage.json')
    if not aligned['valid'].any():raise ValueError('Geometry is an all-zero constraint')
    return aligned,dict(path=str(root),identity=index['identity'],index_sha256=sha256(root/'index.json'),
        geometry_sha256=index['geometry_sha256'],source_feature_cache_identity=index['feature_cache_identity'],
        sample_mapping='scene,token',total=count,valid=coverage['valid'],reasons=coverage['reasons'],
        policy=POLICY,source_model=index['source_model'],stop_gradient=True,
        dt_validation='exact historical float-seconds subtraction for S23; exact integer-microsecond delta for geometry',
        historical_vs_actual_dt_max_abs_s=float((historical-actual).abs().max()))
