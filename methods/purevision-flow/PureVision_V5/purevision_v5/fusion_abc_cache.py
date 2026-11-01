"""One FP32 feature cache shared by all three frozen-encoder pilot readers."""
from pathlib import Path
import math
import time
import torch
from torch.utils.data import default_collate

from .common import atomic_save, code_version, environment, now, object_hash, read_json, sha256, write_json
from .datasets import VelocityDataset
from .training import model_state_sha256

SCHEMA = 'v5_fusion_abc_cache_v1'
SHAPES = {'F_sp': (4, 6, 32), 'F_tm': (4, 32), 'R': (4, 6, 256), 'dt': (4,)}
COUNTS = {'train_core': 7324, 'dev': 1826}
SELECTION_ID = '83abcfc4565b80a06547aef4ad6bb3c0cd8b4d6a3156c5164653d47f91302b5f'
MANIFEST_SHA = 'a74f7963f7c9e0b529fefef75f5be1c191a7a2fccbddd3678b82f825ee0de32a'


def validate_selection(selection):
    if (selection.get('identity')!=SELECTION_ID or selection.get('dataset_sha256')!=MANIFEST_SHA
            or object_hash({k:v for k,v in selection.items() if k!='identity'})!=SELECTION_ID):
        raise ValueError('Cache must use the fixed original 32/8 selection and manifest')


def fp32_policy():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def datasets(manifest, selection, flow, rgb_frames):
    validate_selection(read_json(selection))
    if sha256(manifest)!=MANIFEST_SHA:raise ValueError('Pilot dataset identity changed')
    data = {p: VelocityDataset(manifest, p, 'R03', flow_root=flow,
            selection_path=selection, rgb_policy='finetune', frame_root=rgb_frames)
            for p in COUNTS}
    for p, d in data.items():
        if len(d) != COUNTS[p] or len(d.scene_names) != (32 if p == 'train_core' else 8):
            raise ValueError(f'Fixed selection changed: {p}={len(d)} windows/{len(d.scene_names)} scenes')
    if data['train_core'].scene_names & data['dev'].scene_names or data['train_core'].source_tokens & data['dev'].source_tokens:
        raise ValueError('Selected input windows cross partitions')
    return data


def row_metadata(dataset, index):
    item = dataset.items[index]
    return dict(scene=item['scene'], token=item['token'], time_s=item['time_s'],
                dataset_index=index, image_tokens=[x[0] for x in item['image_frames']],
                pairs=[list(x) for x in item['pairs']])


def sample_indices(count, limit):
    if not limit: return list(range(count))
    if not 1 <= limit <= count: raise ValueError('Invalid preflight sample count')
    return torch.linspace(0, count-1, limit).long().tolist()


def extract_cache(args):
    from .fusion_abc import FrozenFusionExtractor
    from .fusion_abc_runtime import runtime_gpu, peak_memory
    started=time.monotonic()
    fp32_policy(); torch.set_num_threads(args.threads)
    if args.scope == 'pilot' and not args.allow_pilot_cache:
        raise ValueError('Pilot cache requires explicit --allow-pilot-cache')
    if args.scope not in {'pilot','preflight'}:raise ValueError('Unknown extraction scope')
    if args.scope=='preflight' and not (1<=args.train_samples<=64 and 1<=args.dev_samples<=64):
        raise ValueError('Preflight is bounded to 1..64 samples per partition; zero never means full cache')
    if args.batch_size<1:raise ValueError('Extraction batch size must be positive')
    output = Path(args.output)
    output=output.resolve()
    protected=[Path(v).resolve() for v in [args.checkpoint,args.manifest,args.selection,args.flow,args.rgb_frames]]
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must not overlap existing source artifacts')
    if output.exists() and any(output.iterdir()): raise ValueError('Cache output must be new')
    output.mkdir(parents=True, exist_ok=True)
    code = code_version(require_pushed=args.scope == 'pilot')
    hardware=runtime_gpu(args.device)
    if torch.device(args.device).type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
    total=sum(COUNTS.values()) if args.scope=='pilot' else args.train_samples+args.dev_samples
    processed=0
    def progress(state, phase):
        value=dict(status=state,phase=phase,updated_at=now(),completed=processed,total=total,
            processed_windows=processed,elapsed_seconds=time.monotonic()-started,runtime_gpu=hardware,
            scope=args.scope,code=code,**peak_memory(args.device))
        write_json(output/'status.json',value)
        return value
    progress('running','loading_source')
    data = datasets(args.manifest, args.selection, args.flow, args.rgb_frames)
    extractor = FrozenFusionExtractor(args.checkpoint).to(args.device)
    extractor.train()  # Deliberately exercise the permanent eval override.
    before = model_state_sha256(extractor)
    index = dict(schema=SCHEMA, scope=args.scope, created_at=now(), code=code,
                 source=extractor.provenance, selection=read_json(args.selection),
                 selection_file_sha256=sha256(args.selection),
                 source_implementation_sha256={name: sha256(Path(__file__).with_name(name))
                    for name in ['fusion_abc.py','fusion_abc_cache.py','model.py','rgb.py','datasets.py']},
                 shapes={k:list(v) for k,v in SHAPES.items()}, dtype='torch.float32',
                 dt_units='seconds; normalize only inside fusion model by division by 1/12',
                 label_scope='PLANAR_PROXY; annotations excluded from feature files',
                 environment=environment(), partitions={})
    for partition, dataset in data.items():
        limit = 0 if args.scope == 'pilot' else (args.train_samples if partition == 'train_core' else args.dev_samples)
        ids = sample_indices(len(dataset), limit)
        chunks = {k:[] for k in SHAPES}; targets=[]; metadata=[]
        for start in range(0, len(ids), args.batch_size):
            selected = ids[start:start+args.batch_size]
            batch = default_collate([dataset[i] for i in selected])
            inputs = {k:v.to(args.device) for k,v in batch['inputs'].items()}
            features = extractor(inputs)
            for k in chunks: chunks[k].append(features[k].detach().cpu().float().contiguous())
            targets.append(batch['target']); metadata.extend(row_metadata(dataset,i) for i in selected)
            processed+=len(selected)
            if start//args.batch_size%10==0:
                report=progress('running',partition)
                print(f"cache {processed}/{total} windows elapsed={report['elapsed_seconds']:.1f}s",flush=True)
        features = {k:torch.cat(v) for k,v in chunks.items()}
        annotations = dict(targets=torch.cat(targets), metadata=metadata)
        feature_file = output/f'{partition}_features.pt'; annotation_file=output/f'{partition}_annotations.pt'
        atomic_save(feature_file, features); atomic_save(annotation_file, annotations)
        index['partitions'][partition] = dict(file=feature_file.name, sha256=sha256(feature_file),
            annotations=annotation_file.name, annotations_sha256=sha256(annotation_file), count=len(ids),
            full_selection_count=len(dataset), scene_count=len({m['scene'] for m in metadata}),
            sample_manifest_sha256=object_hash(metadata), dataset_selection_hash=dataset.selection_hash,
            input_contract=dataset.contract)
        print(f'{partition}: extracted {len(ids)} windows ({args.scope})', flush=True)
    after = model_state_sha256(extractor)
    if before != after or any(m.training for m in extractor.modules()) or any(p.requires_grad for p in extractor.parameters()):
        raise RuntimeError('Frozen extractor state/mode changed')
    index['frozen_state_before_sha256']=before; index['frozen_state_after_sha256']=after
    index.update(elapsed_seconds=time.monotonic()-started,runtime_gpu=hardware,**peak_memory(args.device))
    index['identity']=object_hash(index); write_json(output/'index.json', index)
    for part in COUNTS: load_cache(output,part,allow_preflight=args.scope=='preflight')
    progress('completed','validated_complete')
    return index


def load_cache(root, partition, allow_preflight=False):
    from .fusion_abc import SOURCE_SHA, FEATURE_TAP_VERSION
    root=Path(root).resolve(); index=read_json(root/'index.json')
    if index.get('schema')!=SCHEMA or index.get('identity')!=object_hash({k:v for k,v in index.items() if k!='identity'}):
        raise ValueError('Invalid feature cache identity')
    validate_selection(index.get('selection',{}))
    source=index.get('source',{})
    if source.get('checkpoint_sha256')!=SOURCE_SHA or source.get('feature_tap_version')!=FEATURE_TAP_VERSION:
        raise ValueError('Cache encoder source or feature taps differ from the locked pilot')
    if index.get('scope') not in {'pilot','preflight'} or (index['scope']=='preflight' and not allow_preflight):
        raise ValueError('Preflight cache cannot be used for formal pilot training')
    if partition not in COUNTS or set(index.get('partitions',{}))!=set(COUNTS):
        raise ValueError('Only fixed train_core/dev partitions allowed')
    if index.get('dtype')!='torch.float32' or index.get('shapes')!={k:list(v) for k,v in SHAPES.items()}:
        raise ValueError('Cache shape/dtype contract mismatch')
    entry=index['partitions'][partition]
    def verified_file(key, digest):
        p=(root/entry[key]).resolve()
        if not p.is_relative_to(root) or sha256(p)!=entry[digest]:raise ValueError('Cache file path/hash changed')
        return p
    features=torch.load(verified_file('file','sha256'),map_location='cpu',weights_only=True)
    annotations=torch.load(verified_file('annotations','annotations_sha256'),map_location='cpu',weights_only=True)
    n=entry['count']
    if index['scope']=='pilot' and n!=COUNTS[partition]: raise ValueError('Incomplete pilot feature cache')
    if set(features)!=set(SHAPES):raise ValueError('Unexpected feature fields')
    for key, shape in SHAPES.items():
        x=features[key]
        if x.dtype!=torch.float32 or tuple(x.shape)!=(n,*shape) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid cached {key}')
    if not (features['dt']>0).all(): raise ValueError('Nonpositive cached dt')
    target=annotations['targets'];meta=annotations['metadata']
    if target.dtype!=torch.float32 or target.shape!=(n,2) or not torch.isfinite(target).all() or len(meta)!=n:
        raise ValueError('Invalid cached annotations')
    if object_hash(meta)!=entry['sample_manifest_sha256'] or len({m['token'] for m in meta})!=n:
        raise ValueError('Changed or duplicate sample identities')
    allowed=set(index['selection']['scenes'][partition])
    for i,m in enumerate(meta):
        if m['scene'] not in allowed or len(m['pairs'])!=4 or len(m['image_tokens'])!=5:raise ValueError('Sample selection mismatch')
        for t,pair in enumerate(m['pairs']):
            if pair[:2]!=m['image_tokens'][t:t+2] or not math.isclose(pair[3]-pair[2],float(features['dt'][i,t]),abs_tol=1e-7):
                raise ValueError('Pair/RGB/dt correspondence mismatch')
    return dict(features=features, **annotations),index
