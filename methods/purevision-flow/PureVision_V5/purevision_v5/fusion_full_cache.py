"""Full 582/103-scene frozen 2x3 cache; pilot caches remain immutable references."""
from pathlib import Path
import math
import time
import torch
from torch.utils.data import default_collate

from .common import atomic_save, code_version, environment, now, object_hash, read_json, sha256, write_json
from .datasets import VelocityDataset
from .fusion_abc import SOURCE_SHA
from .fusion_abc_cache import MANIFEST_SHA, fp32_policy, load_cache as load_pilot, row_metadata
from .fusion_grid_cache import (FrozenGridExtractor, PRECISION, feature_shapes, historical_precision_evidence,
                                ordered_sample_digest, pooling_contract)
from .fusion_abc_runtime import runtime_gpu, peak_memory
from .training import model_state_sha256

SCHEMA = 'v5_fusion_full_cache_v1'
COUNTS = {'train_core':132978, 'dev':23480}
SCENE_COUNTS = {'train_core':582, 'dev':103}
SHAPES = feature_shapes((2,3))
ATOL, RTOL = 1e-6, 1e-5


def sample_key(row):
    return row['scene'], row['token']


def identity_map(metadata):
    result = {}
    for i, row in enumerate(metadata):
        key = sample_key(row)
        if key in result:
            raise ValueError(f'Duplicate (scene, token): {key}')
        result[key] = i
    return result


def semantic_metadata(row):
    # dataset_index is local to each scope; every physical sample field must match.
    return {k:v for k,v in row.items() if k != 'dataset_index'}


def full_datasets(manifest, flow, rgb_frames):
    if sha256(manifest) != MANIFEST_SHA:
        raise ValueError('Full scope must use the unchanged original dataset manifest')
    data = {p:VelocityDataset(manifest,p,'R03',flow_root=flow,rgb_policy='finetune',frame_root=rgb_frames)
            for p in COUNTS}
    for part, dataset in data.items():
        if len(dataset) != COUNTS[part] or len(dataset.scene_names) != SCENE_COUNTS[part] or dataset.scene_selection is not None:
            raise ValueError(f'Full scope incomplete or filtered: {part}')
    if data['train_core'].scene_names & data['dev'].scene_names or data['train_core'].source_tokens & data['dev'].source_tokens:
        raise ValueError('Full scene/frame identities cross original partitions')
    return data


def source_packet(dataset, ids=None):
    ids = list(range(len(dataset))) if ids is None else list(ids)
    return dict(metadata=[row_metadata(dataset,i) for i in ids],
        targets=torch.tensor([dataset.items[i]['target'].tolist() for i in ids],dtype=torch.float32),
        features={'dt':torch.tensor([dataset.items[i]['pair_dt'].tolist() for i in ids],dtype=torch.float32)})


def compare_reference(packet, reference, reference_map, *, features=True):
    """Resolve all reference indices by (scene,token), never by full-scope row IDs."""
    pairs = [(i, reference_map[sample_key(row)]) for i,row in enumerate(packet['metadata'])
             if sample_key(row) in reference_map]
    if not pairs:
        return dict(count=0,identity_mapping='scene,token',feature_max_abs_error={})
    full_rows, pilot_rows = zip(*pairs)
    full_rows, pilot_rows = list(full_rows), list(pilot_rows)
    for a,b in pairs:
        if semantic_metadata(packet['metadata'][a]) != semantic_metadata(reference['metadata'][b]):
            raise ValueError('Matched pilot input frames/pairs/direction/timestamps differ')
    if not torch.equal(packet['targets'][full_rows],reference['targets'][pilot_rows]):
        raise ValueError('Identity-matched pilot labels differ')
    if not torch.equal(packet['features']['dt'][full_rows],reference['features']['dt'][pilot_rows]):
        raise ValueError('Identity-matched pilot dt differs')
    errors = {}
    if features:
        for key in SHAPES:
            left, right = packet['features'][key][full_rows], reference['features'][key][pilot_rows]
            torch.testing.assert_close(left,right,atol=ATOL,rtol=RTOL)
            errors[key] = float((left-right).abs().max())
    return dict(count=len(pairs),identity_mapping='scene,token',metadata_equal=True,targets_exact_equal=True,
                dt_exact_equal=True,feature_max_abs_error=errors,atol=ATOL,rtol=RTOL)


def prepare_sources(args):
    data = full_datasets(args.manifest,args.flow,args.rgb_frames)
    references, maps, source, alignment = {}, {}, {}, {}
    index = None
    for part,dataset in data.items():
        references[part],index = load_pilot(args.reference_cache,part)
        maps[part] = identity_map(references[part]['metadata'])
        source[part] = source_packet(dataset)
        complete = identity_map(source[part]['metadata'])
        missing = set(maps[part])-set(complete)
        if missing:
            raise ValueError(f'Pilot samples missing from full {part}: {sorted(missing)[:5]}')
        alignment[part] = compare_reference(source[part],references[part],maps[part],features=False)
        if alignment[part]['count'] != len(references[part]['metadata']):
            raise ValueError('Pilot coverage is incomplete')
    if set(maps['train_core']) & set(maps['dev']):
        raise ValueError('Pilot identities cross partitions')
    selection = dict(schema='v5_full_scene_selection_v1',scope='full',dataset_sha256=MANIFEST_SHA,
        selection_policy='all original train_core/dev sequences in original manifest order; no filtering',
        scenes={p:list(dict.fromkeys(m['scene'] for m in source[p]['metadata'])) for p in COUNTS},
        counts={p:dict(scenes=SCENE_COUNTS[p],windows=COUNTS[p]) for p in COUNTS})
    selection['identity'] = object_hash(selection)
    scope = dict(schema='v5_fusion_full_scope_v1',status='passed',selection=selection,
        manifest=str(Path(args.manifest).resolve()),manifest_sha256=MANIFEST_SHA,
        original_partition_disjoint=True,pilot_reference_identity=index['identity'],pilot_alignment=alignment,
        partitions={p:dict(count=COUNTS[p],scene_count=SCENE_COUNTS[p],
            sample_manifest_sha256=object_hash(source[p]['metadata']),ordered_sample_digest=ordered_sample_digest(source[p]),
            dataset_selection_hash=data[p].selection_hash,input_contract=data[p].contract) for p in COUNTS})
    scope['identity'] = object_hash(scope)
    return data,references,maps,source,index,scope


def validate_packet(packet, count, ids=None):
    if set(packet['features']) != set(SHAPES):
        raise ValueError('Unexpected full-cache feature fields')
    for key,shape in SHAPES.items():
        x=packet['features'][key]
        if x.dtype != torch.float32 or tuple(x.shape)!=(count,*shape) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid full-cache {key}')
    if packet['targets'].dtype != torch.float32 or packet['targets'].shape!=(count,2) or not torch.isfinite(packet['targets']).all():
        raise ValueError('Invalid full-cache targets')
    if len(packet['metadata']) != count or not (packet['features']['dt']>0).all():
        raise ValueError('Invalid full-cache metadata/dt')
    identity_map(packet['metadata'])
    if ids is not None and [m['dataset_index'] for m in packet['metadata']] != list(ids):
        raise ValueError('Full-cache sample order/coverage differs')
    for i,m in enumerate(packet['metadata']):
        if len(m['image_tokens'])!=5 or len(m['pairs'])!=4:
            raise ValueError('Invalid input frame/pair count')
        for t,pair in enumerate(m['pairs']):
            if pair[:2]!=m['image_tokens'][t:t+2] or not math.isclose(pair[3]-pair[2],float(packet['features']['dt'][i,t]),abs_tol=1e-7):
                raise ValueError('Input frame/time/dt correspondence differs')


def load_full_cache(root, partition, allow_preflight=False):
    root=Path(root).resolve();index=read_json(root/'index.json')
    if index.get('schema')!=SCHEMA or index.get('identity')!=object_hash({k:v for k,v in index.items() if k!='identity'}):
        raise ValueError('Invalid full-cache identity')
    if index.get('scope') not in {'full','full_preflight'} or (index['scope']=='full_preflight' and not allow_preflight):
        raise ValueError('Full training cannot use a pilot/preflight cache')
    if index.get('grid')!=[2,3] or index.get('precision')!=PRECISION or index.get('dtype')!='torch.float32':
        raise ValueError('Full-cache grid/precision/dtype differs')
    if index.get('shapes')!={k:list(v) for k,v in SHAPES.items()} or index['source']['checkpoint_sha256']!=SOURCE_SHA:
        raise ValueError('Full-cache shape/source differs')
    selection=index['selection']
    if (selection.get('identity')!=object_hash({k:v for k,v in selection.items() if k!='identity'})
            or selection.get('scope')!='full' or selection.get('dataset_sha256')!=MANIFEST_SHA
            or any(selection['counts'][p]!=dict(scenes=SCENE_COUNTS[p],windows=COUNTS[p]) for p in COUNTS)):
        raise ValueError('Full selection contract differs')
    if partition not in COUNTS or set(index['partitions'])!=set(COUNTS):
        raise ValueError('Unexpected cache partition')
    entry=index['partitions'][partition]
    def file(key,digest):
        path=(root/entry[key]).resolve()
        if not path.is_relative_to(root) or sha256(path)!=entry[digest]:raise ValueError('Full-cache file path/hash changed')
        return path
    features=torch.load(file('file','sha256'),map_location='cpu',weights_only=True)
    annotations=torch.load(file('annotations','annotations_sha256'),map_location='cpu',weights_only=True)
    packet=dict(features=features,**annotations)
    n=entry['count']
    if index['scope']=='full' and (n!=COUNTS[partition] or entry['scene_count']!=SCENE_COUNTS[partition]):
        raise ValueError('Incomplete full cache')
    validate_packet(packet,n,range(n) if index['scope']=='full' else None)
    scenes={m['scene'] for m in packet['metadata']}
    if not scenes<=set(selection['scenes'][partition]) or (index['scope']=='full' and scenes!=set(selection['scenes'][partition])):
        raise ValueError('Scene coverage differs from full original partition')
    if object_hash(packet['metadata'])!=entry['sample_manifest_sha256'] or ordered_sample_digest(packet)!=entry['ordered_sample_digest']:
        raise ValueError('Full ordered metadata/labels/dt identity differs')
    return packet,index


def _extract(extractor,dataset,ids,batch_size,device):
    chunks={k:[] for k in SHAPES}
    for start in range(0,len(ids),batch_size):
        batch=default_collate([dataset[i] for i in ids[start:start+batch_size]])
        features=extractor({k:v.to(device) for k,v in batch['inputs'].items()})
        for k in SHAPES:chunks[k].append(features[k].cpu().contiguous())
    result=source_packet(dataset,ids)
    result['features']={k:torch.cat(v) for k,v in chunks.items()}
    validate_packet(result,len(ids),ids)
    return result


def save_partition(root,part,packet,dataset,full_count):
    feature_file=root/f'{part}_features.pt';annotation_file=root/f'{part}_annotations.pt'
    atomic_save(feature_file,packet['features'])
    atomic_save(annotation_file,{k:packet[k] for k in ('targets','metadata')})
    return dict(file=feature_file.name,sha256=sha256(feature_file),annotations=annotation_file.name,
        annotations_sha256=sha256(annotation_file),count=len(packet['metadata']),full_selection_count=full_count,
        scene_count=len({m['scene'] for m in packet['metadata']}),sample_manifest_sha256=object_hash(packet['metadata']),
        ordered_sample_digest=ordered_sample_digest(packet),dataset_selection_hash=dataset.selection_hash,
        input_contract=dataset.contract,feature_bytes=sum(x.numel()*x.element_size() for x in packet['features'].values()))


def _base_index(args,scope,selection,extractor,code,hardware,reference):
    return dict(schema=SCHEMA,scope=scope,grid=[2,3],source=extractor.provenance,selection=selection,
        code=code,created_at=now(),runtime_gpu=hardware,environment=environment(),precision=PRECISION,
        shapes={k:list(v) for k,v in SHAPES.items()},dtype='torch.float32',partitions={},
        reference_cache=dict(root=str(Path(args.reference_cache).resolve()),identity=reference['identity'],
                             index_sha256=sha256(Path(args.reference_cache)/'index.json')),
        source_implementation_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('fusion_full_cache.py','fusion_grid_cache.py','fusion_abc.py','datasets.py','model.py','rgb.py')},
        label_scope='PLANAR_PROXY; all original train_core/dev; no holdout/test',
        dt_units='raw seconds; normalize only inside readout by historical DT_SCALE_SECONDS',
        pooling=pooling_contract((2,3)))


def _frozen_check(extractor,before):
    after=model_state_sha256(extractor)
    if before!=after or any(m.training for m in extractor.modules()) or any(p.requires_grad for p in extractor.parameters()):
        raise RuntimeError('Frozen encoder/BN state or eval mode changed')
    return after


def preflight(args):
    started=time.monotonic();fp32_policy();torch.set_num_threads(args.threads)
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Preflight output must be new')
    output.mkdir(parents=True,exist_ok=True)
    data,refs,maps,source,reference,scope=prepare_sources(args)
    write_json(output/'full_scope.json',scope)
    hardware=runtime_gpu(args.device)
    if hardware.get('physical_index')!=0:raise ValueError('This short extraction preflight is assigned GPU0')
    torch.cuda.reset_peak_memory_stats(args.device)
    extractor=FrozenGridExtractor(args.checkpoint,(2,3)).to(args.device)
    torch.nn.Sequential(extractor).train()
    before=model_state_sha256(extractor)
    if before!=reference['frozen_state_before_sha256']:raise ValueError('Frozen source state differs from pilot')
    ids={}
    for part,count in [('train_core',33),('dev',3)]:
        complete=identity_map(source[part]['metadata'])
        reference_keys=list(maps[part])
        selected=[complete[reference_keys[0]],complete[reference_keys[-1]]]
        for value in torch.linspace(0,len(data[part])-1,count).long().tolist()+list(range(count)):
            if value not in selected:selected.append(value)
            if len(selected)==count:break
        ids[part]=sorted(selected)
    baseline={p:_extract(extractor,data[p],ids[p],2,args.device) for p in COUNTS}
    overlap={p:compare_reference(baseline[p],refs[p],maps[p]) for p in COUNTS}
    candidates=[];chosen=2;chosen_packets=baseline
    for batch_size in (32,16):
        candidate={p:_extract(extractor,data[p],ids[p],batch_size,args.device) for p in COUNTS}
        evidence={};passed=True
        for part in COUNTS:
            if (candidate[part]['metadata']!=baseline[part]['metadata'] or not torch.equal(candidate[part]['targets'],baseline[part]['targets'])
                    or not torch.equal(candidate[part]['features']['dt'],baseline[part]['features']['dt'])):
                raise ValueError('Batch-size comparison changed sample identity/label/dt')
            evidence[part]={}
            for key in SHAPES:
                a,b=candidate[part]['features'][key],baseline[part]['features'][key]
                good=bool(torch.isclose(a,b,atol=ATOL,rtol=RTOL).all())
                evidence[part][key]=dict(passed=good,max_abs_error=float((a-b).abs().max()),atol=ATOL,rtol=RTOL)
                passed &= good
        candidates.append(dict(batch_size=batch_size,passed=passed,features=evidence,
            batch_lengths={p:[min(batch_size,len(ids[p])-s) for s in range(0,len(ids[p]),batch_size)] for p in COUNTS}))
        if passed:
            chosen=batch_size;chosen_packets=candidate;break
    after=_frozen_check(extractor,before)
    index=_base_index(args,'full_preflight',scope['selection'],extractor,code_version(),hardware,reference)
    index.update(extraction_config=dict(batch_size=chosen,scope='full_preflight'),frozen_state_before_sha256=before,
                 frozen_state_after_sha256=after,full_scope_identity=scope['identity'])
    for part in COUNTS:index['partitions'][part]=save_partition(output,part,chosen_packets[part],data[part],COUNTS[part])
    index['identity']=object_hash(index);write_json(output/'index.json',index)
    for part in COUNTS:load_full_cache(output,part,allow_preflight=True)
    receipt=dict(schema='v5_fusion_full_extraction_preflight_v1',status='passed',created_at=now(),
        checkpoint_sha256=extractor.provenance['checkpoint_sha256'],source=extractor.provenance,
        full_scope=scope,full_scope_file_sha256=sha256(output/'full_scope.json'),reference_cache_identity=reference['identity'],
        historical_precision_evidence=historical_precision_evidence(args.reference_cache),precision=PRECISION,
        step1_batch2_pilot_overlap=overlap,step2_same_gpu_batch_comparison=candidates,selected_batch_size=chosen,
        fixed_full_dataset_indices=ids,frozen_state_before_sha256=before,frozen_state_after_sha256=after,
        output_cache_root=str(output),output_cache_identity=index['identity'],runtime_gpu=hardware,environment=environment(),
        source_implementation_sha256=index['source_implementation_sha256'],
        all_pilot_samples_contained=True,full_scope_counts=COUNTS,full_scope_scene_counts=SCENE_COUNTS,
        elapsed_seconds=time.monotonic()-started,**peak_memory(args.device))
    write_json(output/'extraction_preflight.json',receipt)
    write_json(output/'status.json',dict(status='completed',scope='full_preflight',count=sum(len(x) for x in ids.values())))
    return receipt


def _verify_source_slice(packet,source,start,end):
    if (packet['metadata']!=source['metadata'][start:end] or not torch.equal(packet['targets'],source['targets'][start:end])
            or not torch.equal(packet['features']['dt'],source['features']['dt'][start:end])):
        raise ValueError('Shard identity/order/labels/dt differ from the complete original manifest')


def _load_shard(root,receipt,contract_hash,source,part):
    if receipt.get('identity')!=object_hash({k:v for k,v in receipt.items() if k!='identity'}) or receipt.get('contract_hash')!=contract_hash:
        raise ValueError('Shard receipt/source contract changed')
    file=(root/receipt['file']).resolve()
    if not file.is_relative_to(root) or sha256(file)!=receipt['sha256']:raise ValueError('Shard path/hash changed')
    packet=torch.load(file,map_location='cpu',weights_only=True)
    start,end=receipt['start'],receipt['end']
    if receipt['partition']!=part:raise ValueError('Shard partition changed')
    validate_packet(packet,end-start,range(start,end));_verify_source_slice(packet,source,start,end)
    if ordered_sample_digest(packet)!=receipt['ordered_sample_digest']:raise ValueError('Shard samples changed')
    return packet


def extract_full(args):
    started=time.monotonic();fp32_policy();torch.set_num_threads(args.threads)
    if not args.allow_full_cache:raise ValueError('Full extraction requires --allow-full-cache')
    output=Path(args.output).resolve()
    protected=[Path(v).resolve() for v in (args.checkpoint,args.manifest,args.flow,args.rgb_frames,args.reference_cache,args.preflight_receipt)]
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):raise ValueError('Output overlaps source artifacts')
    if output.exists() and any(output.iterdir()) and not args.resume:raise ValueError('Nonempty full-cache output requires explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    code=code_version(require_pushed=True)
    if not (output/'index.json').exists():
        write_json(output/'status.json',dict(status='running',scope='full',phase='verifying_full_sources',
            updated_at=now(),completed=0,total=sum(COUNTS.values()),code=code,
            extraction_batch_size=args.batch_size,elapsed_seconds=time.monotonic()-started))
    pre=read_json(args.preflight_receipt)
    if pre.get('status')!='passed' or pre.get('selected_batch_size')!=args.batch_size or pre.get('checkpoint_sha256')!=SOURCE_SHA:
        raise ValueError('Extraction batch/source lacks passed precision evidence')
    for name,digest in pre['source_implementation_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=digest:raise ValueError(f'Extraction implementation changed after preflight: {name}')
    if args.shard_size<args.batch_size or args.shard_size%args.batch_size:raise ValueError('Shard size must be a positive extraction-batch multiple')
    data,refs,maps,source,reference,scope=prepare_sources(args)
    if scope['identity']!=pre['full_scope']['identity'] or reference['identity']!=pre['reference_cache_identity']:
        raise ValueError('Full source/pilot identity changed after preflight')
    hardware=runtime_gpu(args.device)
    if hardware.get('uuid')!=pre['runtime_gpu']['uuid']:
        raise ValueError('Large-batch extraction precision was verified on a different GPU')
    torch.cuda.reset_peak_memory_stats(args.device)
    extractor=FrozenGridExtractor(args.checkpoint,(2,3)).to(args.device);torch.nn.Sequential(extractor).train()
    before=model_state_sha256(extractor)
    if before!=pre['frozen_state_before_sha256']:raise ValueError('Frozen source state changed')
    contract=dict(schema='v5_fusion_full_shard_contract_v1',scope='full',grid=[2,3],code_commit=code['commit'],
        checkpoint_sha256=SOURCE_SHA,frozen_state_sha256=before,precision=PRECISION,batch_size=args.batch_size,
        shard_size=args.shard_size,full_scope_identity=scope['identity'],reference_cache_identity=reference['identity'],
        preflight_receipt_sha256=sha256(args.preflight_receipt),runtime_gpu_uuid=hardware['uuid'],
        software=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
    contract_hash=object_hash(contract);contract_file=output/'extraction_contract.json'
    if contract_file.exists() and read_json(contract_file)!=dict(contract=contract,identity=contract_hash):
        raise ValueError('Resume requires the same complete-shard protocol, code, source and precision')
    if not contract_file.exists():write_json(contract_file,dict(contract=contract,identity=contract_hash))
    if (output/'index.json').exists():
        existing=read_json(output/'index.json')
        if existing.get('extraction_contract_hash')!=contract_hash:raise ValueError('Completed cache belongs to another protocol')
        for part in COUNTS:load_full_cache(output,part)
        write_json(output/'status.json',dict(status='completed',scope='full',phase='validated_existing_complete',
            updated_at=now(),completed=sum(COUNTS.values()),total=sum(COUNTS.values()),
            reused_windows=sum(COUNTS.values()),code=code,runtime_gpu=hardware,
            extraction_batch_size=args.batch_size,elapsed_seconds=time.monotonic()-started))
        return existing
    write_json(output/'full_scope.json',scope)
    index=_base_index(args,'full',scope['selection'],extractor,code,hardware,reference)
    index.update(extraction_contract_hash=contract_hash,extraction_config=dict(batch_size=args.batch_size,shard_size=args.shard_size),
        preflight_receipt_sha256=sha256(args.preflight_receipt),full_scope_identity=scope['identity'],
        frozen_state_before_sha256=before,shards={},pilot_overlap={})
    completed=0;reused=0;last_report=0.
    def progress(phase,status='running'):
        write_json(output/'status.json',dict(status=status,scope='full',phase=phase,updated_at=now(),completed=completed,
            total=sum(COUNTS.values()),reused_windows=reused,elapsed_seconds=time.monotonic()-started,
            code=code,runtime_gpu=hardware,extraction_batch_size=args.batch_size,**peak_memory(args.device)))
    progress('loading_verified_sources')
    for part,dataset in data.items():
        receipts=[];parts=[]
        for start in range(0,len(dataset),args.shard_size):
            end=min(start+args.shard_size,len(dataset));stem=f'{part}_{start:06d}_{end:06d}'
            receipt_file=output/'shards'/f'{stem}.json'
            if receipt_file.exists():
                receipt=read_json(receipt_file)
                if receipt['start']!=start or receipt['end']!=end:raise ValueError('Shard boundaries changed')
                packet=_load_shard(output,receipt,contract_hash,source[part],part);reused+=end-start
            else:
                packet=_extract(extractor,dataset,list(range(start,end)),args.batch_size,args.device)
                _verify_source_slice(packet,source[part],start,end)
                overlap=compare_reference(packet,refs[part],maps[part])
                target=output/'shards'/f'{stem}.pt'
                attempt=1
                while target.exists():
                    attempt+=1;target=output/'shards'/f'{stem}_attempt{attempt}.pt'
                atomic_save(target,packet)
                receipt=dict(schema='v5_fusion_full_shard_v1',partition=part,start=start,end=end,count=end-start,
                    file=str(target.relative_to(output)),sha256=sha256(target),contract_hash=contract_hash,
                    ordered_sample_digest=ordered_sample_digest(packet),sample_manifest_sha256=object_hash(packet['metadata']),
                    pilot_overlap=overlap,created_at=now(),runtime_gpu=hardware)
                receipt['identity']=object_hash(receipt);write_json(receipt_file,receipt)
            parts.append(packet);receipts.append(receipt);completed+=end-start
            if time.monotonic()-last_report>=30 or end==len(dataset):
                progress(part);print(f'full cache {completed}/{sum(COUNTS.values())} reused={reused} elapsed={time.monotonic()-started:.1f}s',flush=True)
                last_report=time.monotonic()
        merged=dict(features={k:torch.cat([p['features'][k] for p in parts]) for k in SHAPES},
                    targets=torch.cat([p['targets'] for p in parts]),metadata=[m for p in parts for m in p['metadata']])
        validate_packet(merged,COUNTS[part],range(COUNTS[part]));_verify_source_slice(merged,source[part],0,COUNTS[part])
        overlap=compare_reference(merged,refs[part],maps[part])
        if overlap['count']!=len(refs[part]['metadata']):raise ValueError('Merged pilot overlap incomplete')
        index['partitions'][part]=save_partition(output,part,merged,dataset,COUNTS[part])
        index['pilot_overlap'][part]=overlap
        index['shards'][part]=[dict(start=r['start'],end=r['end'],file=r['file'],sha256=r['sha256'],identity=r['identity']) for r in receipts]
        del merged,parts
    index.update(frozen_state_after_sha256=_frozen_check(extractor,before),elapsed_seconds=time.monotonic()-started,
                 reused_windows=reused,**peak_memory(args.device))
    index['identity']=object_hash(index);write_json(output/'index.json',index)
    for part in COUNTS:load_full_cache(output,part)
    progress('validated_complete','completed')
    return index
