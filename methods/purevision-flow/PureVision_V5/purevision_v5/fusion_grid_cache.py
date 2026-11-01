"""Direct frozen-map grid pooling and verified reuse of the historical pilot cache."""
from collections.abc import Mapping
from pathlib import Path
import hashlib
import math
import time

import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from .common import atomic_save, code_version, environment, now, object_hash, read_json, sha256, write_json
from .fusion_abc import FrozenFusionExtractor, SOURCE_SHA, adaptive_pool_bounds
from .fusion_abc_cache import (COUNTS, SCHEMA as LEGACY_SCHEMA, datasets, fp32_policy,
                              load_cache as load_legacy_cache, row_metadata, sample_indices,
                              validate_selection)
from .model import FLOW_SHAPE, MASK_SHAPE, _binary_mask, _finite_tensor
from .rgb import RGB_SHAPE
from .training import model_state_sha256

SCHEMA = 'v5_fusion_grid_cache_v1'
FEATURE_TAP_VERSION = 'frozen_r03_appearance_flex_direct_grid_rgb_t0to3_fp32_v1'
PRECISION = dict(amp=False, autocast=False, matmul_tf32=False, cudnn_tf32=False,
                 float32_matmul_precision='highest', cudnn_benchmark=False,
                 input_dtype='torch.float32', output_dtype='torch.float32')


def feature_shapes(grid):
    grid = tuple(grid)
    if grid not in {(2, 3), (4, 6)}:
        raise ValueError('Only the authorized 2x3 and 4x6 grids are supported')
    regions = math.prod(grid)
    return {'F_sp': (4, regions, 32), 'F_tm': (4, 32), 'R': (4, regions, 256), 'dt': (4,)}


def cache_grid(index):
    return (2, 3) if index.get('schema') == LEGACY_SCHEMA else tuple(index['grid'])


def spatial_tokens(feature_map, grid):
    """Pool each raw temporal map directly; output time, row, column, channel."""
    feature_shapes(grid)
    if feature_map.ndim != 5:
        raise ValueError('Expected B,C,T,H,W raw feature map')
    batch, channels, steps, height, width = feature_map.shape
    slices = feature_map.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
    pooled = F.adaptive_avg_pool2d(slices, tuple(grid))
    return pooled.reshape(batch, steps, channels, math.prod(grid)).permute(0, 1, 3, 2).contiguous()


def pooling_contract(grid):
    rows, cols = grid
    result = dict(implementation='adaptive_avg_pool2d_per_temporal_slice',
        bounds='zero_based_end_exclusive; uneven sizes can overlap',
        flow_sfe_input_hw=[9, 16], flow_rows=adaptive_pool_bounds(9, rows),
        flow_cols=adaptive_pool_bounds(16, cols), rgb_input_hw=[72, 128],
        rgb_rows=adaptive_pool_bounds(72, rows), rgb_cols=adaptive_pool_bounds(128, cols),
        flow_temporal='existing_MFE_spatial_mean_preserves_four_time_positions',
        rgb_temporal='preserve_all_five_then_take_positions_0_1_2_3_no_time_pool',
        rgb_legacy_row_flip=False, directly_from_raw_feature_maps=True)
    result['adjacent_overlap_pixels'] = {key: [max(0, b[1] - c[0]) for b, c in zip(result[key], result[key][1:])]
                                         for key in ('flow_rows', 'flow_cols', 'rgb_rows', 'rgb_cols')}
    return result


class FrozenGridExtractor(FrozenFusionExtractor):
    """Same trained source and permanent eval policy; only spatial pooling changes."""
    def __init__(self, checkpoint, grid=(4, 6)):
        feature_shapes(grid)
        super().__init__(checkpoint)
        self.grid = tuple(grid)
        self.provenance.update(grid=list(grid), feature_tap_version=FEATURE_TAP_VERSION,
            shapes={key: ['N', *shape] for key, shape in feature_shapes(grid).items()},
            tap_source_sha256=sha256(__file__), precision=PRECISION,
            raw_feature_shapes={'F_sp': ['N', 32, 4, 9, 16], 'F_tm': ['N', 32, 4, 1, 1],
                                'R': ['N', 256, 5, 72, 128]},
            spatial_alignment='coarse_logical_grid_not_pixel_alignment', pooling=pooling_contract(grid))

    @torch.no_grad()
    def forward(self, inputs, include_legacy=False):
        self.train(False)
        if not isinstance(inputs, Mapping) or set(inputs) != {'images', 'flows', 'valid', 'pair_dt'}:
            raise ValueError('Only images, flows, valid and pair_dt may enter the extractor')
        dt = inputs['pair_dt']
        if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or len(dt) < 1:
            raise ValueError('Expected nonempty B,4 pair_dt')
        batch = len(dt)
        _finite_tensor(dt, (batch, 4), 'pair_dt')
        images = _finite_tensor(inputs['images'], (batch, *RGB_SHAPE), 'images')
        flow = inputs['flows']
        if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (batch, *FLOW_SHAPE):
            raise ValueError('Unexpected flow shape')
        valid = _binary_mask(inputs['valid'], (batch, *MASK_SHAPE), 'valid')
        if any(value.device != dt.device for value in (images, flow, valid)):
            raise ValueError('Inputs must share one device')
        if any(value.dtype != torch.float32 for value in (images, flow, dt)):
            raise ValueError('Historical extraction requires FP32 inputs')
        if bool((dt <= 0).any()) or bool((images.abs() > 1.00001).any()):
            raise ValueError('Invalid seconds or RGB preprocessing')
        if torch.is_autocast_enabled(dt.device.type):
            raise ValueError('AMP extraction is forbidden by the historical cache policy')
        valid = valid & torch.isfinite(flow).all(dim=2, keepdim=True)
        safe = torch.where(valid.expand_as(flow), flow, torch.zeros_like(flow))
        encoded = torch.cat((safe, valid.to(flow.dtype), valid.to(flow.dtype)), dim=2)
        encoder = self.source.flow_encoder
        stem = encoder.spatial(encoded.reshape(batch * 4, 4, 72, 128))
        stem = stem.reshape(batch, 4, 32, *stem.shape[-2:]).transpose(1, 2)
        if tuple(stem.shape[1:]) != (32, 4, 9, 16):
            raise RuntimeError('Unexpected flow stem geometry')
        spatial, temporal = encoder.sfe(stem), encoder.temporal(stem)
        rgb = self.source.rgb_encoder.backbone.forward_feature_map(images)
        if tuple(spatial.shape[1:]) != (32, 4, 9, 16) or tuple(temporal.shape[1:]) != (32, 4, 1, 1):
            raise RuntimeError('Unexpected SFE/MFE geometry')
        if tuple(rgb.shape[1:]) != (256, 5, 72, 128):
            raise RuntimeError('RGB must encode all five frames without temporal pooling')
        def pooled(grid):
            return {'F_sp': spatial_tokens(spatial, grid),
                    'F_tm': temporal.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
                    'R': spatial_tokens(rgb, grid)[:, :4].contiguous(), 'dt': dt.clone()}
        result = pooled(self.grid)
        for key, shape in feature_shapes(self.grid).items():
            value = _finite_tensor(result[key], (batch, *shape), key)
            if value.dtype != torch.float32:
                raise RuntimeError('Extraction output dtype changed')
        return (result, pooled((2, 3))) if include_legacy else result


def ordered_sample_digest(packet):
    """Bind real ordered metadata, all labels and seconds, not shuffle indices."""
    digest = hashlib.sha256(object_hash(packet['metadata']).encode())
    for value in (packet['targets'], packet['features']['dt']):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_grid_cache(root, partition, allow_preflight=False):
    root = Path(root).resolve()
    index = read_json(root / 'index.json')
    if index.get('schema') == LEGACY_SCHEMA:
        return load_legacy_cache(root, partition, allow_preflight=allow_preflight)
    if index.get('schema') != SCHEMA or index.get('identity') != object_hash({k: v for k, v in index.items() if k != 'identity'}):
        raise ValueError('Invalid grid cache identity')
    validate_selection(index.get('selection', {}))
    grid = cache_grid(index)
    shapes = feature_shapes(grid)
    if index.get('shapes') != {k: list(v) for k, v in shapes.items()} or index.get('dtype') != 'torch.float32':
        raise ValueError('Grid cache shape/dtype mismatch')
    if index.get('precision') != PRECISION or index.get('source', {}).get('checkpoint_sha256') != SOURCE_SHA:
        raise ValueError('Extraction precision/source differs from the locked source')
    if index['source'].get('feature_tap_version') != FEATURE_TAP_VERSION or index['source'].get('pooling') != pooling_contract(grid):
        raise ValueError('Raw-map pooling contract differs')
    if index.get('scope') not in {'pilot', 'preflight'} or (index['scope'] == 'preflight' and not allow_preflight):
        raise ValueError('Preflight cache is not a formal pilot')
    if partition not in COUNTS or set(index.get('partitions', {})) != set(COUNTS):
        raise ValueError('Only fixed train_core/dev are supported')
    entry = index['partitions'][partition]
    def verified_file(key, digest):
        path = (root / entry[key]).resolve()
        if not path.is_relative_to(root) or sha256(path) != entry[digest]:
            raise ValueError('Cache file path/hash mismatch')
        return path
    features = torch.load(verified_file('file', 'sha256'), map_location='cpu', weights_only=True)
    annotations = torch.load(verified_file('annotations', 'annotations_sha256'), map_location='cpu', weights_only=True)
    n = entry['count']
    if index['scope'] == 'pilot' and n != COUNTS[partition]:
        raise ValueError('Incomplete pilot cache')
    if set(features) != set(shapes):
        raise ValueError('Unexpected cache feature fields')
    for key, shape in shapes.items():
        value = features[key]
        if value.dtype != torch.float32 or tuple(value.shape) != (n, *shape) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid cached {key}')
    if not (features['dt'] > 0).all():
        raise ValueError('Nonpositive cached dt')
    target, meta = annotations['targets'], annotations['metadata']
    if target.dtype != torch.float32 or target.shape != (n, 2) or not torch.isfinite(target).all() or len(meta) != n:
        raise ValueError('Invalid annotations')
    if object_hash(meta) != entry['sample_manifest_sha256'] or len({m['token'] for m in meta}) != n:
        raise ValueError('Changed or duplicate sample identities')
    allowed = set(index['selection']['scenes'][partition])
    for i, m in enumerate(meta):
        if m['scene'] not in allowed or len(m['pairs']) != 4 or len(m['image_tokens']) != 5:
            raise ValueError('Sample selection mismatch')
        for t, pair in enumerate(m['pairs']):
            if pair[:2] != m['image_tokens'][t:t+2] or not math.isclose(pair[3] - pair[2], float(features['dt'][i, t]), abs_tol=1e-7):
                raise ValueError('Input frames/timestamps/dt mismatch')
    packet = dict(features=features, **annotations)
    if ordered_sample_digest(packet) != entry['ordered_sample_digest']:
        raise ValueError('Ordered samples/labels/dt differ from the declared identity')
    return packet, index


# Same signature as the historical loader, for callers migrating incrementally.
load_cache = load_grid_cache


def historical_precision_evidence(root):
    """Verify the immutable historical extraction source which declares FP32 policy."""
    index = read_json(Path(root) / 'index.json')
    if index.get('schema') != LEGACY_SCHEMA or index.get('scope') != 'pilot':
        raise ValueError('Reference must be the complete original 2x3 pilot')
    # Historical frozen source was preserved by the original pipeline; verify its bytes.
    art = Path(root).parents[1]
    candidates = sorted((art / 'code').glob(f'*{index["code"]["commit"][:7]}*/methods/purevision-flow/PureVision_V5/purevision_v5'))
    expected = index['source_implementation_sha256']
    verified = None
    for source in candidates:
        if all((source / name).is_file() and sha256(source / name) == expected[name]
               for name in ('fusion_abc.py', 'fusion_abc_cache.py')):
            verified = source
            break
    if verified is None:
        raise ValueError('Cannot verify the preserved historical extraction implementation')
    extraction = (verified / 'fusion_abc_cache.py').read_text()
    taps = (verified / 'fusion_abc.py').read_text()
    checks = ["torch.set_float32_matmul_precision('highest')", 'torch.backends.cuda.matmul.allow_tf32 = False',
              'torch.backends.cudnn.allow_tf32 = False', 'fp32_policy();', 'features = extractor(inputs)']
    if not all(text in extraction for text in checks) or 'torch.autocast' in extraction or 'torch.autocast' in taps:
        raise ValueError('Historical extraction precision requires review')
    return dict(precision=PRECISION, verified_source_directory=str(verified),
                source_implementation_sha256={key: expected[key] for key in ('fusion_abc.py', 'fusion_abc_cache.py')},
                reference_cache_identity=index['identity'], reference_extraction_batch_size=2,
                note='Source encoder training AMP is separate; historical feature extraction was FP32 without AMP/TF32')


def _reference_matches_dataset(dataset, reference, entry):
    metadata = [row_metadata(dataset, i) for i in range(len(dataset))]
    targets = torch.tensor([item['target'].tolist() for item in dataset.items], dtype=torch.float32)
    dt = torch.tensor([item['pair_dt'].tolist() for item in dataset.items], dtype=torch.float32)
    if metadata != reference['metadata'] or not torch.equal(targets, reference['targets']) or not torch.equal(dt, reference['features']['dt']):
        raise ValueError('Historical samples/frames/timestamps/labels/dt differ from current fixed dataset')
    if dataset.contract != entry['input_contract'] or dataset.selection_hash != entry['dataset_selection_hash']:
        raise ValueError('Historical raw input or preprocessing contract changed')
    return dict(count=len(dataset), ordered_metadata_equal=True, targets_exact_equal=True, dt_exact_equal=True,
                input_contract_equal=True, ordered_sample_digest=ordered_sample_digest(reference))


def cross_grid_evidence(root23, root46, allow_preflight=False):
    result = {}
    for part in COUNTS:
        a, ia = load_grid_cache(root23, part, allow_preflight)
        b, ib = load_grid_cache(root46, part, allow_preflight)
        if cache_grid(ia) != (2, 3) or cache_grid(ib) != (4, 6):
            raise ValueError('Expected 2x3 and 4x6 caches')
        ids = [m['dataset_index'] for m in b['metadata']]
        if [a['metadata'][i] for i in ids] != b['metadata'] or not torch.equal(a['targets'][ids], b['targets']):
            raise ValueError('Cross-grid actual samples/labels differ')
        if not torch.equal(a['features']['dt'][ids], b['features']['dt']):
            raise ValueError('Cross-grid dt differs')
        result[part] = dict(count=len(ids), metadata_exact_equal=True, labels_exact_equal=True,
                            dt_exact_equal=True, ordered_sample_digest=ordered_sample_digest(b))
    return result


def extract_cache(args):
    from .fusion_abc_runtime import runtime_gpu, peak_memory
    started = time.monotonic()
    fp32_policy()
    torch.set_num_threads(args.threads)
    grid = tuple(args.grid)
    shapes = feature_shapes(grid)
    if grid != (4, 6):
        raise ValueError('Reuse the verified original 2x3 cache; new extraction is only 4x6')
    if args.scope not in {'pilot', 'preflight'} or (args.scope == 'pilot' and not args.allow_pilot_cache):
        raise ValueError('Formal extraction requires the explicit pilot flag')
    if args.scope == 'preflight' and not (1 <= args.train_samples <= 64 and 1 <= args.dev_samples <= 64):
        raise ValueError('Preflight is bounded to 1..64 samples per partition')
    if args.batch_size != 2 or args.threads < 1:
        raise ValueError('Preserve historical extraction batch size 2; threads must be positive')
    output = Path(args.output).resolve()
    protected = [Path(v).resolve() for v in (args.checkpoint, args.manifest, args.selection, args.flow, args.rgb_frames, args.reference_cache)]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('Output must not overlap source artifacts')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Extraction output must be new')
    output.mkdir(parents=True, exist_ok=True)
    code = code_version(require_pushed=args.scope == 'pilot')
    hardware = runtime_gpu(args.device)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    total = sum(COUNTS.values()) if args.scope == 'pilot' else args.train_samples + args.dev_samples
    processed, last_report = 0, 0.0
    def progress(status, phase):
        value = dict(status=status, phase=phase, updated_at=now(), completed=processed, total=total,
                     processed_windows=processed, elapsed_seconds=time.monotonic() - started,
                     runtime_gpu=hardware, scope=args.scope, grid=list(grid), code=code, **peak_memory(args.device))
        write_json(output / 'status.json', value)
        return value
    progress('running', 'verifying_historical_source')
    precision_evidence = historical_precision_evidence(args.reference_cache)
    references, reference_index = {}, None
    for part in COUNTS:
        references[part], reference_index = load_legacy_cache(args.reference_cache, part)
    data = datasets(args.manifest, args.selection, args.flow, args.rgb_frames)
    alignment = {part: _reference_matches_dataset(dataset, references[part], reference_index['partitions'][part])
                 for part, dataset in data.items()}
    extractor = FrozenGridExtractor(args.checkpoint, grid).to(args.device)
    # Exercise the outer train() override and include every BN buffer in the state hash.
    torch.nn.Sequential(extractor).train()
    before = model_state_sha256(extractor)
    if before != reference_index['frozen_state_before_sha256']:
        raise ValueError('Frozen encoder parameter/buffer state differs from historical extraction')
    index = dict(schema=SCHEMA, scope=args.scope, grid=list(grid), created_at=now(), code=code,
        source=extractor.provenance, selection=read_json(args.selection), selection_file_sha256=sha256(args.selection),
        source_implementation_sha256={name: sha256(Path(__file__).with_name(name)) for name in
            ('fusion_abc.py', 'fusion_grid_cache.py', 'fusion_abc_cache.py', 'model.py', 'rgb.py', 'datasets.py')},
        shapes={key: list(shape) for key, shape in shapes.items()}, dtype='torch.float32', precision=PRECISION,
        extraction_config=dict(batch_size=args.batch_size, threads=args.threads, grid=list(grid), scope=args.scope),
        reference_cache=dict(root=str(Path(args.reference_cache).resolve()), identity=reference_index['identity'],
                             index_sha256=sha256(Path(args.reference_cache) / 'index.json')),
        historical_precision_evidence=precision_evidence, full_reference_alignment=alignment,
        dt_units='seconds; normalize only inside fusion model by division by 1/12',
        label_scope='PLANAR_PROXY; annotations excluded from feature files', environment=environment(), partitions={})
    numerical = {}
    for part, dataset in data.items():
        limit = 0 if args.scope == 'pilot' else (args.train_samples if part == 'train_core' else args.dev_samples)
        ids = sample_indices(len(dataset), limit)
        chunks = {key: [] for key in shapes}
        targets, metadata = [], []
        reference = references[part]
        for start in range(0, len(ids), args.batch_size):
            selected = ids[start:start + args.batch_size]
            batch = default_collate([dataset[i] for i in selected])
            inputs = {key: value.to(args.device) for key, value in batch['inputs'].items()}
            check = args.scope == 'preflight' or start == 0
            extracted = extractor(inputs, include_legacy=check)
            features, old = extracted if check else (extracted, None)
            if check:
                evidence = {}
                for key in shapes:
                    observed, expected = old[key].cpu(), reference['features'][key][selected]
                    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
                    evidence[key] = dict(max_abs_error=float((observed - expected).abs().max()),
                                         exact_equal=torch.equal(observed, expected), atol=1e-6, rtol=1e-5)
                numerical[f'{part}:{start}'] = dict(dataset_indices=selected, features=evidence)
            if not torch.equal(features['dt'].cpu(), reference['features']['dt'][selected]) or not torch.equal(batch['target'], reference['targets'][selected]):
                raise ValueError('Extracted actual batch dt/labels differ from the historical cache')
            for key in chunks:
                chunks[key].append(features[key].detach().cpu().float().contiguous())
            targets.append(batch['target'])
            metadata.extend(row_metadata(dataset, i) for i in selected)
            processed += len(selected)
            if time.monotonic() - last_report >= 30 or processed == total:
                report = progress('running', part)
                print(f"cache {processed}/{total} windows elapsed={report['elapsed_seconds']:.1f}s", flush=True)
                last_report = time.monotonic()
        features = {key: torch.cat(values) for key, values in chunks.items()}
        annotations = dict(targets=torch.cat(targets), metadata=metadata)
        packet = dict(features=features, **annotations)
        feature_file, annotation_file = output / f'{part}_features.pt', output / f'{part}_annotations.pt'
        atomic_save(feature_file, features)
        atomic_save(annotation_file, annotations)
        index['partitions'][part] = dict(file=feature_file.name, sha256=sha256(feature_file),
            annotations=annotation_file.name, annotations_sha256=sha256(annotation_file), count=len(ids),
            full_selection_count=len(dataset), scene_count=len({m['scene'] for m in metadata}),
            sample_manifest_sha256=object_hash(metadata), ordered_sample_digest=ordered_sample_digest(packet),
            feature_bytes=sum(t.numel() * t.element_size() for t in features.values()),
            dataset_selection_hash=dataset.selection_hash, input_contract=dataset.contract)
        progress('running', f'{part}_saved')
        print(f'{part}: extracted {len(ids)} windows ({args.scope})', flush=True)
    after = model_state_sha256(extractor)
    if before != after or any(m.training for m in extractor.modules()) or any(p.requires_grad for p in extractor.parameters()):
        raise RuntimeError('Frozen parameter/BN state or mode changed')
    index.update(frozen_state_before_sha256=before, frozen_state_after_sha256=after,
                 frozen_eval=True, frozen_requires_grad=False, historical_2x3_numerical_check=numerical,
                 elapsed_seconds=time.monotonic() - started, runtime_gpu=hardware, **peak_memory(args.device))
    index['identity'] = object_hash(index)
    write_json(output / 'index.json', index)
    cross = cross_grid_evidence(args.reference_cache, output, allow_preflight=args.scope == 'preflight')
    write_json(output / 'cross_grid_evidence.json', dict(created_at=now(), reference_identity=reference_index['identity'],
               grid_identity=index['identity'], partitions=cross))
    progress('completed', 'validated_complete')
    return index
