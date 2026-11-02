"""Twenty-epoch S23 continuation with full-model, atomic epoch-boundary recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .common import (ROOT, atomic_save, code_version, environment, git, now, object_hash,
                     read_json, sha256, write_json)
from .evaluation import validate_lateral_rule
from .fusion_abc import SOURCE_SHA
from .fusion_abc_runtime import peak_memory, runtime_gpu
from .fusion_abc_training import configure_fp32
from .fusion_full_cache import (COUNTS, SCENE_COUNTS, full_datasets, identity_map,
                               load_full_cache, semantic_metadata, source_packet)
from .fusion_grid_cache import ordered_sample_digest
from .fusion_grid_training import (batch, order_record, prediction_rows, resident_packet,
                                   sample_identities, summarize_prediction, write_history)
from .fusion_tail import GROUPS, PARENT_EPE, READOUT_SHA, S23TailModel
from .training import epoch_order, model_state_sha256, restore_rng, rng_state, seed_everything, write_predictions

SCHEMA = 'v5_fusion_tail_epoch_state_v1'
RECOVERY_SCHEMA = 'v5_fusion_tail_recovery_v1'
CONTRACT_SCOPE = 'full_tail_stage'
BOUNDARY = 'validation_then_scheduler_then_advance_epoch_next_batch_zero'
PROTOCOL = dict(schema='v5_fusion_tail_config_v1', scope='full', seed=42, epochs=20,
    batch_size=64, drop_last=False, amp=False, tf32=False, readout_learning_rate=1e-4,
    encoder_learning_rate=1e-5, weight_decay=1e-4, betas=[.9, .999], optimizer_eps=1e-8,
    grad_clip=2., scheduler_epochs=20, eta_min=0., loss_scales_m_s=[5., 1.], loss_beta=1.,
    early_stopping=False, expected_counts=COUNTS, expected_scene_counts=SCENE_COUNTS)


def validate_config(config):
    for key, expected in PROTOCOL.items():
        if config.get(key) != expected or (isinstance(expected, bool) and type(config.get(key)) is not bool):
            raise ValueError(f'Fixed S23 stage protocol differs: {key}')
    if type(config['seed']) is not int or config.get('group') not in GROUPS:
        raise ValueError('Only the two seed42 S23 continuation groups are authorized')
    if validate_lateral_rule(config.get('lateral_rule')) != validate_lateral_rule():
        raise ValueError('The historical lateral rule must be preserved')


def source_digest():
    return {name: sha256(Path(__file__).with_name(name)) for name in
        ('fusion_tail.py', 'fusion_tail_training.py', 'fusion_grid.py', 'datasets.py', 'model.py', 'rgb.py')}


def validate_verification(path, cache_identity, microbatch):
    value = read_json(path)
    if (value.get('schema') != 'v5_s23_tail_model_preflight_v1' or value.get('status') != 'passed'
            or value.get('parent_encoder_sha256') != SOURCE_SHA
            or value.get('parent_readout_sha256') != READOUT_SHA
            or value.get('cache_identity') != cache_identity
            or value.get('selected_microbatch') != microbatch
            or value.get('epoch0_reference', {}).get('reuse_parent_published_metrics') is not True):
        raise ValueError('Required parent, online-path and microbatch preflight evidence differs')
    # The model proof must describe exactly the implementation used for this run.
    for name, digest in value.get('source_sha256', {}).items():
        file = Path(__file__).with_name(name)
        if file.exists() and sha256(file) != digest:
            raise ValueError(f'Model preflight is stale: {name}')
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), identity=value.get('identity'),
                selected_microbatch=microbatch, epoch0_reference=value['epoch0_reference'])


def parent_reference(readout_checkpoint, provenance):
    root = Path(readout_checkpoint).resolve().parent
    result = dict(run_id=provenance['readout']['source_run_id'], selected_parent_epoch=19,
        parent_budget_epochs=20, parent_budget_optimizer_updates=41560, stage_budget_epochs=20,
        stage_budget_optimizer_updates=41560, dev_epe=PARENT_EPE, checkpoint_sha256=READOUT_SHA,
        policy='reuse verified published parent metrics after initial online/cache equivalence', evaluations={})
    for partition in COUNTS:
        base = root / f'eval_{partition}'
        receipt = read_json(base / 'evaluation.json')
        summary = read_json(base / 'metrics.json')
        if (receipt.get('status') != 'completed' or receipt.get('checkpoint_sha256') != READOUT_SHA
                or receipt.get('count') != COUNTS[partition] or summary['overall']['count'] != COUNTS[partition]
                or receipt.get('contract_hash') != provenance['readout']['source_contract_hash']
                or receipt.get('cache_identity') != provenance['readout']['cache_identity']
                or sha256(base / 'predictions.csv') != receipt['predictions_sha256']):
            raise ValueError('Published epoch0 metrics are not bound to the specified full-data parent')
        if partition == 'dev' and summary['overall']['epe'] != PARENT_EPE:
            raise ValueError('Parent dev EPE differs')
        result['dev' if partition == 'dev' else 'train'] = summary
        result['evaluations'][partition] = dict(path=str(base), receipt_sha256=sha256(base / 'evaluation.json'),
            metrics_sha256=sha256(base / 'metrics.json'), predictions_sha256=receipt['predictions_sha256'])
    return result


class CanonicalDataset(Dataset):
    """Resolve physical raw rows explicitly into the verified cache order once."""
    def __init__(self, dataset, packet):
        source = source_packet(dataset)
        mapping = identity_map(source['metadata'])
        canonical = identity_map(packet['metadata'])
        self.raw_indices = [mapping[(m['scene'], m['token'])] for m in packet['metadata']]
        for i, raw in enumerate(self.raw_indices):
            if semantic_metadata(packet['metadata'][i]) != semantic_metadata(source['metadata'][raw]):
                raise ValueError('Raw/cache frame, pair or timestamp identity differs')
        chosen = torch.tensor(self.raw_indices, dtype=torch.long)
        if (not torch.equal(packet['targets'], source['targets'][chosen])
                or not torch.equal(packet['features']['dt'], source['features']['dt'][chosen])):
            raise ValueError('Raw/cache labels or dt differ')
        self.dataset = dataset
        self.evidence = dict(count=len(canonical), identity_mapping='scene,token', metadata_equal=True,
            targets_exact_equal=True, dt_exact_equal=True,
            cache_to_raw_index_sha256=object_hash(self.raw_indices),
            ordered_sample_digest=ordered_sample_digest(packet), raw_input_contract=dataset.contract)

    def __len__(self):
        return len(self.raw_indices)

    def __getitem__(self, index):
        value = self.dataset[self.raw_indices[index]]
        return dict(inputs=value['inputs'], target=value['target'], canonical_index=index)


class EffectiveBatchSampler(Sampler):
    """Microbatches never straddle a 64-sample effective-update boundary."""
    def __init__(self, count, microbatch, effective_batch=64):
        self.order = list(range(count))
        self.microbatch, self.effective_batch = microbatch, effective_batch

    def __iter__(self):
        for start in range(0, len(self.order), self.effective_batch):
            chosen = self.order[start:start + self.effective_batch]
            for offset in range(0, len(chosen), self.microbatch):
                yield chosen[offset:offset + self.microbatch]

    def __len__(self):
        return sum(math.ceil(min(self.effective_batch, len(self.order)-i) / self.microbatch)
                   for i in range(0, len(self.order), self.effective_batch))


def online_loader(dataset, microbatch, workers, device):
    sampler = EffectiveBatchSampler(len(dataset), microbatch)
    # Loader/worker seeds cannot consume model RNG or the independent epoch-order RNG.
    generator = torch.Generator().manual_seed(1042)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
        pin_memory=device.type == 'cuda', persistent_workers=workers > 0, generator=generator,
        **({'prefetch_factor': 2} if workers else {}))
    return loader, sampler, generator


def online_microbatches(iterator, chosen, microbatch, device):
    for offset in range(0, len(chosen), microbatch):
        expected = chosen[offset:offset + microbatch]
        value = next(iterator)
        if value['canonical_index'].tolist() != expected:
            raise RuntimeError('Raw loader changed actual effective-batch sample order')
        yield ({k: v.to(device, non_blocking=True) for k, v in value['inputs'].items()},
               value['target'].to(device, dtype=torch.float32, non_blocking=True))


def effective_update(model, optimizer, microbatches, effective_count, config, cached=False):
    """One true update, including the 50-sample final effective batch."""
    optimizer.zero_grad(set_to_none=True)
    observed = 0
    totals = None
    for values, target in microbatches:
        prediction = model.forward_cached(values) if cached else model(values)
        if prediction.shape != target.shape:
            raise RuntimeError('Prediction shape differs from physical two-axis labels')
        n = len(target)
        scale = target.new_tensor(config['loss_scales_m_s'])
        loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=config['loss_beta'], reduction='mean')
        # Each mean spans both axes; scaling by n / actual_effective_count is
        # essential for a short microbatch and for the final effective batch of 50.
        (loss * (n / effective_count)).backward()
        with torch.no_grad():
            axes = F.smooth_l1_loss(prediction.detach()/scale, target/scale,
                                   beta=config['loss_beta'], reduction='none').mean(0)
            observation = torch.cat((loss.detach().reshape(1), axes)).double() * n
            totals = observation if totals is None else totals + observation
        observed += n
    if observed != effective_count or totals is None or not torch.isfinite(totals).all():
        raise RuntimeError('Incomplete effective batch or nonfinite loss')
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return totals, grad_norm


def evaluate_packet(model, packet, runtime, online, device, config, progress=None):
    model.eval()
    outputs = []
    last_progress = time.monotonic()
    with torch.no_grad():
        if online is None:
            for start in range(0, len(packet['targets']), config['batch_size']):
                values, _ = batch(runtime, slice(start, start+config['batch_size']), device)
                outputs.append(model.forward_cached(values).detach())
                if progress and time.monotonic()-last_progress >= 30:
                    progress(); last_progress = time.monotonic()
        else:
            loader, sampler, _ = online
            sampler.order = list(range(len(packet['targets'])))
            position = 0
            for value in loader:
                ids = value['canonical_index'].tolist()
                if ids != list(range(position, position+len(ids))):
                    raise RuntimeError('Online evaluation order differs from canonical samples')
                inputs = {k: v.to(device, non_blocking=True) for k, v in value['inputs'].items()}
                outputs.append(model(inputs).detach())
                position += len(ids)
                if progress and time.monotonic()-last_progress >= 30:
                    progress(); last_progress = time.monotonic()
    prediction = torch.cat(outputs).float().cpu()
    return summarize_prediction(prediction, packet, config), prediction


def frozen_state_digest(model, manifest):
    names = {p['name'] for p in manifest['parameters'] if not p['trainable']}
    names.update(b['name'] for b in manifest['buffers'])
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in names:
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, config):
    return torch.optim.AdamW(model.optimizer_groups(config['readout_learning_rate'], config['encoder_learning_rate']),
        weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['optimizer_eps'])


def validate_checkpoint(value, contract_hash=None):
    if value.get('schema') != SCHEMA or value.get('contract_hash') != object_hash(value.get('contract', {})):
        raise ValueError('Invalid full S23 stage checkpoint contract')
    if contract_hash is not None and value['contract_hash'] != contract_hash:
        raise ValueError('Resume configuration/data/stage contract differs')
    config, contract = value['config'], value['contract']
    validate_config(config)
    if (config != contract['config'] or value['cache_identity'] != contract['cache_identity']
            or contract.get('scope') not in {CONTRACT_SCOPE, 'tail_preflight'}
            or contract.get('expected_counts') != COUNTS or contract.get('expected_scene_counts') != SCENE_COUNTS
            or value['train_count'] != contract['partition_counts']['train_core']
            or value['dev_count'] != contract['partition_counts']['dev']):
        raise ValueError('Checkpoint lost complete-data stage identity')
    if contract['scope'] == CONTRACT_SCOPE and (contract.get('cache_scope') != 'full'
            or contract['partition_counts'] != COUNTS or contract['scene_counts'] != SCENE_COUNTS):
        raise ValueError('Formal stage cannot consume partial data')
    provenance, freeze = value['provenance'], value['freeze_manifest']
    if (provenance['encoder']['checkpoint_sha256'] != SOURCE_SHA
            or provenance['readout']['checkpoint_sha256'] != READOUT_SHA
            or provenance['readout']['selected_epoch'] != 19 or provenance['stage_budget_epochs'] != 20
            or provenance['stage_optimizer_updates'] != 41560 or freeze['group'] != config['group']
            or freeze['identity'] != object_hash({k: v for k, v in freeze.items() if k != 'identity'})
            or contract['freeze_manifest_identity'] != freeze['identity']):
        raise ValueError('Parent/freeze/stage provenance differs')
    entries = freeze['parameters'] + freeze['buffers']
    shapes = {p['name']: p['shape'] for p in entries}
    if set(value['model']) != set(shapes) or any(list(value['model'][k].shape) != s for k, s in shapes.items()):
        raise ValueError('Checkpoint does not contain the complete encoder, legacy and S23 state')
    if not all(any(k.startswith(prefix) for k in shapes) for prefix in
               ('flow_encoder.', 'rgb_encoder.', 'legacy_head.', 'readout.')):
        raise ValueError('Readout-only checkpoint is not independently deployable')
    completed = value['completed_epochs']
    batches = math.ceil(value['train_count'] / config['batch_size'])
    if (not 1 <= completed <= 20 or value['epoch_index'] != completed or value['next_batch'] != 0
            or value['optimizer_updates'] != completed*batches or value['processed_batches'] != completed*batches
            or value['amp_skips'] != 0 or value['amp_enabled'] is not False or value['scaler'] != {}
            or value['scheduler']['last_epoch'] != completed or value['scheduler']['T_max'] != 20
            or value['scheduler']['eta_min'] != 0 or len(value['history']) != completed
            or len(value['order_records']) != completed or value['boundary'] != BOUNDARY
            or value['dev_prediction'].shape != (value['dev_count'], 2)
            or value['dev_summary']['overall']['count'] != value['dev_count']):
        raise ValueError('Recovery requires the entire validated epoch, scheduler, RNG and optimizer boundary')
    if any(row['epoch'] != i+1 or row['optimizer_updates'] != (i+1)*batches
           or row['dev']['overall']['count'] != value['dev_count'] for i, row in enumerate(value['history'])):
        raise ValueError('Epoch history coverage or effective updates differ')
    if any(row['epoch'] != i+1 or row['sample_count'] != value['train_count']
           or len(row['batch_sizes']) != batches or sum(row['batch_sizes']) != value['train_count']
           or len(row['actual_batch_identity_sha256']) != batches for i, row in enumerate(value['order_records'])):
        raise ValueError('Actual effective-batch identity coverage differs')
    best = min(value['history'], key=lambda row: row['dev']['overall']['epe'])
    if value['best_epoch'] != best['epoch'] or value['best_epe'] != best['dev']['overall']['epe']:
        raise ValueError('Stage best must be earliest strict improvement among epochs 1..20')
    groups = value['optimizer']['param_groups']
    names = ['readout', 'encoder_tail'] if config['group'] == 'S23_TAIL_FT' else ['readout']
    rates = [1e-4, 1e-5] if len(names) == 2 else [1e-4]
    if [g['name'] for g in groups] != names or value['scheduler']['base_lrs'] != rates:
        raise ValueError('New-stage differential learning-rate groups differ')
    for group, rate in zip(groups, rates):
        expected = rate * (1+math.cos(math.pi*completed/20))/2
        if not math.isclose(group['lr'], expected, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Post-validation cosine boundary differs')
    if not value['optimizer']['state'] or any(int(s['step']) != value['optimizer_updates']
                                              for s in value['optimizer']['state'].values()):
        raise ValueError('Adam moments do not match complete effective-update count')
    if not {'python', 'numpy', 'torch', 'cuda'} <= set(value['rng']):
        raise ValueError('Incomplete recovery RNG state')


def load_recovery(path, contract_hash=None):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'recovery.json'
    manifest = read_json(path)
    if (manifest.get('schema') != RECOVERY_SCHEMA
            or manifest.get('identity') != object_hash({k:v for k,v in manifest.items() if k != 'identity'})):
        raise ValueError('Invalid atomic stage recovery manifest')
    loaded, files = {}, {}
    for label in ('current', 'best'):
        entry = manifest[label]
        source = (path.parent / entry['path']).resolve()
        if not source.is_relative_to(path.parent):
            raise ValueError('Recovery file escaped its immutable run directory')
        if source not in files:
            digest = sha256(source)
            state = torch.load(source, map_location='cpu', weights_only=True)
            validate_checkpoint(state, contract_hash)
            files[source] = digest, state
        digest, state = files[source]
        if digest != entry['sha256']:
            raise ValueError('Recovery file hash changed')
        loaded[label] = (source, state)
    current, best = loaded['current'][1], loaded['best'][1]
    if (current['contract_hash'] != best['contract_hash'] or current['code']['commit'] != best['code']['commit']
            or best['completed_epochs'] != current['best_epoch'] or best['best_epe'] != current['best_epe']
            or manifest['completed_epochs'] != current['completed_epochs']
            or manifest['contract_hash'] != current['contract_hash'] or manifest['code_commit'] != current['code']['commit']):
        raise ValueError('Atomic best/current recovery association differs')
    return manifest, loaded


def atomic_link(source, target):
    # Replacing one hardlink with another link to the same inode is an os.rename
    # no-op on POSIX; avoid leaving stale .partial aliases when best is unchanged.
    if target.exists() and os.path.samefile(source, target):
        return
    temporary = target.with_name(target.name+'.partial')
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


def export_recovery(output, current_path, best_path, checkpoint):
    manifest = dict(schema=RECOVERY_SCHEMA, updated_at=now(), run_id=output.name,
        contract_hash=checkpoint['contract_hash'], code_commit=checkpoint['code']['commit'],
        completed_epochs=checkpoint['completed_epochs'], boundary=BOUNDARY,
        current=dict(path=str(current_path.relative_to(output)), sha256=sha256(current_path)),
        best=dict(path=str(best_path.relative_to(output)), sha256=sha256(best_path)))
    manifest['identity'] = object_hash(manifest)
    write_json(output/'recovery.json', manifest)
    atomic_link(current_path, output/'last.pt')
    atomic_link(best_path, output/'best.pt')
    return manifest


def export_evaluation(output, partition, prediction, summary, packet, checkpoint, checkpoint_sha, reused):
    expected = checkpoint['contract']['partition_counts'][partition]
    if len(prediction) != expected or summary['overall']['count'] != expected or len(packet['targets']) != expected:
        raise ValueError('Selected checkpoint evaluation lacks full partition coverage')
    destination = output / f'eval_{partition}'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'metrics.json', summary)
    write_predictions(destination/'predictions.csv', prediction_rows(prediction, packet))
    write_json(destination/'evaluation.json', dict(schema='v5_fusion_tail_evaluation_v1', status='completed',
        partition=partition, count=expected, checkpoint_sha256=checkpoint_sha, training_run=output.name,
        training_code=checkpoint['code'], cache_identity=checkpoint['cache_identity'],
        contract_hash=checkpoint['contract_hash'], config=checkpoint['config'],
        reused_validation_predictions=reused, full_train_inference_passes=0 if reused else 1,
        predictions_sha256=sha256(destination/'predictions.csv'), completed_at=now(), data_scope='full',
        scope=checkpoint['contract']['scope'], expected_counts=COUNTS, expected_scene_counts=SCENE_COUNTS,
        feature_source='current_online_encoders' if checkpoint['config']['group']=='S23_TAIL_FT' else 'verified_frozen_cache'))


def training_contract(config, index, packets, actual_identity, manifest, microbatch, preflight):
    return dict(config=config, scope='tail_preflight' if preflight else CONTRACT_SCOPE, data_scope='full',
        cache_scope=index['scope'], cache_identity=index['identity'], expected_counts=COUNTS,
        expected_scene_counts=SCENE_COUNTS, partition_counts={p:len(v['targets']) for p,v in packets.items()},
        scene_counts={p:len({m['scene'] for m in v['metadata']}) for p,v in packets.items()},
        actual_sample_identity=actual_identity, output_units='m/s', task='PLANAR_PROXY',
        parent_encoder_sha256=SOURCE_SHA, parent_readout_sha256=READOUT_SHA,
        freeze_manifest_identity=manifest['identity'], stage_budget_epochs=20, stage_optimizer_updates=41560,
        selected_parent_epoch=19, optimizer_initialization='new moments; no parent optimizer or scheduler reuse',
        physical_microbatch=microbatch, effective_batch=64,
        accumulation='micro_mean_loss * actual_micro_count / actual_effective_count; clip/step once',
        encoder_training=config['group']=='S23_TAIL_FT',
        epoch_order_policy='torch.randperm(cpu_generator(seed+zero_based_stage_epoch)); independent loader/model RNG')


def train(args):
    started = time.monotonic()
    config_path, cache, output = (Path(v).resolve() for v in (args.config, args.cache, args.output))
    config = read_json(config_path)
    validate_config(config)
    preflight = args.preflight
    if output.exists() and any(output.iterdir()):
        raise ValueError('New output directory required, including recovery attempts')
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('Output overlaps immutable full cache')
    repo = ROOT.parents[2]
    if not preflight:
        if not config_path.is_relative_to(repo) or output.is_relative_to(repo):
            raise ValueError('Formal config must be tracked and output must be external')
        git('ls-files', '--error-unmatch', str(config_path.relative_to(repo)))
    version = code_version(require_pushed=not preflight)
    configure_fp32(args.threads)
    seed_everything(42)
    device = torch.device(args.device)
    hardware, env = runtime_gpu(device), environment()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    packets, indices = {}, {}
    for part in COUNTS:
        packet, index = load_full_cache(cache, part, allow_preflight=preflight)
        if preflight:
            n = min(args.preflight_samples, len(packet['targets']))
            packet = dict(features={k:v[:n] for k,v in packet['features'].items()},
                          targets=packet['targets'][:n], metadata=packet['metadata'][:n])
        packets[part], indices[part] = packet, index
    index = indices['train_core']
    if index['identity'] != indices['dev']['identity']:
        raise ValueError('Train/dev cache identities differ')
    for field in ('scene', 'token'):
        if {m[field] for m in packets['train_core']['metadata']} & {m[field] for m in packets['dev']['metadata']}:
            raise ValueError('Full train/dev partition leakage')
    verification = None if preflight else validate_verification(args.verification, index['identity'], args.microbatch)
    identities = sample_identities(packets['train_core'])
    actual_identity = {p:ordered_sample_digest(v) for p,v in packets.items()}
    loaded, prior_manifest, resume_root = None, None, None
    if args.resume:
        resume_root = Path(args.resume).resolve()
        if resume_root.is_file():
            resume_root = resume_root.parent
        if (resume_root/'result.json').exists() and read_json(resume_root/'result.json').get('status') == 'completed':
            raise ValueError('An already completed stage must be reused, not trained or evaluated again')
        if (resume_root/'recovery.json').exists():
            prior_manifest, loaded = load_recovery(resume_root)
            current = loaded['current'][1]
            if current['code']['commit'] != version['commit']:
                raise ValueError('Epoch-boundary recovery requires the identical fixed source commit')
            model = S23TailModel.from_state(current['model'], config['group'], current['provenance'])
            parent_epoch0 = current['parent_epoch0']
        else:
            if not resume_root.is_dir():
                raise ValueError('Missing interrupted run directory')
            model = S23TailModel(args.encoder_checkpoint, args.readout_checkpoint, config['group'])
            parent_epoch0 = parent_reference(args.readout_checkpoint, model.provenance)
    else:
        model = S23TailModel(args.encoder_checkpoint, args.readout_checkpoint, config['group'])
        parent_epoch0 = parent_reference(args.readout_checkpoint, model.provenance)
    freeze = model.parameter_manifest()
    if model.provenance['readout']['cache_identity'] != index['identity'] and not preflight:
        raise ValueError('Frozen cache differs from the complete-data S23 parent')
    physical_microbatch = args.microbatch if config['group'] == 'S23_TAIL_FT' else 64
    contract = training_contract(config,index,packets,actual_identity,freeze,physical_microbatch,preflight)
    contract_hash = object_hash(contract)
    if loaded:
        validate_checkpoint(loaded['current'][1], contract_hash)
    initial_frozen_sha = frozen_state_digest(model, freeze)
    initialization = dict(policy='fixed_two_parents_new_stage_fresh_adamw_cosine20', seed=42,
        state_sha256=model_state_sha256(model), readout_state_sha256=model_state_sha256(model.readout),
        frozen_state_sha256=initial_frozen_sha,
        parent_encoder_sha256=SOURCE_SHA, parent_readout_sha256=READOUT_SHA)
    model = model.float().to(device)
    runtime, storage, online, alignment = {}, {}, {}, {}
    if config['group'] == 'S23_TAIL_FT':
        datasets = full_datasets(args.manifest, args.flow, args.rgb_frames)
        for part, dataset in datasets.items():
            canonical = CanonicalDataset(dataset, packets[part])
            alignment[part] = canonical.evidence
            online[part] = online_loader(canonical, physical_microbatch, args.workers, device)
            runtime[part], storage[part] = None, dict(mode='current_online_encoders',
                physical_microbatch=physical_microbatch, workers=args.workers,
                pin_memory=device.type=='cuda', prefetch_factor=2 if args.workers else None)
            # No stale frozen features can accidentally enter the online path.
            packets[part]['features'] = {'dt':packets[part]['features']['dt']}
    else:
        for part, packet in packets.items():
            runtime[part], storage[part] = resident_packet(packet, device)
            online[part] = None
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
    completed, updates, history, orders = 0, 0, [], []
    best_epe, best_epoch, best_path = math.inf, None, None
    effective_elapsed, discarded, parent_run, env_difference = 0., [], None, {}
    output.mkdir(parents=True, exist_ok=True)
    (output/'checkpoints').mkdir()
    if loaded:
        current = loaded['current'][1]
        optimizer.load_state_dict(current['optimizer'])
        scheduler.load_state_dict(current['scheduler'])
        completed, updates = current['completed_epochs'], current['optimizer_updates']
        history, orders = current['history'], current['order_records']
        best_epe, best_epoch = current['best_epe'], current['best_epoch']
        initialization = current['initialization']
        if initial_frozen_sha != initialization['frozen_state_sha256']:
            raise ValueError('Recovered encoder BN or frozen state has changed')
        effective_elapsed = current['effective_training_seconds']
        discarded = current.get('discarded_attempts', [])
        for label, (source_path, state) in loaded.items():
            destination = output/'checkpoints'/f"epoch_{state['completed_epochs']:03d}.pt"
            if not destination.exists():
                shutil.copy2(source_path, destination)
            if label == 'best':
                best_path = destination
        current_path = output/'checkpoints'/f'epoch_{completed:03d}.pt'
        export_recovery(output,current_path,best_path,current)
        env_difference = {k:dict(previous=current['environment'].get(k),current=v)
                          for k,v in env.items() if current['environment'].get(k)!=v}
        if current['runtime_gpu'] != hardware:
            env_difference['runtime_gpu'] = dict(previous=current['runtime_gpu'],current=hardware)
        parent_run = dict(run_id=current['run_id'], recovery_manifest=str(resume_root/'recovery.json'),
                          recovery_identity=prior_manifest['identity'])
        for part, loader in online.items():
            if loader:
                loader[2].set_state(current['loader_rng'][part])
        restore_rng(current['rng'])
    elif resume_root:
        parent_run = dict(run_id=resume_root.name, recovery_manifest=None,
                          policy='no completed boundary; fresh prescribed parents and stage optimizer')
    if resume_root and (resume_root/'status.json').exists():
        prior_status = read_json(resume_root/'status.json')
        discarded.append(dict(parent_run=resume_root.name,
            updates=max(0,prior_status.get('attempted_optimizer_updates',updates)-updates),
            elapsed_seconds=max(0.,prior_status.get('training_elapsed_seconds',effective_elapsed)-effective_elapsed),
            evidence=str(resume_root/'status.json'), boundary='only whole committed epochs retained'))
    # Free the extra CPU optimizer/model copies from resume before online allocation.
    loaded = None
    write_json(output/'config.json',config)
    write_json(output/'cache_identity.json',index)
    write_json(output/'freeze_manifest.json',freeze)
    write_json(output/'parent_epoch0.json',parent_epoch0)
    write_history(output,history,orders)
    parameter_count = freeze['trainable_parameter_count']
    receipt = dict(schema='v5_fusion_tail_run_v1',run_id=output.name,group=config['group'],seed=42,status='running',
        started_at=now(),code=version,source_sha256=source_digest(),contract=contract,contract_hash=contract_hash,
        environment=env,runtime_gpu=hardware,environment_difference=env_difference,
        resume_trajectory_claim='complete optimizer/scheduler/RNG boundary; cross-device bitwise equivalence not claimed',
        initialization=initialization,parents=model.provenance,freeze_manifest=freeze,
        parent_epoch0=parent_epoch0,verification=verification,trainable_parameters=parameter_count,
        train_count=len(identities),dev_count=len(packets['dev']['targets']),data_alignment=alignment,
        arithmetic=dict(dtype='float32',amp=False,tf32=False,effective_batch=64,physical_microbatch=physical_microbatch,
            gradient_accumulation='actual micro sample count / actual effective sample count; clip/step once'),
        encoder_training=config['group']=='S23_TAIL_FT',cache_placement=storage,
        command=vars(args),parent_run=parent_run,discarded_attempts=discarded,recovery_boundary='completed_epoch_only')
    write_json(output/'run.json',receipt)
    stopping = {'signal':None}
    previous = {sig:signal.signal(sig,lambda signum,frame:stopping.update(signal=signum))
                for sig in (signal.SIGTERM,signal.SIGINT)}
    loop_start = time.monotonic()
    inherited_elapsed = effective_elapsed
    next_batch, last_status, phase = 0, 0., 'training'
    phase_start = loop_start

    def status(state):
        nonlocal last_status
        last_status = time.monotonic()
        value = dict(status=state,updated_at=now(),run_id=output.name,group=config['group'],seed=42,
            completed_epochs=completed,epoch=completed+1 if completed<20 else None,next_batch=next_batch,
            optimizer_updates=completed*math.ceil(len(identities)/64),
            processed_batches=completed*math.ceil(len(identities)/64),attempted_optimizer_updates=updates,
            amp_skips=0,best_epoch=best_epoch,best_dev_epe=None if best_epoch is None else best_epe,
            elapsed_seconds=time.monotonic()-started,
            training_elapsed_seconds=inherited_elapsed+time.monotonic()-loop_start,
            effective_training_seconds=effective_elapsed,phase=phase,phase_elapsed_seconds=time.monotonic()-phase_start,
            runtime_gpu=hardware,**peak_memory(device))
        write_json(output/'status.json',value)
        return value

    status('running')
    try:
        final_epoch = min(20,completed+1) if preflight else 20
        for epoch in range(completed,final_epoch):
            phase,phase_start = 'training',time.monotonic()
            order,order_hash = epoch_order(len(identities),42,epoch)
            record = order_record(order,order_hash,identities,epoch,64)
            if online['train_core']:
                online['train_core'][1].order = order
                iterator = iter(online['train_core'][0])
            else:
                order_indices = torch.tensor(order,dtype=torch.long,device=runtime['train_core']['placement'])
            learning_rates = [g['lr'] for g in optimizer.param_groups]
            accumulated = torch.zeros(3,dtype=torch.float64,device=device)
            sample_count = 0
            model.train()
            batches = math.ceil(len(order)/64)
            last_log = time.monotonic()
            for b in range(batches):
                chosen = order[b*64:(b+1)*64]
                if online['train_core']:
                    micros = online_microbatches(iterator,chosen,physical_microbatch,device)
                else:
                    micros = [batch(runtime['train_core'],order_indices[b*64:(b+1)*64],device)]
                totals,grad_norm = effective_update(model,optimizer,micros,len(chosen),config,
                                                   cached=online['train_core'] is None)
                accumulated += totals
                updates += 1
                next_batch = b+1
                sample_count += len(chosen)
                clock = time.monotonic()
                if clock-last_log >= 30:
                    print(json.dumps(dict(event='update',group=config['group'],epoch=epoch+1,batch=b+1,
                        optimizer_updates=updates,loss=float(totals[0]/len(chosen)),grad_norm=float(grad_norm),
                        phase='training',phase_elapsed_seconds=clock-phase_start,
                        epoch_samples_per_second=sample_count/(clock-phase_start))),flush=True)
                    last_log = clock
                if clock-last_status >= 30:
                    status('running')
                if stopping['signal'] and b+1<batches:
                    break
            if next_batch != batches:
                break
            training_seconds = time.monotonic()-phase_start
            phase,phase_start = 'dev_validation',time.monotonic()
            status('running')
            dev_summary,dev_prediction = evaluate_packet(model,packets['dev'],runtime['dev'],online['dev'],device,
                                                         config,lambda:status('running'))
            validation_seconds = time.monotonic()-phase_start
            averages = (accumulated/sample_count).cpu().tolist()
            if not math.isclose((averages[1]+averages[2])/2,averages[0],rel_tol=2e-6,abs_tol=2e-7):
                raise RuntimeError('Detached axes disagree with the actual two-axis backward loss')
            if frozen_state_digest(model,freeze) != initialization['frozen_state_sha256']:
                raise RuntimeError('Frozen parameter or BN buffer changed during the stage')
            epe = dev_summary['overall']['epe']
            improved = epe<best_epe
            if improved:
                best_epe,best_epoch = epe,epoch+1
            scheduler.step()
            completed,next_batch = epoch+1,0
            row = dict(epoch=completed,training_loss=averages[0],train_online_loss_x=averages[1],
                train_online_loss_y=averages[2],dev_loss=dev_summary['loss'],dev_loss_x=dev_summary['loss_x'],
                dev_loss_y=dev_summary['loss_y'],dev=dev_summary,learning_rates=learning_rates,
                next_learning_rates=[g['lr'] for g in optimizer.param_groups],sample_order_sha256=order_hash,
                actual_sample_order_sha256=record['actual_sample_order_sha256'],processed_batches=updates,
                optimizer_updates=updates,amp_skips=0,best_epoch=best_epoch,best_dev_epe=best_epe,
                training_seconds=training_seconds,validation_seconds=validation_seconds,updated_at=now())
            history.append(row); orders.append(record)
            effective_elapsed = inherited_elapsed+time.monotonic()-loop_start
            checkpoint = dict(schema=SCHEMA,run_id=output.name,saved_at=now(),model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),
                loader_rng={p:v[2].get_state() for p,v in online.items() if v},
                amp_enabled=False,scaler={},config=config,code=version,source_sha256=source_digest(),
                environment=env,runtime_gpu=hardware,contract=contract,contract_hash=contract_hash,
                cache_identity=index['identity'],provenance=model.provenance,freeze_manifest=freeze,
                parent_epoch0=parent_epoch0,initialization=initialization,epoch_index=completed,next_batch=0,
                completed_epochs=completed,processed_batches=updates,optimizer_updates=updates,amp_skips=0,
                train_count=len(identities),dev_count=len(packets['dev']['targets']),best_epe=best_epe,best_epoch=best_epoch,
                history=history,order_records=orders,dev_summary=dev_summary,dev_prediction=dev_prediction,
                effective_training_seconds=effective_elapsed,discarded_attempts=discarded,parent_run=parent_run,boundary=BOUNDARY)
            validate_checkpoint(checkpoint,contract_hash)
            checkpoint_path = output/'checkpoints'/f'epoch_{completed:03d}.pt'
            if checkpoint_path.exists():
                raise ValueError('Immutable epoch checkpoint already exists')
            phase,phase_start = 'commit_epoch',time.monotonic()
            atomic_save(checkpoint_path,checkpoint)
            if improved:
                best_path = checkpoint_path
            export_recovery(output,checkpoint_path,best_path,checkpoint)
            write_history(output,history,orders)
            print(json.dumps(dict(event='validation',group=config['group'],epoch=completed,
                optimizer_updates=updates,dev_epe=epe,best_epoch=best_epoch,best_dev_epe=best_epe,
                learning_rates=learning_rates,training_seconds=training_seconds,
                validation_seconds=validation_seconds,checkpoint_seconds=time.monotonic()-phase_start)),flush=True)
            status('running')
            if stopping['signal']:
                break
        if stopping['signal']:
            final_state = 'interrupted'
            status_value = status(final_state)
            write_json(output/'interrupted.json',dict(signal=stopping['signal'],boundary_recovery_only=True,
                complete_recovery_exists=(output/'recovery.json').exists(),
                discarded_updates=updates-status_value['optimizer_updates'],
                discarded_elapsed_seconds=max(0.,status_value['training_elapsed_seconds']-effective_elapsed),updated_at=now()))
        else:
            recovery, loaded = load_recovery(output,contract_hash)
            best = loaded['best'][1]
            model.load_state_dict(best['model'],strict=True)
            best_summary,best_sha = best['dev_summary'],recovery['best']['sha256']
            replacement = dict(improved_over_parent=best_epe<PARENT_EPE,
                preferred='stage_best' if best_epe<PARENT_EPE else 'parent_epoch0',tie_rule='parent_on_tie',
                parent_dev_epe=PARENT_EPE,stage_best_dev_epe=best_epe)
            result = dict(schema='v5_fusion_tail_result_v1',run_id=output.name,group=config['group'],seed=42,
                best_epoch=best_epoch,best_dev_epe=best_epe,last_dev_epe=history[-1]['dev']['overall']['epe'],
                best_train_epe=None,dev=best_summary,train=None,trainable_parameters=parameter_count,
                optimizer_updates=updates,completed_epochs=completed,gpu=hardware,elapsed_seconds=time.monotonic()-started,
                training_elapsed_seconds=effective_elapsed,code=version,source_sha256=source_digest(),
                cache_identity=index['identity'],actual_sample_identity=actual_identity,initialization=initialization,
                parents=model.provenance,freeze_manifest=freeze,parent_epoch0=parent_epoch0,
                best_including_parent_epoch=best_epoch if replacement['improved_over_parent'] else 0,
                replacement_decision=replacement,epoch_orders=orders,best_checkpoint_sha256=best_sha,
                last_checkpoint_sha256=recovery['current']['sha256'],contract_hash=contract_hash,
                generalization_gap_dev_minus_train_epe=None,best_train_full_evaluation_count=0,
                best_dev_predictions_reused=True,discarded_attempts=discarded,data_scope='full',scope=contract['scope'],
                train_count=len(identities),dev_count=len(packets['dev']['targets']),expected_counts=COUNTS,
                expected_scene_counts=SCENE_COUNTS,**peak_memory(device))
            export_evaluation(output,'dev',best['dev_prediction'],best_summary,packets['dev'],best,best_sha,True)
            ready_state = 'preflight_dev_ready' if preflight else 'dev_ready'
            write_json(output/'dev_ready.json',dict(result,schema='v5_fusion_tail_dev_ready_v1',status=ready_state,
                train_evaluation_status='pending',updated_at=now()))
            phase,phase_start = 'selected_best_full_train_evaluation',time.monotonic()
            receipt.update(status(ready_state),status=ready_state,best_checkpoint_sha256=best_sha,
                           last_checkpoint_sha256=recovery['current']['sha256'])
            write_json(output/'run.json',receipt)
            print(json.dumps(dict(event=ready_state,group=config['group'],run_id=output.name,
                best_epoch=best_epoch,best_dev_epe=best_epe,optimizer_updates=updates,replacement_decision=replacement)),flush=True)
            reused_train = False
            previous_train = resume_root/'eval_train_core' if resume_root else None
            if previous_train and (previous_train/'evaluation.json').exists():
                proof = read_json(previous_train/'evaluation.json')
                if (proof.get('status')=='completed' and proof.get('checkpoint_sha256')==best_sha
                        and proof.get('contract_hash')==contract_hash and proof.get('count')==len(identities)
                        and sha256(previous_train/'predictions.csv')==proof['predictions_sha256']):
                    train_summary = read_json(previous_train/'metrics.json')
                    if train_summary['overall']['count'] != len(identities):
                        raise ValueError('Prior completed selected-best train evaluation lacks coverage')
                    shutil.copytree(previous_train,output/'eval_train_core')
                    proof.update(training_run=output.name,reused_completed_evaluation_from=str(previous_train))
                    write_json(output/'eval_train_core'/'evaluation.json',proof)
                    reused_train = True
            if not reused_train:
                train_summary,train_prediction = evaluate_packet(model,packets['train_core'],runtime['train_core'],
                    online['train_core'],device,config,lambda:status(ready_state))
                export_evaluation(output,'train_core',train_prediction,train_summary,packets['train_core'],best,best_sha,False)
            final_state = 'preflight_completed' if preflight else 'completed'
            result.update(status=final_state,best_train_epe=train_summary['overall']['epe'],train=train_summary,
                generalization_gap_dev_minus_train_epe=best_epe-train_summary['overall']['epe'],
                best_train_full_evaluation_count=1,reused_completed_train_evaluation=reused_train,
                selected_best_train_evaluation_seconds=time.monotonic()-phase_start,
                elapsed_seconds=time.monotonic()-started,**peak_memory(device))
            write_json(output/'result.json',result)
            status_value = status(final_state)
            receipt.update(best_checkpoint_sha256=best_sha,last_checkpoint_sha256=recovery['current']['sha256'])
        receipt.update(status_value,status=final_state,finished_at=now(),stop_signal=stopping['signal'])
        write_json(output/'run.json',receipt)
        print(json.dumps(dict(event=final_state,group=config['group'],run_id=output.name,
                              best_epoch=best_epoch,best_dev_epe=None if best_epoch is None else best_epe)),flush=True)
        return receipt
    except BaseException as error:
        receipt.update(status('failed'),status='failed',finished_at=now(),error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json',receipt)
        raise
    finally:
        for sig,handler in previous.items():
            signal.signal(sig,handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','cache','manifest','flow','rgb-frames','encoder-checkpoint','readout-checkpoint','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--verification',help='Passed model_preflight.json from this exact implementation')
    parser.add_argument('--resume',help='This-stage recovery.json or interrupted run; always a new output directory')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-samples',type=int,default=64)
    args = parser.parse_args()
    if not 1<=args.microbatch<=64 or not 1<=args.preflight_samples<=114 or args.workers<0 or args.threads<1:
        parser.error('microbatch1..64; preflight_samples1..114; workers>=0; threads>=1')
    if not args.preflight and not args.verification:
        parser.error('Formal stages require --verification model_preflight.json')
    train(args)


if __name__ == '__main__':
    main()
