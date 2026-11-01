"""Local six-group FP32 trainer; recovery is atomic and epoch-boundary only.

An immutable checkpoint owns its completed history, optimizer/scheduler/RNG,
validation predictions and provenance. recovery.json atomically selects both
current and historical best checkpoints. best.pt/last.pt are export links only.
"""
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

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT, atomic_save, code_version, environment, git, now, object_hash, read_json, sha256, write_json
from .evaluation import metrics, summarize, validate_lateral_rule
from .fusion_abc_runtime import peak_memory, runtime_gpu
from .fusion_abc_training import PROTOCOL as HISTORICAL_PROTOCOL, configure_fp32
from .training import epoch_order, model_state_sha256, restore_rng, rng_state, seed_everything, write_predictions

PROTOCOL = dict(HISTORICAL_PROTOCOL, schema='v5_fusion_grid_config_v1')
SUPPORTED_SEEDS = (42, 43, 44)
SCHEMA = 'v5_fusion_grid_epoch_state_v1'
RECOVERY_SCHEMA = 'v5_fusion_grid_recovery_v1'
COUNTS = dict(train_core=7324, dev=1826)


def validate_config(config):
    from .fusion_grid import GRID_VARIANTS
    if type(config.get('seed')) is not int or config['seed'] not in SUPPORTED_SEEDS:
        raise ValueError('Fixed screening/confirmation seeds are integers 42, 43 and 44')
    for key, expected in PROTOCOL.items():
        if key == 'seed':
            continue  # Each seed retains the historical factory and epoch-order policy.
        if config.get(key) != expected or (isinstance(expected, bool) and type(config.get(key)) is not bool):
            raise ValueError(f'Fixed training protocol differs: {key}')
    if config.get('group') not in GRID_VARIANTS:
        raise ValueError('Unknown six-group identifier')
    validate_lateral_rule(config.get('lateral_rule'))


def new_model(config):
    from .fusion_grid import build_six
    return build_six(config['seed'])[config['group']].float()


def sample_identities(packet):
    """Bind actual ordered metadata, input frame/times, labels AND dt."""
    target = packet['targets'].cpu().tolist()
    dt = packet['features']['dt'].cpu().tolist()
    return [object_hash(dict(metadata=m, target=y, dt=d)) for m, y, d in zip(packet['metadata'], target, dt)]


def order_record(order, order_hash, identities, epoch, batch_size):
    # The index digest alone cannot establish correspondence across two caches.
    selected = [identities[i] for i in order]
    batches = [object_hash(selected[i:i + batch_size]) for i in range(0, len(order), batch_size)]
    return dict(epoch=epoch + 1, zero_based_epoch=epoch, sample_count=len(order),
                order_sha256=order_hash, actual_sample_order_sha256=object_hash(selected),
                actual_batch_identity_sha256=batches, batch_sizes=[min(batch_size, len(order) - i)
                    for i in range(0, len(order), batch_size)])


def resident_packet(packet, device):
    """Transfer once; keep annotations and targets for output on the CPU."""
    values = [*packet['features'].values(), packet['targets']]
    required = sum(v.numel() * v.element_size() for v in values)
    resident = device.type == 'cuda'
    free = None
    if resident:
        free, _ = torch.cuda.mem_get_info(device)
        resident = required + 2 * 1024**3 < free
    placement = device if resident else torch.device('cpu')
    runtime = dict(features={k: v.to(placement) for k, v in packet['features'].items()},
                   targets=packet['targets'].to(placement), placement=placement)
    return runtime, dict(mode='gpu_resident' if resident else 'cpu_resident', bytes=required,
                         available_gpu_bytes_at_load=free, reserved_workspace_bytes=2 * 1024**3)


def batch(runtime, indices, device):
    values = {k: v[indices].to(device=device, dtype=torch.float32) for k, v in runtime['features'].items()}
    target = runtime['targets'][indices].to(device=device, dtype=torch.float32)
    return values, target


def summarize_prediction(prediction, packet, config):
    prediction = prediction.detach().float().cpu()
    target = packet['targets'].cpu()
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise RuntimeError('Invalid/nonfinite physical [vx,vy] predictions')
    summary = summarize(prediction.numpy(), target.numpy(), [m['scene'] for m in packet['metadata']],
                        config.get('lateral_rule'))
    summary['nonzero_vy']['status'] = 'sufficient_data' if summary['nonzero_vy']['adequate'] else 'insufficient_data'
    summary['task'] = 'PLANAR_PROXY'
    scale = target.new_tensor(config['loss_scales_m_s'])
    axes = F.smooth_l1_loss(prediction / scale, target / scale, beta=1., reduction='none').mean(0)
    summary.update(loss=float(axes.mean()), loss_x=float(axes[0]), loss_y=float(axes[1]))
    speed = torch.linalg.vector_norm(target, dim=1).numpy()
    summary['speed_bins'] = {}
    for low, high in ((0., 2.), (2., 5.), (5., 10.), (10., 15.), (15., float('inf'))):
        mask = (speed >= low) & (speed < high)
        key = f'{low:g}..{high:g}'
        summary['speed_bins'][key] = dict(lower_m_s=low, upper_m_s=None if math.isinf(high) else high,
            count=int(mask.sum()), metrics=metrics(prediction.numpy()[mask], target.numpy()[mask]) if mask.any() else None)
    summary['speed_definition'] = 'sqrt(target_vx**2 + target_vy**2), half-open bins in m/s'
    return summary


def evaluate_packet(model, packet, runtime, device, config):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(packet['targets']), config['batch_size']):
            values, _ = batch(runtime, slice(start, start + config['batch_size']), device)
            outputs.append(model(values).detach())
    prediction = torch.cat(outputs).float().cpu()  # One evaluation-boundary transfer.
    return summarize_prediction(prediction, packet, config), prediction


def prediction_rows(prediction, packet):
    return [dict(scene=m['scene'], token=m['token'], time_s=m['time_s'], vx=float(p[0]), vy=float(p[1]),
                 target_vx=float(y[0]), target_vy=float(y[1]))
            for m, p, y in zip(packet['metadata'], prediction, packet['targets'])]


def validate_checkpoint(value, contract_hash=None):
    if value.get('schema') != SCHEMA or value.get('contract_hash') != object_hash(value.get('contract', {})):
        raise ValueError('Invalid epoch checkpoint contract')
    if contract_hash is not None and value['contract_hash'] != contract_hash:
        raise ValueError('Resume configuration/cache/scope differs')
    validate_config(value['config'])
    if value['config'] != value['contract']['config'] or value['cache_identity'] != value['contract']['cache_identity']:
        raise ValueError('Checkpoint configuration/cache identity differs')
    completed = value['completed_epochs']
    batches = math.ceil(value['train_count'] / value['config']['batch_size'])
    if (not 1 <= completed <= value['config']['epochs'] or value['epoch_index'] != completed
            or value['next_batch'] != 0 or value['optimizer_updates'] != completed * batches
            or value['processed_batches'] != value['optimizer_updates'] or value['amp_skips'] != 0
            or value['scheduler']['last_epoch'] != completed
            or value['scheduler']['T_max'] != 20 or value['scheduler']['eta_min'] != 0
            or len(value['history']) != completed or len(value['order_records']) != completed
            or value['boundary'] != 'validation_then_scheduler_then_advance_epoch_next_batch_zero'):
        raise ValueError('Recovery supports only a complete validated epoch boundary')
    best = min(value['history'], key=lambda row: row['dev']['overall']['epe'])
    if value['best_epoch'] != best['epoch'] or value['best_epe'] != best['dev']['overall']['epe']:
        raise ValueError('Historical best selection differs')


def load_recovery(path, contract_hash=None):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'recovery.json'
    manifest = read_json(path)
    if (manifest.get('schema') != RECOVERY_SCHEMA
            or manifest.get('identity') != object_hash({k: v for k, v in manifest.items() if k != 'identity'})):
        raise ValueError('Invalid atomic recovery manifest')
    loaded = {}
    for label in ('current', 'best'):
        entry = manifest[label]
        checkpoint_path = (path.parent / entry['path']).resolve()
        if not checkpoint_path.is_relative_to(path.parent) or sha256(checkpoint_path) != entry['sha256']:
            raise ValueError(f'Recovery {label} file/path hash changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint, contract_hash)
        loaded[label] = (checkpoint_path, checkpoint)
    current, best = loaded['current'][1], loaded['best'][1]
    if (current['contract_hash'] != best['contract_hash'] or current['code']['commit'] != best['code']['commit']
            or best['completed_epochs'] != current['best_epoch'] or best['best_epe'] != current['best_epe']
            or manifest['completed_epochs'] != current['completed_epochs']
            or manifest['contract_hash'] != current['contract_hash'] or manifest['code_commit'] != current['code']['commit']):
        raise ValueError('Recovery current/best historical association differs')
    return manifest, loaded


def atomic_link(source, target):
    temporary = target.with_name(target.name + '.partial')
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


def write_history(output, history, orders):
    path = output / 'metrics.jsonl'
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in history))
    os.replace(temporary, path)
    write_json(output / 'epoch_orders.json', orders)


def export_recovery(output, current_path, best_path, checkpoint):
    value = dict(schema=RECOVERY_SCHEMA, updated_at=now(), run_id=output.name,
        contract_hash=checkpoint['contract_hash'], code_commit=checkpoint['code']['commit'],
        completed_epochs=checkpoint['completed_epochs'],
        current=dict(path=str(current_path.relative_to(output)), sha256=sha256(current_path)),
        best=dict(path=str(best_path.relative_to(output)), sha256=sha256(best_path)),
        boundary=checkpoint['boundary'])
    value['identity'] = object_hash(value)
    # This single replace commits the mutually consistent recovery pair.
    write_json(output / 'recovery.json', value)
    atomic_link(current_path, output / 'last.pt')
    atomic_link(best_path, output / 'best.pt')
    return value


def export_evaluation(output, partition, prediction, summary, packet, checkpoint, checkpoint_sha, reused):
    destination = output / f'eval_{partition}'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'metrics.json', summary)
    write_predictions(destination / 'predictions.csv', prediction_rows(prediction, packet))
    write_json(destination / 'evaluation.json', dict(schema='v5_fusion_grid_evaluation_v1', status='completed',
        partition=partition, count=len(prediction), checkpoint_sha256=checkpoint_sha,
        training_run=output.name, training_code=checkpoint['code'], cache_identity=checkpoint['cache_identity'],
        contract_hash=checkpoint['contract_hash'], config=checkpoint['config'],
        reused_validation_predictions=reused, full_train_inference_passes=0 if reused else 1,
        predictions_sha256=sha256(destination / 'predictions.csv'), completed_at=now()))


def train(args):
    from .fusion_grid import EXPECTED_PARAMETERS, GRID_VARIANTS
    from .fusion_grid_cache import cache_grid, load_grid_cache, ordered_sample_digest
    started = time.monotonic()
    config_path, cache, output = (Path(v).resolve() for v in (args.config, args.cache, args.output))
    config = read_json(config_path)
    validate_config(config)
    preflight = args.preflight
    if output.exists() and any(output.iterdir()):
        raise ValueError('New output directory required, including resume attempts')
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('Output overlaps immutable cache')
    repo = ROOT.parents[2]
    if not preflight:
        if not config_path.is_relative_to(repo) or output.is_relative_to(repo):
            raise ValueError('Formal config must be tracked; run output must be external')
        git('ls-files', '--error-unmatch', str(config_path.relative_to(repo)))
    version = code_version(require_pushed=not preflight)
    configure_fp32(args.threads)
    seed_everything(config['seed'])
    device = torch.device(args.device)
    hardware = runtime_gpu(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    env = environment()
    packets, indices = {}, {}
    for part in COUNTS:
        packet, index = load_grid_cache(cache, part, allow_preflight=preflight)
        if cache_grid(index) != tuple(GRID_VARIANTS[config['group']]['grid']):
            raise ValueError('Group grid/cache grid differs')
        if not preflight and (index['scope'] != 'pilot' or len(packet['targets']) != COUNTS[part]):
            raise ValueError('Formal run must consume complete fixed pilot cache')
        if preflight:
            # A disposable bounded epoch exercises all complete-boundary paths.
            n = min(args.preflight_samples, len(packet['targets']))
            packet = dict(features={k: v[:n] for k, v in packet['features'].items()},
                          targets=packet['targets'][:n], metadata=packet['metadata'][:n])
        packets[part], indices[part] = packet, index
    index = indices['train_core']
    if index['identity'] != indices['dev']['identity']:
        raise ValueError('Train/dev cache identity mismatch')
    for key in ('scene', 'token'):
        if {m[key] for m in packets['train_core']['metadata']} & {m[key] for m in packets['dev']['metadata']}:
            raise ValueError('Train/dev sample leakage')
    identities = sample_identities(packets['train_core'])
    actual_identity = {part: ordered_sample_digest(packet) for part, packet in packets.items()}
    contract = dict(config=config, cache_identity=index['identity'], actual_sample_identity=actual_identity,
        scope='preflight' if preflight else 'frozen_fusion_grid_pilot',
        preflight_samples=args.preflight_samples if preflight else None,
        output_units='m/s', task='PLANAR_PROXY', encoder_training=False,
        epoch_order_policy='torch.randperm(cpu_generator(seed+zero_based_epoch)); historical unchanged')
    contract_hash = object_hash(contract)
    runtime, storage = {}, {}
    for part, packet in packets.items():
        runtime[part], storage[part] = resident_packet(packet, device)
    model = new_model(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS[config['group']]:
        raise ValueError('Parameter count differs from fixed architecture')
    initialization = dict(policy='historical_build_triplet_C_then_explicit_six_group_copy_v1',
        state_sha256=model_state_sha256(model), common_modules_sha256={name: model_state_sha256(getattr(model, name))
            for name in ('layers', 'head')}, seed=config['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['optimizer_eps'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
    completed, updates, history, orders = 0, 0, [], []
    best_epe, best_epoch, best_path = math.inf, None, None
    parent_run, env_difference = None, {}
    effective_elapsed, discarded = 0., []
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir()
    if args.resume:
        resume_path = Path(args.resume).resolve()
        manifest_path = resume_path / 'recovery.json' if resume_path.is_dir() else resume_path
        if manifest_path.exists():
            manifest, loaded = load_recovery(manifest_path, contract_hash)
            checkpoint = loaded['current'][1]
            if checkpoint['code']['commit'] != version['commit']:
                raise ValueError('Formal recovery requires identical fixed implementation commit')
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            completed, updates = checkpoint['completed_epochs'], checkpoint['optimizer_updates']
            history, orders = checkpoint['history'], checkpoint['order_records']
            best_epe, best_epoch = checkpoint['best_epe'], checkpoint['best_epoch']
            initialization = checkpoint['initialization']
            effective_elapsed = checkpoint['effective_training_seconds']
            discarded = checkpoint.get('discarded_attempts', [])
            for label, (source_path, state) in loaded.items():
                target_path = output / 'checkpoints' / f"epoch_{state['completed_epochs']:03d}.pt"
                if not target_path.exists():
                    shutil.copy2(source_path, target_path)
                if label == 'best':
                    best_path = target_path
            current_path = output / 'checkpoints' / f'epoch_{completed:03d}.pt'
            export_recovery(output, current_path, best_path, checkpoint)
            env_difference = {key: dict(previous=checkpoint['environment'].get(key), current=value)
                              for key, value in env.items() if checkpoint['environment'].get(key) != value}
            if checkpoint['runtime_gpu'] != hardware:
                env_difference['runtime_gpu'] = dict(previous=checkpoint['runtime_gpu'], current=hardware)
            parent_run = dict(run_id=checkpoint['run_id'], recovery_manifest=str(manifest_path),
                              recovery_identity=manifest['identity'])
            restore_rng(checkpoint['rng'])
        else:
            if not resume_path.is_dir():
                raise ValueError('Missing recovery manifest; pass the interrupted directory to restart initialization')
            parent_run = dict(run_id=resume_path.name, recovery_manifest=None,
                              policy='no complete boundary; restart prescribed initialization')
        prior_status_path = manifest_path.parent / 'status.json'
        if prior_status_path.exists():
            prior_status = read_json(prior_status_path)
            discarded.append(dict(parent_run=parent_run['run_id'],
                updates=max(0, prior_status.get('attempted_optimizer_updates', updates) - updates),
                elapsed_seconds=max(0., prior_status.get('training_elapsed_seconds', effective_elapsed) - effective_elapsed),
                evidence=str(prior_status_path)))
    write_json(output / 'config.json', config)
    write_json(output / 'cache_identity.json', index)
    write_history(output, history, orders)
    receipt = dict(schema='v5_fusion_grid_run_v1', run_id=output.name, group=config['group'], status='running',
        started_at=now(), code=version, contract=contract, contract_hash=contract_hash,
        environment=env, runtime_gpu=hardware, environment_difference=env_difference,
        resume_trajectory_claim='same protocol; cross-device/software restoration is not claimed bitwise identical',
        initialization=initialization, trainable_parameters=parameter_count,
        train_count=len(packets['train_core']['targets']), dev_count=len(packets['dev']['targets']),
        arithmetic=dict(dtype='float32', amp=False, tf32=False, gradient_accumulation=1),
        encoder_training=False, cache_placement=storage, command=vars(args), parent_run=parent_run,
        discarded_attempts=discarded, recovery_boundary='completed_epoch_only')
    write_json(output / 'run.json', receipt)
    stopping = {'signal': None}
    previous = {sig: signal.signal(sig, lambda signum, frame: stopping.update(signal=signum))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    loop_start = time.monotonic()
    inherited_elapsed = effective_elapsed
    next_batch = 0
    last_status = 0.

    def status(state):
        nonlocal last_status
        last_status = time.monotonic()
        value = dict(status=state, updated_at=now(), run_id=output.name, group=config['group'],
            completed_epochs=completed, epoch=completed + 1 if completed < 20 else None,
            next_batch=next_batch, optimizer_updates=completed * math.ceil(len(identities) / config['batch_size']),
            processed_batches=completed * math.ceil(len(identities) / config['batch_size']),
            attempted_optimizer_updates=updates, amp_skips=0, best_epoch=best_epoch,
            best_dev_epe=None if best_epoch is None else best_epe,
            elapsed_seconds=time.monotonic() - started,
            training_elapsed_seconds=inherited_elapsed + time.monotonic() - loop_start,
            effective_training_seconds=effective_elapsed, runtime_gpu=hardware, **peak_memory(device))
        write_json(output / 'status.json', value)
        return value

    status('running')
    try:
        final_epoch = min(20, completed + 1) if preflight else 20
        for epoch in range(completed, final_epoch):
            order, order_hash = epoch_order(len(identities), config['seed'], epoch)
            record = order_record(order, order_hash, identities, epoch, config['batch_size'])
            order_indices = torch.tensor(order, dtype=torch.long, device=runtime['train_core']['placement'])
            learning_rates = [group['lr'] for group in optimizer.param_groups]
            accumulated = torch.zeros(3, dtype=torch.float64, device=device)
            sample_count = 0
            model.train()
            batches = math.ceil(len(order) / config['batch_size'])
            for b in range(batches):
                chosen = order_indices[b * config['batch_size']:(b + 1) * config['batch_size']]
                values, target = batch(runtime['train_core'], chosen, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(values)
                scale = target.new_tensor(config['loss_scales_m_s'])
                loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=1., reduction='mean')
                if prediction.shape != target.shape or not torch.isfinite(loss):
                    raise RuntimeError('Invalid shape/nonfinite loss')
                # Exactly the same forward; axis losses are observation only.
                with torch.no_grad():
                    axes = F.smooth_l1_loss(prediction.detach() / scale, target / scale,
                                           beta=1., reduction='none').mean(0)
                    accumulated += torch.cat((loss.detach().reshape(1), axes)).double() * len(chosen)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                next_batch = b + 1
                sample_count += len(chosen)
                if (b + 1) % 50 == 0:
                    print(json.dumps(dict(event='update', group=config['group'], epoch=epoch + 1, batch=b + 1,
                        optimizer_updates=updates, loss=float(loss.detach()), grad_norm=float(grad_norm))), flush=True)
                if time.monotonic() - last_status >= 30:
                    status('running')
                if stopping['signal'] and b + 1 < batches:
                    break
            if next_batch != batches:
                break  # No partial state can overwrite a complete recovery point.
            dev_summary, dev_prediction = evaluate_packet(model, packets['dev'], runtime['dev'], device, config)
            averages = (accumulated / sample_count).cpu().tolist()
            if not math.isclose((averages[1] + averages[2]) / 2, averages[0], rel_tol=2e-6, abs_tol=2e-7):
                raise RuntimeError('Detached two-axis losses disagree with backward loss')
            epe = dev_summary['overall']['epe']
            improved = epe < best_epe
            if improved:
                best_epe, best_epoch = epe, epoch + 1
            scheduler.step()  # Preserve historical order: validation, scheduler, advance progress.
            completed, next_batch = epoch + 1, 0
            row = dict(epoch=completed, training_loss=averages[0], train_online_loss_x=averages[1],
                train_online_loss_y=averages[2], dev_loss=dev_summary['loss'], dev_loss_x=dev_summary['loss_x'],
                dev_loss_y=dev_summary['loss_y'], dev=dev_summary, learning_rates=learning_rates,
                next_learning_rates=[group['lr'] for group in optimizer.param_groups],
                sample_order_sha256=order_hash, actual_sample_order_sha256=record['actual_sample_order_sha256'],
                processed_batches=updates, optimizer_updates=updates, amp_skips=0,
                best_epoch=best_epoch, best_dev_epe=best_epe, updated_at=now())
            history.append(row)
            orders.append(record)
            effective_elapsed = inherited_elapsed + time.monotonic() - loop_start
            checkpoint = dict(schema=SCHEMA, run_id=output.name, saved_at=now(), model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                amp_enabled=False, scaler={}, config=config, code=version, environment=env, runtime_gpu=hardware,
                contract=contract, contract_hash=contract_hash, cache_identity=index['identity'],
                initialization=initialization, epoch_index=completed, next_batch=0, completed_epochs=completed,
                processed_batches=updates, optimizer_updates=updates, amp_skips=0,
                train_count=len(identities), best_epe=best_epe, best_epoch=best_epoch,
                history=history, order_records=orders, dev_summary=dev_summary, dev_prediction=dev_prediction,
                effective_training_seconds=effective_elapsed, discarded_attempts=discarded,
                parent_run=parent_run, boundary='validation_then_scheduler_then_advance_epoch_next_batch_zero')
            validate_checkpoint(checkpoint, contract_hash)
            checkpoint_path = output / 'checkpoints' / f'epoch_{completed:03d}.pt'
            if checkpoint_path.exists():
                raise ValueError('Immutable epoch checkpoint already exists')
            atomic_save(checkpoint_path, checkpoint)
            if improved:
                best_path = checkpoint_path
            export_recovery(output, checkpoint_path, best_path, checkpoint)
            write_history(output, history, orders)
            print(json.dumps(dict(event='validation', group=config['group'], **row)), flush=True)
            status('running')
            if stopping['signal']:
                break
        if stopping['signal']:
            final_state = 'interrupted'
            status_value = status(final_state)
            write_json(output / 'interrupted.json', dict(signal=stopping['signal'], boundary_recovery_only=True,
                complete_recovery_exists=(output / 'recovery.json').exists(),
                discarded_updates=updates - status_value['optimizer_updates'],
                discarded_elapsed_seconds=max(0., status_value['training_elapsed_seconds'] - effective_elapsed),
                updated_at=now()))
        else:
            # Independently validate the atomic pair, restore best, and reuse its dev prediction.
            recovery, loaded = load_recovery(output / 'recovery.json', contract_hash)
            best = loaded['best'][1]
            model.load_state_dict(best['model'], strict=True)
            train_summary, train_prediction = evaluate_packet(model, packets['train_core'], runtime['train_core'], device, config)
            best_summary = best['dev_summary']
            best_sha = recovery['best']['sha256']
            export_evaluation(output, 'dev', best['dev_prediction'], best_summary, packets['dev'], best, best_sha, True)
            export_evaluation(output, 'train_core', train_prediction, train_summary, packets['train_core'], best, best_sha, False)
            final_state = 'preflight_completed' if preflight else 'completed'
            result = dict(schema='v5_fusion_grid_result_v1', status=final_state, run_id=output.name, group=config['group'],
                best_epoch=best_epoch, best_dev_epe=best_epe, last_dev_epe=history[-1]['dev']['overall']['epe'],
                best_train_epe=train_summary['overall']['epe'], dev=best_summary, train=train_summary,
                trainable_parameters=parameter_count, optimizer_updates=updates, completed_epochs=completed,
                gpu=hardware, elapsed_seconds=time.monotonic() - started, training_elapsed_seconds=effective_elapsed,
                code=version, cache_identity=index['identity'], actual_sample_identity=actual_identity,
                initialization=initialization, epoch_orders=orders, best_checkpoint_sha256=best_sha,
                last_checkpoint_sha256=recovery['current']['sha256'], contract_hash=contract_hash,
                generalization_gap_dev_minus_train_epe=best_epe - train_summary['overall']['epe'],
                best_train_full_evaluation_count=1, best_dev_predictions_reused=True,
                discarded_attempts=discarded, **peak_memory(device))
            write_json(output / 'result.json', result)
            status_value = status(final_state)
            receipt.update(best_checkpoint_sha256=best_sha, last_checkpoint_sha256=recovery['current']['sha256'])
        receipt.update(status_value, status=final_state, finished_at=now(), stop_signal=stopping['signal'])
        write_json(output / 'run.json', receipt)
        print(json.dumps(dict(event=final_state, group=config['group'], run_id=output.name,
                              best_epoch=best_epoch, best_dev_epe=None if best_epoch is None else best_epe)), flush=True)
        return receipt
    except BaseException as error:
        receipt.update(status('failed'), status='failed', finished_at=now(), error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', receipt)
        print(json.dumps(dict(event='failed', group=config['group'], error=receipt['error'])), flush=True)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', help='Atomic recovery.json (or run directory); new output directory always required')
    parser.add_argument('--preflight', action='store_true', help='Disposable single-epoch bounded check; never formal training')
    parser.add_argument('--preflight-samples', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.preflight_samples <= 64 or args.threads < 1:
        parser.error('Preflight sample bound is 1..64 and threads must be positive')
    train(args)


if __name__ == '__main__':
    main()
