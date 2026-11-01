"""Full-scope-only verification with bounded temporary updates and cached reads."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

from purevision_v5.common import ROOT, now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_grid import build_six
from purevision_v5 import fusion_grid_training as training
from purevision_v5.training import epoch_order, model_state_sha256, seed_everything

GROUPS = ('J23', 'S23', 'F23')


def rejected(call):
    try:
        call()
    except (ValueError, KeyError):
        return True
    raise AssertionError('Invalid full-scope contract was accepted')


def verify(args):
    from purevision_v5 import fusion_full_cache
    training.configure_fp32(2)
    packets, indexes = {}, {}
    for partition in training.FULL_COUNTS:
        packets[partition], indexes[partition] = fusion_full_cache.load_full_cache(
            args.cache, partition, allow_preflight=not args.cache_check_only)
    index = indexes['train_core']
    assert index['identity'] == indexes['dev']['identity']
    assert index['scope'] in {'full', 'full_preflight'} and index['grid'] == [2, 3]
    configs = {group: read_json(ROOT / f'configs/next/FUSION_FULL_{group}.json') for group in GROUPS}
    for group, config in configs.items():
        training.validate_config(config)
        pilot = read_json(ROOT / f'configs/next/FUSION_GRID_{group}.json')
        assert config == dict(pilot, schema=training.FULL_CONFIG_SCHEMA, scope='full',
                             expected_counts=training.FULL_COUNTS, expected_scene_counts=training.FULL_SCENES)
    def in_memory_loader(root, partition, allow_preflight=False):
        assert Path(root).resolve() == Path(args.cache).resolve()
        if index['scope'] == 'full_preflight' and not allow_preflight:
            raise ValueError('Full preflight excluded from formal training')
        return packets[partition], indexes[partition]
    with patch.object(fusion_full_cache, 'load_full_cache', in_memory_loader):
        if index['scope'] == 'full':
            for partition in packets:
                packet, _ = training.load_training_partition(args.cache, partition, configs['J23'])
                assert len(packet['targets']) == training.FULL_COUNTS[partition]
            formal_scope_validated = True
        else:
            rejected(lambda: training.load_training_partition(args.cache, 'train_core', configs['J23']))
            formal_scope_validated = False
        bad_index = dict(index, scope='pilot')
        with patch.object(fusion_full_cache, 'load_full_cache', return_value=(packets['train_core'], bad_index)):
            rejected(lambda: training.load_training_partition(args.cache, 'train_core', configs['J23'], True))
    for partition, packet in packets.items():
        keys = [(m['scene'], m['token']) for m in packet['metadata']]
        assert len(keys) == len(set(keys)) == len(packet['targets'])
    assert not ({m['scene'] for m in packets['train_core']['metadata']} & {m['scene'] for m in packets['dev']['metadata']})
    identities = training.sample_identities(packets['train_core'])
    order_receipts = []
    for epoch in range(20):
        before = torch.get_rng_state().clone()
        order, digest = epoch_order(len(identities), 42, epoch)
        assert torch.equal(before, torch.get_rng_state())
        record = training.order_record(order, digest, identities, epoch, 64)
        for group in GROUPS:
            candidate, candidate_digest = epoch_order(len(identities), configs[group]['seed'], epoch)
            assert candidate == order and candidate_digest == digest
        if formal_scope_validated:
            assert len(record['batch_sizes']) == 2078 and record['batch_sizes'][-1] == 50
        order_receipts.append(record)
    assert math.ceil(training.FULL_COUNTS['train_core'] / 64) == 2078
    assert training.FULL_COUNTS['train_core'] % 64 == 50
    assert 20 * 2078 == 41560
    receipt = dict(schema='v5_fusion_full_training_verification_v1', status='passed', observed_at=now(),
        device=args.device, formal_training=False, cache=str(Path(args.cache).resolve()),
        cache_identity=index['identity'], cache_scope=index['scope'],
        cache_index_sha256=sha256(Path(args.cache) / 'index.json'),
        cache_read_policy='each complete partition is read/hashed once, then reused in memory',
        scope='complete_cache_contract_check' if args.cache_check_only else 'bounded_full_scope_training_preflight',
        formal_full_coverage_checked=formal_scope_validated,
        counts={part: len(packet['targets']) for part, packet in packets.items()},
        scene_counts={part: len({m['scene'] for m in packet['metadata']}) for part, packet in packets.items()},
        same_seed_actual_sample_batches_equal=True, actual_epoch_orders=order_receipts,
        expected_formal_counts=training.FULL_COUNTS, expected_formal_scenes=training.FULL_SCENES,
        expected_formal_updates=41560, expected_batches_per_epoch=2078, expected_last_batch_size=50,
        pilot_cache_rejected=True, full_preflight_cache_not_formal=True,
        source_sha256=sha256(ROOT / 'purevision_v5/fusion_grid_training.py'))
    if args.cache_check_only:
        assert formal_scope_validated
        write_json(args.output, receipt)
        print(f'Full cache training contract verified: {args.output}', flush=True)
        return receipt
    models = build_six(42)
    initialization = {}
    all_stores = []
    for group in GROUPS:
        before = torch.get_rng_state().clone()
        model = training.new_model(configs[group])
        assert torch.equal(before, torch.get_rng_state())
        assert model_state_sha256(model) == model_state_sha256(models[group])
        for name in ('layers', 'head'):
            assert model_state_sha256(getattr(model, name)) == model_state_sha256(getattr(models['J23'], name))
        all_stores.extend(p.data_ptr() for p in models[group].parameters())
        initialization[group] = model_state_sha256(model)
    assert len(all_stores) == len(set(all_stores))
    for name, value in models['F23'].named_parameters():
        assert torch.equal(value, dict(models['S23'].named_parameters())[name])
    for overrides in (dict(seed=43), dict(group='S46'), dict(expected_counts=dict(train_core=7324, dev=1826))):
        rejected(lambda overrides=overrides: training.validate_config(dict(configs['J23'], **overrides)))
    count = min(64, len(packets['train_core']['targets']))
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory(prefix='fusion_full_scope_preflight_') as temporary:
        base = Path(temporary)
        evaluation_calls, observed_ready = [], []
        actual_evaluator = training.evaluate_packet
        active_output = {'path': None}
        train_scenes = {m['scene'] for m in packets['train_core']['metadata']}
        def watched_evaluation(model, packet, runtime, device, config):
            is_train = packet['metadata'][0]['scene'] in train_scenes
            evaluation_calls.append('train_core' if is_train else 'dev')
            if is_train:
                ready = read_json(active_output['path'] / 'dev_ready.json')
                assert ready['status'] == 'preflight_dev_ready' and ready['train'] is None
                assert ready['best_train_epe'] is None and ready['best_dev_predictions_reused'] is True
                assert (active_output['path'] / 'eval_dev/predictions.csv').is_file()
                observed_ready.append(True)
            return actual_evaluator(model, packet, runtime, device, config)
        with patch.object(fusion_full_cache, 'load_full_cache', in_memory_loader), patch.object(training, 'evaluate_packet', watched_evaluation):
            for attempt in range(2):
                destination = base / f'attempt{attempt}'
                active_output['path'] = destination
                command = argparse.Namespace(config=str(ROOT / 'configs/next/FUSION_FULL_J23.json'),
                    cache=args.cache, output=str(destination), device=args.device, threads=2,
                    resume=None if attempt == 0 else str(base / 'attempt0/recovery.json'),
                    preflight=True, preflight_samples=count)
                training.train(command)
        assert evaluation_calls == ['dev', 'train_core', 'dev', 'train_core']
        assert observed_ready == [True, True]
        first = training.load_recovery(base / 'attempt0')[1]['current'][1]
        second = training.load_recovery(base / 'attempt1')[1]['current'][1]
        assert first['contract']['scope'] == second['contract']['scope'] == 'full_preflight'
        assert first['contract']['data_scope'] == 'full' and second['optimizer_updates'] == 2
        assert first['dev_count'] == min(count, len(packets['dev']['targets']))
        # Independent uninterrupted two-update reference verifies optimizer/RNG/scheduler recovery.
        seed_everything(42)
        reference = training.new_model(configs['J23']).to(device)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.0001,
                                     betas=(.9, .999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
        small = dict(features={k: v[:count] for k, v in packets['train_core']['features'].items()},
                     targets=packets['train_core']['targets'][:count], metadata=packets['train_core']['metadata'][:count])
        runtime, placement = training.resident_packet(small, device)
        norms, losses = [], []
        for epoch, checkpoint in enumerate((first, second)):
            reference.train()
            order, _ = epoch_order(count, 42, epoch)
            features, target = training.batch(runtime, torch.tensor(order, device=runtime['placement']), device)
            optimizer.zero_grad(set_to_none=True)
            prediction = reference(features)
            scale = target.new_tensor([5., 1.])
            loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=1., reduction='mean')
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reference.parameters())
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 2., error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            assert all(torch.equal(value.cpu(), checkpoint['model'][key]) for key, value in reference.state_dict().items())
            assert scheduler.state_dict() == checkpoint['scheduler']
            actual = optimizer.state_dict()
            assert actual['param_groups'] == checkpoint['optimizer']['param_groups']
            for pid, state in actual['state'].items():
                for key, value in state.items():
                    saved = checkpoint['optimizer']['state'][pid][key]
                    assert torch.equal(value.cpu(), saved) if torch.is_tensor(value) else value == saved
            norms.append(float(norm));losses.append(float(loss.detach()))
        for field, replacement in (('next_batch', 1), ('train_count', count + 1), ('dev_count', first['dev_count'] + 1)):
            broken = dict(first, **{field: replacement})
            rejected(lambda broken=broken: training.validate_checkpoint(broken))
        broken = copy.deepcopy(first)
        broken['contract']['scope'] = 'frozen_fusion_grid_pilot'
        broken['contract_hash'] = object_hash(broken['contract'])
        rejected(lambda: training.validate_checkpoint(broken))
        receipt.update(initialization_matches_fresh_seed42=initialization, independent_group_storage=True,
            shared_transformer_head_exact=True, s23_f23_flow_branch_exact=True,
            model_and_order_rng_independent=True, finite_loss_gradients=True, losses=losses, gradient_norms=norms,
            restored_model_optimizer_scheduler_exact_to_continuous_reference=True,
            complete_epoch_recovery_and_scope_guards=True, wrong_counts_and_partial_rejected=True,
            dev_ready_precedes_selected_train_evaluation=True, selected_train_evaluations_per_run=1,
            best_dev_predictions_reused=True, training_cache_placement=placement,
            temporary_models_and_outputs_only=True)
    write_json(args.output, receipt)
    print(f'Full-scope bounded training preflight passed: {args.output}', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cache-check-only', action='store_true', help='Only new full-cache coverage/orders; skip already verified temporary updates')
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
