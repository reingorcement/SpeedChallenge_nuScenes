"""Bounded seed43/44 preflight, preserving the fixed seed42 implementation.

Uses immutable existing pilot caches, CPU models and two actual windows per
numerical check. Full 20-epoch order receipts only hash real sample identities;
no formal training, feature extraction, or GPU jobs are started.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import random
import types

import numpy as np
import torch
import torch.nn.functional as F

from purevision_v5.common import ROOT, git, now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc import build_triplet, common_initialization_evidence
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_grid import GRID_VARIANTS, build_six, initialization_evidence
from purevision_v5.fusion_grid_cache import load_grid_cache, ordered_sample_digest
from purevision_v5.fusion_grid_training import new_model, order_record, sample_identities, validate_config
from purevision_v5.training import epoch_order, model_state_sha256

GROUPS = ('J23', 'S23', 'F23', 'S46')
SEEDS = (43, 44)


def cpu_rng_identity():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
                numpy=(state[0], state[1].tolist(), *state[2:]))


def assert_rng_equal(before, after):
    assert torch.equal(before['torch'], after['torch'])
    assert before['python'] == after['python'] and before['numpy'] == after['numpy']


def revision_module(revision, filename, name):
    relative = str((ROOT / 'purevision_v5' / filename).relative_to(ROOT.parents[2]))
    source = git('show', f'{revision}:{relative}')
    module = types.ModuleType(f'purevision_v5.{name}')
    module.__package__ = 'purevision_v5'
    exec(compile(source, f'{revision}:{relative}', 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def verify(args):
    configure_fp32(2)
    packets, indices, identities, small, targets = {}, {}, {}, {}, {}
    for grid, location in (((2, 3), args.cache23), ((4, 6), args.cache46)):
        packet, index = load_grid_cache(location, 'train_core')
        assert len(packet['targets']) == 7324 and index['scope'] == 'pilot'
        packets[grid], indices[grid] = packet, index
        identities[grid] = sample_identities(packet)
        # Fixed original first and last windows cover two actual sample identities.
        selected = [0, len(packet['targets']) - 1]
        small[grid] = {key: value[selected].clone() for key, value in packet['features'].items()}
        targets[grid] = packet['targets'][selected].clone()
    assert identities[(2, 3)] == identities[(4, 6)]
    assert torch.equal(targets[(2, 3)], targets[(4, 6)])
    assert torch.equal(small[(2, 3)]['dt'], small[(4, 6)]['dt'])
    fixed = git('rev-parse', args.baseline_commit)
    # Verify the untouched historical factory used by the old and new modules.
    relative = str((ROOT / 'purevision_v5/fusion_abc.py').relative_to(ROOT.parents[2]))
    historical_source = git('show', f'{fixed}:{relative}')
    assert historical_source == (ROOT / 'purevision_v5/fusion_abc.py').read_text().strip()
    old_grid, old_grid_sha = revision_module(fixed, 'fusion_grid.py', '_seed42_reference_grid')
    old_trainer, old_trainer_sha = revision_module(fixed, 'fusion_grid_training.py', '_seed42_reference_training')
    original = old_grid.build_six(42)
    current = build_six(42)
    seed42 = {}
    for group in GRID_VARIANTS:
        config_path = ROOT / f'configs/next/FUSION_GRID_{group}.json'
        config = read_json(config_path)
        old_trainer.validate_config(config)
        validate_config(config)
        relative = str(config_path.relative_to(ROOT.parents[2]))
        assert git('show', f'{fixed}:{relative}') == config_path.read_text().strip()
        assert all(torch.equal(value, current[group].state_dict()[key])
                   for key, value in original[group].state_dict().items())
        with torch.no_grad():
            before = original[group].eval()(small[original[group].grid])
            after = current[group].eval()(small[current[group].grid])
        assert torch.equal(before, after)
        seed42[group] = dict(config_bytes_unchanged=True, complete_state_exact=True,
            forward_exact=True, state_sha256=model_state_sha256(current[group]))
    receipts, orders, initial_hashes = {}, {}, {}
    config_hashes = {}
    for seed in SEEDS:
        rng_before = cpu_rng_identity()
        models = build_six(seed)
        assert_rng_equal(rng_before, cpu_rng_identity())
        historical = build_triplet(seed)
        historical_copy = common_initialization_evidence(historical)
        assert all(historical_copy[name]['exact_equal'] and historical_copy[name]['independent_storage']
                   for name in ('input_projection', 'input_norm', 'head'))
        evidence = initialization_evidence(models, small, device='cpu', seed=seed)
        evidence['historical_B_to_C'] = historical_copy
        evidence['model_initialization_preserves_caller_rng'] = True
        evidence['updates'] = {}
        initial_hashes[seed] = {group: model_state_sha256(models[group]) for group in GROUPS}
        for group in GROUPS:
            config_path = ROOT / f'configs/next/FUSION_GRID_{group}_s{seed}.json'
            config = read_json(config_path)
            validate_config(config)
            baseline_config = read_json(ROOT / f'configs/next/FUSION_GRID_{group}.json')
            assert config == dict(baseline_config, seed=seed)
            config_hashes[f'{group}_s{seed}'] = sha256(config_path)
            # new_model is the same actual entry point used by the formal trainer.
            model = new_model(config).train()
            assert model_state_sha256(model) == initial_hashes[seed][group]
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['optimizer_eps'])
            prediction = model(small[model.grid])
            target = targets[model.grid]
            scale = target.new_tensor(config['loss_scales_m_s'])
            loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=1., reduction='mean')
            axes = F.smooth_l1_loss(prediction.detach() / scale, target / scale, beta=1., reduction='none').mean(0)
            torch.testing.assert_close(loss.detach(), axes.mean(), atol=1e-7, rtol=1e-6)
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                                for p in model.parameters())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            evidence['updates'][group] = dict(actual_windows=2, loss=float(loss.detach()),
                loss_x=float(axes[0]), loss_y=float(axes[1]), gradient_norm=float(norm),
                all_gradients_finite=True, all_updated_parameters_finite=True,
                training_factory_matches_seed_initialization=True, formal_training=False)
        orders[seed] = []
        for epoch in range(20):
            rng_before = cpu_rng_identity()
            order, digest = epoch_order(7324, seed, epoch)
            assert_rng_equal(rng_before, cpu_rng_identity())
            record23 = order_record(order, digest, identities[(2, 3)], epoch, 64)
            record46 = order_record(order, digest, identities[(4, 6)], epoch, 64)
            assert record23 == record46
            assert len(record23['batch_sizes']) == 115 and record23['batch_sizes'][-1] == 28
            # Vary global RNG and consume all factories; isolated data-order RNG is invariant.
            torch.rand(3)
            altered_order, altered_digest = epoch_order(7324, seed, epoch)
            assert altered_order == order and altered_digest == digest
            assert old_trainer.epoch_order(7324, seed, epoch) == (order, digest)
            orders[seed].append(record23)
        evidence['same_seed_all_groups_actual_batches_exact'] = True
        evidence['data_order_rng_independent_of_global_state'] = True
        evidence['historical_epoch_order_policy_unchanged'] = True
        evidence['expected_formal_updates'] = sum(len(row['batch_sizes']) for row in orders[seed])
        assert evidence['expected_formal_updates'] == 2300
        receipts[str(seed)] = evidence
    for group in GROUPS:
        assert initial_hashes[43][group] != initial_hashes[44][group]
    assert all(orders[43][epoch]['actual_sample_order_sha256'] != orders[44][epoch]['actual_sample_order_sha256']
               for epoch in range(20))
    assert all(orders[43][epoch + 1]['actual_sample_order_sha256'] == orders[44][epoch]['actual_sample_order_sha256']
               for epoch in range(19))
    invalid_seeds = [None, 0, 41, 45, True, 43.0, '43']
    sample_config = read_json(ROOT / 'configs/next/FUSION_GRID_J23.json')
    for invalid in invalid_seeds:
        try:
            validate_config(dict(sample_config, seed=invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Unsupported seed was accepted: {invalid!r}')
    for key, changed in (('epochs', 19), ('batch_size', 32), ('amp', True), ('tf32', True),
                         ('loss_scales_m_s', [1, 1])):
        try:
            validate_config(dict(sample_config, seed=43, **{key: changed}))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Protocol mutation accepted: {key}')
    output = dict(schema='v5_fusion_grid_seed_verification_v1', status='passed', verified_at=now(),
        device='cpu', formal_training=False, new_formal_run_scope=[f'{group}_s{seed}' for seed in SEEDS for group in GROUPS],
        baseline_commit=fixed, baseline_grid_source_sha256=old_grid_sha,
        baseline_trainer_source_sha256=old_trainer_sha,
        historical_factory_source_exact=True, seed42_regression=seed42,
        configuration_sha256=config_hashes,
        caches={f'{grid[0]}x{grid[1]}': dict(path=str(Path(location).resolve()), identity=indices[grid]['identity'],
            ordered_sample_digest=ordered_sample_digest(packets[grid]),
            index_sha256=sha256(Path(location) / 'index.json'))
            for grid, location in (((2, 3), args.cache23), ((4, 6), args.cache46))},
        seeds=receipts, actual_epoch_orders={str(seed): value for seed, value in orders.items()},
        cross_seed_initial_states_differ=True, cross_seed_same_numbered_epoch_orders_differ=True,
        order_policy='historical torch.randperm with isolated CPU generator(seed+zero_based_epoch)',
        order_policy_note='Adjacent seeds reuse shifted epoch permutations under the unchanged historical policy; '
                          'same-numbered epochs differ, but cross-seed sequences are not globally disjoint.',
        unsupported_seeds_rejected=True, fixed_protocol_mutations_rejected=True,
        source_file_sha256={name: sha256(ROOT / name) for name in (
            'purevision_v5/fusion_grid.py', 'purevision_v5/fusion_grid_training.py', 'verify_fusion_grid_seeds.py')})
    write_json(args.output, output)
    print(f'Seed43/44 preflight passed; seed42 regression exact; evidence: {args.output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache23', required=True)
    parser.add_argument('--cache46', required=True)
    parser.add_argument('--baseline-commit', default='6d8d6fe')
    parser.add_argument('--output', required=True)
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
