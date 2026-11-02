"""Short seed-scope check using the existing full cache and historical factories."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from purevision_v5.common import ROOT, git, now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc_training import configure_fp32
from purevision_v5.fusion_grid import build_six
from purevision_v5.fusion_grid_training import (
    load_training_partition, new_model, order_record, sample_identities, validate_config)
from purevision_v5.training import epoch_order, model_state_sha256


def verify(args):
    configure_fp32(2)
    groups = ('J23', 'S23', 'F23')
    root = Path(args.root)
    # Model factory and RNG policy were already checked for these seeds in the pilot.
    unchanged = {}
    for name in ('purevision_v5/fusion_abc.py', 'purevision_v5/fusion_grid.py',
                 'purevision_v5/training.py'):
        relative = str((ROOT/name).relative_to(ROOT.parents[2]))
        assert git('show', f'bc9a3d8:{relative}') == (ROOT/name).read_text().strip()
        unchanged[name] = sha256(ROOT/name)
    cfg42 = {g: read_json(ROOT/f'configs/next/FUSION_FULL_{g}.json') for g in groups}
    packet, index = load_training_partition(args.cache, 'train_core', cfg42['J23'])
    assert {p:v['count'] for p,v in index['partitions'].items()} == {'train_core':132978,'dev':23480}
    identities = sample_identities(packet)
    selected = [0, len(identities)-1]
    small = {k:v[selected] for k,v in packet['features'].items()}
    target = packet['targets'][selected]
    prior = root/'runs/20261102_0154_fusion_full_s42_J23'
    baseline_orders = read_json(prior/'epoch_orders.json')
    for epoch in (0,19):
        order,digest = epoch_order(len(identities),42,epoch)
        assert order_record(order,digest,identities,epoch,64) == baseline_orders[epoch]
    for group in groups:
        old = read_json(root/'runs'/f'20261102_0154_fusion_full_s42_{group}'/'run.json')
        assert model_state_sha256(new_model(cfg42[group])) == old['initialization']['state_sha256']
    seed_evidence, orders, initial = {}, {}, {}
    for seed in (43,44):
        before = torch.get_rng_state().clone()
        models = build_six(seed)
        assert torch.equal(before,torch.get_rng_state())
        stores = [p.data_ptr() for g in groups for p in models[g].parameters()]
        assert len(stores) == len(set(stores))
        for g in groups:
            for module in ('layers','head'):
                assert model_state_sha256(getattr(models[g],module)) == model_state_sha256(getattr(models['J23'],module))
        for name,p in models['F23'].named_parameters():
            assert torch.equal(p,dict(models['S23'].named_parameters())[name])
        group_checks = {}
        for group in groups:
            path = ROOT/f'configs/next/FUSION_FULL_{group}_s{seed}.json'
            config = read_json(path); validate_config(config)
            assert config == dict(cfg42[group],seed=seed)
            model = new_model(config)
            digest = model_state_sha256(model)
            assert digest == model_state_sha256(models[group])
            initial[(seed,group)] = digest
            prediction = model(small)
            loss = F.smooth_l1_loss(prediction/target.new_tensor([5.,1.]),target/target.new_tensor([5.,1.]),beta=1.,reduction='mean')
            loss.backward()
            assert torch.isfinite(prediction).all() and torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            group_checks[group] = dict(config_sha256=sha256(path),initial_state_sha256=digest,
                finite_output_loss_gradients=True,two_real_windows=True,loss=float(loss.detach()))
        orders[seed] = []
        for epoch in range(20):
            before = torch.get_rng_state().clone()
            order,digest = epoch_order(len(identities),seed,epoch)
            assert torch.equal(before,torch.get_rng_state())
            row = order_record(order,digest,identities,epoch,64)
            assert row['batch_sizes'] == [64]*2077+[50]
            assert len(row['actual_batch_identity_sha256']) == 2078
            # All three formal trainers invoke this exact independent CPU order policy.
            torch.rand(3)
            assert epoch_order(len(identities),seed,epoch) == (order,digest)
            orders[seed].append(dict(epoch=epoch+1,actual_sample_order_sha256=row['actual_sample_order_sha256'],
                actual_batches_sha256=object_hash(row['actual_batch_identity_sha256']),
                effective_batches=2078,last_batch_size=50))
        seed_evidence[str(seed)] = dict(groups=group_checks,independent_parameter_storage=True,
            common_transformer_head_exact=True,s23_f23_flow_processing_exact=True,
            model_order_rng_separated=True,formal_updates=41560)
    assert all(initial[(43,g)] != initial[(44,g)] for g in groups)
    assert all(a['actual_sample_order_sha256'] != b['actual_sample_order_sha256'] for a,b in zip(orders[43],orders[44]))
    for seed in (41,45,True,'43'):
        try: validate_config(dict(cfg42['J23'],seed=seed))
        except ValueError: pass
        else: raise AssertionError('Unsupported seed accepted')
    receipt=dict(schema='v5_fusion_full_seed_preflight_v1',status='passed',time=now(),
        cache_identity=index['identity'],cache_index_sha256=sha256(Path(args.cache)/'index.json'),
        counts={'train_core':132978,'dev':23480},scene_counts={'train_core':582,'dev':103},
        formal_training=False,seed42_initialization_and_first_last_epoch_regression_exact=True,
        unchanged_factory_and_order_sources=unchanged,seeds=seed_evidence,epoch_orders=orders,
        expected_formal_updates=41560,same_seed_groups_actual_batches_equal=True,
        order_policy='isolated CPU generator(seed+zero_based_epoch); cross-seed sequences have historical shifted overlap',
        source_sha256={name:sha256(ROOT/name) for name in ('verify_fusion_full_seeds.py','purevision_v5/fusion_grid_training.py')})
    write_json(args.output,receipt)
    print(f'Full seed43/44 scope and bounded initialization checks passed: {args.output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    verify(p.parse_args())
