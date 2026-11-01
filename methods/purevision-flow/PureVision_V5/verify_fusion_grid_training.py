"""Bounded GPU-cache and complete-epoch recovery check; never formal training."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import torch
import torch.nn.functional as F

from purevision_v5.common import now, read_json, sha256, write_json
from purevision_v5.fusion_grid_cache import load_grid_cache
from purevision_v5.fusion_grid_training import (batch, configure_fp32, load_recovery, new_model, resident_packet,
    train, validate_checkpoint)
from purevision_v5.training import epoch_order, model_state_sha256, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--first-run', required=True, help='Existing successful bounded preflight epoch1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    first = Path(args.first_run).resolve()
    manifest, loaded = load_recovery(first)
    initial_checkpoint = loaded['current'][1]
    assert initial_checkpoint['completed_epochs'] == 1 and initial_checkpoint['contract']['scope'] == 'preflight'
    config = read_json(args.config)
    configure_fp32(4)
    device = torch.device(args.device)
    packet, _ = load_grid_cache(args.cache, 'train_core')
    n = initial_checkpoint['train_count']
    packet = dict(features={key: value[:n] for key, value in packet['features'].items()},
                  targets=packet['targets'][:n], metadata=packet['metadata'][:n])
    runtime, placement = resident_packet(packet, device)
    assert placement['mode'] == 'gpu_resident'
    cpu_runtime = dict(features=packet['features'], targets=packet['targets'], placement=torch.device('cpu'))
    indices = torch.tensor([0, 17, 31, n - 1])
    cpu_features, cpu_target = batch(cpu_runtime, indices, device)
    gpu_features, gpu_target = batch(runtime, indices.to(device), device)
    assert all(torch.equal(cpu_features[key], gpu_features[key]) for key in cpu_features)
    assert torch.equal(cpu_target, gpu_target)
    model = new_model(config).to(device).eval()
    with torch.no_grad():
        cpu_result, gpu_result = model(cpu_features), model(gpu_features)
    assert torch.equal(cpu_result, gpu_result)
    with tempfile.TemporaryDirectory(prefix='fusion_grid_resume_preflight_') as directory:
        resumed_output = Path(directory) / 'resume'
        resume_args = argparse.Namespace(config=args.config, cache=args.cache, output=str(resumed_output),
            device=args.device, threads=4, resume=str(first / 'recovery.json'), preflight=True, preflight_samples=n)
        train(resume_args)
        resumed_manifest, resumed = load_recovery(resumed_output)
        resumed_checkpoint = resumed['current'][1]
        assert resumed_checkpoint['completed_epochs'] == 2 and resumed_checkpoint['optimizer_updates'] == 2
        assert resumed_checkpoint['next_batch'] == 0 and resumed_checkpoint['scheduler']['last_epoch'] == 2
        # Independent, uninterrupted two-update reference starts at prescribed initialization.
        seed_everything(42)
        reference = new_model(config).to(device)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.0001,
                                     betas=(.9, .999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
        losses, norms, states = [], [], []
        for epoch in range(2):
            reference.train()
            order, _ = epoch_order(n, 42, epoch)
            features, target = batch(runtime, torch.tensor(order, device=device), device)
            optimizer.zero_grad(set_to_none=True)
            prediction = reference(features)
            loss = F.smooth_l1_loss(prediction / target.new_tensor([5., 1.]), target / target.new_tensor([5., 1.]),
                                    beta=1., reduction='mean')
            assert torch.isfinite(loss)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in reference.parameters())
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            reference.eval()
            scheduler.step()
            checkpoint = initial_checkpoint if epoch == 0 else resumed_checkpoint
            assert all(torch.equal(value.cpu(), checkpoint['model'][key])
                       for key, value in reference.state_dict().items())
            assert scheduler.state_dict() == checkpoint['scheduler']
            ref_optimizer = optimizer.state_dict()
            assert ref_optimizer['param_groups'] == checkpoint['optimizer']['param_groups']
            for pid, state in ref_optimizer['state'].items():
                for key, value in state.items():
                    saved = checkpoint['optimizer']['state'][pid][key]
                    assert torch.equal(value.cpu(), saved) if torch.is_tensor(value) else value == saved
            losses.append(float(loss.detach()))
            norms.append(float(norm))
            states.append(model_state_sha256(reference))
        partial = copy.copy(resumed_checkpoint)
        partial['next_batch'] = 1
        partial_rejected = False
        try:
            validate_checkpoint(partial)
        except ValueError:
            partial_rejected = True
        assert partial_rejected
        # Export aliases have no authority: a damaged alias cannot alter recovery.
        (resumed_output / 'last.pt').unlink()
        (resumed_output / 'last.pt').write_bytes(b'non-authoritative export')
        alias_manifest, _ = load_recovery(resumed_output)
        assert alias_manifest['identity'] == resumed_manifest['identity']
        receipt = dict(schema='v5_fusion_grid_training_verification_v1', status='passed', verified_at=now(),
            gpu=read_json(first / 'run.json')['runtime_gpu'], formal_training=False,
            preflight_sample_count=n, formal_config_unchanged=True, gpu_cache_placement=placement,
            gpu_cpu_actual_batch_exact=True, gpu_cpu_forward_exact=True, finite_loss_and_all_gradients=True,
            losses=losses, gradient_norms=norms, initial_epoch_state_exact_to_uninterrupted_reference=True,
            restored_next_epoch_state_exact_to_uninterrupted_reference=True,
            restored_optimizer_exact=True, restored_scheduler_exact=True,
            state_sha256=states, complete_epoch_boundary=True, partial_state_rejected=True,
            recovery_ignores_export_aliases=True, best_dev_prediction_reused=True,
            train_evaluation_passes_per_selected_best=1,
            initial_run=str(first), initial_recovery_sha256=sha256(first / 'recovery.json'),
            resumed_recovery_identity=resumed_manifest['identity'],
            actual_sample_order_sha256=[row['actual_sample_order_sha256'] for row in resumed_checkpoint['order_records']])
        write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
