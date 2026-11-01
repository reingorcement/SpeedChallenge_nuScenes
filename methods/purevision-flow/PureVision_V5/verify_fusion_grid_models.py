"""One disposable six-model initialization/gradient/save-load preflight."""
from __future__ import annotations

import argparse
import copy
import io
from pathlib import Path

import torch
from torch.nn import functional as F

from purevision_v5.common import now, read_json, sha256, write_json
from purevision_v5.fusion_grid import build_six, initialization_evidence
from purevision_v5.fusion_abc_training import configure_fp32


def verify(cache23, cache46, device, output):
    configure_fp32(2)
    caches = {(2, 3): Path(cache23), (4, 6): Path(cache46)}
    features, annotations, provenance, packets = {}, {}, {}, {}
    for grid, root in caches.items():
        index = read_json(root / 'index.json')
        entry = index['partitions']['dev']
        feature_path, annotation_path = root / entry['file'], root / entry['annotations']
        if sha256(feature_path) != entry['sha256'] or sha256(annotation_path) != entry['annotations_sha256']:
            raise RuntimeError('Preflight input file hash mismatch')
        raw_features = torch.load(feature_path, map_location='cpu', weights_only=True)
        annotation = torch.load(annotation_path, map_location='cpu', weights_only=True)
        packets[grid] = (raw_features, annotation)
        provenance[f'{grid[0]}x{grid[1]}'] = dict(path=str(root), identity=index['identity'],
            index_sha256=sha256(root / 'index.json'), source=index['source']['checkpoint_sha256'],
            partition='dev', sample_scope='first two 4x6 rows matched by full metadata to the 2x3 cache')
    chosen = packets[(4, 6)][1]['metadata'][:2]
    for grid, (raw_features, annotation) in packets.items():
        positions = {entry['token']: i for i, entry in enumerate(annotation['metadata'])}
        ids = [positions[entry['token']] for entry in chosen]
        if [annotation['metadata'][i] for i in ids] != chosen:
            raise RuntimeError('Fixed preflight samples differ across grids')
        features[grid] = {key: value[ids].clone() for key, value in raw_features.items()}
        annotations[grid] = annotation['targets'][ids].clone()
        provenance[f'{grid[0]}x{grid[1]}']['fixed_sample_metadata'] = chosen
    if not torch.equal(annotations[(2, 3)], annotations[(4, 6)]) or not torch.equal(features[(2, 3)]['dt'], features[(4, 6)]['dt']):
        raise RuntimeError('Fixed preflight labels/dt differ across grids')
    models = build_six()
    evidence = initialization_evidence(models, features, device=device)
    evidence.update(schema='v5_fusion_grid_model_preflight_v1', observed_at=now(), status='passed',
        inputs=provenance, device=device, precision=dict(dtype='float32', amp=False, tf32=False),
        formal_restart='disposable preflight models/optimizers; formal training recreates all prescribed initial state',
        source_file_sha256={name: sha256(Path(__file__).parent / 'purevision_v5' / name)
                            for name in ('fusion_abc.py', 'fusion_grid.py')}, updates={})
    for name, original in models.items():
        model = copy.deepcopy(original).to(device).train()
        values = {key: value.to(device) for key, value in features[model.grid].items()}
        target = annotations[model.grid].to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,
                                     betas=(.9, .999), eps=1e-8)
        scale = torch.tensor([5., 1.], device=device)
        prediction = model(values)
        loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=1, reduction='mean')
        axis_losses = F.smooth_l1_loss(prediction.detach() / scale, target / scale, beta=1, reduction='none').mean(0)
        torch.testing.assert_close(loss.detach(), axis_losses.mean(), atol=1e-7, rtol=1e-6)
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Nonfinite loss/gradient in {name}')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        other = copy.deepcopy(original).to(device)
        other.load_state_dict(torch.load(stream, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            a, b = model.eval()(values), other.eval()(values)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        evidence['updates'][name] = dict(loss=float(loss.detach()), loss_x=float(axis_losses[0]),
            loss_y=float(axis_losses[1]), gradient_norm_before_clip=float(norm), gradients_finite=True,
            checkpoint_model_roundtrip_exact=True, input_token_count=model.token_count)
    write_json(output, evidence)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache23', required=True)
    parser.add_argument('--cache46', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.cache23, args.cache46, args.device, args.output)
    print(f"Six-model preflight {result['status']}; evidence: {args.output}", flush=True)
