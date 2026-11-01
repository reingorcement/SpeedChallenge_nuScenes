"""Lock the three user-selected historical best checkpoints and proven predictions.

Read-only verification reuses completed independent evaluations, binds their
checkpoint receipts and identical training-best CSVs, and checks the original
full-dev labels by (scene, token). It never trains or evaluates a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from collect_results import check_summary, require, target_identity
from fusion_grid_comparison import read_history, read_predictions, summarize_rows
from purevision_v5.common import ROOT, now, object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import summarize, validate_lateral_rule
from purevision_v5.fusion_abc_cache import MANIFEST_SHA
from purevision_v5.selection import label_rows


ART = Path('/home/zhangyu/.local/share/speedest-v5-next')
MANIFEST = Path('/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1/dataset.json')
HISTORY_SPECS = {
    'FLEX_original': dict(run_id='20261031_r02_base_flex_s42_R02_FLEX', budget=40, best_epoch=37,
        sha256='4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823',
        role='original flow source supplemental reference', parent_flow_sha256=None),
    'FLEX_continued': dict(run_id='20261031_r03_flex_s42_R02_FLEX_CONT', budget=40, best_epoch=29,
        sha256='6bb8a1b3563c7137a6af8073888f6bcf7b99e8b2dad5a79689aef61a86691ac9',
        role='historical main flow reference', parent_flow_sha256='4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823'),
    'RGB_appearance': dict(run_id='20261101_rgb_path_s42_R03_APPEAR_FT_s42_18ep', budget=18, best_epoch=1,
        sha256='ab25fd6c4a7cd14a99a189f56e6b47913ed0f946a60cff2d0c9102479bc30ff8',
        role='historical dual-path reference and frozen encoder source',
        parent_flow_sha256='4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823'),
}


def identity_map(rows):
    result = {}
    for row in rows:
        key = (row['scene'], row['token'])
        require(key not in result, f'Duplicate sample identity {key}')
        result[key] = (float(row['time_s']), float(row['target_vx']), float(row['target_vy']))
    return result


def lateral_labels(rows, rule):
    chosen = [row for row in rows if abs(float(row['target_vy'])) >= rule['nonzero_threshold_m_s']]
    scenes = len({row['scene'] for row in chosen})
    adequate = len(chosen) >= rule['min_subset_windows'] and scenes >= rule['min_subset_scenes']
    return dict(count=len(chosen), scene_count=scenes, threshold_m_s=rule['nonzero_threshold_m_s'],
                status='sufficient_data' if adequate else 'insufficient_data', independent_windows=True)


def verify_reference(root, name, spec, labels):
    directory = root / 'runs' / spec['run_id']
    checkpoint = directory / 'best.pt'
    require(sha256(checkpoint) == spec['sha256'], f'Fixed historical checkpoint SHA differs: {name}')
    packet = torch.load(checkpoint, map_location='cpu', weights_only=True)
    run = read_json(directory / 'run.json')
    config = packet['config']
    require(packet['schema'] == 'v5_training_state_v1' and packet['contract_hash'] == object_hash(packet['contract'])
            and config == packet['contract']['config'], 'Historical checkpoint contract differs')
    require(packet['run_id'] == spec['run_id'] == run['run_id'] and run['contract'] == packet['contract']
            and run['code'] == packet['code'], 'Historical run/checkpoint provenance differs')
    require(config['seed'] == 42 and config['epochs'] == spec['budget'] and
            packet['best_epoch'] + 1 == spec['best_epoch'] == packet['epoch'] and packet['next_batch'] == 0,
            'Historical seed/budget/best epoch differs from user lock')
    require(run['status'] == 'completed' and run['scope'] == 'full_train_dev' and
            (run['train_windows'], run['dev_windows'], run['train_scenes'], run['dev_scenes']) ==
            (132978, 23480, 582, 103), 'Historical full-data coverage differs')
    require(run['initialization'] == packet['initialization'], 'Initialization receipt differs')
    require(packet['contract']['data']['dataset_sha256'] == MANIFEST_SHA, 'Historical dataset differs')
    history = read_history(directory / 'metrics.jsonl')
    require(len(history) == spec['budget'] and [row['epoch'] for row in history] == list(range(spec['budget'])),
            'Historical stage budget was not completed')
    best = min(history, key=lambda row: row['validation']['overall']['epe'])
    require(best['epoch'] + 1 == spec['best_epoch'] and best['validation']['overall']['epe'] == packet['best_epe'],
            'Historical selected checkpoint is not the recorded best/earlier tie')
    parents = {}
    for key in ('flow_initialization', 'rgb_initialization'):
        if key in config:
            parent = config[key]
            path = Path(parent['path'])
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            require(sha256(path) == parent['sha256'], 'Historical parent checkpoint hash differs')
            parents[key] = dict(path=str(path), sha256=parent['sha256'])
    if spec['parent_flow_sha256']:
        require(parents['flow_initialization']['sha256'] == spec['parent_flow_sha256'], 'Wrong historical flow parent')
    else:
        require('flow_initialization' not in parents, 'Original FLEX should use its recorded default initialization')
    evaluation = root / 'evaluations' / spec['run_id']
    receipt = read_json(evaluation / 'evaluation.json')
    require(receipt['status'] == 'completed' and receipt['checkpoint_sha256'] == spec['sha256'] and
            Path(receipt['checkpoint']).resolve() == checkpoint.resolve() and receipt['training_run'] == spec['run_id'],
            'Independent historical evaluation does not identify the locked checkpoint')
    require(receipt['training_code'] == packet['code'] and receipt['contract_hash'] == packet['contract_hash'] and
            receipt['data'] == packet['contract']['data'] and receipt['selection_hash'] == packet['contract']['dev_selection'],
            'Historical independent evaluation provenance differs')
    require(receipt['scope'] == 'full_train_dev' and receipt.get('partition', 'dev') == 'dev' and
            receipt['count'] == 23480 and receipt['runtime'].get('selection') is None,
            'Historical evaluation does not cover original full dev')
    predictions = evaluation / 'predictions.csv'
    require(sha256(predictions) == sha256(directory / 'dev_predictions.csv'),
            'Independent fixed-checkpoint predictions differ from training-best predictions')
    rows = read_predictions(predictions)
    require(identity_map(rows) == identity_map(labels), 'Historical prediction identities/labels/times/full coverage differ')
    rule = validate_lateral_rule(config['lateral_rule'])
    actual = summarize([[float(row['vx']), float(row['vy'])] for row in rows],
                       [[float(row['target_vx']), float(row['target_vy'])] for row in rows],
                       [row['scene'] for row in rows], rule)
    for recorded in (read_json(evaluation / 'metrics.json'), read_json(directory / 'best_metrics.json'), best['validation']):
        check_summary(recorded, actual)
    require(actual['overall']['epe'] == packet['best_epe'], 'Prediction EPE differs from locked checkpoint selection')
    return dict(name=name, role=spec['role'], run_id=spec['run_id'], seed=42,
        checkpoint=str(checkpoint), checkpoint_sha256=spec['sha256'], selected_epoch_one_based=spec['best_epoch'],
        stage_budget_epochs=spec['budget'], scheduler_epochs=config.get('scheduler_epochs', config['epochs']),
        budget_interpretation='this stage only; weight initialization ancestry is recorded separately, not all training from scratch',
        parent_checkpoints=parents, initialization=packet['initialization'], config=config,
        training_code=packet['code'], contract_hash=packet['contract_hash'],
        best_updates=packet['optimizer_updates'], best_amp_skips=packet['amp_skips'],
        predictions=str(predictions), predictions_sha256=sha256(predictions),
        training_best_predictions=str(directory / 'dev_predictions.csv'),
        independent_evaluation_receipt=str(evaluation / 'evaluation.json'),
        independent_evaluation_receipt_sha256=sha256(evaluation / 'evaluation.json'),
        proof=dict(independent_receipt_binds_exact_checkpoint_sha=True, training_best_csv_exact_match=True,
            recorded_best_epoch_and_metrics_exact_match=True, identity_key=['scene', 'token'],
            complete_unique_dev_identity_labels_timestamps_exact_match=True, no_new_model_inference=True),
        count=len(rows), scene_count=len({row['scene'] for row in rows}), target_identity=target_identity(rows),
        metrics=summarize_rows(rows, config))


def create_manifest(root=ART, manifest=MANIFEST):
    root, manifest = Path(root), Path(manifest)
    require(sha256(manifest) == MANIFEST_SHA, 'Full original dataset manifest changed')
    labels = {part: label_rows(manifest, part)[0] for part in ('train_core', 'dev')}
    require(len(labels['train_core']) == 132978 and len(labels['dev']) == 23480, 'Full label counts differ')
    require(len({row['scene'] for row in labels['train_core']}) == 582 and
            len({row['scene'] for row in labels['dev']}) == 103, 'Full scene counts differ')
    maps = {part: identity_map(rows) for part, rows in labels.items()}
    require(not (set(maps['train_core']) & set(maps['dev'])), 'Full partition identity conflict')
    references = {name: verify_reference(root, name, spec, labels['dev']) for name, spec in HISTORY_SPECS.items()}
    rule = next(iter(references.values()))['config']['lateral_rule']
    value = dict(schema='v5_fusion_full_history_manifest_v1', status='locked_verified', created_at=now(),
        lock_policy='user-predeclared checkpoints; immutable before full training; no post-result substitution',
        dataset_manifest=str(manifest), dataset_sha256=MANIFEST_SHA,
        full_labels={part: dict(count=len(rows), scene_count=len({row['scene'] for row in rows}),
            target_identity=target_identity(rows), lateral=lateral_labels(rows, rule)) for part, rows in labels.items()},
        references=references, new_model_inference=False, matched_ablation=False,
        comparison_scope='historical accuracy context; architecture, encoder training, precision and stage budgets differ')
    value['identity'] = object_hash(value)
    return value


def load_manifest(path):
    value = read_json(path)
    require(value.get('status') == 'locked_verified' and value.get('identity') ==
            object_hash({k: v for k, v in value.items() if k != 'identity'}), 'Historical manifest lock differs')
    require(set(value['references']) == set(HISTORY_SPECS), 'Historical reference set changed')
    for name, row in value['references'].items():
        require(row['checkpoint_sha256'] == HISTORY_SPECS[name]['sha256'] and
                sha256(row['predictions']) == row['predictions_sha256'], 'Locked historical prediction/source differs')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ART)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Historical lock must use a new output; never overwrite a prior lock')
    result = create_manifest(args.root, args.manifest)
    write_json(args.output, result)
    print(f"Historical references locked: {args.output}; identity={result['identity']}", flush=True)


if __name__ == '__main__':
    main()
