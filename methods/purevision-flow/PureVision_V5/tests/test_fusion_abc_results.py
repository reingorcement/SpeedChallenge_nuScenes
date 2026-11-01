"""Bounded reports must not promote partial ABC training into final evidence."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import fusion_abc_comparison as comparison
from purevision_v5.common import object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import summarize
from purevision_v5.fusion_abc_training import PROTOCOL, SCHEMA


def test_relative_delta_handles_zero_baseline():
    baseline = {key: 0. for key in comparison.METRICS}
    candidate = {key: 1. for key in comparison.METRICS}
    delta = comparison.difference(candidate, baseline)
    assert delta['epe'] == {'delta': 1., 'absolute_magnitude': 1., 'relative_percent': None}


def test_partial_group_withholds_all_comparisons(tmp_path):
    jobs = []
    for v in comparison.VARIANTS:
        out = tmp_path / v; out.mkdir()
        write_json(out / 'run.json', {'status': 'completed' if v != 'C' else 'running', 'completed_epochs': 20 if v != 'C' else 19})
        for part in ('train_core', 'dev'):
            write_json(out / f'eval_{part}' / 'evaluation.json', {'status': 'completed'})
        jobs.append(dict(name=v, variant=v, job_id=f'pilot_{v}', kind='training', run_dir=str(out)))
    plan = dict(prefix='pilot', code=dict(commit='a' * 40), jobs=jobs)
    plan['identity'] = object_hash(plan); write_json(tmp_path / 'plan.json', plan)
    report, curves, predictions = comparison.collect(tmp_path / 'plan.json')
    assert report['status'] == 'incomplete'
    assert report['comparisons'] == {} and curves == predictions == {}
    assert report['rows']['C']['completed_epochs'] == 19


def finished_fixture(tmp_path):
    directory = tmp_path / 'pilot_A'; directory.mkdir()
    config = {**PROTOCOL, 'variant': 'A'}
    config_path = tmp_path / 'A.json'; write_json(config_path, config)
    index = dict(identity='test_cache_identity', source=dict(checkpoint_sha256=comparison.SOURCE_SHA))
    contract = dict(config=config, cache_identity=index['identity'], scope='frozen_fusion_pilot', smoke_steps=0)
    code = dict(commit='a' * 40, verified_remote_commit='a' * 40, dirty=False)
    annotations = {}; metrics = {}
    for part, count in [('train_core', 7324), ('dev', 1826)]:
        out = directory / f'eval_{part}'; out.mkdir()
        values = [dict(scene=f'scene_{i % (32 if part == "train_core" else 8)}', token=f'{part}_{i}', time_s=float(i),
                       target_vx=0., target_vy=0., vx=.2, vy=.01) for i in range(count)]
        annotations[part] = values
        actual = summarize([[p['vx'], p['vy']] for p in values], [[0., 0.]] * count, [p['scene'] for p in values])
        actual['nonzero_vy']['status'] = 'insufficient_data'; actual['task'] = 'PLANAR_PROXY'
        metrics[part] = actual; write_json(out / 'metrics.json', actual)
        with (out / 'predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    history = []
    for epoch in range(1, 21):
        history.append(dict(epoch=epoch, dev=metrics['dev'], processed_batches=115 * epoch, optimizer_updates=115 * epoch,
                            amp_skips=0, learning_rates=[.001 * (1 + math.cos(math.pi * (epoch - 1) / 20)) / 2],
                            sample_order_sha256=f'order_{epoch}', training_loss=.01, next_learning_rates=[0.]))
    (directory / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
    write_json(directory / 'epoch_orders.json', [dict(epoch=i, order_sha256=f'order_{i}') for i in range(1, 21)])
    packet = dict(schema=SCHEMA, contract=contract, contract_hash=object_hash(contract), config=config,
                  cache_identity=index['identity'], amp_enabled=False, amp_skips=0,
                  processed_batches=2300, optimizer_updates=2300, run_id=directory.name,
                  code=code, completed_epochs=20, next_batch=0, best_epoch=1, best_epe=metrics['dev']['overall']['epe'],
                  scheduler=dict(T_max=20, eta_min=0, last_epoch=20))
    torch.save(packet, directory / 'last.pt'); torch.save(packet, directory / 'best.pt')
    run = dict(run_id=directory.name, code=code, contract=contract, contract_hash=object_hash(contract), train_count=7324,
               dev_count=1826, trainable_parameters=124354, encoder_training=False,
               arithmetic=dict(dtype='float32', amp=False, tf32=False, gradient_accumulation=1),
               status='completed', completed_epochs=20, processed_batches=2300, optimizer_updates=2300,
               best_checkpoint_sha256=sha256(directory / 'best.pt'), last_checkpoint_sha256=sha256(directory / 'last.pt'), initialization={})
    write_json(directory / 'run.json', run); write_json(directory / 'status.json', run); write_json(directory / 'config.json', config)
    for part in ('train_core', 'dev'):
        write_json(directory / f'eval_{part}' / 'evaluation.json', dict(status='completed', partition=part,
            checkpoint_sha256=run['best_checkpoint_sha256'], training_run=directory.name, training_code=code,
            evaluation_code=code, cache_identity=index['identity'], contract_hash=run['contract_hash'], count=len(annotations[part])))
    plan = dict(code=code, config_hashes={'A': sha256(config_path)})
    job = dict(run_dir=str(directory), job_id=directory.name, config_path=str(config_path), name='A')
    return plan, job, index, annotations, directory


def test_complete_inspection_independently_recomputes_predictions(tmp_path):
    args = finished_fixture(tmp_path)
    row, history, predictions = comparison.inspect(args[0], 'A', args[1], args[2], args[3])
    assert row['processed_batches'] == row['optimizer_updates'] == 2300
    assert row['best_epoch'] == 1  # Equal EPE preserves the earlier epoch.
    assert row['evaluations']['dev']['metrics']['nonzero_vy']['status'] == 'insufficient_data'
    assert len(history) == 20 and len(predictions['train_core']) == 7324


def test_claimed_complete_but_partial_checkpoint_rejected(tmp_path):
    plan, job, index, annotations, directory = finished_fixture(tmp_path)
    packet = torch.load(directory / 'last.pt', weights_only=True)
    packet.update(completed_epochs=19, next_batch=41, processed_batches=2226, optimizer_updates=2226)
    torch.save(packet, directory / 'last.pt')
    with pytest.raises(ValueError, match='Last checkpoint is not completed20'):
        comparison.inspect(plan, 'A', job, index, annotations)


def test_independent_prediction_target_mismatch_rejected(tmp_path):
    plan, job, index, annotations, directory = finished_fixture(tmp_path)
    path = directory / 'eval_dev' / 'predictions.csv'
    text = path.read_text().replace('dev_0,', 'wrong_token,', 1); path.write_text(text)
    with pytest.raises(ValueError, match='sample/target mismatch'):
        comparison.inspect(plan, 'A', job, index, annotations)


def test_pair_export_joins_by_scene_token_not_row_order(tmp_path):
    predictions = {}
    for variant, dx in [('A', 0.), ('B', .1), ('C', .2)]:
        rows = [dict(scene='s', token=f't{i}', time_s=str(i), target_vx='0.0', target_vy='0.0', vx=str(i + dx), vy='0.0') for i in (1, 2)]
        if variant == 'B': rows.reverse()
        predictions[variant] = {p: rows for p in ('dev', 'train_core')}
    histories = {v: [dict(epoch=1, training_loss=.1, dev=dict(overall=dict(epe=.2)), learning_rates=[.001], next_learning_rates=[.0009], processed_batches=115, optimizer_updates=115)] for v in comparison.VARIANTS}
    report = dict(comparisons={'B_minus_A': dict(per_scene={'s': comparison.difference({k:.1 for k in comparison.METRICS}, {k:.2 for k in comparison.METRICS})})})
    comparison.export_tables(tmp_path, report, histories, predictions)
    with (tmp_path / 'paired_dev_predictions.csv').open() as stream: rows = list(csv.DictReader(stream))
    assert [r['token'] for r in rows] == ['t1', 't2']
    assert float(rows[0]['B_minus_A_epe']) == pytest.approx(.1)
    assert (tmp_path / 'learning_curves.svg').is_file()


def test_public_collector_excludes_cache_arrays_weights_and_private_controls(tmp_path):
    from collect_fusion_abc import collect_public
    root = tmp_path / 'art'; root.mkdir()
    cache = root / 'cache'; cache.mkdir()
    write_json(cache / 'index.json', {'identity': 'public_index'})
    (cache / 'train_core_features.pt').write_bytes(b'private-feature-bytes')
    write_json(root / 'monitoring/network.json', {'password': 'must-not-export'})
    jobs = []
    for v in comparison.VARIANTS:
        directory = root / 'runs' / f'pilot_{v}'; directory.mkdir(parents=True)
        write_json(directory / 'run.json', {'status': 'running'})
        write_json(directory / 'status.json', {'status': 'running', 'processed_batches': 1})
        (directory / 'last.pt').write_bytes(b'weights-should-only-be-hashed')
        log = root / f'{v}.log'; log.write_text(json.dumps({'event': 'update', 'loss': .1}) + '\n')
        jobs.append(dict(name=v, variant=v, job_id=f'pilot_{v}', kind='training', run_dir=str(directory),
                         status_path=str(root / f'{v}.json'), device='cuda:0', log_path=str(log)))
    plan = dict(stage='frozen_fusion_abc_pilot', prefix='pilot', code=dict(commit='a' * 40), jobs=jobs,
                cache_root=str(cache), completion_report=str(root / 'report/summary.json'))
    monitoring = root / 'monitoring/fusion_abc_pilot'
    for name in ('schedule_install.json', 'schedule_trigger.json', 'lifecycle.json', 'cleanup_receipt.json'):
        write_json(monitoring / name, {'current_plan': 'pilot', 'receipt': name})
    for name in ('cache', 'A', 'B', 'C', 'compare'):
        write_json(monitoring / 'startup' / f'{name}.json', {'run': f'pilot_{name}'})
    write_json(monitoring / 'network.json', {'password': 'scoped-private-control'})
    plan_path = root / 'plan.json'; write_json(plan_path, plan)
    files, rows, sources = collect_public(root, [plan_path])
    assert len(rows) == 3
    assert files['fusion_abc/pilot/cache/index.json']
    assert all(not name.endswith('.pt') and 'network.json' not in name for name in files)
    assert rows[0]['checkpoint_hashes']['last.pt']['representation'] == 'hash_only'
    assert all(b'private-feature-bytes' not in blob and b'must-not-export' not in blob for blob in files.values())

    for name in ('schedule_install.json', 'schedule_trigger.json', 'lifecycle.json', 'cleanup_receipt.json'):
        assert json.loads(files[f'fusion_abc/pilot/monitoring/{name}'])['current_plan'] == 'pilot'
    for name in ('cache', 'A', 'B', 'C', 'compare'):
        assert json.loads(files[f'fusion_abc/pilot/monitoring/startup/{name}.json'])['run'] == f'pilot_{name}'
    assert all(b'scoped-private-control' not in blob for blob in files.values())
