"""Isolated publisher retry checks: never invoke Git, tmux or real publishing."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name('completion_20261030_full_nuscenes_front1024_v2.py')
if not SCRIPT.exists():  # The byte-identical repository audit snapshot.
    SCRIPT = Path(__file__).with_name('completion_20261030_full_nuscenes_front1024.py')


@pytest.mark.parametrize('first_failure', ['timeout', 'pending', 'stale_zero_exit'])
def test_retry_survives_timeout_and_pending_receipt(tmp_path, monkeypatch, first_failure):
    spec = importlib.util.spec_from_file_location('completion_retry_fixture', SCRIPT)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    plan_path = tmp_path / 'pipeline' / watcher.PREFIX / 'plan.json'
    monkeypatch.setattr(watcher, 'PLAN', plan_path)
    (tmp_path / 'monitoring').mkdir()
    plan_path.parent.mkdir(parents=True)
    jobs = [dict(job_id=f'{watcher.PREFIX}_{i}') for i in range(5)]
    plan_path.write_text(json.dumps(dict(identity=watcher.IDENTITY, code=dict(commit=watcher.COMMIT), jobs=jobs)))
    (plan_path.parent / 'status.json').write_text(json.dumps(dict(plan_identity=watcher.IDENTITY, status='completed')))
    publication = tmp_path / 'monitoring/publication.json'
    initial = dict(status='published', training_commits=[watcher.COMMIT], report_commit='a' * 40,
                   verified_remote_commit='a' * 40, all_terminal=False, started_at='2000-01-01T00:00:00+00:00')
    publication.write_text(json.dumps(initial))
    status = tmp_path / 'report-worktree/methods/purevision-flow/PureVision_V5/docs/next_results/status.json'
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps(dict(rows=[dict(run_id=j['job_id'], training_commit=watcher.COMMIT,
                                                status='completed') for j in jobs])))
    calls, sleeps = [], []
    def fake_publish(command, timeout):
        assert command == watcher.COMMAND and timeout == 660
        calls.append(command)
        if len(calls) == 1:
            if first_failure == 'stale_zero_exit':
                return SimpleNamespace(returncode=0)
            publication.write_text(json.dumps(dict(status='pending', report_commit='b' * 40)))
            if first_failure == 'timeout':
                raise subprocess.TimeoutExpired(command, timeout)
            return SimpleNamespace(returncode=1)
        assert len(calls) == 2
        publication.write_text(json.dumps(dict(initial, report_commit='b' * 40,
            verified_remote_commit='b' * 40, all_terminal=True, started_at=watcher.now())))
        return SimpleNamespace(returncode=0)
    def no_real_wait(seconds):
        sleeps.append(seconds)
        assert seconds == 300  # No return to initial-publication waiting after a failure.
        current = json.loads((tmp_path / 'monitoring' / f'completion_publish_{watcher.PREFIX}.json').read_text())
        assert current['status'] == 'publication_pending' and current['initial_publication_latched']
        if first_failure == 'timeout':
            assert current['error'].startswith('TimeoutExpired:')
    monkeypatch.setattr(watcher.subprocess, 'run', fake_publish)
    monkeypatch.setattr(watcher.time, 'sleep', no_real_wait)
    watcher.main()
    assert len(calls) == 2 and sleeps == [300]
    proof = json.loads((tmp_path / 'monitoring' / f'completion_initial_publication_{watcher.PREFIX}.json').read_text())
    assert proof['commit'] == watcher.COMMIT and proof['plan_identity'] == watcher.IDENTITY
    assert proof['publication']['verified_remote_commit'] == 'a' * 40
    final = json.loads((tmp_path / 'monitoring' / f'completion_publish_{watcher.PREFIX}.json').read_text())
    assert final['status'] == 'published' and final['publication']['verified_remote_commit'] == 'b' * 40
