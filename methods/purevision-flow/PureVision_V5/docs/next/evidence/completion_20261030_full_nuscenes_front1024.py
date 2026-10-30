"""Recorded one-plan completion publisher; no job mutation or algorithm action."""
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/zhangyu/.local/share/speedest-v5-next')
PREFIX = '20261030_full_nuscenes_front1024'
COMMIT = '87d7468f911da5579bbbc39f79ec6bc85fa16dd6'
IDENTITY = 'cbc303bbcd40f1189db741cab7cce6f3938d6a3be3e3326ce182555ce95cf620'
FIXED = ROOT / 'code/87d7468f911d/methods/purevision-flow/PureVision_V5'
COMMAND = ['bash', str(FIXED / 'scripts/publish_results.sh'), str(ROOT),
           str(ROOT / 'monitoring' / f'publisher_command_{PREFIX}.json'), 'manual']
PLAN = ROOT / 'pipeline' / PREFIX / 'plan.json'
TERMINAL = {'completed', 'failed', 'interrupted', 'blocked', 'cancelled'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def record(**values):
    value = dict(time_utc=now(), prefix=PREFIX, commit=COMMIT,
                 plan_identity=IDENTITY, watcher_version=2, **values)
    write(ROOT / 'monitoring' / f'completion_publish_{PREFIX}.json', value)
    print(json.dumps(value), flush=True)


def exported_rows():
    value = load(ROOT / 'report-worktree/methods/purevision-flow/PureVision_V5/docs/next_results/status.json')
    return {row.get('run_id'): row for row in value.get('rows', [])}


def published(receipt):
    return (receipt.get('status') == 'published' and COMMIT in receipt.get('training_commits', [])
            and bool(receipt.get('report_commit'))
            and receipt.get('verified_remote_commit') == receipt.get('report_commit'))


def latch_initial_publication(plan):
    """Keep the proven initial publication despite later pending receipts."""
    proof_path = ROOT / 'monitoring' / f'completion_initial_publication_{PREFIX}.json'
    proof = load(proof_path)
    if (proof.get('commit') == COMMIT and proof.get('plan_identity') == IDENTITY
            and published(proof.get('publication', {}))):
        return proof
    path = ROOT / 'monitoring/publication.json'
    receipt, rows = load(path), exported_rows()
    covered = all(rows.get(job['job_id'], {}).get('training_commit') == COMMIT for job in plan['jobs'])
    if published(receipt) and covered:
        proof = dict(time_utc=now(), commit=COMMIT, plan_identity=IDENTITY,
                     publication=receipt, current_job_ids=[job['job_id'] for job in plan['jobs']],
                     receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write(proof_path, proof)
        return proof
    return None


def main():
    initial_proof = None
    record(status='waiting', command=COMMAND, initial_publication_latched=False)
    while True:
        plan = load(PLAN)
        jobs = plan.get('jobs', [])
        valid_plan = plan.get('identity') == IDENTITY and plan.get('code', {}).get('commit') == COMMIT and len(jobs) == 5
        if valid_plan and initial_proof is None:
            initial_proof = latch_initial_publication(plan)
            if initial_proof:
                record(status='waiting', command=COMMAND, initial_publication_latched=True,
                       initial_verified_remote_commit=initial_proof['publication']['verified_remote_commit'])
        state = load(PLAN.parent / 'status.json')
        if (valid_plan and initial_proof and state.get('plan_identity') == IDENTITY
                and state.get('status') in {'completed', 'finished_with_failures'}):
            started, error, exit_code = now(), None, None
            try:
                exit_code = subprocess.run(COMMAND, timeout=660).returncode
            except (subprocess.TimeoutExpired, OSError) as exc:
                error = f'{type(exc).__name__}: {exc}'
            receipt, rows = load(ROOT / 'monitoring/publication.json'), exported_rows()
            covered = all(rows.get(job['job_id'], {}).get('status') in TERMINAL
                          and rows.get(job['job_id'], {}).get('training_commit') == COMMIT for job in jobs)
            verified = (exit_code == 0 and error is None and published(receipt)
                        and receipt.get('all_terminal') is True and receipt.get('started_at', '') >= started and covered)
            record(status='published' if verified else 'publication_pending', command=COMMAND,
                   subprocess_exit=exit_code, error=error, publication=receipt,
                   current_plan_terminal_rows=covered, initial_publication_latched=True,
                   next_retry_seconds=None if verified else 300)
            if verified:
                return
            time.sleep(300)
            continue
        time.sleep(30)


if __name__ == '__main__':
    main()
