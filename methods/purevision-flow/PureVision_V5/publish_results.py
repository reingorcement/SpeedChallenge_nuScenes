"""Publish selected task reports in an isolated worktree; never control training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from monitor import FLOW_PROGRESS_FIELDS

REPORT_DIRECTORY = Path("methods/purevision-flow/PureVision_V5/docs/next_results")
REPORT_BRANCH = "results/purevision-v5-next"
TERMINAL = {"completed", "failed", "interrupted", "cancelled", "blocked"}
MAX_BYTES = 16 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_bytes(content)
    temporary.replace(path)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"api_key", "openai_api_key", "access_token", "refresh_token", "authorization", "password", "network.env", "network.json"}:
                raise ValueError("Private credential/control field excluded from publication")
            check_public(item)
    elif isinstance(value, list):
        for item in value:
            check_public(item)
    elif isinstance(value, str) and re.search(r"(?:sk-[A-Za-z0-9_-]{24,}|(?:OPENAI_API_KEY|access_token|refresh_token|password)\s*[:=]\s*[^\s,}]+)", value):
        raise ValueError("Possible credential value excluded from publication")


def source_path(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()) or path.name in {"network.env", "network.json"}:
        raise ValueError("Publication source escaped artifact root")
    return path


def read_source(root, path, *, lines=False):
    path = source_path(root, path)
    if not path.exists():
        return None, None
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        blob = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(blob) > MAX_BYTES:
        raise ValueError("Metadata exceeded publication size limit")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) and not lines:
        raise ValueError("Source changed during read; retry next publication")
    if lines:
        complete = blob[:blob.rfind(b"\n") + 1]
        value = [json.loads(line) for line in complete.splitlines() if line.strip()]
        blob = complete
    else:
        value = json.loads(blob)
    check_public(value)
    return value, dict(path=str(path), source_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                       representation="complete_jsonl_records" if lines else "parsed_json")


def command_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def collect(root, plans):
    """Export original small JSON, complete epoch records and bounded predictions."""
    root = Path(root).resolve()
    files, rows, sources = {}, [], []
    for plan_path in plans:
        plan, origin = read_source(root, plan_path)
        if not plan or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", plan.get("prefix", "")):
            raise ValueError("Invalid/missing task plan")
        prefix = plan["prefix"]
        files[f"plans/{prefix}.json"] = encoded(plan)
        sources.append(origin)
        coordinator, coordinator_origin = read_source(root, root / "pipeline" / prefix / "status.json")
        if coordinator and coordinator.get("plan_identity") == plan.get("identity") and plan.get("identity"):
            files[f"plans/{prefix}_coordinator.json"] = encoded(coordinator)
            sources.append(coordinator_origin)
        else:
            coordinator = {}
        for job in plan["jobs"]:
            run_id = job["job_id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
                raise ValueError("Unsafe run identity")
            raw, source = read_source(root, job["status_path"])
            state = (raw or {}).get("status", "queued")
            effective = coordinator.get("jobs", {}).get(job["name"], {})
            if state not in TERMINAL and effective.get("status") == "unknown":
                state = "unknown"
            elif state not in TERMINAL and effective.get("status") == "interrupted":
                from monitor import confirmed_pipeline_terminal
                if confirmed_pipeline_terminal({**job, "pipeline_id": prefix}, root, raw or {}):
                    state = "interrupted"
                else:
                    state = "unknown"
            row = dict(run_id=run_id, name=job["name"], experiment=job.get("experiment", job["name"]),
                       status=state, kind=job["kind"], training_commit=plan["code"]["commit"],
                       plan=prefix, device=job.get("device"), log_path=job.get("log_path"))
            if raw:
                files[f"experiments/{run_id}/job_status.json"] = encoded(raw)
                sources.append(source)
            if job["kind"] != "training":
                row["progress"] = {k: raw[k] for k in FLOW_PROGRESS_FIELDS if raw and k in raw}
                if job.get("progress_path"):
                    progress, progress_origin = read_source(root, job["progress_path"])
                    if progress is not None:
                        if not isinstance(progress, dict):
                            raise ValueError("Non-training progress report must be an object")
                        files[f"experiments/{run_id}/progress_report.json"] = encoded(progress)
                        sources.append(progress_origin)
                        row["progress"].update({k: progress[k] for k in FLOW_PROGRESS_FIELDS if k in progress})
                        if "pairs_processed" in progress:
                            row["progress"]["completed"] = progress["pairs_processed"]
                        row["progress_report_status"] = progress.get("status")
                if job.get("expected_pairs") is not None:
                    row["progress"]["expected_pairs"] = job["expected_pairs"]
                rows.append(row)
                continue
            directory = source_path(root, job["run_dir"])
            run, _ = read_source(root, directory / "run.json")
            if run and (run.get("run_id") != run_id or run.get("code", {}).get("commit") != plan["code"]["commit"]):
                raise ValueError("Training identity differs from planned run")
            for name in ("run.json", "status.json", "metrics.jsonl"):
                value, origin = read_source(root, directory / name, lines=name.endswith("jsonl"))
                if value is not None:
                    files[f"experiments/{run_id}/{name}"] = (
                        b"".join((json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n").encode() for v in value)
                        if name.endswith("jsonl") else encoded(value))
                    sources.append(origin)
                    if name == "status.json":
                        row["training_status"] = value.get("status")
                        row["progress"] = {k: value[k] for k in ("epoch", "global_step", "optimizer_updates", "amp_skips", "best_epe", "best_epoch") if k in value}
            evaluation_command = next((c for c in job.get("commands", []) if any(Path(str(v)).name == "evaluate.py" for v in c)), [])
            evaluation_dir = command_value(evaluation_command, "--output")
            if evaluation_dir:
                evaluation_dir = source_path(root, evaluation_dir)
                for name in ("evaluation.json", "metrics.json"):
                    value, origin = read_source(root, evaluation_dir / name)
                    if value is not None:
                        files[f"experiments/{run_id}/evaluation/{name}"] = encoded(value)
                        sources.append(origin)
                        if name == "evaluation.json":
                            row["evaluation_status"] = value.get("status")
                        else:
                            row["dev_metrics"] = value.get("overall")
                prediction = evaluation_dir / "predictions.csv"
                if row.get("evaluation_status") == "completed" and prediction.is_file():
                    if prediction.stat().st_size > MAX_BYTES:
                        row["predictions"] = dict(path=str(prediction), sha256=checksum(prediction), representation="hash_only_size_limit")
                    else:
                        blob = prediction.read_bytes()
                        files[f"experiments/{run_id}/evaluation/predictions.csv"] = blob
                        row["predictions"] = dict(path=str(prediction), sha256=hashlib.sha256(blob).hexdigest(), representation="full_csv")
            if state == "completed" and row.get("evaluation_status") != "completed":
                row["status"] = "evaluation_pending"
            for name in ("last.pt", "best.pt"):
                checkpoint = directory / name
                if checkpoint.is_file():
                    # A single open inode is stable under the training writer's atomic replacement.
                    with checkpoint.open("rb") as stream:
                        info = os.fstat(stream.fileno())
                        sha = hashlib.file_digest(stream, "sha256").hexdigest()
                    row.setdefault("checkpoint_hashes", {})[name] = dict(path=str(checkpoint), sha256=sha, size_bytes=info.st_size, representation="hash_only")
            proof = root / "monitoring/startup" / run_id / "startup_proof.json"
            value, origin = read_source(root, proof)
            if value is not None:
                files[f"experiments/{run_id}/startup_proof.json"] = encoded(value)
                sources.append(origin)
            rows.append(row)
    # Explicit additional evidence allowlist; never recurse into logs, data or monitoring secrets.
    for evidence in sorted(root.glob("evidence/*.json")):
        value, origin = read_source(root, evidence)
        files[f"evidence/{evidence.name}"] = encoded(value)
        sources.append(origin)
    # Publish small verified patrol receipts; never publish private network controls or raw AI logs.
    for name in ("latest.json","last_astra.json","schedule.json","probe_schedule.json","publisher_schedule.json"):
        value,origin=read_source(root,root/"monitoring"/name)
        if value is not None:
            files[f"monitoring/{name}"]=encoded(value);sources.append(origin)
            if name=="last_astra.json" and value.get("answer"):
                path=source_path(root,value["answer"])
                if path.is_file():
                    blob=path.read_bytes()
                    if len(blob)>24000:raise ValueError("Patrol answer exceeds bounded report size")
                    check_public(blob.decode('utf-8'))
                    files["monitoring/last_astra.txt"]=blob
                    files[f"monitoring/patrols/{path.name}"]=blob
    for plan_path in plans:
        plan=json.loads(Path(plan_path).read_text())
        value,origin=read_source(root,root/"monitoring"/f"lifecycle_{plan['prefix']}.json")
        if value is not None:
            files[f"monitoring/lifecycle_{plan['prefix']}.json"]=encoded(value);sources.append(origin)
    reports = {*root.glob("reports/resolution*/summary.*"), *root.glob("reports/full_flow_*/summary.*"),
               *root.glob("reports/encoder_comparison_*/summary.*"), *root.glob("reports/fusion_comparison_*/summary.*")}
    for report in sorted(reports):
        if report.suffix not in {".json", ".md"}:
            continue
        if report.suffix == ".json":
            value, origin = read_source(root, report)
            blob = encoded(value)
        else:
            report = source_path(root, report)
            with report.open("rb") as stream:
                before = os.fstat(stream.fileno())
                blob = stream.read(MAX_BYTES + 1)
                after = os.fstat(stream.fileno())
            if len(blob) > MAX_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Comparison summary is oversized or changed during reading")
            check_public(blob.decode())
            origin = dict(path=str(report), source_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), representation="original_markdown")
        files[f"comparisons/{report.parent.name}/{report.name}"] = blob
        sources.append(origin)
    summary = dict(schema="v5_next_public_report_v1", rows=rows,
                   all_terminal=bool(rows) and all(r["status"] in TERMINAL for r in rows),
                   limits=["PLANAR_PROXY labels; dev selection and dev recomputation are not held-out testing.",
                           "Checkpoint hashes identify server files; weights are not uploaded.",
                           "Source records are read snapshots, not an independent model rerun."])
    files["status.json"] = encoded(summary)
    lines = ["# V5 下一阶段实验状态", "", "这是自动发布的工件快照；算法决策由用户确认。作业源码与结果发布提交分别记录。", "", "| run ID | 阶段作业 | 状态 | 代码 commit | 光流已处理/总数 | dev EPE |", "|---|---|---|---|---|---|"]
    for row in rows:
        progress = row.get("progress", {})
        pairs = "—" if row["kind"] == "training" else f"{progress.get('completed', '—')}/{progress.get('total', progress.get('expected_pairs', '—'))}"
        lines.append(f"| {row['run_id']} | {row['name']} | {row['status']} | `{row['training_commit']}` | {pairs} | {row.get('dev_metrics', {}).get('epe', '—')} |")
    lines += ["", "训练作业目录包含实际配置、完整 epoch JSONL、状态与可用的 dev 评价；非训练作业包含 worker 状态及已生成的 progress_report.json。原始大日志、数据/光流和权重不上传。", "`source_index.json` 记录所读文件与摘要；`status.json` 记录 checkpoint 哈希和预测公开形式。", "没有结果的组仍是未开始/运行中/失败，不能按计划填入指标。", ""]
    files["README.md"] = "\n".join(lines).encode()
    files["source_index.json"] = encoded(dict(
        sources=sorted(sources, key=lambda v: v["path"]),
        exports=[dict(path=name, export_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)) for name, blob in sorted(files.items())],
        note="source_sha256 hashes source bytes (JSONL complete prefix); export_sha256 hashes the published representation. JSON may be reformatted; index excludes itself."))
    return files, summary


def git(repo, *args, check=True):
    from purevision_v5.common import proxy_ssh_command
    transport = ["-c", f"core.sshCommand={proxy_ssh_command()}"] if args and args[0] in {"ls-remote", "push", "fetch"} else []
    result = subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo),
                             "-c", "user.name=Codex", "-c", "user.email=codex@openai.com", *transport, *args],
                            text=True, capture_output=True, timeout=120, check=False)
    if check and result.returncode:
        raise RuntimeError(f"git {args[0]} failed (exit {result.returncode}): {result.stderr[-1200:]}")
    return result.stdout.strip()


@contextmanager
def publication_lock(root):
    path = root / "monitoring/publisher.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def publish(root, worktree, plans, *, push_url=None, branch=REPORT_BRANCH, stop_when_terminal=False, trigger="manual"):
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    receipt_path = root / "monitoring/publication.json"
    with publication_lock(root):
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt = {**{k: previous[k] for k in ("report_commit", "remote_base") if k in previous},
                   "started_at": now(), "worktree": str(worktree), "branch": branch, "status": "pending", "trigger": trigger}
        try:
            if not worktree.is_relative_to(root) or worktree == root:
                raise ValueError("Results worktree must be isolated beneath the private artifact root")
            if git(worktree, "branch", "--show-current") != branch:
                raise ValueError("Unexpected results branch")
            if git(worktree, "status", "--porcelain", "--untracked-files=all"):
                raise ValueError("Results worktree contains unknown changes; refusing automatic staging")
            remote = push_url or "origin"
            remote_ref = f"refs/heads/{branch}"
            remote_line = git(worktree, "ls-remote", remote, remote_ref)
            remote_sha = remote_line.split()[0] if remote_line else None
            local_sha = git(worktree, "rev-parse", "HEAD")
            pending_local = previous.get("status") == "pending" and previous.get("report_commit") == local_sha
            if remote_sha and remote_sha != local_sha:
                # Only retry our exact prior commit from a failed push; never merge/rebase automatically.
                if not pending_local or previous.get("remote_base") != remote_sha:
                    raise ValueError("Remote results branch changed; manual reconciliation required")
                git(worktree, "push", remote, f"HEAD:{remote_ref}")
                if git(worktree, "ls-remote", remote, remote_ref).split()[0] != local_sha:
                    raise ValueError("Pending publication could not be verified remotely")
                remote_sha = local_sha
            files, summary = collect(root, plans)
            target = worktree / REPORT_DIRECTORY
            target.mkdir(parents=True, exist_ok=True)
            for relative, blob in files.items():
                path = target / relative
                if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                    raise ValueError("Report output escaped its allowlisted directory")
                if not path.exists() or path.read_bytes() != blob:
                    atomic(path, blob)
            # Stage exactly generated paths; no repository-wide add, no removal of historical reports.
            changed = git(worktree, "status", "--porcelain", "--untracked-files=all")
            if changed:
                paths = [str(REPORT_DIRECTORY / relative) for relative in files]
                git(worktree, "add", "--", *paths)
                staged = git(worktree, "diff", "--cached", "--name-only").splitlines()
                if any(not p.startswith(str(REPORT_DIRECTORY) + "/") for p in staged):
                    raise ValueError("Unexpected staged file outside results allowlist")
                # csv.DictWriter produces CRLF. Preserve the original prediction
                # bytes/hash while treating that line ending as valid for this check.
                git(worktree, "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
                    "diff", "--cached", "--check")
                git(worktree, "commit", "-m", "docs(v5-next): publish experiment progress")
            report_commit = git(worktree, "rev-parse", "HEAD")
            receipt.update(report_commit=report_commit, remote_base=remote_sha or local_sha,
                           plan_identities=[value["identity"] for p in plans
                                            if "identity" in (value := json.loads(Path(p).read_text()))],
                           changed=bool(changed), training_commits=sorted({r["training_commit"] for r in summary["rows"]}))
            atomic(receipt_path, encoded(receipt))
            if changed or not remote_sha or pending_local:
                git(worktree, "push", remote, f"HEAD:{remote_ref}")
            observed = git(worktree, "ls-remote", remote, remote_ref)
            if not observed or observed.split()[0] != report_commit:
                raise ValueError("Remote report commit verification failed")
            receipt.update(status="published", verified_remote_commit=report_commit, finished_at=now(), all_terminal=summary["all_terminal"])
            atomic(receipt_path, encoded(receipt))
            schedule_path = root / "monitoring/publisher_schedule.json"
            if trigger == "scheduled" and schedule_path.exists():
                schedule = json.loads(schedule_path.read_text())
                schedule.update(status="verified_trigger", last_result=receipt)
                atomic(schedule_path, encoded(schedule))
            if stop_when_terminal and summary["all_terminal"]:
                from monitor import modify_cron
                modify_cron(root, publisher_only=True)
            return receipt
        except Exception as error:
            receipt.update(status="pending", error=f"{type(error).__name__}: {error}", finished_at=now())
            atomic(receipt_path, encoded(receipt))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--plan", required=True, action="append", type=Path)
    parser.add_argument("--push-url")
    parser.add_argument("--branch", default=REPORT_BRANCH)
    parser.add_argument("--stop-when-terminal", action="store_true")
    parser.add_argument("--trigger", choices=("manual", "scheduled"), default="manual")
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.worktree, args.plan, push_url=args.push_url, branch=args.branch,
                             stop_when_terminal=args.stop_when_terminal, trigger=args.trigger)))


if __name__ == "__main__":
    main()
