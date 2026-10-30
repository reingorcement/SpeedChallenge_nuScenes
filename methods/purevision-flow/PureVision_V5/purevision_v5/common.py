"""Small artifact and provenance helpers; outputs live outside the source tree."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def git(*arguments):
    repo = ROOT.parents[2]
    return subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *arguments], text=True).strip()


def code_version(require_pushed=False):
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    remote = None
    if require_pushed:
        if dirty:
            raise ValueError("Formal training requires a clean fixed checkout")
        lines = git("-c", "http.proxy=", "ls-remote", "origin", "refs/heads/method/purevision-v5",
                    f"refs/tags/v5-code-{commit[:12]}").splitlines()
        if not any(line.split()[0] == commit for line in lines):
            raise ValueError("Formal training commit must match a pushed V5 branch tip or immutable code tag")
        remote = commit
        if git("rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
            raise ValueError("Formal training must run from a detached training worktree")
    return dict(commit=commit, dirty=dirty, verified_remote_commit=remote)


def environment():
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, platform=platform.platform(), hostname=platform.node(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
