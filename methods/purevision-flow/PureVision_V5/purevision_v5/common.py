"""Small artifact and provenance helpers; outputs live outside the source tree."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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


def git(*arguments, timeout=45):
    repo = ROOT.parents[2]
    return subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *arguments],
                                   text=True, timeout=timeout).strip()


def proxy_ssh_command():
    """Build scoped SSH arguments without copying proxy credentials into argv."""
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1"]
    # GitHub is the only network SSH host used here. Respect the same bypass
    # policy as Git/libcurl; otherwise an explicit ProxyCommand ignores NO_PROXY.
    bypass = False
    for item in (os.environ.get("NO_PROXY", "") + "," + os.environ.get("no_proxy", "")).split(","):
        host = item.strip().lower()
        if ":" in host:
            host, port = host.rsplit(":", 1)
            if port != "22":
                continue
        host = host.lstrip(".")
        if host == "*" or (host and ("github.com" == host or "github.com".endswith("." + host))):
            bypass = True
            break
    if bypass:
        return shlex.join(ssh)
    proxy = urlsplit(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "")
    if proxy.scheme == "http" and proxy.hostname and not proxy.username and not proxy.password:
        proxy_command = shlex.join(["nc", "-X", "connect", "-x",
                                   f"{proxy.hostname}:{proxy.port or 80}", "%h", "%p"])
        ssh += ["-o", f"ProxyCommand={proxy_command}"]
    return shlex.join(ssh)


def pushed_refs(commit):
    """Verify the actual remote, with a bounded SSH fallback for HTTPS outages.

    Scope transport options to this invocation; never rewrite global Git/SSH
    settings or weaken host-key checks. Only an unauthenticated HTTP proxy can
    be passed to nc; credentials remain in the normal HTTPS transport.
    """
    refs = ("refs/heads/method/purevision-v5", "refs/heads/method/purevision-v5-next",
            f"refs/tags/v5-code-{commit[:12]}")
    try:
        return git("-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=15",
                   "ls-remote", "origin", *refs, timeout=25).splitlines()
    except (subprocess.SubprocessError, OSError):
        pass
    # GitHub can transiently time out even when the immutable tag was just
    # pushed. Retry real remote observations, never substitute a cached SHA.
    for attempt in range(3):
        try:
            return git("-c", "url.git@github.com:.insteadOf=https://github.com/",
                       "-c", f"core.sshCommand={proxy_ssh_command()}",
                       "ls-remote", "origin", *refs, timeout=25).splitlines()
        except (subprocess.SubprocessError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def code_version(require_pushed=False):
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    remote = None
    if require_pushed:
        if dirty:
            raise ValueError("Formal training requires a clean fixed checkout")
        lines = pushed_refs(commit)
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
