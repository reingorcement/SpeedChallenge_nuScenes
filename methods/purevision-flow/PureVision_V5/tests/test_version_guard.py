"""Formal work cannot bypass a clean, detached, remotely published code version."""
import subprocess

import pytest

from purevision_v5 import common


def fake_git(monkeypatch, *, dirty=False, branch="HEAD"):
    monkeypatch.setattr(common, "git", lambda *args, **kwargs: {
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain", "--untracked-files=normal"): " M bad.py" if dirty else "",
        ("rev-parse", "--abbrev-ref", "HEAD"): branch,
    }[args])


@pytest.mark.parametrize("ref", ["refs/heads/method/purevision-v5-next", "refs/tags/v5-code-aaaaaaaaaaaa"])
def test_verified_pushed_next_branch_or_code_tag(monkeypatch, ref):
    fake_git(monkeypatch)
    monkeypatch.setattr(common, "pushed_refs", lambda commit: [f"{commit}\t{ref}"])
    assert common.code_version(True)["verified_remote_commit"] == "a" * 40


@pytest.mark.parametrize("dirty,branch,remote", [(True, "HEAD", "a" * 40),
                                                (False, "method/purevision-v5-next", "a" * 40),
                                                (False, "HEAD", "b" * 40)])
def test_unpublished_dirty_or_mutable_worktree_rejected(monkeypatch, dirty, branch, remote):
    fake_git(monkeypatch, dirty=dirty, branch=branch)
    monkeypatch.setattr(common, "pushed_refs", lambda commit: [f"{remote}\trefs/heads/method/purevision-v5-next"])
    with pytest.raises(ValueError):
        common.code_version(True)


def test_https_failure_uses_scoped_proxy_ssh_and_does_not_accept_stale_cache(monkeypatch):
    calls = []
    def call(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("git", 25)
        return "a" * 40 + "\trefs/heads/method/purevision-v5-next"
    monkeypatch.setattr(common, "git", call)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7900")
    assert common.pushed_refs("a" * 40)
    assert len(calls) == 2
    assert any("ProxyCommand=" in arg and "127.0.0.1:7900" in arg for arg in calls[1])
    assert not any("StrictHostKeyChecking=no" in arg for arg in calls[1])


def test_authenticated_proxy_not_copied_to_command(monkeypatch):
    calls = []
    def call(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("unavailable")
        raise OSError("still unavailable")
    monkeypatch.setattr(common, "git", call)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:do_not_expose@127.0.0.1:7900")
    with pytest.raises(OSError):
        common.pushed_refs("a" * 40)
    assert "do_not_expose" not in repr(calls)
