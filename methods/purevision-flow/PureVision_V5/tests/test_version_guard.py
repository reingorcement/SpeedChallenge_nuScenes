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
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
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
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:do_not_expose@127.0.0.1:7900")
    with pytest.raises(OSError):
        common.pushed_refs("a" * 40)
    assert "do_not_expose" not in repr(calls)


@pytest.mark.parametrize("key,bypass", [("NO_PROXY", "localhost,github.com"), ("no_proxy", ".github.com"),
                                        ("NO_PROXY", "*"), ("NO_PROXY", "github.com:22")])
def test_github_no_proxy_bypasses_ssh_proxy_without_removing_other_proxy_environment(monkeypatch, key, bypass):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7900")
    monkeypatch.setenv(key, bypass)
    assert "ProxyCommand=" not in common.proxy_ssh_command()
    assert common.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7900"


@pytest.mark.parametrize("bypass", ["notgithub.com", "github.com:443", "example.com"])
def test_unrelated_or_other_port_no_proxy_does_not_silently_bypass_ssh(monkeypatch, bypass):
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", bypass)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7900")
    assert "ProxyCommand=" in common.proxy_ssh_command()


def test_transient_remote_failures_retry_actual_remote_before_accepting(monkeypatch):
    calls = []
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    def network(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise OSError("temporary network failure")
        return "a" * 40 + "\trefs/tags/v5-code-" + "a" * 12
    monkeypatch.setattr(common, "git", network)
    assert common.pushed_refs("a" * 40) == ["a" * 40 + "\trefs/tags/v5-code-" + "a" * 12]
    assert len(calls) == 3
    assert all("ls-remote" in call for call in calls)


def test_remote_retries_are_bounded_and_never_accept_stale_evidence(monkeypatch):
    calls = []
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    def network(*args, **kwargs):
        calls.append(args)
        raise OSError("offline")
    monkeypatch.setattr(common, "git", network)
    with pytest.raises(OSError, match="offline"):
        common.pushed_refs("a" * 40)
    assert len(calls) == 4
