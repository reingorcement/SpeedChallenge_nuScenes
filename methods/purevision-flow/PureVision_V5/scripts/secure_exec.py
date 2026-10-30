"""Execute an argument vector with JSON proxy values; never source shell text."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import stat

PROXY_KEYS = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}


def trusted_file(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Control file must not be a symlink")
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("Control file requires task ownership and private permissions")
    for directory in path.absolute().parents:
        info = directory.stat()
        if info.st_mode & 0o022 or info.st_uid not in {0, os.getuid()}:
            raise ValueError("Control file has an untrusted writable parent")
    return path


def proxy_environment(path, base=None):
    result = dict(os.environ if base is None else base)
    if not Path(path).exists():
        return result
    values = json.loads(trusted_file(path).read_text())
    if not isinstance(values, dict) or set(values) - PROXY_KEYS:
        raise ValueError("Proxy JSON contains unsupported variable names")
    for key, value in values.items():
        if not isinstance(value, str) or any(c in value for c in "\n\r\0"):
            raise ValueError("Proxy variable must be a single-line string")
        # These values are passed directly to execve, never parsed by a shell.
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not Path(command[0]).is_absolute():
        raise ValueError("Executable must be an absolute path")
    os.execve(command[0], command, proxy_environment(args.network))


if __name__ == "__main__":
    main()
