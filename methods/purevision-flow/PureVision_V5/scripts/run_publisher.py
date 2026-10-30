"""Load a private, fixed publisher argument vector; no shell evaluation."""
import argparse
import json
import os
from pathlib import Path
from secure_exec import trusted_file, proxy_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("command_file", type=Path)
    parser.add_argument("--trigger", choices=("manual", "scheduled"), default="manual")
    args = parser.parse_args()
    root = args.root.resolve()
    path = trusted_file(args.command_file)
    if not path.resolve().is_relative_to(root / "monitoring"):
        raise ValueError("Publisher control file must be in private monitoring directory")
    command = json.loads(path.read_text())["command"]
    prefix = ["/home/zhangyu/.conda/envs/speedest/bin/python", "-B", str(Path(__file__).resolve().parents[1] / "publish_results.py")]
    if command[:3] != prefix or not all(isinstance(v, str) for v in command) or "--root" not in command:
        raise ValueError("Publisher command must target this fixed code version")
    if Path(command[command.index("--root") + 1]).resolve() != root:
        raise ValueError("Publisher root differs from scheduled task")
    os.execve(command[0], command + ["--trigger", args.trigger], proxy_environment(root / "monitoring/network.json"))


if __name__ == "__main__":
    main()
