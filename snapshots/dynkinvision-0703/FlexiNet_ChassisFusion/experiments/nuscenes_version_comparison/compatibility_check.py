from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from bootstrap import (  # noqa: E402
    HOST_PROJECT_ROOT,
    SOURCE_ROOT_ENV,
    compatibility_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a source tree can host all comparison algorithms."
    )
    parser.add_argument("--project-root", default=str(HOST_PROJECT_ROOT))
    parser.add_argument("--profile", choices=["legacy", "full"], default="legacy")
    args = parser.parse_args()
    report = compatibility_report(args.project_root, profile=args.profile)
    report["optional_full_profile_source_variable"] = SOURCE_ROOT_ENV
    print(json.dumps(report, indent=2))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
