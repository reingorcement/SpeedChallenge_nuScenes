#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -u -B run_batch.py --output "${1:-runs/development}" --labels evaluation/labels --ablation --resume
