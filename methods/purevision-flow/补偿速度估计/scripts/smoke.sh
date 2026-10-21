#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -B run.py --input data/HW_DEV_SINE40_A12 --output "${1:-runs/smoke}" --frames 8
