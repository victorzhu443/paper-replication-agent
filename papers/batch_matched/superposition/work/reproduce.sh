#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-15}
"$PY" toy.py
