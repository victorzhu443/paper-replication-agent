#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
export RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-16200}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG:-{\}}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY="${PYTHON:-python}"
"$PY" run_lth.py
