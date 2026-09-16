#!/usr/bin/env bash
set -euo pipefail
DEFCFG='{}'
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG:-$DEFCFG}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}"
PY="${PYTHON:-python}"
"$PY" run_experiments.py
