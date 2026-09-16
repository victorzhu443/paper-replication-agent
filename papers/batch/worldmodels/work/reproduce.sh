#!/usr/bin/env bash
set -euo pipefail
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
export RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-1500}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then
  export REPLICATOR_CONFIG='{}'
fi
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
PY="${PYTHON:-python}"
"$PY" wm_main.py
