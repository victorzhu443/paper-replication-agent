#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python}
export SEED=${SEED:-0} SMOKE=${SMOKE:-0} SCALE=${SCALE:-1.0}
export METRICS_OUT=${METRICS_OUT:-metrics.json}
export REPLICATOR_CONFIG=${REPLICATOR_CONFIG:-'{}'}
exec "$PY" train.py
