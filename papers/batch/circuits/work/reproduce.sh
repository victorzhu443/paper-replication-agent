#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python}
export METRICS_OUT=${METRICS_OUT:-metrics.json}
export SEED=${SEED:-0}
export SMOKE=${SMOKE:-0}
export SCALE=${SCALE:-1.0}
export REPLICATOR_CONFIG=${REPLICATOR_CONFIG:-'{}'}
$PY train.py
