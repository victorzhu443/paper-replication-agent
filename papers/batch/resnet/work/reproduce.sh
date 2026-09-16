#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export METRICS_OUT="${METRICS_OUT:-$PWD/metrics.json}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then export REPLICATOR_CONFIG='{}'; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}"
export TORCH_THREADS="${TORCH_THREADS:-15}"
"$PY" train.py
