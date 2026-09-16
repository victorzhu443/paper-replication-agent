#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export SEED="${SEED:-0}" SMOKE="${SMOKE:-0}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG:-}"
export METRICS_OUT="${METRICS_OUT:-$PWD/metrics.json}"
exec "$PY" gan_mnist.py
