#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-python}"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then export REPLICATOR_CONFIG='{}'; fi
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
"$PY" train_mnist_bn.py
