#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG:-{\}}"
export TORCH_THREADS="${TORCH_THREADS:-8}"
PY="${PYTHON:-python}"
"$PY" lottery.py
