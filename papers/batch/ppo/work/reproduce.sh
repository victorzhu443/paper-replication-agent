#!/usr/bin/env bash
set -euo pipefail
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG:-{\}}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
PY="${PYTHON:-python}"
"$PY" ppo_cartpole.py
