#!/usr/bin/env bash
set -euo pipefail
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then REPLICATOR_CONFIG='{}'; fi
export REPLICATOR_CONFIG
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export HF_HOME="${HF_HOME:-$(pwd)/data_cache/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}"
PY="${PYTHON:-python}"
"$PY" trace_rome.py
