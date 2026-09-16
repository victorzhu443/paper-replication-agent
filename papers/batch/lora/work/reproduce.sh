#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export SCALE="${SCALE:-1.0}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then export REPLICATOR_CONFIG='{}'; fi
export HF_HOME="${HF_HOME:-$(cd ../../../.. && pwd)/data_cache/hf}"
export TOKENIZERS_PARALLELISM=false
export NTHREADS="${NTHREADS:-15}"
PY="${PYTHON:-python3}"
"$PY" train.py
