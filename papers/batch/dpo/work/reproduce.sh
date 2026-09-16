#!/usr/bin/env bash
set -euo pipefail
export SEED="${SEED:-0}" SMOKE="${SMOKE:-0}" SCALE="${SCALE:-1.0}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export TOKENIZERS_PARALLELISM=false
export NTHREADS="${NTHREADS:-8}"
"${PYTHON:-python}" run_dpo.py
