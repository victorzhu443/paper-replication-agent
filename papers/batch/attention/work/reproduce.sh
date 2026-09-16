#!/usr/bin/env bash
# Reduced-scale reproduction of "Attention Is All You Need" (CPU, compute tier 2).
# Env: SEED, SMOKE (1 => tiny subset / few steps), REPLICATOR_CONFIG (json), METRICS_OUT.
set -euo pipefail

export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
if [ -z "${REPLICATOR_CONFIG:-}" ]; then REPLICATOR_CONFIG='{}'; fi
export REPLICATOR_CONFIG
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$(pwd)/data_cache/hf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-15}"

PY="${PYTHON:-python}"
cd "$(dirname "$0")"
"$PY" train_eval.py
echo "wrote $METRICS_OUT"
