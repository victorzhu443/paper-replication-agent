#!/usr/bin/env bash
# Layer Normalization (Ba et al., 2016) reduced-scale replication -- CPU only, compute tier 2.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
export SEED="${SEED:-0}"
export SMOKE="${SMOKE:-0}"
export REPLICATOR_CONFIG="${REPLICATOR_CONFIG-}"
export METRICS_OUT="${METRICS_OUT:-metrics.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-15}"
export TOKENIZERS_PARALLELISM=false

"$PY" train_eval.py
