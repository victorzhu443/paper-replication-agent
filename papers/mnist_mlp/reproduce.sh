#!/usr/bin/env bash
# Contract: reads SEED, SMOKE, REPLICATOR_CONFIG, METRICS_OUT; writes METRICS_OUT.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
exec "$PY" train.py
