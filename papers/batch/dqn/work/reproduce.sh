#!/usr/bin/env bash
set -euo pipefail
: "${SEED:=0}"
: "${SMOKE:=0}"
: "${REPLICATOR_CONFIG:={}}"
: "${METRICS_OUT:=metrics.json}"
PY="${PYTHON:-python}"
export SEED SMOKE REPLICATOR_CONFIG METRICS_OUT
"$PY" dqn_cartpole.py
