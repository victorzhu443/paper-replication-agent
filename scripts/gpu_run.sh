#!/usr/bin/env bash
# Launch the paper-scale sweep on a GPU box inside tmux so it survives SSH disconnects.
#   bash scripts/gpu_run.sh                 # all 14 CS papers at paper scale
#   bash scripts/gpu_run.sh batchnorm lora  # a subset
# Progress: tmux attach -t sweep ; tail -f runs/batch_gpu/batch.log ; runs/batch_gpu/SUMMARY.md
set -euo pipefail
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p runs/batch_gpu
tmux new-session -d -s sweep "cd $PWD && REPLICATOR_GPU=1 REPLICATOR_RUN_MINUTES=${REPLICATOR_RUN_MINUTES:-180} uv run python -m evals.batch $* 2>&1 | tee -a runs/batch_gpu/batch.stdout; echo SWEEP-DONE; sleep 86400"
echo "started in tmux session 'sweep'; follow with: tail -f runs/batch_gpu/batch.log"
