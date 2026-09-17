#!/usr/bin/env bash
# One-shot setup for a fresh Linux GPU box (Ubuntu 22.04/24.04 with NVIDIA drivers, e.g. a
# Lambda / RunPod / Vast / AWS g5-g6 "PyTorch" image). Run from the repo root:
#   bash scripts/gpu_bootstrap.sh
set -euo pipefail
echo "== nvidia-smi"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "no NVIDIA driver visible"; exit 1; }
echo "== system deps"; sudo apt-get update -qq && sudo apt-get install -y -qq git tmux swig python3-dev build-essential > /dev/null
echo "== uv"; command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH="$HOME/.local/bin:$PATH"
echo "== python env"; uv sync -q
echo "== CUDA torch (replaces the CPU wheel)"
CUDA_TAG=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | tr -d . | cut -c1-3)
case "$CUDA_TAG" in 12[0-3]) IDX=cu121;; 12[4-9]) IDX=cu124;; 13*) IDX=cu128;; *) IDX=cu124;; esac
uv pip install -q --index-url "https://download.pytorch.org/whl/$IDX" torch torchvision
echo "== RL extras (MuJoCo, Box2D, Atari ROMs)"; uv pip install -q "gymnasium[mujoco,box2d]" ale-py autorom 2>/dev/null || true
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('torch', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
echo "== tests"; uv run pytest -q tests 2>&1 | tail -1
echo "== envelope"; uv run python -c "from replicator.triage import COMPUTE_ENVELOPE as e; print(e)"
echo; echo "Ready. Set ANTHROPIC_API_KEY, then: bash scripts/gpu_run.sh [paper ...]"
