#!/usr/bin/env bash
# GPU serve path for Clef (gold eval, one model at a time).
# Does not hide the GPU. Refuses to start if CUDA is unavailable.
# Do not use start-clef.sh for gold — that path forces CPU + disk offload.
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export CLEF_HOST="${CLEF_HOST:-0.0.0.0}"
export CLEF_PORT="${CLEF_PORT:-8012}"
export CLEF_MODEL_ID="${CLEF_MODEL_ID:-clef}"
export CLEF_MODEL_DIR="${CLEF_MODEL_DIR:-/home/gx10/ge2340/serve/clef}"
export CLEF_MAX_LENGTH="${CLEF_MAX_LENGTH:-2048}"
export CLEF_THREADS="${CLEF_THREADS:-8}"
export CLEF_DEVICE=cuda
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-clef-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "start-clef-gpu.sh: torch.cuda.is_available() is False; refusing CPU fallback" >&2
  exit 1
fi
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_clef_systemone.py
