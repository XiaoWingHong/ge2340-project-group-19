#!/usr/bin/env bash
# GPU serve path for Kev (gold eval, one model at a time among eval arms).
# Does not hide the GPU. Refuses to start if CUDA is hidden or unavailable.
# Do not use start-kev.sh for gold — that path forces CPU + disk offload.
# After start, GET /health must show device starting with cuda (actual
# compute device from loaded weights, not a preference). A cpu device is a HOLD.
#
# This serve uses torch (laya venv, 2.15.0+cu132). Gate on
# torch.cuda.is_available() — that is the real inference device check.
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export KEV_HOST="${KEV_HOST:-0.0.0.0}"
export KEV_PORT="${KEV_PORT:-8013}"
export KEV_MODEL_ID="${KEV_MODEL_ID:-kev-27b}"
export KEV_MODEL_DIR="${KEV_MODEL_DIR:-/home/gx10/ge2340/serve/kev}"
export KEV_THREADS="${KEV_THREADS:-8}"
export KEV_DEVICE=cuda
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-kev-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "start-kev-gpu.sh: torch.cuda.is_available() is False; refusing CPU fallback" >&2
  exit 1
fi
mkdir -p /home/gx10/ge2340/logs
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_kev_systemone.py
