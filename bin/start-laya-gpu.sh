#!/usr/bin/env bash
# GPU serve path for Laya (gold eval, one model at a time).
# Does not set LAYA_DEVICE=cpu. Refuses to start if CUDA is hidden or unavailable.
# Do not use start-laya.sh for gold — that path forces CPU.
# After start, GET /health must show device=cuda (actual compute device).
# LAYA_DEVICE is only a preference in laya-serve; a silent CPU fallback is a HOLD.
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export USE_TF=0
export LAYA_HOST="${LAYA_HOST:-0.0.0.0}"
export LAYA_PORT="${LAYA_PORT:-8010}"
export LAYA_DEVICE=cuda
export LAYA_PRELOAD=1
export LAYA_MAX_LOADED=1
export LAYA_MODELS=english
export LAYA_DEFAULT_MODEL=english
export LAYA_EXTRA_MODELS='{"english":"/home/gx10/ge2340/serve/laya"}'
export LAYA_LOG_LEVEL=info
export LAYA_THREADS=8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-laya-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "start-laya-gpu.sh: torch.cuda.is_available() is False; refusing CPU fallback" >&2
  exit 1
fi
cd /home/gx10/ge2340
exec laya-serve
