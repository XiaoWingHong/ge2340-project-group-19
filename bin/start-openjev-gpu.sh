#!/usr/bin/env bash
# GPU serve path for OpenJev (gold eval, one model at a time among eval arms).
# Does not hide the GPU. Refuses to start if CUDA is hidden or unavailable.
# Do not use start-openjev.sh for gold — that path forces CPU + disk offload.
# After start, GET /health must show device starting with cuda (actual
# compute device from loaded weights, not a preference). A cpu device is a HOLD.
#
# This serve uses torch (laya venv, 2.15.0+cu132). Gate on
# torch.cuda.is_available() — that is the real inference device check.
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export OPENJEV_HOST="${OPENJEV_HOST:-0.0.0.0}"
export OPENJEV_PORT="${OPENJEV_PORT:-8014}"
export OPENJEV_MODEL_ID="${OPENJEV_MODEL_ID:-openjev}"
export OPENJEV_MODEL_DIR="${OPENJEV_MODEL_DIR:-/home/gx10/ge2340/serve/openjev}"
export OPENJEV_MAX_LENGTH="${OPENJEV_MAX_LENGTH:-4096}"
export OPENJEV_THREADS="${OPENJEV_THREADS:-8}"
export OPENJEV_DEVICE=cuda
# Official SERVE.md calibration
export READOUT_T="${READOUT_T:-0.85}"
export READOUT_NOUL_T="${READOUT_NOUL_T:-1.829074}"
export READOUT_NOUL_BIAS="${READOUT_NOUL_BIAS:-0}"
export READOUT_INSTR_STYLE="${READOUT_INSTR_STYLE:-pyrepr}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-openjev-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "start-openjev-gpu.sh: torch.cuda.is_available() is False; refusing CPU fallback" >&2
  exit 1
fi
mkdir -p /home/gx10/ge2340/logs
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_openjev_systemone.py
