#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export CLEF_HOST=0.0.0.0
export CLEF_PORT="${CLEF_PORT:-8012}"
export CLEF_MODEL_ID="${CLEF_MODEL_ID:-clef}"
export CLEF_MODEL_DIR="${CLEF_MODEL_DIR:-/home/gx10/ge2340/serve/clef}"
export CLEF_OFFLOAD_DIR="${CLEF_OFFLOAD_DIR:-/home/gx10/ge2340/serve/clef-offload}"
export CLEF_MAX_LENGTH="${CLEF_MAX_LENGTH:-2048}"
export CLEF_CPU_MEM="${CLEF_CPU_MEM:-28GiB}"
export CLEF_THREADS="${CLEF_THREADS:-8}"
# Optional: CLEF_AUTH_ENV=<env name> and export that env with the bearer token
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Hide GPU from this process so Flash-Next keeps unified RAM
export CUDA_VISIBLE_DEVICES=""
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_clef_systemone.py
