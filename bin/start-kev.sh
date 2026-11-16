#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export KEV_HOST=0.0.0.0
export KEV_PORT="${KEV_PORT:-8013}"
export KEV_MODEL_ID="${KEV_MODEL_ID:-kev-27b}"
export KEV_MODEL_DIR="${KEV_MODEL_DIR:-/home/gx10/ge2340/serve/kev}"
export KEV_OFFLOAD_DIR="${KEV_OFFLOAD_DIR:-/home/gx10/ge2340/serve/kev-offload}"
export KEV_CPU_MEM="${KEV_CPU_MEM:-6GiB}"
export KEV_THREADS="${KEV_THREADS:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Hide GPU from this process so Flash-Next keeps unified RAM
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$KEV_OFFLOAD_DIR" /home/gx10/ge2340/logs
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_kev_systemone.py
