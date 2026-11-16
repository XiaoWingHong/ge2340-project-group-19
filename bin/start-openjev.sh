#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export OPENJEV_HOST=0.0.0.0
export OPENJEV_PORT="${OPENJEV_PORT:-8014}"
export OPENJEV_MODEL_ID="${OPENJEV_MODEL_ID:-openjev}"
export OPENJEV_MODEL_DIR="${OPENJEV_MODEL_DIR:-/home/gx10/ge2340/serve/openjev}"
export OPENJEV_OFFLOAD_DIR="${OPENJEV_OFFLOAD_DIR:-/home/gx10/ge2340/serve/openjev-offload}"
# Aggressive default so Clef(12GiB)+Kev(6GiB)+OpenJev can coexist at load time.
# Eng: stop Clef and/or Kev before n≈10 smoke if OOM / thrash.
export OPENJEV_CPU_MEM="${OPENJEV_CPU_MEM:-4GiB}"
export OPENJEV_MAX_LENGTH="${OPENJEV_MAX_LENGTH:-4096}"
export OPENJEV_THREADS="${OPENJEV_THREADS:-4}"
# Official SERVE.md calibration
export READOUT_T="${READOUT_T:-0.85}"
export READOUT_NOUL_T="${READOUT_NOUL_T:-1.829074}"
export READOUT_NOUL_BIAS="${READOUT_NOUL_BIAS:-0}"
export READOUT_INSTR_STYLE="${READOUT_INSTR_STYLE:-pyrepr}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Hide GPU from this process so Flash-Next keeps unified RAM
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OPENJEV_OFFLOAD_DIR" /home/gx10/ge2340/logs
cd /home/gx10/ge2340/serve
exec python /home/gx10/ge2340/serve/serve_openjev_systemone.py
