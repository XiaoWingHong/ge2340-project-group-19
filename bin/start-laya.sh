#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/laya/bin:$PATH"
export USE_TF=0
export LAYA_HOST=0.0.0.0
export LAYA_PORT=8010
export LAYA_DEVICE=cpu
export LAYA_PRELOAD=1
export LAYA_MAX_LOADED=1
export LAYA_MODELS=english
export LAYA_DEFAULT_MODEL=english
export LAYA_EXTRA_MODELS='{"english":"/home/gx10/ge2340/serve/laya"}'
export LAYA_LOG_LEVEL=info
export LAYA_THREADS=8
cd /home/gx10/ge2340
exec laya-serve
