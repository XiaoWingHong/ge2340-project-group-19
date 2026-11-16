#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/semif/bin:$PATH"
export SEMIF_HOST=0.0.0.0
export SEMIF_PORT="${SEMIF_PORT:-8011}"
export SEMIF_MODEL_ID="${SEMIF_MODEL_ID:-semif-qwen3.5-4b-gguf}"
export SEMIF_GGUF="${SEMIF_GGUF:-/home/gx10/ge2340/serve/semif-weights/gguf/Qwen_Qwen3.5-4B-Q4_K_M.gguf}"
# Prefer local BF16/tokenizer tree when present (offline-safe)
if [[ -d /home/gx10/ge2340/serve/semif-weights/Qwen3.5-4B ]]; then
  export SEMIF_TOKENIZER="/home/gx10/ge2340/serve/semif-weights/Qwen3.5-4B"
else
  export SEMIF_TOKENIZER="${SEMIF_TOKENIZER:-Qwen/Qwen3.5-4B}"
fi
export SEMIF_REVISION="${SEMIF_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
export SEMIF_THREADS="${SEMIF_THREADS:-8}"
export SEMIF_MAX_TOKENS="${SEMIF_MAX_TOKENS:-4096}"
# Optional: set SEMIF_AUTH_ENV=SOME_ENV_NAME and export that env with the bearer token
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
cd /home/gx10/ge2340/serve/semif
exec python /home/gx10/ge2340/serve/semif/serve_systemone.py
