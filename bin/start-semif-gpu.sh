#!/usr/bin/env bash
# GPU serve path for SemIf (gold eval, one model at a time).
# Does not force n_gpu_layers=0. Refuses to start if CUDA is hidden or
# this llama.cpp build cannot offload to GPU.
# Do not use start-semif.sh for gold — that path is CPU-only.
# After start, GET /health must show device=cuda (actual compute device
# from load metadata, not a preference). A cpu device is a HOLD.
#
# This serve uses llama.cpp, not torch. The SemIf venv's torch is CPU-only
# (2.10.0+cpu); do not gate on torch.cuda.is_available().
set -euo pipefail
export PATH="/home/gx10/ge2340/venvs/semif/bin:$PATH"
export SEMIF_HOST="${SEMIF_HOST:-0.0.0.0}"
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
export SEMIF_DEVICE=cuda
export SEMIF_N_GPU_LAYERS="${SEMIF_N_GPU_LAYERS:--1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-semif-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if ! python -c 'from llama_cpp import llama_cpp as c; import sys; sys.exit(0 if c.llama_supports_gpu_offload() else 1)'; then
  echo "start-semif-gpu.sh: llama.cpp llama_supports_gpu_offload() is False; refusing CPU fallback" >&2
  exit 1
fi
cd /home/gx10/ge2340/serve/semif
exec python /home/gx10/ge2340/serve/semif/serve_systemone.py
