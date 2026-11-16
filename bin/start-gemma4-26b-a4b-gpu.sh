#!/usr/bin/env bash
# GPU serve path for Gemma 4 26B-A4B (official gold eval Arm C, one model at a time).
# Docker SGLang path used in the Oct 30 Arm C smoke (triton attention).
# Does not hide the GPU. Refuses to start if CUDA is hidden or unavailable.
# Do not serve gold through the Flash-Next gateway (https://llm.wingx.icu)
# or any CPU / disk-offload path.
# After start, GET /health must show device starting with cuda (resolved
# runtime device from the loaded SGLang ServerArgs, not a hardcoded
# string). A cpu device is a HOLD.
#
# Host gate uses the laya venv torch (2.15.0+cu132) plus nvidia-smi.
# The container repeats the CUDA check before bind.
# This script only starts ge2340-gemma4-26b-a4b on :8101. It must not be
# used to start, stop, restart, or unload Qwen, PaddleOCR, FN, or Muse.
set -euo pipefail

NAME="${GEMMA4_CONTAINER_NAME:-ge2340-gemma4-26b-a4b}"
IMAGE="${GEMMA4_IMAGE:-qwen38-27b-nvfp4-dflash2-sglang:local}"
# Same image id as paddleocr-vl-sglang:local. Never replace or restart
# paddleocr-vl-worker or ge2340-qwen38-27b; a second container from this
# image is OK.
EXPECTED_IMAGE_ID_PREFIX="${GEMMA4_IMAGE_ID_PREFIX:-3ede270d2e15}"
WEIGHTS="${GEMMA4_MODEL_DIR:-/home/gx10/ge2340/serve/gemma4-26b-a4b}"
WRAPPER="${GEMMA4_WRAPPER:-/home/gx10/ge2340/serve/serve_gemma4_26b_a4b_health.py}"
CACHE="${GEMMA4_CACHE:-/home/gx10/.config/ge2340/sglang-cache}"
HOST="${GEMMA4_HOST:-0.0.0.0}"
PORT="${GEMMA4_PORT:-8101}"
MODEL_ID="${GEMMA4_MODEL_ID:-gemma-4-26b-a4b}"
MEM_FRACTION="${GEMMA4_MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${GEMMA4_CONTEXT_LENGTH:-131072}"
LAYA_PYTHON="${GEMMA4_LAYA_PYTHON:-/home/gx10/ge2340/venvs/laya/bin/python}"

# A leftover empty CUDA_VISIBLE_DEVICES would hide the GPU; refuse rather than inherit it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 1
fi
if [[ "${SGLANG_USE_CPU_ENGINE:-0}" == "1" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: SGLANG_USE_CPU_ENGINE=1; refusing CPU fallback" >&2
  exit 1
fi
if [[ ! -x "$LAYA_PYTHON" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: laya python not found at $LAYA_PYTHON" >&2
  exit 1
fi
if ! "$LAYA_PYTHON" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "start-gemma4-26b-a4b-gpu.sh: torch.cuda.is_available() is False; refusing CPU fallback" >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "start-gemma4-26b-a4b-gpu.sh: nvidia-smi -L failed; refusing CPU fallback" >&2
  exit 1
fi
if [[ ! -d "$WEIGHTS" || ! -f "$WEIGHTS/model.safetensors.index.json" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: weights missing under $WEIGHTS" >&2
  exit 1
fi
# Local chat_template.jinja + patched tokenizer_config.json are already in
# the weight tree (smoke-era). Do not re-patch or overwrite them here.
if [[ ! -f "$WEIGHTS/chat_template.jinja" || ! -f "$WEIGHTS/tokenizer_config.json" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: chat_template.jinja or tokenizer_config.json missing under $WEIGHTS (do not re-patch here)" >&2
  exit 1
fi
if [[ ! -f "$WRAPPER" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: wrapper missing at $WRAPPER" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "start-gemma4-26b-a4b-gpu.sh: image $IMAGE is not present locally; refusing to pull" >&2
  exit 1
fi
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE_ID" != sha256:"$EXPECTED_IMAGE_ID_PREFIX"* && "$IMAGE_ID" != "$EXPECTED_IMAGE_ID_PREFIX"* ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: image $IMAGE id $IMAGE_ID is not ${EXPECTED_IMAGE_ID_PREFIX}…; refusing start (would also risk confusion with paddleocr-vl-worker)" >&2
  exit 1
fi

# Do not stop or replace anyone else's container. Only remove our own leftover if it is not running.
# Never docker rm a running container, and never any name other than ge2340-gemma4-26b-a4b.
if [[ "$NAME" != "ge2340-gemma4-26b-a4b" ]]; then
  echo "start-gemma4-26b-a4b-gpu.sh: container name $NAME is not ge2340-gemma4-26b-a4b; refusing start" >&2
  exit 1
fi
if docker inspect "$NAME" >/dev/null 2>&1; then
  ST="$(docker inspect -f '{{.State.Status}}' "$NAME")"
  if [[ "$ST" == "running" || "$ST" == "paused" || "$ST" == "restarting" ]]; then
    echo "start-gemma4-26b-a4b-gpu.sh: container $NAME is already $ST; not stopping it" >&2
    exit 1
  fi
  echo "start-gemma4-26b-a4b-gpu.sh: removing leftover (not running) container $NAME (status=$ST)" >&2
  docker rm "$NAME"
fi

if ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  echo "start-gemma4-26b-a4b-gpu.sh: port $PORT is already bound; refusing to start (not killing the occupant)" >&2
  exit 1
fi

mkdir -p /home/gx10/ge2340/logs "$CACHE"

echo "start-gemma4-26b-a4b-gpu.sh: starting $NAME on ${HOST}:${PORT} model=$MODEL_ID image=$IMAGE" >&2
echo "start-gemma4-26b-a4b-gpu.sh: after load, GET http://127.0.0.1:${PORT}/health must show device starting with cuda, error=null" >&2

# Foreground docker run so the operator's nohup pid tracks this serve.
# Flags match the journalled 2026-10-30 16:49:05 Asia/Hong_Kong smoke
# docker run (triton, not flashinfer), plus --device=cuda,
# --cpu-offload-gb=0, and default enable_thinking=false (see GPU-PATH notes).
# Do not pass --gpus to any other container; do not pull.
exec docker run --name "$NAME" --gpus all --ipc=host --network=host \
  --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_USE_CPU_ENGINE=0 \
  -v "$WEIGHTS":/model:ro \
  -v "$CACHE":/cache \
  -v "$WRAPPER":/opt/ge2340/serve_gemma4_26b_a4b_health.py:ro \
  --entrypoint python3 \
  "$IMAGE" \
  /opt/ge2340/serve_gemma4_26b_a4b_health.py \
    --model-path=/model \
    --served-model-name="$MODEL_ID" \
    --trust-remote-code \
    --tp-size=1 \
    --mem-fraction-static="$MEM_FRACTION" \
    --context-length="$CONTEXT_LENGTH" \
    --attention-backend=triton \
    --chunked-prefill-size=8192 \
    --disable-prefill-cuda-graph \
    --default-chat-template-kwargs='{"enable_thinking":false}' \
    --device=cuda \
    --cpu-offload-gb=0 \
    --host="$HOST" \
    --port="$PORT"
