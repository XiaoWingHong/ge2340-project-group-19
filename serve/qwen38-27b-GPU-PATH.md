# Qwen3.8-27B GPU serve path

**When:** 2026-11-15 ~03:21 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. Arm C official gold is OpenAI-compatible chat on this host bind, not SystemOne and not the Flash-Next gateway. No `start-qwen*` script existed in `bin` (only Arm B `start-*-gpu.sh`). an earlier weights-only note is a historical WEIGHTS_ONLY note and was left unchanged at the time.

## How to start (after OpenJev / other eval backends are fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Leave `qwen38-flash-next-tf` Exited — do not restore it. Do not start or restart Qwen until authorized. Do not use `https://llm.wingx.icu/v1` / Flash-Next / leftover `LLM_BASE_URL` for gold. Do not invent a transformers CPU or disk-offload chat wrapper as the official path.

```bash
nohup bin/start-qwen38-27b-gpu.sh \
  > logs/qwen38-27b-serve.log 2>&1 &
echo $! > logs/qwen38-27b-serve.pid
```

- Bind: `0.0.0.0:8100`
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, official `GET /health` (wrapper JSON, same port)
- Client model id: `qwen3.8-27b`
- Weights: `serve/qwen38-27b` (HF `Qwen/Qwen3.8-27B`, ~52 GiB BF16, 18 shards) — kept, ro-mounted at `/model`
- Image (already on host, do not pull): `qwen38-27b-nvfp4-dflash2-sglang:local` (id `3ede270d2e15`, 38.7GB). Same image id as `paddleocr-vl-sglang:local`. A second container from this image is OK; never replace or restart `paddleocr-vl-worker`.
- Container name: `ge2340-qwen38-27b` (never `paddleocr-vl-worker` / `qwen38-flash-next-tf`)
- Wrapper: `serve/serve_qwen38_27b_health.py` (ro-mounted as the container entrypoint)
- Auth: none by default (gold treats none / `-` / `noauth` / `n/a` as no-auth)
- Env already in GX10 `.env.eval` (do not rewrite): `LLM_QWEN38_27B_BASE_URL=http://127.0.0.1:8100/v1`, `LLM_QWEN38_27B_MODEL=qwen3.8-27b`

### Serve flags

Smoke-proven SGLang flags taken from the journalled `docker run` that actually served Arm C Qwen on 2026-10-30 15:55 Asia/Hong_Kong (and a later 2026-11-14 serve that advertised `max_model_len=131072`). These match the Oct 30 READY.md draft. Later Arm C smokes for other models sometimes used `context-length=32768` or `attention-backend=triton` for those architectures; Qwen smoke used flashinfer + 131072 + `mem-fraction-static=0.85` + `disable-prefill-cuda-graph` + `chunked-prefill-size=8192`.

Added for the gold path (not in the smoke command, do not force thinking on):

- `--device=cuda` and `--cpu-offload-gb=0` so auto-detect cannot pick CPU and offload stays off
- `--default-chat-template-kwargs='{"enable_thinking":false}'` because the Qwen3.8 chat template treats an *undefined* `enable_thinking` as thinking-on
- The wrapper also copies a top-level request field `enable_thinking` into `chat_template_kwargs` before SGLang parses the body. Stock `ChatCompletionRequest` in this image has no `enable_thinking` field, so the eval script's top-level `enable_thinking=false` would otherwise be dropped. Per-request `chat_template_kwargs` still wins if already set. Temperature 0 is already sent by `eval_three_arms.py` and is not overridden here.
- `--reasoning-parser=qwen3` is kept from the smoke. It parses `<think>` tags if they appear; it is not `--reasoning-parser=qwen3-thinking` and does not enable thinking by itself. Do not pass `--enable-strict-thinking`.

`mem-fraction-static=0.85` is tight next to PaddleOCR's existing `mem-fraction-static=0.13` (~15.4–15.5 GiB) on a 121.63 GiB GB10, but that pairing is what the Arm C smokes used while paddleocr-vl-worker was up. Leave PaddleOCR headroom; do not raise the fraction; do not require Flash-Next.

If CUDA is hidden, `SGLANG_USE_CPU_ENGINE=1`, `torch.cuda.is_available()` is false, nvidia-smi fails, `--device` is not cuda, or `cpu_offload_gb>0`, the start script / wrapper exits and does not fall back to CPU. The start script also refuses to pull a missing image, refuses a different image id than `3ede270d2e15…`, refuses to start if `:8100` is already bound, and will not stop a *running* `ge2340-qwen38-27b` (it only `docker rm`s that exact name if it is not running).

Do not use a start script without `-gpu` for this gold run — there isn't one, and one must not be invented as a CPU path.

## Health check (required — device must be the actual compute device)

Stock SGLang `GET /health` is an empty HTTP 200 (or 503 while starting). That is not enough. Official gold uses the wrapper:

`GET http://127.0.0.1:8100/health`

After the process is up, the JSON must show:

- `status=ok`
- `model=qwen3.8-27b`
- `backend=sglang-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — source of truth is the loaded `ServerArgs.device` (plus `base_gpu_id` / `torch.cuda.current_device()` for the index), not a hardcoded string. `ServerArgs.device` is what this image resolved and used to launch the engine (`--device=cuda`; auto-detect would also call `get_device()`, which returns `cpu` only if `SGLANG_USE_CPU_ENGINE=1`).
- `device_mode=cuda`
- `port=8100`
- `error=null`
- `cpu_offload_gb=0` (informational)

While weights are still loading, the same URL returns HTTP 503 with `status=starting` and `error=not_loaded`. That is not a GO.

If `device` is `cpu`, anything that does not start with `cuda`, or `cpu_offload_gb>0`, this is a HOLD: do not serve gold eval, do not silently accept CPU. HTTP status is 503 with `status=error`.

`GET /get_server_info` or `GET /server_info` also includes `device` (index-stripped, e.g. `cuda`) among the launch ServerArgs. The official pre-gold check is still `GET /health` above. Independently confirm `device` starts with `cuda` before gold.

## Evidence Qwen can use GPU (no serve process was started)

- This exact image and flag set already served Arm C Qwen smoke on GX10: 2026-10-30 15:55 Asia/Hong_Kong `docker run --name ge2340-qwen38-27b --gpus all --ipc=host --network=host` … `--port=8100`. Smoke n=10, model=`qwen3.8-27b`, Acc_hard=0.70, mean latency ~121.5s, 1/10 read-timeout at 300s (`eval_smoke_qwen38_27b.json`). A later process on 2026-11-14 21:47 Asia/Hong_Kong bound the same container name and advertised `max_model_len=131072`. Both of those serves were torn down; this script was not present then.
- Host: NVIDIA GB10 ~121.63 GiB, CUDA 13.0, aarch64, driver 580.178.04. Laya venv torch is `2.15.0+cu132` and `torch.cuda.is_available()` is the host-side gate. Inside the image, `is_cpu()` is true only when `SGLANG_USE_CPU_ENGINE=1`; otherwise `get_device()` returns `cuda` when `torch.cuda.is_available()`, and raises if no accelerator is present. The wrapper exits before bind if CUDA is missing.
- BF16 27B (~52 GiB weights) previously co-resided with PaddleOCR's ~15.5 GiB sglang footprint at `mem-fraction-static=0.85`. Disk is tight (~68G free); Muse weights must stay; this path does not pull images or install another CUDA stack.
- No in-process / container weight load was done here (would allocate tens of GiB and is reserved until after authorization). Image routes were inspected with `docker run --rm` and a non-serving entrypoint, without `--gpus` and without `--network=host`, and without `launch_server`.

No extra package, CUDA-torch install, or Docker pull was needed or performed.
