# Gemma 4 26B-A4B GPU serve path

**When:** 2026-11-15 ~04:10 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. Arm C official gold is OpenAI-compatible chat on this host bind, not SystemOne and not the Flash-Next gateway. No `start-gemma*` script existed in `bin`. an earlier weights-only note is a historical smoke note and was left unchanged, like an earlier weights-only note.

**Do not start this serve from the path-prep step.** Start it only after authorization, and only after Qwen is confirmed down. This document is the start path, not a start.

## How to start (after Qwen is fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Leave `qwen38-flash-next-tf` Exited — do not restore it. Do not start, stop, restart, or unload Qwen (`ge2340-qwen38-27b`, `:8100`) from this script. Do not touch Label Studio, Muse weights, or PaddleOCR. Do not use `https://llm.wingx.icu/v1` / Flash-Next / leftover `LLM_BASE_URL` for gold. Do not invent a transformers CPU or disk-offload chat wrapper as the official path.

```bash
nohup bin/start-gemma4-26b-a4b-gpu.sh \
  > logs/gemma4-26b-a4b-serve.log 2>&1 &
echo $! > logs/gemma4-26b-a4b-serve.pid
```

- Bind: `0.0.0.0:8101`
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, official `GET /health` (wrapper JSON, same port)
- Client model id: `gemma-4-26b-a4b`
- Weights: `serve/gemma4-26b-a4b` (HF `google/gemma-4-26B-A4B`, ~49 GiB, `Gemma4ForConditionalGeneration`) — kept, ro-mounted at `/model`. Local `chat_template.jinja` + patched `tokenizer_config.json` are already in that tree (`tokenizer_config.json.bak-pre-template` is the pre-patch copy). Do not re-patch or overwrite weights.
- Image (already on host, do not pull): `qwen38-27b-nvfp4-dflash2-sglang:local` (id `3ede270d2e15`, 38.7GB). Same image id as `paddleocr-vl-sglang:local`. A second container from this image is OK; never replace or restart `paddleocr-vl-worker` or `ge2340-qwen38-27b`.
- Container name: `ge2340-gemma4-26b-a4b` only (never `paddleocr-vl-worker` / `qwen38-flash-next-tf` / `ge2340-qwen38-27b`)
- Wrapper: `serve/serve_gemma4_26b_a4b_health.py` (ro-mounted as the container entrypoint)
- Auth: none by default (gold treats none / `-` / `noauth` / `n/a` as no-auth)
- Env already in GX10 `.env.eval` (do not rewrite): `LLM_GEMMA4_26B_A4B_BASE_URL=http://127.0.0.1:8101/v1`, `LLM_GEMMA4_26B_A4B_API_KEY=none`, `LLM_GEMMA4_26B_A4B_MODEL=gemma-4-26b-a4b`

### Serve flags

Smoke-proven SGLang flags taken from the journalled `docker run` that actually served Arm C Gemma on 2026-10-30 16:49:05 Asia/Hong_Kong, recovered from `tailscaled` journal on GX10 (a Mac hop was used; `--cmd=` is complete in the journal). That successful command was:

```text
docker run -d --name ge2340-gemma4-26b-a4b --gpus all --ipc=host --network=host --shm-size=32g
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -v serve/gemma4-26b-a4b:/model:ro
  -v /home/gx10/.config/ge2340/sglang-cache:/cache
  qwen38-27b-nvfp4-dflash2-sglang:local
  python3 -m sglang.launch_server
    --model-path=/model --served-model-name=gemma-4-26b-a4b --trust-remote-code
    --tp-size=1 --mem-fraction-static=0.85 --context-length=131072
    --attention-backend=triton --chunked-prefill-size=8192
    --disable-prefill-cuda-graph --host=0.0.0.0 --port=8101
```

**Attention backend MUST be `triton`.** Do not invent flashinfer. The first journalled attempt at 16:48:37 Asia/Hong_Kong used `--attention-backend=flashinfer` and was removed at 16:49:05. Evidence: `serve/logs/gemma4-flashinfer-fail-20261030.log` (Gemma4 hybrid SWA detected; the saved log is truncated after the GPU-memory fallback line and does not itself contain the assertion text) and an earlier weights-only note. The immediately following journalled run switched to triton and is the smoke-proven command.

`mem-fraction-static`, `context-length`, `chunked-prefill-size`, and `disable-prefill-cuda-graph` **are recorded** in that 16:49:05 command: `0.85`, `131072`, `8192`, and `disable-prefill-cuda-graph` respectively. Those values also match the Arm C Qwen smoke pairing that co-resided with PaddleOCR. Do not raise `mem-fraction-static`.

Flag-history gap (documented, not invented around): at 16:56 Asia/Hong_Kong the local `chat_template.jinja` + `tokenizer_config.json` patch landed, and dockerd shows a container recreate (`sbJoin` for `ge2340-gemma4-26b-a4b`) at 16:56:49. That restart was sent as `bash -s` (stdin heredoc), so the exact `docker run` argv is **not** in the journal command line. There is no later journalled flashinfer or flag-changed `docker run`. Official smoke (`/v1/models` + chat ping 200, then n=10) ran against the post-template serve. This path freezes the 16:49:05 triton flag set, mounts the already-patched weight tree read-only, and does not re-patch.

Added for the gold path (not in the smoke command, do not force thinking on):

- `--device=cuda` and `--cpu-offload-gb=0` so auto-detect cannot pick CPU and offload stays off
- `--default-chat-template-kwargs='{"enable_thinking":false}'` because official gold sends `enable_thinking=false`. The same SGLang image already accepted this switch on the Qwen gold path. The smoke-era Gemma `chat_template.jinja` does not itself branch on `enable_thinking` (no think-tag wrappers); the switch is still set so the eval flag is not dropped at the server default. Do not turn thinking on. Do not pass a Qwen `--reasoning-parser` / `--tool-call-parser` (those were not in the Gemma smoke command).
- The wrapper also copies a top-level request field `enable_thinking` into `chat_template_kwargs` before SGLang parses the body. Stock `ChatCompletionRequest` in this image has no `enable_thinking` field, so the eval script's top-level `enable_thinking=false` would otherwise be dropped. Per-request `chat_template_kwargs` still wins if already set. Temperature 0 is already sent by `eval_three_arms.py` and is not overridden here.

`mem-fraction-static=0.85` is tight next to PaddleOCR's existing `mem-fraction-static=0.13` on a 121.63 GiB GB10, but that pairing is what the Arm C smokes used while paddleocr-vl-worker was up. Leave PaddleOCR headroom; do not raise the fraction; do not require Flash-Next.

The gold start script is foreground `exec docker run` (no `-d`) so the operator's nohup pid tracks the serve. If CUDA is hidden, `SGLANG_USE_CPU_ENGINE=1`, `torch.cuda.is_available()` is false, nvidia-smi fails, `--device` is not cuda, or `cpu_offload_gb>0`, the start script / wrapper exits and does not fall back to CPU. The start script also refuses to pull a missing image, refuses a different image id than `3ede270d2e15…`, refuses to start if `:8101` is already bound, and will not stop a *running* `ge2340-gemma4-26b-a4b` (it only `docker rm`s that exact name if it is not running). It never `docker rm`s any other name.

Do not use a start script without `-gpu` for this gold run — there isn't one, and one must not be invented as a CPU path.

## Health check (required — device must be the actual compute device)

Stock SGLang `GET /health` is an empty HTTP 200 (or 503 while starting). That is not enough. Official gold uses the wrapper:

`GET http://127.0.0.1:8101/health`

After the process is up, the JSON must show:

- `status=ok`
- `model=gemma-4-26b-a4b`
- `backend=sglang-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — source of truth is the loaded `ServerArgs.device` (plus `base_gpu_id` / `torch.cuda.current_device()` for the index), not a hardcoded string. `ServerArgs.device` is what this image resolved and used to launch the engine (`--device=cuda`; auto-detect would also call `get_device()`, which returns `cpu` only if `SGLANG_USE_CPU_ENGINE=1`).
- `device_mode=cuda`
- `port=8101`
- `error=null`
- `cpu_offload_gb=0` (informational)

While weights are still loading, the same URL returns HTTP 503 with `status=starting` and `error=not_loaded`. That is not a GO.

If `device` is `cpu`, anything that does not start with `cuda`, or `cpu_offload_gb>0`, this is a HOLD: do not serve gold eval, do not silently accept CPU. HTTP status is 503 with `status=error`.

`GET /get_server_info` or `GET /server_info` also includes `device` (index-stripped, e.g. `cuda`) among the launch ServerArgs. The official pre-gold check is still `GET /health` above. Independently confirm `device` starts with `cuda` before gold.

## Evidence Gemma can use GPU (no serve process was started for this path prep)

- This exact image and the 16:49:05 triton flag set already served Arm C Gemma smoke on GX10: container `ge2340-gemma4-26b-a4b`, bind `:8101`, model `gemma-4-26b-a4b`, auth none. Chat ping + `/v1/models` were 200 around 17:03 Asia/Hong_Kong. smoke n=10 acc_hard=0.30 after a `max_tokens=256` patch (`eval_smoke_gemma4_26b_a4b.json` / `.md`). That smoke JSON is not official gold and must not be overwritten. The container was torn down ~17:59 the same day; weights were kept.
- Host: NVIDIA GB10 ~121.63 GiB, CUDA 13.0, aarch64, driver 580.178.04. Laya venv torch is `2.15.0+cu132` and `torch.cuda.is_available()` is the host-side gate. Inside the image, `is_cpu()` is true only when `SGLANG_USE_CPU_ENGINE=1`; otherwise `get_device()` returns `cuda` when `torch.cuda.is_available()`, and raises if no accelerator is present. The wrapper exits before bind if CUDA is missing.
- The ~49 GiB BF16 MoE weights previously co-resided with PaddleOCR at `mem-fraction-static=0.85`. Disk is tight; Muse weights must stay; this path does not pull images or install another CUDA stack.
- No in-process / container weight load was done here (would allocate tens of GiB and is reserved until after authorization). The earlier weights-only note was left unchanged at the time.

No extra package, CUDA-torch install, or Docker pull was needed or performed.
