# Nemotron 3.5 Lightning GPU serve path

**When:** 2026-11-15 ~10:40 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. Arm C official gold is OpenAI-compatible chat on this host bind, not SystemOne and not the Flash-Next gateway. No `start-nemotron*` script existed in `bin`. an earlier weights-only note is a historical smoke note and was left unchanged, like an earlier weights-only note and an earlier weights-only note.

**Do not start this serve from the path-prep step.** Start it only after authorization, and only after Gemma is confirmed down. This document is the start path, not a start. Official Nemotron gold JSON/MD must not be written until a later PM GO. Do not treat `eval_smoke_nemotron35_lightning.json` / `.md`, or the aborted Gemma run, as official gold.

## How to start (after Gemma is fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Leave `qwen38-flash-next-tf` Exited — do not restore it. Do not start, stop, restart, or unload Gemma (`ge2340-gemma4-26b-a4b`, `:8101`) or Qwen (`ge2340-qwen38-27b`, `:8100`) from this script. Do not touch Label Studio, Muse weights, or PaddleOCR. Do not use `https://llm.wingx.icu/v1` / Flash-Next / leftover `LLM_BASE_URL` for gold. Do not invent a transformers CPU or disk-offload chat wrapper as the official path.

```bash
nohup bin/start-nemotron35-lightning-gpu.sh \
  > logs/nemotron35-lightning-serve.log 2>&1 &
echo $! > logs/nemotron35-lightning-serve.pid
```

- Bind: `0.0.0.0:8102`
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, official `GET /health` (wrapper JSON, same port)
- Client model id: `nemotron-3.5-lightning`
- Weights: `serve/nemotron35-lightning` (HF `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, ~62 GiB, 14/14 shards, `NemotronHForCausalLM`) — kept, ro-mounted at `/model`. `chat_template.jinja` is already in that tree. Do not re-patch or overwrite weights.
- Image (already on host, do not pull): `qwen38-27b-nvfp4-dflash2-sglang:local` (id `3ede270d2e15`, 38.7GB). Same image id as `paddleocr-vl-sglang:local`. A second container from this image is OK; never replace or restart `paddleocr-vl-worker`, `ge2340-qwen38-27b`, or `ge2340-gemma4-26b-a4b`.
- Container name: `ge2340-nemotron35-lightning` only (never `paddleocr-vl-worker` / `qwen38-flash-next-tf` / `ge2340-qwen38-27b` / `ge2340-gemma4-26b-a4b`)
- Wrapper: `serve/serve_nemotron35_lightning_health.py` (ro-mounted as the container entrypoint)
- Auth: none by default (gold treats none / `-` / `noauth` / `n/a` as no-auth)
- Env already in GX10 `.env.eval` (do not rewrite): `LLM_NEMOTRON35_LIGHTNING_BASE_URL=http://127.0.0.1:8102/v1`, `LLM_NEMOTRON35_LIGHTNING_API_KEY=none`, `LLM_NEMOTRON35_LIGHTNING_MODEL=nemotron-3.5-lightning`

### Serve flags

Smoke-proven SGLang flags taken from the journalled `docker run` that actually served Arm C Nemotron on 2026-10-30 18:00:17 Asia/Hong_Kong, recovered from the `tailscaled` journal on GX10 (a Mac hop was used; the `--cmd=` was split across consecutive journal lines at the same timestamp and reconstructs completely). There was no later journalled `docker run`, restart, or `bash -s` recreate before teardown at 19:05:34 Asia/Hong_Kong (`docker stop` then `docker rm`). That successful command was:

```text
MODEL=serve/nemotron35-lightning
docker rm -f ge2340-nemotron35-lightning 2>/dev/null
docker run -d --name ge2340-nemotron35-lightning --gpus all --ipc=host --network=host
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -v "$MODEL":/model:ro
  qwen38-27b-nvfp4-dflash2-sglang:local
  python3 -m sglang.launch_server
    --model-path=/model --served-model-name=nemotron-3.5-lightning --trust-remote-code
    --tp-size=1 --mem-fraction-static=0.85 --context-length=32768
    --host=0.0.0.0 --port=8102
    --mamba-ssm-dtype=float16 --reasoning-parser=nemotron_3
    --tool-call-parser=qwen3_coder
```

`mem-fraction-static=0.85`, `context-length=32768`, `mamba-ssm-dtype=float16`, `--reasoning-parser=nemotron_3`, and `--tool-call-parser=qwen3_coder` **are recorded** in that command and match an earlier weights-only note. Do not raise `mem-fraction-static`. Context is 32768, not the 131072 used for Qwen/Gemma.

**Flag-history gap (documented, not invented around):** READY.md lists "Attention backend | flashinfer (default)". The journalled command does **not** pass `--attention-backend=flashinfer` (or any other attention backend). This path does not invent that flag; it leaves the image default, as the smoke did. The journalled command also did **not** pass `--shm-size=32g`, a cache volume, `--chunked-prefill-size`, or `--disable-prefill-cuda-graph` (those were in the Qwen/Gemma smoke commands, not this one). Historical smoke used `docker run -d` and a leading `docker rm -f`; the gold start script uses foreground `exec docker run` (no `-d`) and will only `docker rm` an *exited* leftover named `ge2340-nemotron35-lightning` — never a running container, never `docker rm -f`, never any other name.

Added for the gold path (not in the smoke command, do not force thinking on):

- `--device=cuda` and `--cpu-offload-gb=0` so auto-detect cannot pick CPU and offload stays off
- `--default-chat-template-kwargs='{"enable_thinking":false}'` because official gold sends `enable_thinking=false`. The same SGLang image already accepted this switch on the Qwen and Gemma gold paths. The on-disk `chat_template.jinja` does branch on `enable_thinking` and treats an *undefined* flag as thinking-on (`enable_thinking if enable_thinking is defined else True`; thinking-on opens `<think>\n`, thinking-off emits `<think></think>`). The switch is required so the server default is not thinking-on. Do not turn thinking on.
- The wrapper also copies a top-level request field `enable_thinking` into `chat_template_kwargs` before SGLang parses the body. Stock `ChatCompletionRequest` in this image has no `enable_thinking` field, so the eval script's top-level `enable_thinking=false` would otherwise be dropped. Per-request `chat_template_kwargs` still wins if already set. Temperature 0 is already sent by `eval_three_arms.py` and is not overridden here.
- `--reasoning-parser=nemotron_3` and `--tool-call-parser=qwen3_coder` are kept from the smoke. The reasoning parser does not itself enable thinking, but it will split think-text into `reasoning_content` if thinking tags appear. See FLAGS/RISK below. Do not add a different parser.

`mem-fraction-static=0.85` is tight next to PaddleOCR on a 121.63 GiB GB10, but that pairing is what the Arm C smokes used while paddleocr-vl-worker was up. Leave PaddleOCR headroom; do not raise the fraction; do not require Flash-Next.

The gold start script is foreground `exec docker run` (no `-d`) so the operator's nohup pid tracks the serve. If CUDA is hidden, `SGLANG_USE_CPU_ENGINE=1`, `torch.cuda.is_available()` is false, nvidia-smi fails, `--device` is not cuda, or `cpu_offload_gb>0`, the start script / wrapper exits and does not fall back to CPU. The start script also refuses to pull a missing image, refuses a different image id than `3ede270d2e15…`, refuses to start if `:8102` is already bound, and will not stop a *running* `ge2340-nemotron35-lightning` (it only `docker rm`s that exact name if it is not running). It never `docker rm`s any other name.

Do not use a start script without `-gpu` for this gold run — there isn't one, and one must not be invented as a CPU path.

## FLAGS/RISK — thinking, reasoning-parser, and runaway generation

This is the highest-risk Arm C serve in the remaining gold queue. Historical Oct 30 facts (do not use smoke as official gold; leave smoke files untouched):

- Reasoning parser was on for smoke (`--reasoning-parser=nemotron_3`). A prior draft note at ~18:09 Asia/Hong_Kong noted that a chat ping returned empty `content` with filled `reasoning_content` (reasoning parser on). HF-recommended sampling on the ping was temperature 1.0 / top_p 0.95; official gold will still send temperature 0 / `enable_thinking=false` via the locked eval script.
- First n=10 smoke at `max_tokens=256`: empty `content`, `reasoning_content` truncated, `finish_reason=length`, Acc_hard 0.20, all promo. That 256-cap artifact is not a separate file on disk today.
- Later resmoke at `max_tokens=1024` (the file left on disk): `eval_smoke_nemotron35_lightning.json` sha256 `b9c3d5900bd571843788cf8a721932c762530668ab0a1effe9e89fd139ae7087`, `.md` sha256 `df66bc9471307a1095cc88d585b02a0360a633d8a397f646ed7afe3e3418193d`. n=10, Acc_hard 0.40, Acc_conf 0.50, macro_f1≈0.207, mean latency ~30113.5 ms, via `:8102` model=`nemotron-3.5-lightning`. Notes say `enable_thinking=false`, but the smoke serve had no health wrapper and no `--default-chat-template-kwargs`, so a top-level `enable_thinking=false` from the eval script may have been dropped by stock `ChatCompletionRequest`, leaving the template default (thinking-on). That is a plausible cause of the empty-`content` / `reasoning_content` / long-latency pattern. Do not treat that resmoke as official gold.
- Official gold still uses the locked `eval_three_arms.py` (`max_tokens=1024`, `enable_thinking=false`, temperature 0) and must not be rewritten here. The gold path adds the default kwargs switch and the wrapper copy so that flag is not dropped. Even with thinking-off, the template emits an empty `<think></think>` prelude, and `nemotron_3` remains the smoke-proven parser.
- **If later official responses look thinking-on, put text only in `reasoning_content`, hit the 1024 cap, or run away (smoke-like multi-minute latency or 300s timeouts), that is a HOLD, not official gold.** Do not start a second run, do not raise `max_tokens` in the locked script, and do not unload the serve from this path.

## Health check (required — device must be the actual compute device)

Stock SGLang `GET /health` is an empty HTTP 200 (or 503 while starting). That is not enough. Official gold uses the wrapper:

`GET http://127.0.0.1:8102/health`

After the process is up, the JSON must show:

- `status=ok`
- `model=nemotron-3.5-lightning`
- `backend=sglang-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — source of truth is the loaded `ServerArgs.device` (plus `base_gpu_id` / `torch.cuda.current_device()` for the index), not a hardcoded string. `ServerArgs.device` is what this image resolved and used to launch the engine (`--device=cuda`; auto-detect would also call `get_device()`, which returns `cpu` only if `SGLANG_USE_CPU_ENGINE=1`).
- `device_mode=cuda`
- `port=8102`
- `error=null`
- `cpu_offload_gb=0` (informational)

While weights are still loading, the same URL returns HTTP 503 with `status=starting` and `error=not_loaded`. That is not a GO. HTTP 503 starting is not READY.

If `device` is `cpu`, anything that does not start with `cuda`, or `cpu_offload_gb>0`, this is a HOLD: do not serve gold eval, do not silently accept CPU. HTTP status is 503 with `status=error`.

`GET /get_server_info` or `GET /server_info` also includes `device` (index-stripped, e.g. `cuda`) among the launch ServerArgs. The official pre-gold check is still `GET /health` above. Independently confirm `device` starts with `cuda` before gold.

## Evidence Nemotron can use GPU (no serve process was started for this path prep)

- This exact image and the 18:00:17 flag set already served Arm C Nemotron smoke on GX10: container `ge2340-nemotron35-lightning`, bind `:8102`, model `nemotron-3.5-lightning`, auth none, host network, weights ro-mounted. `/v1/models` + chat ping were HTTP 200 around 18:08 Asia/Hong_Kong. A later resmoke n=10 acc_hard=0.40 at `max_tokens=1024` (`eval_smoke_nemotron35_lightning.json` / `.md`). That smoke JSON is not official gold and must not be overwritten. The container was torn down ~19:05–19:06 the same day; weights were kept. The earlier weights-only note was left unchanged at the time.
- Host: NVIDIA GB10 ~121.63 GiB, CUDA 13.0, aarch64, driver 580.178.04. Laya venv torch is `2.15.0+cu132` and `torch.cuda.is_available()` is the host-side gate. Inside the image, `is_cpu()` is true only when `SGLANG_USE_CPU_ENGINE=1`; otherwise `get_device()` returns `cuda` when `torch.cuda.is_available()`, and raises if no accelerator is present. The wrapper exits before bind if CUDA is missing.
- The ~62 GiB BF16 MoE weights previously co-resided with PaddleOCR at `mem-fraction-static=0.85`. Disk is tight; Muse weights must stay; this path does not pull images or install another CUDA stack.
- No in-process / container weight load was done here (would allocate tens of GiB and is reserved until after authorization). No container was started just to test.

No extra package, CUDA-torch install, or Docker pull was needed or performed.
