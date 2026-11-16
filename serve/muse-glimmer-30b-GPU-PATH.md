# Muse Glimmer 30B GPU serve path

**When:** 2026-11-15 ~11:55 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. Arm C official gold is OpenAI-compatible chat on this host bind, not SystemOne and not the Flash-Next gateway. No `start-muse*` script existed in `bin`. an earlier weights-only note is a historical smoke note and was left unchanged, like an earlier weights-only note, an earlier weights-only note, an earlier weights-only note, and an earlier weights-only note.

**Do not start this serve from the path-prep step.** Start it only after authorization, and only after LFM is confirmed down. This document is the start path, not a start. Official Muse gold JSON/MD must not be written until a later PM GO. Do not treat `eval_smoke_muse_glimmer_30b.json` / `.md`, the stopped LFM run, or official Nemotron/Qwen gold as official Muse gold. Official LFM gold (`eval_gold_masked_v2_C_lfm25_24b_a2b.json` / `.md`) was skipped and must stay absent; do not write it here.

## How to start (after LFM is fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Leave `qwen38-flash-next-tf` Exited — do not restore it. Do not start, stop, restart, or unload LFM (`ge2340-lfm25-24b-a2b`, `:8103`), Nemotron (`ge2340-nemotron35-lightning`, `:8102`), Gemma (`ge2340-gemma4-26b-a4b`, `:8101`), or Qwen (`ge2340-qwen38-27b`, `:8100`) from this script. Do not touch Label Studio, LFM/Gemma/Nemotron weights, or PaddleOCR. Do not use `https://llm.wingx.icu/v1` / Flash-Next / leftover `LLM_BASE_URL` for gold. Do not invent a transformers CPU or disk-offload chat wrapper as the official path.

```bash
nohup bin/start-muse-glimmer-30b-gpu.sh \
  > logs/muse-glimmer-30b-serve.log 2>&1 &
echo $! > logs/muse-glimmer-30b-serve.pid
```

- Bind: `0.0.0.0:8104`
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, official `GET /health` (wrapper JSON, same port)
- Client model id: `muse-glimmer-30b`
- Weights: `serve/muse-glimmer-30b` (HF `meta-models/Muse-Glimmer-30B`, ~56 GiB, 2/2 shards, `MuseGlimmerForConditionalGeneration` / `muse_glimmer`; text_config `muse_glimmer_text` is hybrid SWA: 39 `sliding_attention` + 13 `full_attention` layers, `sliding_window=2048`) — kept, ro-mounted at `/model`. `chat_template.jinja` (sha256 `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`, 9992 B) and `model.safetensors.index.json` (1436 tensors, `total_parameters=29776626688`, `total_size=59553253376`) are already in that tree. `special_tokens_map.json` is absent (non-blocking, noted in historical READY). Do not re-patch or overwrite weights. LFM / Gemma / Nemotron weight trees must stay.
- Image (already on host, do not pull): `qwen38-27b-nvfp4-dflash2-sglang:local` (id `3ede270d2e15`, full `sha256:3ede270d2e1532abae4378b026f805d5fb66caee17a459fe7b9e2e9e94fe68f8`, 38.7GB). Same image id as `paddleocr-vl-sglang:local`. A second container from this image is OK; never replace or restart `paddleocr-vl-worker` or any `ge2340-*-gpu` leftover.
- Container name: `ge2340-muse-glimmer-30b` only (never `paddleocr-vl-worker` / `qwen38-flash-next-tf` / `ge2340-lfm25-24b-a2b` / `ge2340-nemotron35-lightning` / `ge2340-qwen38-27b` / `ge2340-gemma4-26b-a4b`)
- Wrapper: `serve/serve_muse_glimmer_30b_health.py` (ro-mounted as the container entrypoint)
- Auth: none by default (gold treats none / `-` / `noauth` / `n/a` as no-auth)
- Env already in GX10 `experiments/.env.eval` (do not rewrite; sha256 `fe2eabc9e03c7480a0821f2f0f2ab78e0d0f284f5238961eaeb7433673c3c4b7`): `LLM_MUSE_GLIMMER_30B_BASE_URL=http://127.0.0.1:8104/v1`, `LLM_MUSE_GLIMMER_30B_API_KEY` is none-like, `LLM_MUSE_GLIMMER_30B_MODEL=muse-glimmer-30b`

### Serve flags

Smoke-proven SGLang flags taken from the journalled `docker run` that actually served Arm C Muse on 2026-10-30 21:15:54 Asia/Hong_Kong, recovered from the `tailscaled` journal on GX10 (a Mac hop was used; the `--cmd=` was split across consecutive journal lines at the same timestamp and reconstructs completely). The same SSH session first wrote a `WEIGHTS_READY` draft of an earlier weights-only note and `echo SERVED` was not yet used; then `docker rm -f ge2340-muse-glimmer-30b 2>/dev/null || true` and the `docker run` below. Dockerd joined it as host-network endpoint `ge2340-muse-glimmer-30b` at 21:15:54 (later `task-delete` container id `e5866368b670…`). There was no earlier executed Muse `docker run` and no later journalled `docker run` before teardown.

```text
docker rm -f ge2340-muse-glimmer-30b 2>/dev/null || true
docker run -d --name ge2340-muse-glimmer-30b --gpus all --ipc=host --network=host \
  --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -v serve/muse-glimmer-30b:/model:ro \
  qwen38-27b-nvfp4-dflash2-sglang:local \
  python3 -m sglang.launch_server \
    --model-path=/model \
    --served-model-name=muse-glimmer-30b \
    --trust-remote-code \
    --tp-size=1 \
    --mem-fraction-static=0.85 \
    --context-length=32768 \
    --chat-template=/model/chat_template.jinja \
    --host=0.0.0.0 \
    --port=8104
```

`--ipc=host`, `--shm-size=32g`, `mem-fraction-static=0.85`, `context-length=32768`, `--trust-remote-code`, `--tp-size=1`, and `--chat-template=/model/chat_template.jinja` **are recorded** in that command. Do not raise `mem-fraction-static`. Context is 32768, not the 131072 used for Qwen/Gemma (the model config allows 131072; smoke froze 32768). Keep `--chat-template`: on-disk `tokenizer_config.json` has no embedded `chat_template` field, so the jinja file would not be picked up from the tokenizer config alone. Keep `--shm-size=32g` because it was in this smoke command (unlike LFM).

There was no later journalled `docker run` or container recreate before teardown. `/v1/models` polling started immediately after the 21:15:54 start and first succeeded around 21:23:36 Asia/Hong_Kong; a one-word chat ping (`max_tokens=16`, `temperature=0`, model `muse-glimmer-30b`) ran 21:23:49–21:23:53. A 21:23:53 `SERVED` READY (overwritten at 21:35 by the current `TORN_DOWN` note; recovered from the same journal) recorded container Up, Base URL `http://gx10:8104`, `/v1/models` 200, `/v1/chat/completions` smoke OK, image `qwen38-27b-nvfp4-dflash2-sglang:local`, and flags "as Arm C mirror (host net, tp=1, mem-fraction-static=0.85, context=32768, chat-template=/model/chat_template.jinja) — no multimodal flag changes needed". smoke n=10 was started at 21:26:10 (`eval_three_arms.py --arms C --llm-backends muse_glimmer_30b --llm-max-n 10`); artifacts mtime 21:29. Dockerd `task-delete` for `e5866368b670…` is at 21:34:06 Asia/Hong_Kong; the stop/rm itself was sent as `bash -s` (stdin not journalled). FN `qwen38-flash-next-tf` was then restored at 21:34:28 (leave it Exited now). Current on-disk an earlier weights-only note is the later 21:35 `TORN_DOWN` note and is sparse (no mem-fraction, context, attention backend, or parsers). Where that SERVED note and the journalled argv disagree, this path follows the argv.

**Flag-history gap (documented, not invented around):** The 21:23:53 SERVED READY does **not** record an attention backend, tool-call parser, reasoning parser, or CUDA-graph setting. No `docker logs` / `GET /get_server_info` output from the smoke serve was persisted on disk or in the journal (tailscaled logs the `--cmd=`, not stdout). The auto-selected attention backend is therefore **not a measured fact** for Muse smoke. The journalled command does **not** pass `--attention-backend`, `--tool-call-parser`, `--reasoning-parser`, `--chunked-prefill-size`, `--disable-prefill-cuda-graph`, or a cache volume. This path does not invent those flags; it leaves the image defaults, as the smoke did.

**Do not copy flashinfer or triton from Gemma/Qwen.** Muse text_config is hybrid SWA (`layer_types` 39 sliding + 13 full, `sliding_window=2048`), in the same family as Gemma 4, where the first journalled attempt used `--attention-backend=flashinfer`, flashinfer asserted, and the smoke-proven Gemma gold path required `--attention-backend=triton`. Muse smoke did not pass a backend at all and still reached `/v1/models` 200 and finished n=10 with `errors=0`. Until a later start measures the runtime default, do not pin one. Historical smoke used `docker run -d` and led with `docker rm -f`; the gold start script uses foreground `exec docker run` (no `-d`) and will only `docker rm` an *exited* leftover named `ge2340-muse-glimmer-30b` — never a running container, never `docker rm -f`, never any other name.

Added for the gold path (not in the smoke command, do not force thinking on):

- `--device=cuda` and `--cpu-offload-gb=0` so auto-detect cannot pick CPU and offload stays off
- `--default-chat-template-kwargs='{"enable_thinking":false}'` because official gold sends `enable_thinking=false`. The same SGLang image already accepted this switch on the Qwen, Gemma, Nemotron, and LFM gold paths. **The on-disk `chat_template.jinja` does not mention `enable_thinking` at all** (0 hits in the jinja file; 0 hits in `tokenizer_config.json`, which also has no embedded `chat_template`). On `add_generation_prompt` it emits only `<|start|>assistant` — an unclosed assistant-start, with no `<think>` tag, no ` to=user` / ` to=self`, and no `<|message|>`. The kwargs switch is still added so a server default exists if SGLang itself consults the key, but it should not be assumed to select the user channel, suppress the ATEM `to=self` reasoning channel, or override the template's default `reasoning_strength` of `high`. Do not turn thinking on. Do not invent a `reasoning_strength` kwarg that was not in the smoke command.
- The wrapper also copies a top-level request field `enable_thinking` into `chat_template_kwargs` before SGLang parses the body. Stock `ChatCompletionRequest` in this image has no `enable_thinking` field, so the eval script's top-level `enable_thinking=false` would otherwise be dropped. Per-request `chat_template_kwargs` still wins if already set. Temperature 0 is already sent by `eval_three_arms.py` and is not overridden here. Do not add a reasoning or tool-call parser that was not in the smoke command.

`mem-fraction-static=0.85` is tight next to PaddleOCR on a 121.63 GiB GB10, but that pairing is what the Arm C smokes used while paddleocr-vl-worker was up. Leave PaddleOCR headroom; do not raise the fraction; do not require Flash-Next.

The gold start script is foreground `exec docker run` (no `-d`) so the operator's nohup pid tracks the serve. If CUDA is hidden, `SGLANG_USE_CPU_ENGINE=1`, `torch.cuda.is_available()` is false, nvidia-smi fails, `--device` is not cuda, or `cpu_offload_gb>0`, the start script / wrapper exits and does not fall back to CPU. The start script also refuses to pull a missing image, refuses a different image id than `3ede270d2e15…`, refuses to start if `:8104` is already bound, and will not stop a *running* `ge2340-muse-glimmer-30b` (it only `docker rm`s that exact name if it is not running). It never `docker rm`s any other name.

Do not use a start script without `-gpu` for this gold run — there isn't one, and one must not be invented as a CPU path.

## FLAGS/RISK — ATEM `to=self` reasoning, ignored `enable_thinking`, and 22.5s smoke latency

Historical Oct 30 facts (do not use smoke as official gold; leave smoke files untouched):

- `chat_template.jinja` is the "Muse Glimmer ATEM Chat Template". It renders a reasoning channel as `<|start|>assistant to=self<|message|>…<|eom|>` when an assistant message already has `reasoning_content`. `tokenizer_config.json` `response_template` maps `open_pattern: to=self<\|message\|>` to `reasoning_content` and `to=user<\|message\|>` to `content`. The generation prompt does not force either recipient: `add_generation_prompt` is exactly:

```jinja
{%- if add_generation_prompt -%}
    {{- '<|start|>assistant' -}}
{%- endif -%}
```

  The model is then expected to continue with ` to=user<|message|>` or ` to=self<|message|>`. If it chooses `to=self`, text can land in `reasoning_content`.
- `render_reasoning()` defaults `reasoning_strength` to `'high'` and always injects `Reasoning strength: high.` into the system block unless a system prompt already contains "reasoning strength" (it also normalizes "Reasoning effort" to "Reasoning strength"). This is independent of `enable_thinking`.
- Smoke serve had no health wrapper and no `--default-chat-template-kwargs`. The locked eval script (and the Oct 30 smoke notes) send top-level `enable_thinking=false`. Stock `ChatCompletionRequest` in this image drops that field, and this template does not read it anyway.
- the operator ping at 21:23:49 (journalled `max_tokens=16`, `temperature=0`, "Say hi in one word.", no `enable_thinking` in the body) completed at 21:23:53 (~4s, not a multi-minute runaway). The raw ping body was not journalled (tailscaled logs the `--cmd=`, not stdout). The immediately following 21:23:53 SERVED READY says `/v1/chat/completions` smoke OK. A 21:25:44 follow-up ping (`max_tokens=32`, `temperature=0`, "Reply with exactly: ok") printed `finish_reason`, `content`, `reasoning_content` / `reasoning`, and message keys in a Python snippet; a 21:32:37 post-smoke probe (`max_tokens=256`, `temperature=0`) printed `finish_reason`, `content_head`, and `usage` (not `reasoning_content`). Stdout of those snippets was also not journalled, so whether `reasoning_content` was empty cannot be verified from the journal.
- smoke left on disk (GX10 and box hashes match; do not overwrite): `eval_smoke_muse_glimmer_30b.json` sha256 `bdf8afe448fd12a1c356666d2849e16b4b8c2de2fe0ccecca152a1e479a97e93` (1831 B, mtime 2026-10-30 21:29 Asia/Hong_Kong), `.md` sha256 `2c46e95a295c03b7a68d6d7ec348a0af67e928db831706d125f3732082268acd` (992 B). Log sha256 `b63de4ae6fcf1bf5acb3bc3751f7297c72ffbf43d4733695a4d32ba0e3cd3bb2` (1142 B). n=10, Acc_hard 0.40, Acc_conf 1.00, macro_f1≈0.2619, mean latency ~22476.3 ms, errors=0, model=`muse-glimmer-30b`, notes say `enable_thinking=false`. Mean ~22.5s is slower than thinking-off LFM smoke (~13s) / Qwen gold (~4s) / Nemotron gold (~0.9s). The smoke JSON/MD/log contain predictions (label list only) and aggregate metrics — no raw `content`, `reasoning_content`, think tags, or per-item finish reasons — so thinking-on vs cap-hit vs slower thinking-off decode cannot be proven from those files. The locked `eval_three_arms.py` falls back to `reasoning_content` / `reasoning` when `content` is empty, so a thinking-on smoke could still have scored with `errors=0`.
- No reasoning parser and no tool-call parser were on for smoke, so ATEM `to=self` text would stay inside whatever field the engine populated (not verified).
- Official gold still uses the locked `eval_three_arms.py` (sha256 `033032419a0b4abbd2e8fde81762bcd7713847a3b1434020e1786b8c0700ff2d`; `max_tokens=1024`, `enable_thinking=false`, temperature 0) and must not be rewritten here. The gold path adds the default kwargs switch and the wrapper copy so that flag is not dropped; that does not, by itself, change this template's unclosed `<|start|>assistant` generation prompt or its default `reasoning_strength=high`.

**If later official responses look thinking-on, put text only in `reasoning_content`, hit the 1024 cap, or run away (smoke-like multi-minute latency or 300s timeouts), that is a HOLD, not official gold.** Do not start a second run, do not raise `max_tokens` in the locked script, and do not unload the serve from this path.

Official output path when later GO'd (do not write these files now): `experiments/models/eval_gold_masked_v2_C_muse_glimmer_30b.json` and `.md`.

## Health check (required — device must be the actual compute device)

Stock SGLang `GET /health` is an empty HTTP 200 (or 503 while starting). That is not enough. Official gold uses the wrapper:

`GET http://127.0.0.1:8104/health`

After the process is up, the JSON must show:

- `status=ok`
- `model=muse-glimmer-30b`
- `backend=sglang-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — source of truth is the loaded `ServerArgs.device` (plus `base_gpu_id` / `torch.cuda.current_device()` for the index), not a hardcoded string. `ServerArgs.device` is what this image resolved and used to launch the engine (`--device=cuda`; auto-detect would also call `get_device()`, which returns `cpu` only if `SGLANG_USE_CPU_ENGINE=1`).
- `device_mode=cuda`
- `port=8104`
- `error=null`
- `cpu_offload_gb=0` (informational)

While weights are still loading, the same URL returns HTTP 503 with `status=starting` and `error=not_loaded`. That is not a GO. HTTP 503 starting is not READY.

If `device` is `cpu`, anything that does not start with `cuda`, or `cpu_offload_gb>0`, this is a HOLD: do not serve gold eval, do not silently accept CPU. HTTP status is 503 with `status=error`.

`GET /get_server_info` or `GET /server_info` also includes `device` (index-stripped, e.g. `cuda`) among the launch ServerArgs. The official pre-gold check is still `GET /health` above. Independently confirm `device` starts with `cuda` before gold.

## Evidence Muse can use GPU (no serve process was started for this path prep)

- This exact image and the 21:15:54 flag set already served Arm C Muse smoke on GX10: container `ge2340-muse-glimmer-30b`, bind `:8104`, model `muse-glimmer-30b`, auth none, host network, weights ro-mounted, `--shm-size=32g`. `/v1/models` was HTTP 200 around 21:23 Asia/Hong_Kong. Chat ping completed ~21:23:53 as above. smoke n=10 acc_hard=0.40 (`eval_smoke_muse_glimmer_30b.json` / `.md`). That smoke JSON is not official gold and must not be overwritten. The container was torn down ~21:34 the same day; weights were kept. The earlier weights-only note was left unchanged at the time.
- Host: NVIDIA GB10 ~121.63 GiB, CUDA 13.0, aarch64, driver 580.178.04. Laya venv torch is `2.15.0+cu132` and `torch.cuda.is_available()` is the host-side gate. Inside the image, `is_cpu()` is true only when `SGLANG_USE_CPU_ENGINE=1`; otherwise `get_device()` returns `cuda` when `torch.cuda.is_available()`, and raises if no accelerator is present. The wrapper exits before bind if CUDA is missing.
- The ~56 GiB weights previously co-resided with PaddleOCR at `mem-fraction-static=0.85`. Disk is tight; LFM / Gemma / Nemotron weights must stay; this path does not pull images or install another CUDA stack.
- No in-process / container weight load was done here (would allocate tens of GiB and is reserved until after authorization). No container was started just to test.

No extra package, CUDA-torch install, or Docker pull was needed or performed.

As of this path-prep check (2026-11-15 11:55:04 Asia/Hong_Kong): LFM is already down (`GET http://127.0.0.1:8103/health` curl 7 connection refused; `docker inspect ge2340-lfm25-24b-a2b` → no such object). `:8104` is also not listening (`GET http://127.0.0.1:8104/health` curl 7; no `ge2340-muse-glimmer-30b` container). Official LFM gold JSON/MD still absent. LFM smoke hashes unchanged (`523f23352af861aa48181b529ca6c5adebb267ea3306df8902fa6db385270adc` / `deb27a47a36c38ac023d9e28d3449bfd770eecdc0dbad8427c2f3b61a00760c7`). Locked `eval_three_arms.py` sha256 still `033032419a0b4abbd2e8fde81762bcd7713847a3b1434020e1786b8c0700ff2d`. `.env.eval` sha256 still `fe2eabc9e03c7480a0821f2f0f2ab78e0d0f284f5238961eaeb7433673c3c4b7`. PaddleOCR `:8003` was HTTP 200. FN `qwen38-flash-next-tf` was Exited (0). Those were left as they are.
