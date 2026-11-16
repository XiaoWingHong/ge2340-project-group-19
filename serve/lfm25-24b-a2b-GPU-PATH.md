# LFM 2.5 24B A2B GPU serve path

**When:** 2026-11-15 ~11:15 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. Arm C official gold is OpenAI-compatible chat on this host bind, not SystemOne and not the Flash-Next gateway. No `start-lfm*` script existed in `bin`. an earlier weights-only note is a historical smoke note and was left unchanged, like an earlier weights-only note, an earlier weights-only note, and an earlier weights-only note.

**Do not start this serve from the path-prep step.** Start it only after authorization, and only after Nemotron is confirmed down. This document is the start path, not a start. Official LFM gold JSON/MD must not be written until a later PM GO. Do not treat `eval_smoke_lfm25_24b_a2b.json` / `.md`, or the skipped/absent Gemma gold run, as official gold.

## How to start (after Nemotron is fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Leave `qwen38-flash-next-tf` Exited — do not restore it. Do not start, stop, restart, or unload Nemotron (`ge2340-nemotron35-lightning`, `:8102`), Gemma (`ge2340-gemma4-26b-a4b`, `:8101`), or Qwen (`ge2340-qwen38-27b`, `:8100`) from this script. Do not touch Label Studio, Muse weights, or PaddleOCR. Do not use `https://llm.wingx.icu/v1` / Flash-Next / leftover `LLM_BASE_URL` for gold. Do not invent a transformers CPU or disk-offload chat wrapper as the official path.

```bash
nohup bin/start-lfm25-24b-a2b-gpu.sh \
  > logs/lfm25-24b-a2b-serve.log 2>&1 &
echo $! > logs/lfm25-24b-a2b-serve.pid
```

- Bind: `0.0.0.0:8103`
- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, official `GET /health` (wrapper JSON, same port)
- Client model id: `lfm2.5-24b-a2b`
- Weights: `serve/lfm25-24b-a2b` (HF `LiquidAI/LFM2.5-24B-A2B`, ~45 GiB, 10/10 shards, `Lfm2MoeForCausalLM` / `lfm2_moe`) — kept, ro-mounted at `/model`. `chat_template.jinja` and `model.safetensors.index.json` are already in that tree. Do not re-patch or overwrite weights. Muse weights at `serve/muse-glimmer-30b` (~56 GiB) must stay.
- Image (already on host, do not pull): `qwen38-27b-nvfp4-dflash2-sglang:local` (id `3ede270d2e15`, full `sha256:3ede270d2e1532abae4378b026f805d5fb66caee17a459fe7b9e2e9e94fe68f8`, 38.7GB). Same image id as `paddleocr-vl-sglang:local`. A second container from this image is OK; never replace or restart `paddleocr-vl-worker` or any `ge2340-*-gpu` leftover.
- Container name: `ge2340-lfm25-24b-a2b` only (never `paddleocr-vl-worker` / `qwen38-flash-next-tf` / `ge2340-nemotron35-lightning` / `ge2340-qwen38-27b` / `ge2340-gemma4-26b-a4b`)
- Wrapper: `serve/serve_lfm25_24b_a2b_health.py` (ro-mounted as the container entrypoint)
- Auth: none by default (gold treats none / `-` / `noauth` / `n/a` as no-auth)
- Env already in GX10 `.env.eval` (do not rewrite): `LLM_LFM25_24B_A2B_BASE_URL=http://127.0.0.1:8103/v1`, `LLM_LFM25_24B_A2B_API_KEY` is none-like, `LLM_LFM25_24B_A2B_MODEL=lfm2.5-24b-a2b`

### Serve flags

Smoke-proven SGLang flags taken from the journalled `docker run` that actually served Arm C LFM on 2026-10-30 19:06:55 Asia/Hong_Kong, recovered from the `tailscaled` journal on GX10 (a Mac hop was used; the `--cmd=` was split across consecutive journal lines at the same timestamp and reconstructs completely). Nemotron was torn down first at 19:05:34 (`docker stop` then `docker rm` `ge2340-nemotron35-lightning`; dockerd force-killed the leftover at 19:05:44). The successful LFM command, which dockerd joined as host-network endpoint `ge2340-lfm25-24b-a2b` at 19:06:55 (container id `7f74aa22aba7…`), was:

```text
MODEL=serve/lfm25-24b-a2b
docker run -d --name ge2340-lfm25-24b-a2b --gpus all --ipc=host --network=host
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -v "$MODEL":/model:ro
  qwen38-27b-nvfp4-dflash2-sglang:local
  python3 -m sglang.launch_server
    --model-path=/model
    --served-model-name=lfm2.5-24b-a2b
    --trust-remote-code
    --tp-size=1
    --mem-fraction-static=0.85
    --context-length=32768
    --host=0.0.0.0
    --port=8103
    --chat-template=/model/chat_template.jinja
```

`--ipc=host`, `mem-fraction-static=0.85`, `context-length=32768`, `--trust-remote-code`, `--tp-size=1`, and `--chat-template=/model/chat_template.jinja` **are recorded** in that command. Do not raise `mem-fraction-static`. Context is 32768, not the 131072 used for Qwen/Gemma. Keep `--chat-template`: on-disk `tokenizer_config.json` has no embedded `chat_template` field, so the jinja file would not be picked up from the tokenizer config alone.

There was no later journalled `docker run` or container recreate before teardown at 19:21:30 Asia/Hong_Kong (`docker stop` then `docker rm`; dockerd `task-delete` for `7f74aa22aba7…` at 19:21:38). Later `bash -s` sessions appear at 19:14:40, 19:16:52, 19:16:59, and 19:20:07/20/29 (stdin was not journalled; 19:16 coincides with smoke JSON mtime). Dockerd recorded no second container create, so those sessions did not rebuild the serve.

A 19:04:27 `WEIGHTS_READY` draft (written while Nemotron still held `:8102`) included a *planned* recipe that was **not executed**: it omitted `--ipc=host` and `PYTORCH_CUDA_ALLOC_CONF`, used `--network host` (no equals), and had a shorter argv. Follow the 19:06:55 journalled command, not that draft. Current an earlier weights-only note is the later 19:21 `TORN_DOWN` note and is sparse (no mem-fraction, context, attention backend, or parsers). A 19:13:18 `SERVED` READY (overwritten at 19:21; recovered from the same journal) documented runtime observations after the 19:06:55 start; it is historical and was not rewritten on disk. Where that SERVED note and the journalled argv disagree, this path follows the argv.

**Flag-history gap (documented, not invented around):** The 19:13:18 SERVED READY lists "Attention backend | flashinfer (default)", "Linear attn | triton (decode/prefill/verify)", and "tool-call-parser | auto-detected `lfm2` (not passed explicitly)". It also notes that prefill CUDA graph was disabled automatically ("some layers do not apply Standard GQA") while decode CUDA graph was captured. The journalled command does **not** pass `--attention-backend`, `--tool-call-parser=lfm2`, or `--disable-prefill-cuda-graph`. This path does not invent those flags; it leaves the image defaults, as the smoke did. The journalled command also did **not** pass `--shm-size=32g`, a cache volume, `--chunked-prefill-size`, or a reasoning parser (those were in the Qwen/Gemma and/or Nemotron smoke commands, not this one). Historical smoke used `docker run -d` and did not lead with `docker rm -f`; the gold start script uses foreground `exec docker run` (no `-d`) and will only `docker rm` an *exited* leftover named `ge2340-lfm25-24b-a2b` — never a running container, never `docker rm -f`, never any other name.

Added for the gold path (not in the smoke command, do not force thinking on):

- `--device=cuda` and `--cpu-offload-gb=0` so auto-detect cannot pick CPU and offload stays off
- `--default-chat-template-kwargs='{"enable_thinking":false}'` because official gold sends `enable_thinking=false`. The same SGLang image already accepted this switch on the Qwen, Gemma, and Nemotron gold paths. **The on-disk `chat_template.jinja` does not mention `enable_thinking` at all.** On `add_generation_prompt` it always emits `<|im_start|>assistant\n<think>` — an unclosed think-start, not an empty `<think></think>` prelude, and not a branch that can default the flag to True or False. `preserve_thinking` (default false) only affects how prior assistant messages are re-rendered. The kwargs switch is still added so a server default exists if SGLang itself consults the key, but it should not be assumed to close that `<think>` tag or turn thinking off inside this template. Do not turn thinking on.
- The wrapper also copies a top-level request field `enable_thinking` into `chat_template_kwargs` before SGLang parses the body. Stock `ChatCompletionRequest` in this image has no `enable_thinking` field, so the eval script's top-level `enable_thinking=false` would otherwise be dropped. Per-request `chat_template_kwargs` still wins if already set. Temperature 0 is already sent by `eval_three_arms.py` and is not overridden here. Do not add a reasoning or tool-call parser that was not in the smoke command.

`mem-fraction-static=0.85` is tight next to PaddleOCR on a 121.63 GiB GB10, but that pairing is what the Arm C smokes used while paddleocr-vl-worker was up. Leave PaddleOCR headroom; do not raise the fraction; do not require Flash-Next.

The gold start script is foreground `exec docker run` (no `-d`) so the operator's nohup pid tracks the serve. If CUDA is hidden, `SGLANG_USE_CPU_ENGINE=1`, `torch.cuda.is_available()` is false, nvidia-smi fails, `--device` is not cuda, or `cpu_offload_gb>0`, the start script / wrapper exits and does not fall back to CPU. The start script also refuses to pull a missing image, refuses a different image id than `3ede270d2e15…`, refuses to start if `:8103` is already bound, and will not stop a *running* `ge2340-lfm25-24b-a2b` (it only `docker rm`s that exact name if it is not running). It never `docker rm`s any other name.

Do not use a start script without `-gpu` for this gold run — there isn't one, and one must not be invented as a CPU path.

## FLAGS/RISK — thinking, unclosed `<think>`, and 13s smoke latency

Historical Oct 30 facts (do not use smoke as official gold; leave smoke files untouched):

- Smoke serve had no health wrapper and no `--default-chat-template-kwargs`. The locked eval script (and the Oct 30 smoke notes) send top-level `enable_thinking=false`. Stock `ChatCompletionRequest` in this image drops that field, and this template does not read it anyway, so the generation prompt still opens `<think>`.
- the operator ping at 19:13:00 (journalled `max_tokens=64`, no temperature / no `enable_thinking` in the body) returned HTTP 200 with assistant content filled and `finish_reason=length`, recorded in the 19:13:18 SERVED READY. That session completed in ~2s, so it was not a multi-minute runaway, but a 64-token `length` stop on a "ping" is consistent with the unclosed think-start filling the cap. The raw ping body was not journalled (tailscaled logs the `--cmd=`, not stdout). A 19:13:53 follow-up ping used `max_tokens=64`, `temperature=0`, and printed `reasoning_content` / message keys in a Python snippet; stdout of that snippet was also not journalled, so whether `reasoning_content` was empty cannot be verified from the journal.
- smoke left on disk: `eval_smoke_lfm25_24b_a2b.json` sha256 `523f23352af861aa48181b529ca6c5adebb267ea3306df8902fa6db385270adc` (1830 B, mtime 2026-10-30 19:16), `.md` sha256 `deb27a47a36c38ac023d9e28d3449bfd770eecdc0dbad8427c2f3b61a00760c7` (984 B). n=10, Acc_hard 0.60, Acc_conf 0.885, macro_f1≈0.4167, mean latency ~13240.5 ms, errors=0, model=`lfm2.5-24b-a2b`, notes say `enable_thinking=false`. Mean ~13s is slower than thinking-off Qwen gold (~4s) / Nemotron gold (~0.9s). The smoke JSON/MD/log contain predictions and aggregate metrics only — no raw `content`, `reasoning_content`, think tags, or per-item finish reasons — so thinking-on vs cap-hit vs slower thinking-off decode cannot be proven from those files. The locked `eval_three_arms.py` falls back to `reasoning_content` / `reasoning` when `content` is empty, so a thinking-on smoke could still have scored with `errors=0`.
- No reasoning parser was on for smoke, so think-text would stay inside `content` rather than being split out (unless the engine still populated `reasoning_content` on its own; not verified).
- Official gold still uses the locked `eval_three_arms.py` (sha256 `033032419a0b4abbd2e8fde81762bcd7713847a3b1434020e1786b8c0700ff2d`; `max_tokens=1024`, `enable_thinking=false`, temperature 0) and must not be rewritten here. The gold path adds the default kwargs switch and the wrapper copy so that flag is not dropped; that does not, by itself, change this template's unclosed `<think>` generation prompt.

**If later official responses look thinking-on, put text only in `reasoning_content`, hit the 1024 cap, or run away (smoke-like multi-minute latency or 300s timeouts), that is a HOLD, not official gold.** Do not start a second run, do not raise `max_tokens` in the locked script, and do not unload the serve from this path.

Official output path when later GO'd (do not write these files now): `experiments/models/eval_gold_masked_v2_C_lfm25_24b_a2b.json` and `.md`.

## Health check (required — device must be the actual compute device)

Stock SGLang `GET /health` is an empty HTTP 200 (or 503 while starting). That is not enough. Official gold uses the wrapper:

`GET http://127.0.0.1:8103/health`

After the process is up, the JSON must show:

- `status=ok`
- `model=lfm2.5-24b-a2b`
- `backend=sglang-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — source of truth is the loaded `ServerArgs.device` (plus `base_gpu_id` / `torch.cuda.current_device()` for the index), not a hardcoded string. `ServerArgs.device` is what this image resolved and used to launch the engine (`--device=cuda`; auto-detect would also call `get_device()`, which returns `cpu` only if `SGLANG_USE_CPU_ENGINE=1`).
- `device_mode=cuda`
- `port=8103`
- `error=null`
- `cpu_offload_gb=0` (informational)

While weights are still loading, the same URL returns HTTP 503 with `status=starting` and `error=not_loaded`. That is not a GO. HTTP 503 starting is not READY.

If `device` is `cpu`, anything that does not start with `cuda`, or `cpu_offload_gb>0`, this is a HOLD: do not serve gold eval, do not silently accept CPU. HTTP status is 503 with `status=error`.

`GET /get_server_info` or `GET /server_info` also includes `device` (index-stripped, e.g. `cuda`) among the launch ServerArgs. The official pre-gold check is still `GET /health` above. Independently confirm `device` starts with `cuda` before gold.

## Evidence LFM can use GPU (no serve process was started for this path prep)

- This exact image and the 19:06:55 flag set already served Arm C LFM smoke on GX10: container `ge2340-lfm25-24b-a2b`, bind `:8103`, model `lfm2.5-24b-a2b`, auth none, host network, weights ro-mounted. `/v1/models` was HTTP 200 around 19:13 Asia/Hong_Kong (`max_model_len=32768`). Chat ping HTTP 200 as above. smoke n=10 acc_hard=0.60 (`eval_smoke_lfm25_24b_a2b.json` / `.md`). That smoke JSON is not official gold and must not be overwritten. The container was torn down ~19:21 the same day; weights were kept. The earlier weights-only note was left unchanged at the time.
- Host: NVIDIA GB10 ~121.63 GiB, CUDA 13.0, aarch64, driver 580.178.04. Laya venv torch is `2.15.0+cu132` and `torch.cuda.is_available()` is the host-side gate. Inside the image, `is_cpu()` is true only when `SGLANG_USE_CPU_ENGINE=1`; otherwise `get_device()` returns `cuda` when `torch.cuda.is_available()`, and raises if no accelerator is present. The wrapper exits before bind if CUDA is missing.
- The ~45 GiB MoE weights previously co-resided with PaddleOCR at `mem-fraction-static=0.85`. Disk is tight; Muse weights must stay; this path does not pull images or install another CUDA stack.
- No in-process / container weight load was done here (would allocate tens of GiB and is reserved until after authorization). No container was started just to test.

No extra package, CUDA-torch install, or Docker pull was needed or performed.

As of this path-prep check (2026-11-15 ~11:11 Asia/Hong_Kong): Nemotron is already down (`GET http://127.0.0.1:8102/health` curl 7 connection refused; `docker inspect ge2340-nemotron35-lightning` → no such object; no nemotron process). `:8103` is also not listening (`GET http://127.0.0.1:8103/health` curl 7; no `ge2340-lfm25-24b-a2b` container). PaddleOCR `:8003` was HTTP 200. FN `qwen38-flash-next-tf` was Exited (0). Those were left as they are. Official LFM and Gemma gold JSON/MD were still absent.
