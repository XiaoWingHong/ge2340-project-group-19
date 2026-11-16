# OpenJev GPU serve path

**When:** 2026-11-15 ~02:43 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. The original `start-openjev.sh` sets `CUDA_VISIBLE_DEVICES=""` and the wrapper loads with accelerate CPU + disk offload so Flash-Next could keep unified RAM. That path is invalid for official gold latency.

## How to start (after Kev / other eval backends are fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Do not start or restart OpenJev until authorized. Do not use official `vllm serve` + `helper/shim.py` for this gold run (vLLM is not installed in the laya venv; installing it would be a large extra CUDA stack on a tight disk). Gold uses this wrapper's in-process letter-logprob readout, the same SystemOne contract as the historic CPU path, on CUDA.

```bash
nohup bin/start-openjev-gpu.sh \
  > logs/openjev-serve.log 2>&1 &
echo $! > logs/openjev-serve.pid
```

- Bind: `0.0.0.0:8014`
- Client model id: `openjev` (aliases accepted: `jev-latest`)
- Weights: `serve/openjev` (full-weight `openjev/openjev`, ~50.96 GiB BF16, `Qwen3_5ForConditionalGeneration`, 27.36B params, 64 layers, hidden 5120, includes vision tower)
- Venv: `venvs/laya` (torch `2.15.0+cu132`, transformers `5.22.0`, accelerate `1.15.0`)
- Auth: none by default; optional `OPENJEV_API_KEY` or `OPENJEV_AUTH_ENV=<env name>` + Bearer
- `OPENJEV_DEVICE=cuda`, `HF_HUB_OFFLINE=1`
- Calibration (official SERVE.md): `READOUT_T=0.85`, `READOUT_NOUL_T=1.829074`, `READOUT_NOUL_BIAS=0`, `READOUT_INSTR_STYLE=pyrepr`
- Load: `Qwen3_5ForConditionalGeneration.from_pretrained(..., torch_dtype=bfloat16, device_map="cuda", local_files_only=True)` — no disk offload, no CPU placement. After load, the wrapper refuses to serve if any non-meta parameter is not on CUDA, or if `hf_device_map` contains `cpu`/`disk` placements. Inputs are moved to the loaded parameter device at inference time.
- If CUDA is hidden or `torch.cuda.is_available()` is false, the script exits and does not fall back to CPU.

Do not use `bin/start-openjev.sh` for this gold run.

The original CPU start script is unchanged. The CPU branch in `serve_openjev_systemone.py` is unchanged in behavior. Backup of the pre-change wrapper: `serve_openjev_systemone.py.bak-20261115-0243` next to the script.

## Health check (required — device must be the actual compute device)

After the process is up, `GET /health` must show:

- `status=ok`
- `model=openjev`
- `backend=torch-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — this is the source of truth, taken from a loaded backbone parameter device, not a hardcoded string
- `device_mode=cuda`
- `port=8014`
- `error=null` / omitted-none

If `device` is `cpu`, `cpu+disk_offload`, or anything that does not start with `cuda`, this is a HOLD: do not serve gold eval, do not silently accept CPU. The start script also exits before bind if `CUDA_VISIBLE_DEVICES` is empty or `torch.cuda.is_available()` is false.

## Evidence OpenJev can use GPU (no serve process was started)

- Laya venv torch is `2.15.0+cu132` (`torch.version.cuda=13.2`), `torch.cuda.is_available()=True`, device `NVIDIA GB10`, `total_mem=121.63 GiB`. OpenJev scores with this torch, so the `torch.cuda.is_available()` gate is the correct one (unlike SemIf, which scores through llama.cpp).
- The same venv and `torch.cuda` already served Clef and Kev gold as full-weight BF16 Qwen3.5-class models.
- This checkpoint is full-weight BF16 (`dtype=bfloat16`, 64 layers, hidden 5120, `total_parameters=27356728560`, safetensors ~50.96 GiB). That fits the GB10 with PaddleOCR's existing ~15.5 GiB sglang footprint left running (Kev's similar 27B BF16 load was ~61.5 GiB resident).
- Official card numbers used vLLM 0.29.0 + `helper/shim.py` with online FP8 on an H100. vLLM is not in the laya venv. It was not installed: disk has ~68 GiB free, Muse weights must stay, and a second CUDA torch / vLLM pin would be a large, risky install. The locked gold contract is this wrapper's `POST /v1/systemone` + `GET /health`, not the official shim (`GET /v1/version`, default bind `127.0.0.1:8765`).
- No in-process weight load was done here (would allocate tens of GiB; reserved until after authorization). Tokenizer/config are already readable `local_files_only` on the historic CPU path. `Qwen3_5ForConditionalGeneration.from_pretrained` accepts `device_map` (transformers 5.22.0).

No extra package or CUDA-torch install was needed or performed.
