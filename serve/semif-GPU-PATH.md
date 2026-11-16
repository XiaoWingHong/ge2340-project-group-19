# SemIf GPU serve path

**When:** 2026-11-15 ~02:04 Asia/Hong_Kong (script revised)
**Why:** Gold eval policy is GPU only, one model at a time. The original `start-semif.sh` leaves `n_gpu_layers=0`, and `/health` previously hard-coded `device=cpu`. The llama-cpp-python 0.3.35 in the SemIf venv was also a CPU-only build (`llama_supports_gpu_offload()=False`, no `libggml-cuda`). That path is invalid for official gold latency.

## How to start (after Laya is fully unloaded, and after authorization)
```bash
nohup bin/start-semif-gpu.sh \
  > logs/semif-serve.log 2>&1 &
echo $! > logs/semif-serve.pid
```

- Bind: `0.0.0.0:8011`
- Client model id: `semif-qwen3.5-4b-gguf`
- GGUF: `serve/semif-weights/gguf/Qwen_Qwen3.5-4B-Q4_K_M.gguf`
- Auth: none by default
- `SEMIF_DEVICE=cuda`, `SEMIF_N_GPU_LAYERS=-1` (all layers), `HF_HUB_OFFLINE=1`
- Same pinned `llama-cpp-python==0.3.35`, rebuilt in `venvs/semif` with `GGML_CUDA=on` and `CMAKE_CUDA_ARCHITECTURES=121` (NVIDIA GB10, compute 12.1). After the rebuild, `llama_supports_gpu_offload()` is True and `libggml-cuda.so` is present. The previous CPU-only install is backed up at `venvs/semif-llama-cpp-cpu-0.3.35.bak`. `numpy` was pinned back to `2.2.6` (semif-phase1 requirement) after pip's force-reinstall briefly pulled 2.5.4.
- `serve_systemone.py` is 0.2.0: `/health` reports actual `device` and `n_gpu_layers` from load metadata. `llamacpp_backend.py` honors `SEMIF_DEVICE` / `SEMIF_N_GPU_LAYERS` and refuses a CUDA request if offload is unavailable. Originals: `serve_systemone.py.bak-20261115-0156` and `llamacpp_backend.py.bak-20261115-0156`.
- The original `bin/start-semif.sh` is unchanged and still defaults to CPU (`n_gpu_layers=0`). Do not use it for this gold run.

## Start-script revision (2026-11-15 ~02:04 Asia/Hong_Kong)

The first `start-semif-gpu.sh` (sha256 `65c477d5358737649a729afd074254f16a729a3f9640938e059d4716e81a1014`, kept as `start-semif-gpu.sh.bak-20261115-0157`) exited immediately at 02:00:11 because it gated on `torch.cuda.is_available()`. That check is wrong for this serve: SemIf scores through llama.cpp, and the SemIf venv's torch is CPU-only (`2.10.0+cpu`, `torch.version.cuda=None`). No CUDA torch was installed (disk-careful; not required). The revised script drops the torch gate and still refuses to start if `CUDA_VISIBLE_DEVICES` is empty or `llama_supports_gpu_offload()` is false. After start, `GET /health` remains the required actual-device check.

## Health check (required)

After the process is up, `GET /health` must show the **actual** compute device as `cuda` (status ok, `n_gpu_layers` not 0). `device` comes from load metadata, not a hardcoded string. If `device` is `cpu` or anything other than `cuda`, this is a HOLD: do not serve gold eval, do not silently accept CPU. The start script also exits before bind if CUDA is hidden or `llama_supports_gpu_offload()` is false.

## the operator preflight (not the official serve)

On 2026-11-15 01:58:17–01:58:25 Asia/Hong_Kong the GGUF was loaded once in-process with `SEMIF_DEVICE=cuda` (no HTTP bind, `:8011` left free). Result: `device=cuda`, `n_gpu_layers=-1`, `offloaded 34/34 layers to GPU`, CUDA0 model buffer 2740.75 MiB. The process exited and released that allocation. Official start remains after authorization.
