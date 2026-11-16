# Clef GPU serve path

**When:** 2026-11-15 ~01:18 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one model at a time. The original `start-clef.sh` sets `CUDA_VISIBLE_DEVICES=""` and the wrapper loads with CPU + disk offload so Flash-Next could keep unified RAM. That path is invalid for official gold latency.

## How to start (after all co-resident eval backends are down)
```bash
# stop any leftover Clef/Laya/SemIf/qwen38-27b first
nohup bin/start-clef-gpu.sh \
  > logs/clef-serve.log 2>&1 &
echo $! > logs/clef-serve.pid
```

- Bind: `0.0.0.0:8012`
- Model id: `clef`
- Auth: none
- Load: official `load_release_model(..., device="cuda")` — full ~51 GiB BF16 on the GB10, no disk offload
- Health: `GET /v1` is not used; `GET /health` must show `backend=torch-cuda` and `device` starting with `cuda`
- If CUDA is hidden or unavailable, the script exits and does not fall back to CPU

Do not use `bin/start-clef.sh` for this gold run.

The original CPU start script and the CPU branch in `serve_clef_systemone.py` are unchanged. Backup of the pre-change wrapper: `serve_clef_systemone.py.cpu-offload.bak` next to the script.

Do not restore Flash-Next. After Clef is stopped later, `serve/clef-offload` may be deleted to reclaim ~24G.
