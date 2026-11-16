# Kev GPU serve path

**When:** 2026-11-15 ~02:17 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one eval-arm model at a time. The original `start-kev.sh` sets `CUDA_VISIBLE_DEVICES=""` and the wrapper loads with CPU + disk offload so Flash-Next could keep unified RAM. That path is invalid for official gold latency.

## How to start (after SemIf / other eval backends are fully unloaded, and after authorization)
Leave PaddleOCR-VL on `:8003` running. Do not start or restart Kev until authorized. Do not use official `python -m kev.serve` for this gold run (it binds `127.0.0.1:8008` by default and has no `GET /health`).

```bash
nohup bin/start-kev-gpu.sh \
  > logs/kev-serve.log 2>&1 &
echo $! > logs/kev-serve.pid
```

- Bind: `0.0.0.0:8013`
- Client model id: `kev-27b` (aliases accepted: `kev-latest`, `jev-latest`)
- Weights: `serve/kev` (full-weight `jaredpalmer/kev-27b`, ~48 GiB BF16, hybrid Qwen3.5 / `Qwen3_5TextModel`)
- Venv: `venvs/laya` (editable `kev==0.1.0` from `serve/kev-src`)
- Auth: none by default
- `KEV_DEVICE=cuda`, `HF_HUB_OFFLINE=1`
- Load: official `Checkpoint._load_torch("cuda", LoadOptions)` — the same NVIDIA torch path `kev.serve` uses (bf16, CUDA graphs on by default). Tokenizer is read from the local tree so the load stays offline. `fla` is not installed, so fused Qwen3.5 kernels stay off (`fused_available()=False`); that is optional, not a HOLD. `KEV_CUDA_GRAPHS=0` declines graphs; `KEV_DTYPE=fp32` restores the exact path.
- If CUDA is hidden or `torch.cuda.is_available()` is false, the script exits and does not fall back to CPU. After load, the wrapper also refuses to serve if backbone weights are not on CUDA.

Do not use `bin/start-kev.sh` for this gold run.

The original CPU start script is unchanged. The CPU branch in `serve_kev_systemone.py` is unchanged in behavior. Backup of the pre-change wrapper: `serve_kev_systemone.py.bak-20261115-0217` next to the script.

## Health check (required — device must be the actual compute device)

After the process is up, `GET /health` must show:

- `status=ok`
- `model=kev-27b`
- `backend=torch-cuda`
- **`device` starting with `cuda`** (typically `cuda:0`) — this is the source of truth, taken from the loaded backbone parameter device, not a hardcoded string
- `device_mode=cuda`
- `port=8013`

If `device` is `cpu`, `cpu+disk_offload`, or anything that does not start with `cuda`, this is a HOLD: do not serve gold eval, do not silently accept CPU. The start script also exits before bind if `CUDA_VISIBLE_DEVICES` is empty or `torch.cuda.is_available()` is false.

## Evidence Kev can use GPU (no serve process was started)

- Laya venv torch is `2.15.0+cu132` (`torch.version.cuda=13.2`), `torch.cuda.is_available()=True`, device `NVIDIA GB10`, `total_mem=121.63 GiB`.
- Official `kev.serve` / `Checkpoint.load` is a torch CUDA path. This checkpoint is full-weight BF16 (`weights_dtype=bf16`, 64 layers, hidden 5120). READY notes say official serve wants ~65 GB resident; that fits the GB10 with PaddleOCR's existing ~15 GiB sglang footprint left running.
- Clef already used this same venv and `torch.cuda` successfully for gold (full ~51 GiB BF16).
- No in-process weight load was done here (would allocate tens of GiB and is reserved until after authorization). Tokenizer/config import with `local_files_only=True` succeeds. `peft` is not installed; it is not required for this full-weight checkpoint.

No extra package or CUDA-torch install was needed or performed.
