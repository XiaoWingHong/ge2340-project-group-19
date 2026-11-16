# Laya GPU serve path

**When:** 2026-11-15 ~01:44 Asia/Hong_Kong
**Why:** Gold eval policy is GPU only, one model at a time. The original `start-laya.sh` sets `LAYA_DEVICE=cpu`. That path is invalid for official gold latency.

## How to start (after Clef is fully unloaded)
```bash
nohup bin/start-laya-gpu.sh \
  > logs/laya-serve.log 2>&1 &
echo $! > logs/laya-serve.pid
```

- Bind: `0.0.0.0:8010`
- Client model id: `english` (local tree `serve/laya`)
- Auth: none
- `LAYA_DEVICE=cuda`, `USE_TF=0`, `HF_HUB_OFFLINE=1`
- Do not use `bin/start-laya.sh` for this gold run

## Health check (required — LAYA_DEVICE is only a preference)

laya-serve may silently fall back to CPU if it cannot get a GPU, and would still answer. After the process is up, `GET /health` must show the **actual** compute device as `cuda` (status ok, loaded includes `english`). If `device` is `cpu` or anything other than `cuda`, this is a HOLD: do not serve gold eval, do not silently accept CPU.
