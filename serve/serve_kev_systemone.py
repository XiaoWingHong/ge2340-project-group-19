#!/usr/bin/env python3
"""SystemOne-compatible HTTP wrapper around jaredpalmer/kev-27b.

Two load paths, selected by KEV_DEVICE:

- cuda (or gpu / cuda:0): official kev.checkpoint torch load onto the GPU
  (same path kev.serve uses on NVIDIA). Used for gold eval when this process
  is the only eval-arm model on the GB10. Refuses to start if CUDA is hidden
  or unavailable — never silently falls back to CPU.
- cpu (default): historic CPU + accelerate disk-offload path, for the old
  start-kev.sh that hid the GPU so Flash-Next could keep unified RAM.

Official python -m kev.serve has no GET /health; this wrapper keeps that
contract for official gold latency. Do not use start-kev.sh for official gold latency.
"""
from __future__ import annotations

import asyncio
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

MODEL_DIR = Path(os.environ.get("KEV_MODEL_DIR", "/home/gx10/ge2340/serve/kev"))
OFFLOAD_DIR = Path(os.environ.get("KEV_OFFLOAD_DIR", "/home/gx10/ge2340/serve/kev-offload"))
MODEL_ID = os.environ.get("KEV_MODEL_ID", "kev-27b")
HOST = os.environ.get("KEV_HOST", "0.0.0.0")
PORT = int(os.environ.get("KEV_PORT", "8013"))
CPU_MEM = os.environ.get("KEV_CPU_MEM", "14GiB")
AUTH_ENV = os.environ.get("KEV_AUTH_ENV", "")
THREADS = int(os.environ.get("KEV_THREADS", "8"))
API_KEY = os.environ.get("KEV_API_KEY")
DEVICE_MODE = os.environ.get("KEV_DEVICE", "cpu").strip().lower()

_tok = None
_model = None
_loaded_at: Optional[float] = None
_gate: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None
_meta = None


def _auth_ok(authorization: Optional[str]) -> bool:
    expected = None
    if AUTH_ENV:
        expected = os.environ.get(AUTH_ENV, "") or None
    if API_KEY:
        expected = API_KEY
    if not expected:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return authorization[len("Bearer ") :] == expected


def _is_gpu_mode() -> bool:
    if DEVICE_MODE in {"cuda", "gpu", "cuda:0"}:
        return True
    return DEVICE_MODE.startswith("cuda:")


def _cuda_device() -> str:
    """Return 'cuda' or exit. Never fall back to CPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        raise SystemExit(
            "KEV_DEVICE=cuda but CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "KEV_DEVICE=cuda but torch.cuda.is_available() is False; refusing CPU fallback"
        )
    return "cuda"


def _load_gpu() -> None:
    """Official full-weight CUDA load (kev.checkpoint.Checkpoint._load_torch)."""
    global _tok, _model, _loaded_at, _load_error, _meta
    from dataclasses import replace

    from transformers import AutoTokenizer

    from kev.checkpoint import Checkpoint, LoadOptions, fused_available

    device = _cuda_device()
    ck = Checkpoint(str(MODEL_DIR))
    _meta = ck.meta

    # Local tree carries a copy of the base tokenizer; prefer it so
    # HF_HUB_OFFLINE=1 does not depend on a Hub fetch of meta.base.
    if (MODEL_DIR / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    else:
        from kev.model import load_tokenizer

        tok = load_tokenizer(_meta.base, revision=_meta.base_revision)

    opts = LoadOptions.from_env()
    if opts.backend in (None, "auto"):
        opts = replace(opts, backend="torch")
    if opts.dtype is None:
        opts = replace(opts, dtype=torch.bfloat16)
    # Same NVIDIA serving defaults as kev.serve: CUDA graphs on, fused
    # kernels only when flash-linear-attention is installed at the pin.
    if opts.cuda_graphs is None:
        opts = replace(opts, cuda_graphs=True)
    if opts.fused is None:
        opts = replace(opts, fused=fused_available())

    model = ck._load_torch(tok, device, opts)
    model.head.load_state_dict(_meta.head)
    model.eval()
    model.head.temperature = (
        _meta.temperature if opts.temperature is None else opts.temperature
    )

    param = next(model.lm.parameters())
    if param.device.type != "cuda":
        raise SystemExit(
            f"KEV_DEVICE=cuda but weights landed on {param.device}; refusing to serve"
        )

    _tok = tok
    _model = model
    _loaded_at = time.time()
    _load_error = None
    graphs = bool(getattr(model, "graphs", None))
    print(
        f"[kev] loaded on {param.device} name={torch.cuda.get_device_name(0)} "
        f"backend={model.backend} dtype={model.dtype} cuda_graphs={graphs} "
        f"T={model.head.temperature}",
        flush=True,
    )


def _load_cpu_offload() -> None:
    """Historic CPU + disk-offload path. Unchanged from the 2026-10-30 wrapper."""
    global _tok, _model, _loaded_at, _load_error, _meta
    torch.set_num_threads(THREADS)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from kev.checkpoint import Checkpoint
    from kev.model import DecisionModel, PointerHead, is_hybrid, pad_id

    ck = Checkpoint(str(MODEL_DIR))
    _meta = ck.meta

    if (MODEL_DIR / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(
            _meta.base, revision=_meta.base_revision, local_files_only=False
        )

    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    text_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config

    with init_empty_weights():
        empty = AutoModel.from_config(config, torch_dtype=dtype)

    no_split = getattr(empty, "_no_split_modules", None) or [
        "Qwen3_5DecoderLayer",
        "Qwen3_5TextDecoderLayer",
        "Qwen3DecoderLayer",
    ]
    device_map = infer_auto_device_map(
        empty,
        max_memory={"cpu": CPU_MEM},
        no_split_module_classes=no_split,
        dtype=dtype,
    )
    forced_cpu_prefixes = (
        "embed_tokens",
        "norm",
        "rotary_emb",
    )
    for key in list(device_map.keys()):
        leaf = key.split(".")[-1]
        if leaf in forced_cpu_prefixes or any(
            key.endswith("." + p) or key == p for p in forced_cpu_prefixes
        ):
            device_map[key] = "cpu"

    backbone = load_checkpoint_and_dispatch(
        empty,
        str(MODEL_DIR),
        device_map=device_map,
        offload_folder=str(OFFLOAD_DIR),
        dtype=dtype,
        no_split_module_classes=no_split,
    )

    m = DecisionModel.__new__(DecisionModel)
    nn.Module.__init__(m)
    m.lm = backbone
    m.pad_id = pad_id(tok)
    m.hybrid = is_hybrid(text_cfg)
    m.option_isolation = bool(_meta.option_isolation)
    m.head = PointerHead(backbone.config.hidden_size, dp=int(_meta.head_dim))
    m.head.load_state_dict(_meta.head)
    m.head.temperature = float(_meta.temperature)
    m.head.to(device="cpu", dtype=torch.float32)
    m.device = "cpu"
    m.graphs = None
    m.eval()

    _tok = tok
    _model = m
    _loaded_at = time.time()
    _load_error = None
    try:
        dm = getattr(backbone, "hf_device_map", {}) or {}
        cpu_n = sum(1 for v in dm.values() if v == "cpu")
        disk_n = sum(1 for v in dm.values() if v == "disk")
        print(
            f"[kev] device_map cpu={cpu_n} disk={disk_n} keys={len(dm)} T={m.head.temperature}",
            flush=True,
        )
    except Exception:
        pass


def _load() -> None:
    global _load_error
    if _model is not None:
        return
    try:
        if _is_gpu_mode():
            torch.set_num_threads(THREADS)
            _load_gpu()
            return
        _load_cpu_offload()
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise


def _actual_device() -> str:
    """Compute device to report. After a GPU load, this is the backbone parameter device."""
    if _is_gpu_mode():
        if _model is not None:
            return str(next(_model.lm.parameters()).device)
        return "cuda"
    return "cpu+disk_offload"


def _predict(body: dict[str, Any]) -> dict:
    from kev.api import SystemOneRequest, to_record, to_answers, with_date_facts, output_tokens
    from kev.model import admit

    req = SystemOneRequest.model_validate(body)
    if os.environ.get("KEV_DATE_FACTS", "0") == "1":
        req = req.model_copy(update={"state": with_date_facts(req.state)})
    rec, meta = to_record(req)
    t0 = time.perf_counter()
    enc = admit(
        _model,
        _tok,
        rec,
        truncate=os.environ.get("KEV_TRUNCATE_STATES", "0") == "1",
    )
    with torch.inference_mode():
        probs = _model.probs(enc)
    ps = []
    for p in probs:
        if hasattr(p, "detach"):
            ps.append([float(x) for x in p.detach().cpu().tolist()])
        else:
            ps.append([float(x) for x in p])
    answers = to_answers(ps, meta)
    ms = (time.perf_counter() - t0) * 1000.0
    model_name = body.get("model") or MODEL_ID
    if model_name in ("kev-latest", "jev-latest"):
        model_name = MODEL_ID
    return {
        "model": model_name,
        "answers": answers,
        "usage": {
            "input_tokens": len(enc["ids"]),
            "output_tokens": output_tokens(_tok, answers),
        },
        "latency_ms": round(ms, 1),
    }


app = FastAPI(title="Kev SystemOne", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict:
    gpu = _is_gpu_mode()
    body = {
        "status": "ok" if _model is not None else ("error" if _load_error else "starting"),
        "model": MODEL_ID,
        "backend": "torch-cuda" if gpu else "torch-cpu-disk-offload",
        "device": _actual_device(),
        "device_mode": DEVICE_MODE,
        "port": PORT,
        "model_dir": str(MODEL_DIR),
        "offload_dir": None if gpu else str(OFFLOAD_DIR),
        "cpu_mem_budget": None if gpu else CPU_MEM,
        "temperature": None if _model is None else float(_model.head.temperature),
        "loaded_at": _loaded_at,
        "error": _load_error,
    }
    if gpu and _model is not None:
        body["dtype"] = getattr(_model, "dtype", None)
        body["cuda_graphs"] = bool(getattr(_model, "graphs", None))
    return body


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
                "aliases": ["kev-latest", "jev-latest"],
            }
        ],
    }


@app.post("/v1/systemone")
async def systemone_route(
    request: Request, authorization: Optional[str] = Header(default=None)
):
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    if _model is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    if "questions" not in body:
        raise HTTPException(status_code=400, detail="questions is required")
    if "state" not in body:
        raise HTTPException(status_code=400, detail="state is required")
    global _gate
    if _gate is None:
        _gate = asyncio.Lock()
    try:
        async with _gate:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(None, partial(_predict, body))
            ms = (time.perf_counter() - t0) * 1000.0
        rid = request.headers.get("x-typesafe-request-id") or ""
        return JSONResponse(
            content=result,
            headers={
                "Server-Timing": f"inference;dur={ms:.2f}",
                "X-Inference-Time-Ms": f"{ms:.2f}",
                "x-typesafe-request-id": rid,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"inference failed: {type(e).__name__}: {e}"
        ) from None


def main() -> None:
    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
