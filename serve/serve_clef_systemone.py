#!/usr/bin/env python3
"""SystemOne-compatible HTTP wrapper around Cloudflare/clef.

Two load paths, selected by CLEF_DEVICE:

- cuda (or gpu / cuda:0): official load_release_model onto the GPU. Used for
  gold eval when this process is the only model on the GB10. Refuses to start
  if CUDA is hidden or unavailable — never silently falls back to CPU.
- cpu (default): historic CPU + accelerate disk-offload path, for the old
  start-clef.sh that hid the GPU so Flash-Next could keep unified RAM.

ClefModel.forward dives into inner language_model and bypasses accelerate
hooks, so the CPU path wraps forward to (1) execute layers via the hooked
top-level module and (2) materialize embedding weights onto CPU before the
joint head runs. The GPU path does not need that patch.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

MODEL_DIR = Path(os.environ.get("CLEF_MODEL_DIR", "/home/gx10/ge2340/serve/clef"))
OFFLOAD_DIR = Path(os.environ.get("CLEF_OFFLOAD_DIR", "/home/gx10/ge2340/serve/clef-offload"))
MODEL_ID = os.environ.get("CLEF_MODEL_ID", "clef")
HOST = os.environ.get("CLEF_HOST", "0.0.0.0")
PORT = int(os.environ.get("CLEF_PORT", "8012"))
MAX_LENGTH = int(os.environ.get("CLEF_MAX_LENGTH", "2048"))
CPU_MEM = os.environ.get("CLEF_CPU_MEM", "28GiB")
AUTH_ENV = os.environ.get("CLEF_AUTH_ENV", "")
THREADS = int(os.environ.get("CLEF_THREADS", "8"))
DEVICE_MODE = os.environ.get("CLEF_DEVICE", "cpu").strip().lower()

_model = None
_processor = None
_loaded_at: Optional[float] = None
_gate: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _auth_ok(authorization: Optional[str]) -> bool:
    if not AUTH_ENV:
        return True
    expected = os.environ.get(AUTH_ENV, "")
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
            "CLEF_DEVICE=cuda but CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "CLEF_DEVICE=cuda but torch.cuda.is_available() is False; refusing CPU fallback"
        )
    return "cuda"


def _materialize_cpu(t: torch.Tensor) -> torch.Tensor:
    if t.device.type == "meta":
        raise RuntimeError("parameter still on meta after offload; cannot materialize")
    return t.detach().to(device="cpu", dtype=torch.bfloat16)


def _patch_forward(clef_model: Any) -> None:
    """Replace ClefModel.forward so accelerate disk-offload hooks actually run."""

    def forward(batch: dict[str, Any]):
        media = batch.get("media") or {}
        lm = clef_model.language_model
        # Prefer text-only path (matches upstream) but go through modules that
        # still carry accelerate AlignDevicesHook when possible.
        base_model = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        text_model = base_model.model
        if not media and hasattr(text_model, "language_model"):
            inner = text_model.language_model
        else:
            inner = text_model

        outputs = inner(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        hidden = outputs.last_hidden_state
        if hidden.device.type != "cpu":
            hidden = hidden.to("cpu")

        emb = base_model.get_output_embeddings()
        weight = emb.weight
        # If lm_head was disk-offloaded, accelerate leaves a meta tensor here.
        # Force a CPU copy from the tied embed_tokens when possible.
        if weight.device.type == "meta":
            embed = None
            try:
                embed = base_model.model.language_model.embed_tokens.weight
            except Exception:
                try:
                    embed = base_model.model.embed_tokens.weight
                except Exception:
                    embed = None
            if embed is None or embed.device.type == "meta":
                # Last resort: reload lm_head shard from safetensors index
                from safetensors import safe_open

                index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
                weight_key = None
                for k, v in index.get("weight_map", {}).items():
                    if k.endswith("lm_head.weight") or k == "lm_head.weight":
                        weight_key = (k, v)
                        break
                if weight_key is None:
                    # tied weights: use embed_tokens
                    for k, v in index.get("weight_map", {}).items():
                        if "embed_tokens.weight" in k:
                            weight_key = (k, v)
                            break
                if weight_key is None:
                    raise RuntimeError("cannot locate lm_head/embed_tokens in index")
                key, shard = weight_key
                with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
                    weight = f.get_tensor(key).to(dtype=torch.bfloat16)
            else:
                weight = _materialize_cpu(embed)
        else:
            weight = _materialize_cpu(weight)

        return clef_model.head(
            hidden,
            batch["input_ids"].to("cpu"),
            batch["attention_mask"].to("cpu"),
            batch["records"],
            weight,
        )

    clef_model.forward = forward  # type: ignore[method-assign]


def _load() -> None:
    global _model, _processor, _loaded_at, _load_error
    if _model is not None:
        return
    try:
        torch.set_num_threads(THREADS)
        sys.path.insert(0, str(MODEL_DIR))
        if _is_gpu_mode():
            device = _cuda_device()
            from joint_schema_model import load_release_model  # type: ignore

            clef, processor = load_release_model(
                MODEL_DIR, device=device, dtype=torch.bfloat16
            )
            param = next(clef.parameters())
            if param.device.type != "cuda":
                raise SystemExit(
                    f"CLEF_DEVICE=cuda but weights landed on {param.device}; refusing to serve"
                )
            _model = clef
            _processor = processor
            _loaded_at = time.time()
            _load_error = None
            print(
                f"[clef] loaded on {param.device} name={torch.cuda.get_device_name(0)}",
                flush=True,
            )
            return

        OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        from joint_schema_model import ClefModel, JointSchemaHead  # type: ignore
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

        # Build device map that keeps embeddings + early layers on CPU and the
        # rest on disk — and crucially keeps lm_head / norm on CPU for the head.
        with init_empty_weights():
            empty = Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL_DIR, torch_dtype=torch.bfloat16
            )
        no_split = getattr(empty, "_no_split_modules", None) or [
            "Qwen3_5DecoderLayer",
            "Qwen3_5VisionBlock",
        ]
        device_map = infer_auto_device_map(
            empty,
            max_memory={"cpu": CPU_MEM},
            no_split_module_classes=no_split,
            dtype=torch.bfloat16,
        )
        # Force critical tensors onto CPU (not disk)
        forced_cpu_prefixes = (
            "model.visual",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        )
        for key in list(device_map.keys()):
            if any(key == p or key.startswith(p + ".") for p in forced_cpu_prefixes):
                device_map[key] = "cpu"
        # Also force first N layers to CPU if mapped to disk — already handled by max_memory

        backbone = load_checkpoint_and_dispatch(
            empty,
            str(MODEL_DIR),
            device_map=device_map,
            offload_folder=str(OFFLOAD_DIR),
            dtype=torch.bfloat16,
            no_split_module_classes=no_split,
        )
        backbone.config.use_cache = False

        head_config = json.loads((MODEL_DIR / "joint_head_config.json").read_text())
        head = JointSchemaHead(**head_config)
        head.load_state_dict(load_file(MODEL_DIR / "joint_head.safetensors"), strict=True)
        head = head.to(device="cpu", dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        clef = ClefModel(backbone, head).eval()
        _patch_forward(clef)
        _model = clef
        _processor = processor
        _loaded_at = time.time()
        _load_error = None
        # Log a short device map summary
        try:
            dm = getattr(backbone, "hf_device_map", {}) or {}
            cpu_n = sum(1 for v in dm.values() if v == "cpu")
            disk_n = sum(1 for v in dm.values() if v == "disk")
            print(f"[clef] device_map cpu={cpu_n} disk={disk_n} keys={len(dm)}", flush=True)
        except Exception:
            pass
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise


def _predict(body: Dict[str, Any]) -> dict:
    from joint_schema_model import systemone  # type: ignore

    req = dict(body)
    if not isinstance(req.get("model"), str) or not req.get("model"):
        req["model"] = MODEL_ID
    return systemone(_model, _processor, req, max_length=MAX_LENGTH)


app = FastAPI(title="Clef SystemOne", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else ("error" if _load_error else "starting"),
        "model": MODEL_ID,
        "backend": "torch-cuda" if _is_gpu_mode() else "torch-cpu-disk-offload",
        "device": (
            str(next(_model.parameters()).device)
            if _model is not None
            else ("cuda" if _is_gpu_mode() else "cpu+disk_offload")
        ),
        "device_mode": DEVICE_MODE,
        "port": PORT,
        "model_dir": str(MODEL_DIR),
        "offload_dir": None if _is_gpu_mode() else str(OFFLOAD_DIR),
        "max_length": MAX_LENGTH,
        "cpu_mem_budget": None if _is_gpu_mode() else CPU_MEM,
        "loaded_at": _loaded_at,
        "error": _load_error,
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
    if not isinstance(body, dict) or "questions" not in body:
        raise HTTPException(
            status_code=400,
            detail="request body must be an object with a 'questions' field",
        )
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
        return JSONResponse(
            content=result,
            headers={
                "Server-Timing": f"inference;dur={ms:.2f}",
                "X-Inference-Time-Ms": f"{ms:.2f}",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # Surface type+message for Eng debugging (no secrets)
        raise HTTPException(
            status_code=500, detail=f"inference failed: {type(e).__name__}: {e}"
        ) from None


def main() -> None:
    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
