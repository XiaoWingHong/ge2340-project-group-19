#!/usr/bin/env python3
"""SystemOne-compatible HTTP wrapper around openjev/openjev.

Two load paths, selected by OPENJEV_DEVICE:

- cuda (or gpu / cuda:0): full-weight torch load onto the GPU via
  Qwen3_5ForConditionalGeneration.from_pretrained(..., device_map="cuda").
  Used for official gold eval when OpenJev is the only eval-arm model on the
  GB10. Refuses to start if CUDA is hidden or unavailable, and refuses to
  serve if weights did not land on CUDA — never silently falls back to CPU.
- cpu (default): historic accelerate CPU + disk-offload path, unchanged, so
  start-openjev.sh keeps its previous behavior.

Official OpenJev card numbers were measured with vLLM 0.29.0 + helper/shim.py
(FP8 on H100). That stack is not installed here. This wrapper keeps letter-
logprob readout in-process with the same SERVE.md calibration constants, and
exposes GET /health plus POST /v1/systemone (Jev-shaped). Text-only for
screenshot/image lane not wired.
"""
from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

MODEL_DIR = Path(os.environ.get("OPENJEV_MODEL_DIR", "/home/gx10/ge2340/serve/openjev"))
OFFLOAD_DIR = Path(os.environ.get("OPENJEV_OFFLOAD_DIR", "/home/gx10/ge2340/serve/openjev-offload"))
MODEL_ID = os.environ.get("OPENJEV_MODEL_ID", "openjev")
HOST = os.environ.get("OPENJEV_HOST", "0.0.0.0")
PORT = int(os.environ.get("OPENJEV_PORT", "8014"))
CPU_MEM = os.environ.get("OPENJEV_CPU_MEM", "4GiB")
MAX_LENGTH = int(os.environ.get("OPENJEV_MAX_LENGTH", "4096"))
AUTH_ENV = os.environ.get("OPENJEV_AUTH_ENV", "")
API_KEY = os.environ.get("OPENJEV_API_KEY", "")
THREADS = int(os.environ.get("OPENJEV_THREADS", "4"))
DEVICE_MODE = os.environ.get("OPENJEV_DEVICE", "cpu").strip().lower()

# Official SERVE.md calibration (measured card numbers)
TEMP = float(os.environ.get("READOUT_T", "0.85"))
NOUL_T = float(os.environ.get("READOUT_NOUL_T", "1.829074"))
NOUL_BIAS = float(os.environ.get("READOUT_NOUL_BIAS", "0"))
INSTR_STYLE = os.environ.get("READOUT_INSTR_STYLE", "pyrepr")
PERMS = int(os.environ.get("READOUT_PERMS", "1"))

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]

_model = None
_tok = None
_letter_ids: dict[str, int] = {}
_loaded_at: Optional[float] = None
_gate: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None
_device_map_summary: dict[str, int] = {}
_shim_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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
            "OPENJEV_DEVICE=cuda but CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "OPENJEV_DEVICE=cuda but torch.cuda.is_available() is False; refusing CPU fallback"
        )
    return "cuda"


def _load_gpu() -> None:
    """Full-weight CUDA load. No disk offload, no CPU placement."""
    global _model, _tok, _letter_ids, _loaded_at, _load_error, _device_map_summary
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    device = _cuda_device()
    torch.set_num_threads(THREADS)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    _letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in LETTERS}

    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    backbone.config.use_cache = False
    backbone.eval()

    param_devices = {p.device for p in backbone.parameters() if p.device.type != "meta"}
    if not param_devices or any(d.type != "cuda" for d in param_devices):
        raise SystemExit(
            "OPENJEV_DEVICE=cuda but weights landed on "
            f"{sorted(str(d) for d in param_devices) or ['<none>']}; refusing to serve"
        )
    param = next(p for p in backbone.parameters() if p.device.type == "cuda")

    dm = getattr(backbone, "hf_device_map", {}) or {}
    _device_map_summary = {
        "cuda": sum(
            1 for v in dm.values() if v == 0 or str(v).startswith("cuda")
        ),
        "cpu": sum(1 for v in dm.values() if v == "cpu"),
        "disk": sum(1 for v in dm.values() if v == "disk"),
        "keys": len(dm),
    }
    if _device_map_summary["cpu"] or _device_map_summary["disk"]:
        raise SystemExit(
            "OPENJEV_DEVICE=cuda but device_map has cpu/disk placements "
            f"{_device_map_summary}; refusing to serve"
        )
    print(
        f"[openjev] loaded on {param.device} name={torch.cuda.get_device_name(0)} "
        f"dtype=bfloat16 device_map={_device_map_summary}",
        flush=True,
    )

    _tok = tok
    _model = backbone
    _loaded_at = time.time()
    _load_error = None


def _load_cpu_offload() -> None:
    """Historic CPU + disk-offload path. Unchanged from the 2026-10-30 wrapper."""
    global _model, _tok, _letter_ids, _loaded_at, _load_error, _device_map_summary
    torch.set_num_threads(THREADS)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    _letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in LETTERS}

    with init_empty_weights():
        empty = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16, local_files_only=True
        )
    no_split = getattr(empty, "_no_split_modules", None) or [
        "Qwen3_5DecoderLayer",
        "Qwen3_5TextDecoderLayer",
        "Qwen3_5VisionBlock",
    ]
    device_map = infer_auto_device_map(
        empty,
        max_memory={"cpu": CPU_MEM},
        no_split_module_classes=no_split,
        dtype=torch.bfloat16,
    )
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

    backbone = load_checkpoint_and_dispatch(
        empty,
        str(MODEL_DIR),
        device_map=device_map,
        offload_folder=str(OFFLOAD_DIR),
        dtype=torch.bfloat16,
        no_split_module_classes=no_split,
    )
    backbone.config.use_cache = False
    backbone.eval()

    dm = getattr(backbone, "hf_device_map", {}) or {}
    _device_map_summary = {
        "cpu": sum(1 for v in dm.values() if v == "cpu"),
        "disk": sum(1 for v in dm.values() if v == "disk"),
        "keys": len(dm),
    }
    print(f"[openjev] device_map cpu={_device_map_summary['cpu']} disk={_device_map_summary['disk']} keys={_device_map_summary['keys']}", flush=True)

    _tok = tok
    _model = backbone
    _loaded_at = time.time()
    _load_error = None


def _load() -> None:
    global _load_error
    if _model is not None:
        return
    try:
        if _is_gpu_mode():
            _load_gpu()
        else:
            _load_cpu_offload()
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise


def _actual_device() -> str:
    """Compute device to report. After a GPU load, this is a loaded parameter device."""
    if _is_gpu_mode():
        if _model is not None:
            for p in _model.parameters():
                if p.device.type != "meta":
                    return str(p.device)
        return "cuda"
    return "cpu+disk_offload"


def _build_prompt(state_text: str, instructions: str, options: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    return (
        f"State:\n{state_text}"
        f"\n\nQuestion: {instructions}\nOptions:\n{lines}\n\n"
        "Answer with the letter of the best option only."
    )


def _next_token_logprobs(prompt: str, allowed: list[int]) -> tuple[list[float], int]:
    """Return logprobs for allowed token ids at the first output position + prompt length."""
    assert _model is not None and _tok is not None
    messages = [{"role": "user", "content": prompt}]
    try:
        text = _tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = _tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    enc = _tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] > MAX_LENGTH:
        input_ids = input_ids[:, -MAX_LENGTH:]
    if _is_gpu_mode():
        input_ids = input_ids.to(next(p.device for p in _model.parameters() if p.device.type != "meta"))
    attn = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = _model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits[0, -1].float().cpu()
    log_probs = torch.log_softmax(logits, dim=-1)
    raw = [float(log_probs[i].item()) for i in allowed]
    return raw, int(input_ids.shape[1])


def _readout_once(state_text: str, instructions: str, options: list[tuple[str, str]]) -> tuple[list[float], int]:
    prompt = _build_prompt(state_text, instructions, options)
    allowed = [_letter_ids[LETTERS[i]] for i in range(len(options))]
    if len(set(allowed)) != len(allowed):
        raise RuntimeError("candidate token ids are not unique")
    raw, ntok = _next_token_logprobs(prompt, allowed)
    if any(not math.isfinite(v) for v in raw):
        raise RuntimeError("non-finite candidate logprobs")
    z = [v / TEMP for v in raw]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e], ntok


def _readout(state_text: str, instructions: str, options: list[tuple[str, str]]) -> tuple[list[float], int]:
    if PERMS <= 1:
        return _readout_once(state_text, instructions, options)
    import random

    acc = [0.0] * len(options)
    tokens = 0
    for j in range(PERMS):
        order = list(range(len(options)))
        random.Random(j).shuffle(order)
        p, t = _readout_once(state_text, instructions, [options[i] for i in order])
        tokens += t
        for pos, i in enumerate(order):
            acc[i] += p[pos] / PERMS
    return acc, tokens


def _distribution(state_text: str, instr: str, opts: list[tuple[str, str]]) -> tuple[list[float], int]:
    if len(opts) <= 52:
        return _readout(state_text, instr, opts)
    k = -(-len(opts) // 52)
    size = -(-len(opts) // k)
    chunks = [opts[i : i + size] for i in range(0, len(opts), size)]
    parts: list[tuple[list[float], int]] = []
    tokens = 0
    for chunk in chunks:
        p, t = _readout(state_text, instr, chunk)
        tokens += t
        parts.append((p, max(range(len(p)), key=p.__getitem__)))
    pf, t = _readout(state_text, instr, [chunk[w] for chunk, (_, w) in zip(chunks, parts)])
    tokens += t
    raw = [pf[c] * pi / p[w] for c, (p, w) in enumerate(parts) for pi in p]
    s = sum(raw)
    return [v / s for v in raw], tokens


class BadQuestion(ValueError):
    pass


def _instr(q: dict) -> str:
    i = q.get("instructions")
    if i is None:
        raise BadQuestion("instructions is required")
    if isinstance(i, str):
        return i
    return str(i) if INSTR_STYLE == "pyrepr" else json.dumps(i, ensure_ascii=False)


def _desc(v: Any) -> str:
    return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))


def _choice_confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    u = 1.0 / len(p)
    return max(0.0, (max(p) - u) / (1.0 - u))


def _score_confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2
    umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        # Drop image fields for the text-only path
        rest = {k: v for k, v in state.items() if k not in ("screenshot", "image")}
        if "document" in rest and len(rest) == 1:
            return str(rest["document"])
        return json.dumps(rest, ensure_ascii=False)
    return str(state)


def _answer_choice(state_text: str, q: dict) -> tuple[dict, int]:
    crit = q.get("criteria")
    if not isinstance(crit, dict) or not crit:
        raise BadQuestion("choice.criteria must be a non-empty map of option -> description|null")
    opts = [(k, _desc(v)) for k, v in crit.items()]
    p, tokens = _distribution(state_text, _instr(q), opts)
    choice = opts[max(range(len(p)), key=lambda i: p[i])][0]
    probs = {k: round(v, 4) for (k, _), v in zip(opts, p)}
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probs,
        "confidence": round(_choice_confidence(p), 4),
    }, tokens


def _answer_score(state_text: str, q: dict) -> tuple[dict, int]:
    levels = q.get("criteria")
    if not isinstance(levels, list) or len(levels) < 2:
        raise BadQuestion("score.criteria must be an ordered array of at least two levels")
    opts = [(str(i), _desc(l)) for i, l in enumerate(levels)]
    p, tokens = _distribution(
        state_text, _instr(q) + " Rate along the ordered levels below (lowest first).", opts
    )
    return {
        "type": "score",
        "score": round(sum(i * pi for i, pi in enumerate(p)), 4),
        "legend": {str(i): l for i, l in enumerate(levels)},
        "probabilities": {str(i): round(pi, 4) for i, pi in enumerate(p)},
        "confidence": round(_score_confidence(p), 4),
    }, tokens


def _answer_noul(state_text: str, q: dict) -> tuple[dict, int]:
    crit = q.get("criteria") or {}
    yes = _desc(crit.get("true")) or "The statement is true."
    no = _desc(crit.get("false")) or "The statement is false."
    p, tokens = _distribution(state_text, _instr(q), [("yes", yes), ("no", no)])
    py = min(max(p[0], 1e-4), 1 - 1e-4)
    z = math.log(py / (1 - py)) / NOUL_T + NOUL_BIAS
    return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}, tokens


ANSWER = {"choice": _answer_choice, "score": _answer_score, "noul": _answer_noul}


def _model_string() -> str:
    flags = {
        "perms": PERMS,
        "targeted": True,
        "instr_style": INSTR_STYLE,
        "backend": "torch-cuda" if _is_gpu_mode() else "torch-cpu-disk-offload",
    }
    return (
        f"{MODEL_ID} T={TEMP} noul={NOUL_T},{NOUL_BIAS} flags="
        + json.dumps(flags, separators=(",", ":"))
        + f" shim=serve_openjev_systemone.py@{_shim_sha[:12]}"
    )


def _predict(body: dict[str, Any]) -> dict:
    state = body.get("state")
    qs = body.get("questions")
    if state is None or not isinstance(qs, dict) or not qs:
        raise ValueError("state and a non-empty questions map are required")
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in ANSWER:
            raise ValueError(f"questions.{qid}.type must be one of choice, score, noul")
    state_text = _state_text(state)
    answers = {}
    toks = 0
    # Sequential (CPU+disk cannot usefully parallelize without thrashing)
    for qid, q in qs.items():
        a, t = ANSWER[q["type"]](state_text, q)
        answers[qid] = a
        toks += t
    return {
        "id": f"openjev-{int(time.time() * 1000)}",
        "model": body.get("model") or _model_string(),
        "answers": answers,
        "usage": {"input_tokens": toks, "output_tokens": 0},
    }


app = FastAPI(title="OpenJev SystemOne", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict:
    gpu = _is_gpu_mode()
    return {
        "status": "ok" if _model is not None else ("error" if _load_error else "starting"),
        "model": MODEL_ID,
        "backend": "torch-cuda" if gpu else "torch-cpu-disk-offload",
        "device": _actual_device(),
        "device_mode": DEVICE_MODE,
        "port": PORT,
        "model_dir": str(MODEL_DIR),
        "offload_dir": None if gpu else str(OFFLOAD_DIR),
        "cpu_mem_budget": None if gpu else CPU_MEM,
        "max_length": MAX_LENGTH,
        "calibration": {"T": TEMP, "noul_t": NOUL_T, "noul_bias": NOUL_BIAS},
        "device_map": _device_map_summary,
        "loaded_at": _loaded_at,
        "error": _load_error,
        "note": (
            "Text-only SystemOne; torch CUDA full-weight load (no vLLM)."
            if gpu
            else "Text-only SystemOne; official vLLM+shim path not used (FN keeps GPU)."
        ),
    }


@app.get("/v1/version")
def version() -> dict:
    return {
        "model_dir": MODEL_ID,
        "T": TEMP,
        "noul_t": NOUL_T,
        "noul_bias": NOUL_BIAS,
        "flags": {"perms": PERMS, "targeted": True, "instr_style": INSTR_STYLE},
        "shim_file": "serve_openjev_systemone.py",
        "shim_sha256": _shim_sha,
        "cpu_mem_budget": CPU_MEM,
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local", "aliases": ["openjev", "jev-latest"]}],
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
    global _gate
    if _gate is None:
        _gate = asyncio.Lock()
    try:
        async with _gate:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(None, partial(_predict, body))
            ms = (time.perf_counter() - t0) * 1000.0
        result["latency_ms"] = round(ms, 1)
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
    except BadQuestion as e:
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
