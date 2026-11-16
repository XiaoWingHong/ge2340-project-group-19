#!/usr/bin/env python3
"""Thin SystemOne-compatible HTTP wrapper around SemIf (llamacpp/direct).

Exposes TypeSafe Jev-style POST /v1/systemone (and a native POST /v1/score)
so SemIf can be smoked the same way as Laya. Not affiliated with Jev.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from semif_phase1 import llamacpp_backend
from semif_phase1.core import validate_row

MODEL_ID = os.environ.get("SEMIF_MODEL_ID", "semif-qwen3.5-4b-gguf")
HOST = os.environ.get("SEMIF_HOST", "0.0.0.0")
PORT = int(os.environ.get("SEMIF_PORT", "8011"))
GGUF = Path(os.environ["SEMIF_GGUF"])
TOKENIZER_SRC = os.environ.get("SEMIF_TOKENIZER", "Qwen/Qwen3.5-4B")
REVISION = os.environ.get(
    "SEMIF_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
THREADS = int(os.environ.get("SEMIF_THREADS", "8"))
MAX_TOKENS = int(os.environ.get("SEMIF_MAX_TOKENS", "4096"))
AUTH_ENV = os.environ.get("SEMIF_AUTH_ENV", "")  # if set, require Bearer from that env

_model = None
_tokenizer = None
_metadata: Dict[str, Any] = {}
_gate: Optional[asyncio.Lock] = None
_loaded_at: Optional[float] = None


def _auth_ok(authorization: Optional[str]) -> bool:
    if not AUTH_ENV:
        return True
    expected = os.environ.get(AUTH_ENV, "")
    if not expected:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return authorization[len("Bearer ") :] == expected


def _load() -> None:
    global _model, _tokenizer, _metadata, _loaded_at
    if _model is not None:
        return
    if not GGUF.is_file():
        raise RuntimeError(f"GGUF not found: {GGUF}")
    # Prefer local tokenizer dir when TOKENIZER_SRC is a path
    src = TOKENIZER_SRC
    rev = REVISION
    _model, _tokenizer, _metadata = llamacpp_backend.load_model(
        src, rev, GGUF, threads=THREADS, context_tokens=MAX_TOKENS
    )
    _loaded_at = time.time()


def _state_text(state: Any) -> Any:
    # SemIf accepts str/dict/list; keep structured state for prompt fidelity
    if state is None or state == "":
        raise ValueError("state must be nonempty")
    return state


def _noul_row(qid: str, state: Any, question: Dict[str, Any]) -> dict:
    instructions = question.get("instructions") or question.get("question") or qid
    return {
        "id": qid,
        "state": state,
        "question": str(instructions),
        "options": [
            {"id": "yes", "description": "Yes — the criterion holds for the evidence."},
            {"id": "no", "description": "No — the criterion does not hold for the evidence."},
        ],
    }


def _choice_row(qid: str, state: Any, question: Dict[str, Any]) -> dict:
    instructions = question.get("instructions") or question.get("question") or qid
    criteria = question.get("criteria")
    options: List[dict] = []
    if isinstance(criteria, dict):
        for label, desc in criteria.items():
            options.append(
                {
                    "id": str(label),
                    "description": str(desc) if desc is not None else str(label),
                }
            )
    elif isinstance(criteria, list):
        for item in criteria:
            if isinstance(item, str):
                options.append({"id": item, "description": item})
            elif isinstance(item, dict) and "id" in item:
                options.append(
                    {
                        "id": str(item["id"]),
                        "description": str(item.get("description", item["id"])),
                    }
                )
            else:
                raise ValueError(
                    f"question {qid!r}: criteria list entries must be strings or {{id, description}}"
                )
    else:
        raise ValueError(
            f"question {qid!r}: a choice question takes 'criteria' as a dict of label -> description, or a list of labels"
        )
    if not (2 <= len(options) <= 16):
        raise ValueError(f"question {qid!r}: choice needs 2-16 options (SemIf limit)")
    return {
        "id": qid,
        "state": state,
        "question": str(instructions),
        "options": options,
    }


def _score_row(qid: str, state: Any, question: Dict[str, Any]) -> dict:
    instructions = question.get("instructions") or question.get("question") or qid
    legend = question.get("legend") or {
        "1": "Very low",
        "2": "Low",
        "3": "Medium",
        "4": "High",
        "5": "Very high",
    }
    if not isinstance(legend, dict) or not (2 <= len(legend) <= 16):
        raise ValueError(f"question {qid!r}: score legend must be a dict with 2-16 entries")
    options = [
        {"id": str(k), "description": str(v)} for k, v in legend.items()
    ]
    return {
        "id": qid,
        "state": state,
        "question": str(instructions),
        "options": options,
    }


def _row_for_question(qid: str, state: Any, question: Any) -> dict:
    if not isinstance(question, dict) or "type" not in question:
        raise ValueError(f"question {qid!r}: must be an object with a type field")
    kind = question["type"]
    if kind == "noul":
        return _noul_row(qid, state, question)
    if kind == "choice":
        return _choice_row(qid, state, question)
    if kind == "score":
        return _score_row(qid, state, question)
    raise ValueError(f"question {qid!r}: unsupported type {kind!r} (want noul|choice|score)")


def _answer_from_result(kind: str, result: dict, question: dict) -> dict:
    ids = result["option_ids"]
    probs = result["probabilities"]
    mapping = {oid: float(p) for oid, p in zip(ids, probs)}
    best = max(mapping, key=mapping.get)
    conf = float(mapping[best])
    if kind == "noul":
        # noul = P(yes); fall back if labels differ
        noul = float(mapping.get("yes", mapping.get("true", conf)))
        return {"type": "noul", "noul": round(noul, 4), "confidence": round(noul, 4)}
    if kind == "choice":
        return {
            "type": "choice",
            "choice": best,
            "probabilities": {k: round(v, 4) for k, v in mapping.items()},
            "confidence": round(conf, 4),
        }
    # score
    try:
        score_val = float(best)
    except ValueError:
        score_val = best
    return {
        "type": "score",
        "score": score_val,
        "probabilities": {k: round(v, 4) for k, v in mapping.items()},
        "confidence": round(conf, 4),
        "legend": question.get("legend")
        or {k: v for k, v in zip(ids, [None] * len(ids))},
    }


def _predict(state: Any, questions: Dict[str, Any]) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    state = _state_text(state)
    answers: Dict[str, Any] = {}
    input_tokens = 0
    for qid, question in questions.items():
        row = _row_for_question(str(qid), state, question)
        validate_row(row)
        result = llamacpp_backend.score(
            _model, _tokenizer, row, _metadata, MAX_TOKENS
        )
        answers[str(qid)] = _answer_from_result(question["type"], result, question)
        input_tokens += int(result.get("input_tokens") or 0)
    return {
        "model": MODEL_ID,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "routing": {
            "model": MODEL_ID,
            "backend": "llamacpp",
            "gguf": str(GGUF),
            "tokenizer": TOKENIZER_SRC,
            "revision": REVISION,
        },
    }


def _score_native(row: dict) -> dict:
    validate_row(row)
    return llamacpp_backend.score(_model, _tokenizer, row, _metadata, MAX_TOKENS)


app = FastAPI(title="SemIf SystemOne", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict:
    device = (_metadata or {}).get("device", "cpu")
    n_gpu_layers = (_metadata or {}).get("n_gpu_layers", 0)
    return {
        "status": "ok" if _model is not None else "starting",
        "model": MODEL_ID,
        "backend": "llamacpp",
        "gguf": GGUF.name if GGUF else None,
        "port": PORT,
        "loaded_at": _loaded_at,
        "device": device,
        "n_gpu_layers": n_gpu_layers,
        "semif_device": os.environ.get("SEMIF_DEVICE", ""),
    }


@app.post("/v1/systemone")
async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    body = await request.json()
    if not isinstance(body, dict) or "questions" not in body:
        raise HTTPException(
            status_code=400, detail="request body must be an object with a 'questions' field"
        )
    global _gate
    if _gate is None:
        _gate = asyncio.Lock()
    try:
        async with _gate:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None, partial(_predict, body.get("state"), body["questions"])
            )
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
    except Exception:
        raise HTTPException(status_code=500, detail="inference failed") from None


@app.post("/v1/score")
async def score_native(request: Request, authorization: Optional[str] = Header(default=None)):
    """Native SemIf row: {id, state, question, options:[{id,description},...]}."""
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    if "id" not in body:
        body = {**body, "id": str(uuid.uuid4())}
    global _gate
    if _gate is None:
        _gate = asyncio.Lock()
    try:
        async with _gate:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, partial(_score_native, body))
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        raise HTTPException(status_code=500, detail="inference failed") from None


def main() -> None:
    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
