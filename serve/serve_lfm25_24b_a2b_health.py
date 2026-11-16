#!/usr/bin/env python3
"""Health-contract wrapper for official Arm C LFM 2.5 24B A2B SGLang GPU serve.

Stock SGLang GET /health returns an empty HTTP 200 and does not report the
compute device. GET /get_server_info (and /server_info) includes ServerArgs.device
after resolution, but the official gold contract is GET /health JSON on the
same port as /v1/chat/completions.

This file is the container entrypoint. It refuses a CPU or offload load,
installs the health JSON, and maps a top-level enable_thinking request field
into chat_template_kwargs (stock ChatCompletionRequest drops that field).
The on-disk LFM chat_template.jinja does not branch on enable_thinking; it
always opens an unclosed <think> tag in the generation prompt. The copy is
still required so a request-level flag is not dropped. It then runs the same
sglang.launch_server path as the Oct 30 Arm C smoke
(--chat-template=/model/chat_template.jinja kept from that command).
"""
from __future__ import annotations

import json
import os
import sys


def _die(msg: str, code: int = 1) -> None:
    print(f"serve_lfm25_24b_a2b_health.py: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _argv_value(argv: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def refuse_cpu_and_offload(argv: list[str]) -> None:
    if os.environ.get("SGLANG_USE_CPU_ENGINE", "0") == "1":
        _die("SGLANG_USE_CPU_ENGINE=1; refusing CPU fallback")
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"].strip() == "":
        _die("CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback")

    device = _argv_value(argv, "--device")
    if device is not None and not device.strip().lower().startswith("cuda"):
        _die(f"--device={device!r} is not cuda; refusing CPU fallback")

    offload = _argv_value(argv, "--cpu-offload-gb")
    if offload is not None:
        try:
            offload_gb = float(offload)
        except ValueError:
            _die(f"--cpu-offload-gb={offload!r} is not a number; refusing start")
        if offload_gb > 0:
            _die(f"--cpu-offload-gb={offload}; refusing CPU offload")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import environment
        _die(f"torch import failed; refusing start: {exc}")
    if not torch.cuda.is_available():
        _die("torch.cuda.is_available() is False; refusing CPU fallback")


def _health_json():
    """Build the official health contract from the loaded runtime, not a constant."""
    import sglang.srt.entrypoints.http_server as http_server
    from sglang.srt.managers.tokenizer_manager import ServerStatus

    port = int(os.environ.get("LFM25_PORT", "8103"))
    model_id = os.environ.get("LFM25_MODEL_ID", "lfm2.5-24b-a2b")
    state = getattr(http_server, "_global_state", None)
    tm = getattr(state, "tokenizer_manager", None) if state is not None else None
    if tm is None:
        return {
            "status": "starting",
            "model": model_id,
            "backend": None,
            "device": None,
            "device_mode": None,
            "port": port,
            "error": "not_loaded",
            "cpu_offload_gb": None,
        }, 503

    args = getattr(tm, "server_args", None)
    if args is not None:
        model_id = getattr(args, "served_model_name", None) or model_id
        port = int(getattr(args, "port", port) or port)

    status_enum = getattr(tm, "server_status", None)
    exiting = bool(getattr(tm, "gracefully_exit", False))
    if exiting or status_enum == ServerStatus.Starting:
        return {
            "status": "starting",
            "model": model_id,
            "backend": None,
            "device": None,
            "device_mode": None,
            "port": port,
            "error": "not_loaded",
            "cpu_offload_gb": None,
        }, 503
    if status_enum == ServerStatus.UnHealthy:
        return {
            "status": "error",
            "model": model_id,
            "backend": None,
            "device": None,
            "device_mode": None,
            "port": port,
            "error": "unhealthy",
            "cpu_offload_gb": None,
        }, 503

    device = str(getattr(args, "device", "") or "")
    cpu_offload_gb = float(getattr(args, "cpu_offload_gb", 0) or 0)
    if device.lower() == "cuda":
        idx = getattr(args, "base_gpu_id", 0)
        try:
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
        except Exception:
            pass
        device = f"cuda:{idx}"

    device_ok = device.lower().startswith("cuda") and cpu_offload_gb <= 0
    if cpu_offload_gb > 0:
        device_mode = f"{device}+cpu_offload" if device else "cpu_offload"
        error = f"cpu_offload_gb={cpu_offload_gb}"
        status = "error"
        backend = "sglang-cuda+cpu_offload"
        http_status = 503
    elif not device_ok:
        device_mode = device or "unknown"
        error = f"device is {device!r}, not cuda"
        status = "error"
        backend = f"sglang-{device or 'unknown'}"
        http_status = 503
    else:
        device_mode = "cuda"
        error = None
        status = "ok"
        backend = "sglang-cuda"
        http_status = 200

    return {
        "status": status,
        "model": model_id,
        "backend": backend,
        "device": device,
        "device_mode": device_mode,
        "port": port,
        "error": error,
        "cpu_offload_gb": cpu_offload_gb,
    }, http_status


def _rewrite_enable_thinking(body: bytes) -> bytes:
    """Copy top-level enable_thinking into chat_template_kwargs if missing there."""
    try:
        data = json.loads(body.decode("utf-8") or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(data, dict) or "enable_thinking" not in data:
        return body
    ctk = data.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    if "enable_thinking" not in ctk:
        ctk["enable_thinking"] = data["enable_thinking"]
        data["chat_template_kwargs"] = ctk
        return json.dumps(data).encode("utf-8")
    return body


def install_middleware(app) -> None:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.middleware("http")
    async def ge2340_health_and_thinking(request: Request, call_next):
        if request.method == "GET" and request.url.path == "/health":
            payload, code = _health_json()
            return JSONResponse(payload, status_code=code)
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            body = await request.body()
            new_body = _rewrite_enable_thinking(body)
            if new_body is not body:
                headers = [
                    (k, v) if k != b"content-length" else (k, str(len(new_body)).encode("ascii"))
                    for k, v in request.scope["headers"]
                ]
                if not any(k == b"content-length" for k, _ in headers):
                    headers.append((b"content-length", str(len(new_body)).encode("ascii")))
                scope = dict(request.scope)
                scope["headers"] = headers

                async def receive():
                    return {"type": "http.request", "body": new_body, "more_body": False}

                request = Request(scope, receive)
        return await call_next(request)


def main() -> None:
    refuse_cpu_and_offload(sys.argv[1:])

    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    from sglang.srt.entrypoints.http_server import app

    install_middleware(app)

    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.launch_server import run_server

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    device = getattr(server_args, "device", None)
    if not device or not str(device).lower().startswith("cuda"):
        _die(f"resolved device={device!r} is not cuda; refusing CPU fallback")
    offload = float(getattr(server_args, "cpu_offload_gb", 0) or 0)
    if offload > 0:
        _die(f"cpu_offload_gb={offload}; refusing CPU offload")

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
