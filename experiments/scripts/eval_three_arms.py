#!/usr/bin/env python3
"""
Eval multi-model arms: LogReg / SystemOne decision backends / multi-LLM.

HUMAN GOLD (masked) — emails_labeled_gold_masked-2026-11-14-v2.jsonl
and split_gold-2026-11-14.json (task_id, seed 2340).

Never print secret values. Source experiments/.env.eval before running, or
pass --env-file.

Arm A: official native CUDA logistic regression, sequential single-email
  evaluation of the frozen cuML model plus the original sklearn 1.9.1 TF-IDF
  bridge. Historical CPU LogReg is not an official path.
Arm B: SystemOne-style POST {BASE}/v1/systemone — pluggable backends:
  clef | laya | kev | semif | openjev
  Env: DECISION_BACKENDS (comma list); per-backend {B}_BASE_URL / {B}_API_KEY / {B}_MODEL
  Online Jev is NOT in the set (SemIf replaces it).
Arm C+: OpenAI-compatible POST {BASE}/chat/completions — pluggable LLMs:
  qwen38_27b | gemma4_26b_a4b | nemotron35_lightning | lfm25_24b_a2b | muse_glimmer_30b
  Env: LLM_BACKENDS; per-backend LLM_{M}_BASE_URL / LLM_{M}_API_KEY / LLM_{M}_MODEL
  Legacy LLM_* falls back for qwen38_27b only.
  Shared few-shot; enable_thinking=false; JSON {"label","confidence"}.

Metrics:
  Acc_hard = N_correct / N
  Acc_conf = sum_i 1[correct and valid] * confidence_i / N
             (failed/incorrect contribute 0; this is not mean top-choice confidence)
  mean_top_choice_confidence is reported separately.
  Also: mean latency_ms, errors / error_rate as mean counts; six-label Macro-F1.

Caps: DECISION_MAX_N, LLM_MAX_N; CLI overrides.
Dry-run: --dry-run loads data+env, reports present/MISSING keys by name, no API POSTs.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
import socket
import re
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

def _load_runtime_dependencies() -> None:
    """Load the eval stack. Skipped until after --help / --times parsing when this file is the main program."""
    global joblib, np, pd, requests
    global SEMANTICS, explicit_error, score_records
    global accuracy_score, classification_report, confusion_matrix, f1_score
    if globals().get("pd") is not None:
        return
    import joblib as joblib_
    import numpy as np_
    import pandas as pd_
    import requests as requests_
    from eval_scoring import SEMANTICS as SEMANTICS_
    from eval_scoring import explicit_error as explicit_error_
    from eval_scoring import score_records as score_records_
    from sklearn.metrics import accuracy_score as accuracy_score_
    from sklearn.metrics import classification_report as classification_report_
    from sklearn.metrics import confusion_matrix as confusion_matrix_
    from sklearn.metrics import f1_score as f1_score_
    joblib = joblib_
    np = np_
    pd = pd_
    requests = requests_
    SEMANTICS = SEMANTICS_
    explicit_error = explicit_error_
    score_records = score_records_
    accuracy_score = accuracy_score_
    classification_report = classification_report_
    confusion_matrix = confusion_matrix_
    f1_score = f1_score_


joblib = np = pd = requests = None
SEMANTICS = explicit_error = score_records = None
accuracy_score = classification_report = confusion_matrix = f1_score = None
if __name__ != "__main__":
    _load_runtime_dependencies()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
GOLD_JSONL = DATA / "emails_labeled_gold_masked-2026-11-14-v2.jsonl"
GOLD_SPLIT = DATA / "split_gold-2026-11-14.json"

LOCKS = {
    "experiments/data/emails_labeled_gold_masked-2026-11-14-v2.jsonl": "2868b95fa79fe7d3e1f90f0cb8d0b9d3707cdff9542aa7b7801b8a5530258779",
    "experiments/data/split_gold-2026-11-14.json": "b2e067b9995202df22bb30e5255e575ca0191e57b03a1340df858460493ef835",
}
GOLD_NOTE = (
    "HUMAN GOLD (masked) — labels from "
    "emails_labeled_gold_masked-2026-11-14-v2.jsonl "
    "(Label Studio project 1, human gold, name-masked v2); "
    "split_gold-2026-11-14.json, seed 2340, stratified 70/15/15 on label."
)
LABELS = ["promo", "admin", "course", "personal", "news", "other"]
LABEL_RE = re.compile(r"\b(promo|admin|course|personal|news|other)\b", re.I)

# Locked backend ids.
DECISION_BACKEND_IDS = ("clef", "laya", "kev", "semif", "openjev")
LLM_BACKEND_IDS = (
    "qwen38_27b",
    "gemma4_26b_a4b",
    "nemotron35_lightning",
    "lfm25_24b_a2b",
    "muse_glimmer_30b",
)

# Criteria strings must match the gold-train GO verbatim, including punctuation
# and the capital V on Vendor.
EMAIL_TYPE_CRITERIA = {
    "promo": "promotional mail (e.g. scholarships, application calls, optional surveys, events, recruitment).",
    "admin": "official university administration (e.g. announcement digests, IT or security notices, required surveys, programme changes).",
    "course": "course instructional mail (e.g. assignments, class arrangements, programme or project process).",
    "personal": "one-to-one non-official mail (e.g. personal replies, mentor meetings, participant logistics).",
    "news": "achievement or recognition announcements with no call to apply or sign up (e.g. awards, prizes, voting results).",
    "other": "Vendor or system mail that does not fit the other types (e.g. product updates, account notices, form auto-replies, automated results).",
}

SYSTEMONE_QUESTIONS = {
    "email_type": {
        "type": "choice",
        "instructions": "Classify this university email into exactly one type.",
        "criteria": dict(EMAIL_TYPE_CRITERIA),
    }
}

SYSTEM_PROMPT = (
    "You are an email classifier for a university mailbox. "
    "Classify each email into exactly one label: promo, admin, course, personal, news, other. "
    "Return ONLY a JSON object: {\"label\": \"<one of six>\", \"confidence\": <float in [0,1]>}. "
    "confidence is your estimated probability that the chosen label is correct. "
    "Definitions: "
    + "; ".join(f"{k}={v}" for k, v in EMAIL_TYPE_CRITERIA.items())
)


def load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without printing values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def redact(text: str, *secrets: str) -> str:
    out = text or ""
    for s in secrets:
        if s:
            out = out.replace(s, "<REDACTED>")
    return out


def load_data():
    rows = []
    with GOLD_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["id"] = df["task_id"].astype(int)
    df = df.set_index("id", drop=False)
    split = json.loads(GOLD_SPLIT.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        split[key] = [int(x) for x in split[key]]
    return df, split


def build_text_arm_a(df: pd.DataFrame) -> pd.Series:
    from_tok = df["from"].fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True)
    return (
        from_tok.str.strip()
        + " "
        + df["subject"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )


def email_state(row, max_chars: int = 6000) -> str:
    """Subject + body truncated for SystemOne state field."""
    subj = str(row.get("subject") or "")
    body = str(row.get("body") or "")
    text = f"Subject: {subj}\n\n{body}".strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n...[truncated]"
    return text


def email_user_payload(row, body_limit: int = 2500) -> str:
    body = (row.get("body") or "")[:body_limit]
    return f"From: {row.get('from')}\nSubject: {row.get('subject')}\nBody:\n{body}"


def _clamp_conf(raw: Any) -> float | None:
    """Parse confidence/score into [0,1] or None if absent/invalid."""
    if raw is None:
        return None
    try:
        c = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(c):
        return None
    if c < 0.0:
        c = 0.0
    if c > 1.0:
        # allow accidental percentages
        if c <= 100.0:
            c = c / 100.0
        else:
            c = 1.0
    return float(c)


def parse_label_confidence(text: str) -> tuple[str | None, float | None]:
    """Parse LLM JSON {\"label\", \"confidence\"}; confidence may be None if missing.

    Missing confidence is treated as 0.0 when accumulating Acc_conf so the
    denominator stays N (Acc_conf = (1/N) sum c_i).

    After strip, a single leading to=user prefix is ignored (whitespace
    before it is allowed). The remainder must json.loads to a dict with a
    valid label; confidence is taken from that same object. Regex / LABEL_RE
    fallbacks are not used; longer dumps return (None, None).
    """
    if not text:
        return None, None
    text = text.strip()
    if text.startswith("to=user"):
        text = text[len("to=user"):].strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None, None
    if not isinstance(obj, dict) or "label" not in obj or explicit_error(obj):
        return None, None
    lab = str(obj["label"]).strip().lower()
    if lab not in LABELS:
        return None, None
    conf = _clamp_conf(obj.get("confidence", obj.get("score")))
    return lab, conf

def parse_label(text: str) -> str | None:
    lab, _ = parse_label_confidence(text)
    return lab


def parse_systemone_choice_score(data: dict) -> tuple[str | None, float | None]:
    """Extract (choice, score) from SystemOne response (answers.email_type)."""
    answers = data.get("answers") or {}
    eth = answers.get("email_type")
    label: str | None = None
    score: float | None = None
    if isinstance(eth, dict):
        for key in ("choice", "value", "answer", "label"):
            if key in eth:
                lab = str(eth[key]).strip().lower()
                if lab in LABELS:
                    label = lab
                    break
        if label is None:
            for nest in ("result", "selected"):
                if nest in eth and isinstance(eth[nest], (str, dict)):
                    cand = (
                        eth[nest]
                        if isinstance(eth[nest], str)
                        else eth[nest].get("choice") or eth[nest].get("value")
                    )
                    if cand:
                        lab = str(cand).strip().lower()
                        if lab in LABELS:
                            label = lab
                            break
        for sk in ("score", "confidence", "prob", "probability"):
            if sk in eth:
                score = _clamp_conf(eth[sk])
                if score is not None:
                    break
        if score is None and isinstance(eth.get("result"), dict):
            for sk in ("score", "confidence"):
                if sk in eth["result"]:
                    score = _clamp_conf(eth["result"][sk])
                    if score is not None:
                        break
    if label is None:
        lab2, conf2 = parse_label_confidence(json.dumps(data))
        return lab2, conf2
    return label, score


class EvaluationReplyError(RuntimeError):
    def __init__(self, message: str, raw_reply: str | None):
        super().__init__(message)
        self.raw_reply = raw_reply


class ChatReply(str):
    """Unchanged parser input plus full redacted HTTP response for raw provenance."""
    def __new__(cls, content, raw_http_body):
        obj=super().__new__(cls,content);obj.raw_http_body=raw_http_body;return obj


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    temperature: float = 0.0,
    timeout: float = 300.0,
    stop: list[str] | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    # Auth optional: empty / none / - / noauth → no Authorization header (e.g. local serve)
    if api_key and api_key.lower() not in {"none", "-", "noauth", "n/a"}:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "enable_thinking": False,
        "max_tokens": 1024,
    }
    # Gemma-only request-level stop is passed in by run_llm_backend; other backends leave stop=None.
    if stop:
        payload["stop"] = stop
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        snippet = redact(r.text[:500], api_key)
        raise EvaluationReplyError(f"HTTP {r.status_code}: {snippet}", redact(r.text or "", api_key))
    try:
        data = r.json()
        if explicit_error(data):
            raise ValueError(explicit_error(data))
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        return ChatReply((content or "").strip(), redact(r.text or "", api_key if api_key.lower() not in {"none", "-", "noauth", "n/a"} else ""))
    except Exception as e:
        raise EvaluationReplyError(redact(str(e), api_key), redact(r.text or "", api_key)) from e


def systemone_classify(
    base_url: str,
    api_key: str,
    model: str,
    state: str,
    timeout: float = 300.0,
) -> tuple[str, float | None, str]:
    """Shared SystemOne POST /v1/systemone → (choice, score, redacted raw body)."""
    url = base_url.rstrip("/") + "/v1/systemone"
    headers = {"Content-Type": "application/json"}
    # Auth optional: empty / none / - / noauth → no Authorization header (e.g. local Laya)
    if api_key and api_key.lower() not in {"none", "-", "noauth", "n/a"}:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "state": state, "questions": SYSTEMONE_QUESTIONS}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    raw_body = redact(r.text or "", api_key)
    if not r.ok:
        snippet = raw_body[:500]
        raise EvaluationReplyError(f"HTTP {r.status_code}: {snippet}", redact(r.text or "", api_key))
    try:
        data = r.json()
        if explicit_error(data):
            raise ValueError(explicit_error(data))
        lab, score = parse_systemone_choice_score(data)
        if not lab:
            raise ValueError("Could not parse SystemOne choice from response keys=" + str(list(data.keys()) if isinstance(data, dict) else type(data)))
        return lab, score, raw_body
    except Exception as e:
        raise EvaluationReplyError(redact(str(e), api_key), raw_body) from e


# --- credential resolution (names only; never echo values) ---


def _env_present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def resolve_decision_creds(backend_id: str) -> dict[str, Any]:
    """Resolve URL/KEY/MODEL for a decision backend. Values kept internal."""
    bid = backend_id.strip().lower()
    prefix = bid.upper()  # CLEF, LAYA, KEV, SEMIF, OPENJEV
    base = os.environ.get(f"{prefix}_BASE_URL", "").strip()
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    # Empty API key allowed when explicitly set to none/- /noauth (auth-less local servers)
    no_auth_tokens = {"none", "-", "noauth", "n/a"}
    missing: list[str] = []
    if not base:
        missing.append(f"{prefix}_BASE_URL")
    if not key:
        # treat unset as missing; set LAYA_API_KEY=none for no-auth backends
        missing.append(f"{prefix}_API_KEY")
    elif key.lower() in no_auth_tokens:
        key = ""  # strip sentinel before requests
    if not model:
        missing.append(f"{prefix}_MODEL")
    return {
        "backend_id": bid,
        "base_url": base,
        "api_key": key,
        "model": model,
        "ready": len(missing) == 0,
        "missing_keys": missing,
        "key_names": {
            "base_url": f"{prefix}_BASE_URL",
            "api_key": f"{prefix}_API_KEY",
            "model": f"{prefix}_MODEL",
        },
    }


def resolve_llm_creds(backend_id: str) -> dict[str, Any]:
    """Resolve URL/KEY/MODEL for an LLM backend. qwen38_27b falls back to LLM_*."""
    bid = backend_id.strip().lower()
    m = bid.upper()  # QWEN38_27B, GEMMA4_26B_A4B, ...
    base = os.environ.get(f"LLM_{m}_BASE_URL", "").strip()
    key = os.environ.get(f"LLM_{m}_API_KEY", "").strip()
    model = os.environ.get(f"LLM_{m}_MODEL", "").strip()
    used_legacy = False
    if bid == "qwen38_27b":
        if not base:
            base = os.environ.get("LLM_BASE_URL", "").strip()
            if base:
                used_legacy = True
        if not key:
            key = os.environ.get("LLM_API_KEY", "").strip()
            if key:
                used_legacy = True
        if not model:
            model = os.environ.get("LLM_MODEL", "").strip()
            if model:
                used_legacy = True
    missing: list[str] = []
    if not base:
        missing.append(
            f"LLM_{m}_BASE_URL" + (" (or LLM_BASE_URL)" if bid == "qwen38_27b" else "")
        )
    if not key:
        missing.append(
            f"LLM_{m}_API_KEY" + (" (or LLM_API_KEY)" if bid == "qwen38_27b" else "")
        )
    if not model:
        missing.append(
            f"LLM_{m}_MODEL" + (" (or LLM_MODEL)" if bid == "qwen38_27b" else "")
        )
    return {
        "backend_id": bid,
        "base_url": base,
        "api_key": key,
        "model": model,
        "ready": len(missing) == 0,
        "missing_keys": missing,
        "used_legacy_llm": used_legacy,
        "key_names": {
            "base_url": f"LLM_{m}_BASE_URL",
            "api_key": f"LLM_{m}_API_KEY",
            "model": f"LLM_{m}_MODEL",
        },
    }


def parse_backend_list(
    raw: str | None, allowed: tuple[str, ...], default: tuple[str, ...]
) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    out: list[str] = []
    for part in str(raw).split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise SystemExit(
                f"Unknown backend id {p!r}; allowed: {', '.join(allowed)}"
            )
        if p not in out:
            out.append(p)
    return out if out else list(default)


def resolve_max_n(cli_val, env_keys: list[str], full_n: int) -> int:
    """CLI overrides env; first non-empty env key wins; <=0 or unset → full test."""
    if cli_val is not None:
        n = int(cli_val)
    else:
        n = full_n
        for ek in env_keys:
            if os.environ.get(ek, "").strip() != "":
                n = int(os.environ[ek])
                break
    return full_n if n <= 0 else n


def _endpoint_reachable(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Cheap TCP check so a down backend is not called 300 times at 300s timeout."""
    from urllib.parse import urlparse
    u = urlparse(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if (u.scheme or "").lower() == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"tcp {host}:{port} open"
    except Exception as e:
        return False, f"tcp {host}:{port} unreachable ({type(e).__name__})"


def _acc_hard(y_true: list, preds: list) -> float:
    """Acc_hard = N_correct / N."""
    assert len(y_true) == len(preds)
    return sum(p in LABELS and p == y for y, p in zip(y_true, preds)) / len(y_true)


def _acc_conf(confidences: list[float | None], n: int) -> float | None:
    """Deprecated helper: mean top-choice confidence, not official Acc_conf.

    Official Acc_conf is computed by score_records after attach_scored_records.
    """
    if n <= 0:
        return None
    # Acc_conf = (1/N) Σ_{i=1}^{N} c_i   where c_i is confidence of selected label
    total = 0.0
    for c in confidences:
        total += float(c) if c is not None else 0.0
    # pad if caller passed fewer scores than n (should not happen)
    if len(confidences) < n:
        # remaining treated as 0
        pass
    return float(total / n)


# --- arms ---


def _metric_predictions(preds: list) -> list:
    return [p if p in LABELS else "__abstention__" for p in preds]


def _confusion(y_true: list, preds: list) -> dict[str, Any]:
    cm = confusion_matrix(y_true, _metric_predictions(preds), labels=LABELS)
    return {"labels": list(LABELS), "matrix": cm.tolist()}


def attach_scored_records(
    arm: dict[str, Any],
    subset: pd.DataFrame,
    preds: list,
    confs: list,
    latencies: list,
    raw_replies: list,
    row_errors: list | None = None,
) -> dict[str, Any]:
    """Attach ids, gold, per-email latency, redacted raw replies, and items[]."""
    if "task_id" in subset.columns:
        task_ids = [int(x) for x in subset["task_id"].tolist()]
    else:
        task_ids = [int(i) for i in subset.index.tolist()]
    gold = subset["label"].tolist()
    n = len(task_ids)
    items = []
    for i in range(n):
        items.append(
            {
                "task_id": task_ids[i],
                "gold": gold[i] if i < len(gold) else None,
                "prediction": preds[i] if i < len(preds) else None,
                "confidence": confs[i] if i < len(confs) else None,
                "latency_ms": latencies[i] if i < len(latencies) else None,
                "raw_reply": raw_replies[i] if i < len(raw_replies) else None,
                "error": row_errors[i] if row_errors else None,
            }
        )
    arm["task_ids"] = task_ids
    arm["gold_labels"] = gold
    arm["predictions"] = list(preds)
    arm["confidences"] = list(confs)
    arm["latency_ms_per_email"] = list(latencies)
    arm["raw_replies"] = list(raw_replies)
    arm["items"] = items
    if (
        arm.get("status") in {"ok", "partial"}
        and preds
        and gold
        and len(preds) == len(gold)
    ):
        if "report" in arm and arm["report"] is not None:
            arm["classification_report"] = arm["report"]
        else:
            arm["classification_report"] = classification_report(
                gold, _metric_predictions(preds), labels=LABELS, zero_division=0, output_dict=True
            )
            arm["report"] = arm["classification_report"]
        arm["confusion_matrix"] = _confusion(gold, preds)
    arm["row_errors"] = [r["error"] for r in items]
    arm.update(score_records(items))
    return arm


def run_arm_a(df_test: pd.DataFrame, y_true: list) -> dict[str, Any]:
    raise SystemExit("Historical CPU LogReg is not an official evaluation path; official Arm A is sequential single-email native CUDA")
    arm: dict[str, Any] = {
        "arm": "A_logreg",
        "status": "ok",
        "notes": "local sklearn; no API cost; private; conf=max predict_proba",
        "n_scored": len(df_test),
        "errors": 0,
        "error_rate": 0.0,
    }
    t0 = time.perf_counter()
    try:
        vectorizer = joblib.load(MODELS / "tfidf.joblib")
        clf = joblib.load(MODELS / "logreg.joblib")
        text = build_text_arm_a(df_test)
        X = vectorizer.transform(text)
        pred = clf.predict(X).tolist()
        # LogReg confidence = max class probability (top-choice)
        proba = clf.predict_proba(X)
        confs = [float(row.max()) for row in proba]
        total_ms = (time.perf_counter() - t0) * 1000.0
        n = len(pred)
        mean_ms = float(total_ms / n) if n else None
        # Batch inference: record the shared per-email share and the batch total.
        latencies = [mean_ms] * n
        raw_replies = [None] * n
        arm.update(
            {
                "backend_id": "logreg",
                "acc_hard": _acc_hard(y_true, pred),
                "acc_conf": _acc_conf(confs, n),
                # legacy aliases kept for older readers
                "accuracy": _acc_hard(y_true, pred),
                "macro_f1": float(
                    f1_score(y_true, pred, average="macro", labels=LABELS, zero_division=0)
                ),
                "latency_ms": mean_ms,
                "latency_ms_batch_total": float(total_ms),
                "predictions": pred,
                "confidences": confs,
                "report": classification_report(
                    y_true, pred, labels=LABELS, zero_division=0, output_dict=True
                ),
            }
        )
        attach_scored_records(arm, df_test, pred, confs, latencies, raw_replies)
        print(classification_report(y_true, pred, labels=LABELS, zero_division=0))
        print(
            f"Arm A n={n} acc_hard={arm['acc_hard']:.4f} "
            f"acc_conf={arm['acc_conf']:.4f} macro_f1={arm['macro_f1']:.4f} "
            f"mean_lat_ms={mean_ms:.1f}"
        )
    except Exception as e:
        arm.update(
            {
                "status": "error",
                "notes": str(e),
                "backend_id": "logreg",
                "acc_hard": None,
                "acc_conf": None,
                "accuracy": None,
                "macro_f1": None,
                "latency_ms": None,
                "error_rate": None,
            }
        )
        print("Arm A failed:", e)
    return arm


def run_decision_backend(
    backend_id: str, df_test: pd.DataFrame, max_n: int
) -> dict[str, Any]:
    arm_name = f"B_{backend_id}"
    creds = resolve_decision_creds(backend_id)
    arm: dict[str, Any] = {
        "arm": arm_name,
        "status": "skipped",
        "notes": (
            f"needs {creds['key_names']['base_url']} + "
            f"{creds['key_names']['api_key']} + {creds['key_names']['model']}"
        ),
        "acc_hard": None,
        "acc_conf": None,
        "accuracy": None,
        "macro_f1": None,
        "latency_ms": None,
        "n_scored": 0,
        "errors": 0,
        "error_rate": None,
        "backend_id": backend_id,
    }
    if not creds["ready"]:
        arm["notes"] = f"skipped; missing: {', '.join(creds['missing_keys'])}"
        print(f"Arm {arm_name} skipped: missing {', '.join(creds['missing_keys'])}")
        return arm

    base, key, model = creds["base_url"], creds["api_key"], creds["model"]
    ok, reason = _endpoint_reachable(base)
    if not ok:
        arm["status"] = "skipped"
        arm["notes"] = f"skipped; endpoint not responding: {base} ({reason})"
        print(f"Arm {arm_name} skipped: {arm['notes']}")
        return arm
    subset = df_test.head(max_n)
    print(f"Arm {arm_name} scoring n={len(subset)} via {base}/v1/systemone model={model}")
    preds: list[str | None] = []
    confs: list[float | None] = []
    latencies: list[float] = []
    raw_replies: list[str | None] = []
    row_errors: list[str | None] = []
    errors = 0
    last_err = ""
    for _, row in subset.iterrows():
        t0 = time.perf_counter()
        raw_reply: str | None = None
        row_error: str | None = None
        try:
            lab, score, raw_reply = systemone_classify(base, key, model, email_state(row))
            preds.append(lab)
            confs.append(score)
        except Exception as e:
            errors += 1
            preds.append(None)
            confs.append(None)  # → 0 in Acc_conf
            last_err = redact(str(e), key)
            row_error = last_err
            raw_reply = getattr(e, "raw_reply", raw_reply)
            if errors <= 3:
                print(f"Arm {arm_name} call error:", last_err)
        row_errors.append(row_error)
        raw_replies.append(raw_reply)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    y_sub = subset["label"].tolist()
    n = len(preds)
    err_rate = float(errors / n) if n else None
    if errors == n and n > 0:
        arm.update(
            {
                "status": "error",
                "notes": f"all calls failed; last_error={last_err}",
                "n_scored": n,
                "latency_ms": float(np.mean(latencies)) if latencies else None,
                "predictions": preds,
                "confidences": confs,
                "errors": errors,
                "error_rate": err_rate,
            }
        )
        attach_scored_records(arm, subset, preds, confs, latencies, raw_replies, row_errors)
        print(f"Arm {arm_name} failed: all calls errored")
        return arm

    acc_h = _acc_hard(y_sub, preds)
    acc_c = _acc_conf(confs, n)
    arm.update(
        {
            "status": "ok" if errors == 0 else "partial",
            "acc_hard": acc_h,
            "acc_conf": acc_c,
            "accuracy": acc_h,
            "macro_f1": float(
                f1_score(y_sub, _metric_predictions(preds), average="macro", labels=LABELS, zero_division=0)
            ),
            "latency_ms": float(np.mean(latencies)) if latencies else None,
            "notes": (
                f"SystemOne POST /v1/systemone backend={backend_id} model={model}; "
                f"parse answers.email_type.choice+score; errors={errors}/{n}; n={n}"
            ),
            "predictions": preds,
            "confidences": confs,
            "n_scored": n,
            "errors": errors,
            "error_rate": err_rate,
            "report": classification_report(
                y_sub, _metric_predictions(preds), labels=LABELS, zero_division=0, output_dict=True
            ),
        }
    )
    attach_scored_records(arm, subset, preds, confs, latencies, raw_replies, row_errors)
    print(classification_report(y_sub, _metric_predictions(preds), labels=LABELS, zero_division=0))
    print(
        f"Arm {arm_name} n={n} acc_hard={acc_h:.4f} "
        f"acc_conf={acc_c:.4f} macro_f1={arm['macro_f1']:.4f} "
        f"mean_lat_ms={arm['latency_ms']:.1f} error_rate={err_rate:.4f}"
    )
    return arm


def build_fewshot_msgs(df: pd.DataFrame, train_ids: list) -> list[dict]:
    fewshot_msgs: list[dict] = []
    train_df = df.loc[train_ids]
    for lab in LABELS:
        cand = train_df[train_df["label"] == lab]
        if len(cand) == 0:
            continue
        ex = cand.iloc[0]
        fewshot_msgs.append({"role": "user", "content": email_user_payload(ex)})
        # demo confidence 1.0 (gold demo); teaches schema with confidence field
        fewshot_msgs.append(
            {
                "role": "assistant",
                "content": json.dumps({"label": lab, "confidence": 1.0}),
            }
        )
    return fewshot_msgs


def run_llm_backend(
    backend_id: str,
    df: pd.DataFrame,
    train_ids: list,
    df_test: pd.DataFrame,
    max_n: int,
    fewshot_msgs: list[dict] | None = None,
) -> dict[str, Any]:
    arm_name = f"C_{backend_id}"
    creds = resolve_llm_creds(backend_id)
    arm: dict[str, Any] = {
        "arm": arm_name,
        "status": "skipped",
        "notes": (
            f"needs {creds['key_names']['base_url']} + "
            f"{creds['key_names']['api_key']} + {creds['key_names']['model']}"
        ),
        "acc_hard": None,
        "acc_conf": None,
        "accuracy": None,
        "macro_f1": None,
        "latency_ms": None,
        "n_scored": 0,
        "errors": 0,
        "error_rate": None,
        "backend_id": backend_id,
    }
    if not creds["ready"]:
        arm["notes"] = f"skipped; missing: {', '.join(creds['missing_keys'])}"
        print(f"Arm {arm_name} skipped: missing {', '.join(creds['missing_keys'])}")
        return arm

    base, key, model = creds["base_url"], creds["api_key"], creds["model"]
    ok, reason = _endpoint_reachable(base)
    if not ok:
        arm["status"] = "skipped"
        arm["notes"] = f"skipped; endpoint not responding: {base} ({reason})"
        print(f"Arm {arm_name} skipped: {arm['notes']}")
        return arm
    if fewshot_msgs is None:
        fewshot_msgs = build_fewshot_msgs(df, train_ids)

    subset = df_test.head(max_n)
    legacy_note = " (legacy LLM_*)" if creds.get("used_legacy_llm") else ""
    print(
        f"Arm {arm_name} scoring n={len(subset)} via {base} model={model}{legacy_note}"
    )
    preds: list[str | None] = []
    confs: list[float | None] = []
    latencies: list[float] = []
    raw_replies: list[str | None] = []
    row_errors: list[str | None] = []
    errors = 0
    last_err = ""
    # Official gold: Gemma 4 26B A4B only. Request-level stop=["<end_of_turn>"] to
    # halt post-JSON <end_of_turn> <start_of_turn>user continuation dumps. Do not
    # also send <start_of_turn>. Do not apply this stop to any other backend.
    gemma_stop = (
        ["<end_of_turn>"]
        if backend_id.strip().lower() == "gemma4_26b_a4b"
        else None
    )
    for _, row in subset.iterrows():
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + fewshot_msgs
            + [{"role": "user", "content": email_user_payload(row)}]
        )
        t0 = time.perf_counter()
        raw_reply: str | None = None
        row_error: str | None = None
        try:
            content = chat_completion(base, key, model, messages, stop=gemma_stop)
            raw_reply = content.raw_http_body if isinstance(content, ChatReply) else redact(content or "", key)
            lab, conf = parse_label_confidence(content)
            if lab is None:
                errors += 1
                preds.append(None)
                confs.append(None)
                last_err = "JSON parse error"
                row_error = last_err
                if errors <= 3:
                    print(f"Arm {arm_name} JSON parse error")
            else:
                preds.append(lab)
                confs.append(conf)
        except Exception as e:
            errors += 1
            preds.append(None)
            confs.append(None)
            last_err = redact(str(e), key)
            row_error = last_err
            raw_reply = getattr(e, "raw_reply", raw_reply)
            if errors <= 3:
                print(f"Arm {arm_name} call error:", last_err)
        row_errors.append(row_error)
        raw_replies.append(raw_reply)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    y_sub = subset["label"].tolist()
    n = len(preds)
    err_rate = float(errors / n) if n else None
    if errors == n and n > 0:
        arm.update(
            {
                "status": "error",
                "notes": f"all calls failed; last_error={last_err}",
                "n_scored": n,
                "latency_ms": float(np.mean(latencies)) if latencies else None,
                "predictions": preds,
                "confidences": confs,
                "errors": errors,
                "error_rate": err_rate,
            }
        )
        attach_scored_records(arm, subset, preds, confs, latencies, raw_replies, row_errors)
        print(f"Arm {arm_name} failed: all calls errored")
        return arm

    acc_h = _acc_hard(y_sub, preds)
    acc_c = _acc_conf(confs, n)
    arm.update(
        {
            "status": "ok" if errors == 0 else "partial",
            "acc_hard": acc_h,
            "acc_conf": acc_c,
            "accuracy": acc_h,
            "macro_f1": float(
                f1_score(y_sub, _metric_predictions(preds), average="macro", labels=LABELS, zero_division=0)
            ),
            "latency_ms": float(np.mean(latencies)) if latencies else None,
            "notes": (
                f"few-shot ({len(fewshot_msgs)//2} demos); OpenAI-compatible chat; "
                f"enable_thinking=false; model={model}; "
                f"errors={errors}/{n}; n={n}"
                + (
                    '; request-level stop=["<end_of_turn>"] only (Gemma-only; no <start_of_turn>)'
                    if gemma_stop
                    else ""
                )
            ),
            "predictions": preds,
            "confidences": confs,
            "n_scored": n,
            "errors": errors,
            "error_rate": err_rate,
            "report": classification_report(
                y_sub, _metric_predictions(preds), labels=LABELS, zero_division=0, output_dict=True
            ),
        }
    )
    attach_scored_records(arm, subset, preds, confs, latencies, raw_replies, row_errors)
    print(classification_report(y_sub, _metric_predictions(preds), labels=LABELS, zero_division=0))
    print(
        f"Arm {arm_name} n={n} acc_hard={acc_h:.4f} "
        f"acc_conf={acc_c:.4f} macro_f1={arm['macro_f1']:.4f} "
        f"mean_lat_ms={arm['latency_ms']:.1f} error_rate={err_rate:.4f}"
    )
    return arm


def summarize_arm(arm: dict) -> dict:
    return {
        "arm": arm["arm"],
        "status": arm.get("status"),
        "n_scored": arm.get("n_scored"),
        "acc_hard": arm.get("acc_hard", arm.get("accuracy")),
        "acc_conf": arm.get("acc_conf"),
        "mean_top_choice_confidence": arm.get("mean_top_choice_confidence"),
        "accuracy": arm.get("accuracy"),  # alias of acc_hard
        "macro_f1": arm.get("macro_f1"),
        "latency_ms": arm.get("latency_ms"),
        "errors": arm.get("errors"),
        "error_rate": arm.get("error_rate"),
        "notes": arm.get("notes"),
        "backend_id": arm.get("backend_id"),
    }


def _arm_record(a: dict) -> dict[str, Any]:
    """Full per-arm record for script-generated JSON (no hand assembly)."""
    keys = [
        "arm",
        "backend_id",
        "status",
        "notes",
        "n_scored",
        "errors",
        "error_rate",
        "acc_hard",
        "acc_conf",
        "mean_top_choice_confidence",
        "acc_conf_definition",
        "accuracy",
        "macro_f1",
        "latency_ms",
        "latency_ms_batch_total",
        "task_ids",
        "gold_labels",
        "predictions",
        "confidences",
        "latency_ms_per_email",
        "raw_replies",
        "row_errors",
        "n_correct",
        "scoring_semantics",
        "items",
        "classification_report",
        "confusion_matrix",
        "report",
    ]
    rec = {k: a[k] for k in keys if k in a}
    if "classification_report" not in rec and rec.get("report") is not None:
        rec["classification_report"] = rec["report"]
    return rec



def check_requested_configuration(arms_filter, decision_backends, llm_backends):
    """Fail before any inference if a requested official backend is unconfigured."""
    missing=[]
    if "A" in arms_filter:
        if not (MODELS / "tfidf.joblib").is_file():
            missing.append("experiments/models/tfidf.joblib")
        if not (MODELS / "cuda_A_20261115" / "cuml_logreg.joblib").is_file():
            missing.append("experiments/models/cuda_A_20261115/cuml_logreg.joblib")
    if "B" in arms_filter:
        for backend in decision_backends:missing.extend(resolve_decision_creds(backend)["missing_keys"])
    if "C" in arms_filter:
        for backend in llm_backends:missing.extend(resolve_llm_creds(backend)["missing_keys"])
    if missing:raise SystemExit("Missing requested backend configuration: "+", ".join(missing))


def completed_inference(arms, required_n=300):
    """Actual scored item rows, not merely invoking a script or probing health."""
    return bool(arms) and all(a.get("status")!="skipped" and a.get("n_scored")==required_n and len(a.get("items",[]))==required_n and len({r["task_id"] for r in a["items"]})==required_n for a in arms)


def write_outputs(
    arms: list[dict],
    out_json: Path,
    out_md: Path | None = None,
    meta: dict | None = None,
) -> None:
    rows = [summarize_arm(a) for a in arms]
    provenance = dict(meta or {})
    provenance.update({
        "completed_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "inference_performed": completed_inference(arms, int(provenance.get("required_n", 300))),
        "actual_inference_rows": sum(len(a.get("items", [])) for a in arms),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_sha256": hashlib.sha256((Path(__file__).parent / "eval_scoring.py").read_bytes()).hexdigest(),
        "gold_hashes": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (GOLD_JSONL, GOLD_SPLIT)},
        "baseline_artifact_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (MODELS / "logreg.joblib", MODELS / "tfidf.joblib")},
    })
    payload = {
        "scoring_semantics": SEMANTICS,
        "note": GOLD_NOTE,
        "source_jsonl": GOLD_JSONL.name,
        "split": GOLD_SPLIT.name,
        "taxonomy": LABELS,
        "metrics_note": (
            "Acc_hard=N_correct/N; "
            "Acc_conf=sum(correct valid confidence)/N (failed and incorrect contribute 0); "
            "mean_top_choice_confidence is a distinct all-row mean with missing/failed as 0; "
            "failed replies have null prediction, are always incorrect, and contribute false negatives to six-class Macro-F1."
        ),
        "caps": {
            "DECISION_MAX_N": os.environ.get("DECISION_MAX_N"),
            "LLM_MAX_N": os.environ.get("LLM_MAX_N"),
        },
        "selection": meta or {},
        "provenance": provenance,
        "summary": rows,
        "arms": [_arm_record(a) for a in arms],
        "api": {
            "arm_a": "offline TF-IDF + LogReg from models/; conf=max predict_proba",
            "arm_b": (
                "POST {BASE}/v1/systemone for backends clef|laya|kev|semif|openjev; "
                "parse answers.email_type.choice+score; env {B}_BASE_URL/{B}_API_KEY/{B}_MODEL"
            ),
            "arm_c": (
                "POST {BASE}/chat/completions; few-shot JSON {label,confidence}; "
                "enable_thinking=false; backends "
                "qwen38_27b|gemma4_26b_a4b|nemotron35_lightning|lfm25_24b_a2b|muse_glimmer_30b; "
                "LLM_{M}_* (legacy LLM_* → qwen38_27b)"
            ),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if out_md is None:
        print("Wrote", out_json)
        return

    lines = [
        "# Eval — multi-model arms (human gold, masked)",
        "",
        f"> {GOLD_NOTE}",
        "",
        "Metrics: Acc_hard = N_correct/N; Acc_conf = (1/N) Σ c_i (mean top-choice confidence).",
        "",
        "| Arm | Status | n | Acc_hard | Acc_conf | Macro-F1 | Mean latency (ms) | Errors | Error rate | Notes |",
        "|-----|--------|---|----------|----------|----------|-------------------|--------|------------|-------|",
    ]
    for r in rows:
        ah = r.get("acc_hard", r.get("accuracy"))
        ac = r.get("acc_conf")
        acc = f"{ah:.4f}" if isinstance(ah, float) else "—"
        conf = f"{ac:.4f}" if isinstance(ac, float) else "—"
        f1 = f"{r['macro_f1']:.4f}" if isinstance(r.get("macro_f1"), float) else "—"
        lat = f"{r['latency_ms']:.1f}" if isinstance(r.get("latency_ms"), float) else "—"
        er = f"{r['error_rate']:.4f}" if isinstance(r.get("error_rate"), float) else "—"
        errs = r.get("errors") if r.get("errors") is not None else "—"
        notes = (r.get("notes") or "").replace("|", "/")
        lines.append(
            f"| {r['arm']} | {r['status']} | {r.get('n_scored')} | {acc} | {conf} | "
            f"{f1} | {lat} | {errs} | {er} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## API notes",
            "- Arm A: offline LogReg + TF-IDF (conf = max predict_proba)",
            "- Arm B: SystemOne `POST /v1/systemone` — clef, laya, kev, semif, openjev "
            "(no online Jev; SemIf replaces it)",
            "- Arm C+: OpenAI-compatible chat (`enable_thinking: false`) — "
            "qwen38_27b, gemma4_26b_a4b, nemotron35_lightning, lfm25_24b_a2b, muse_glimmer_30b",
            "",
            f"JSON: `{out_json}`",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_json)
    print("Wrote", out_md)


def dry_run_report(
    decision_backends: list[str],
    llm_backends: list[str],
    test_n: int,
    decision_max: int,
    llm_max: int,
    arms_filter: set[str],
    out_json: Path | None,
) -> None:
    print("=== DRY-RUN (no API POSTs; no model inference) ===")
    print(f"test_n={test_n} decision_max_n={decision_max} llm_max_n={llm_max}")
    print(f"arms_filter={sorted(arms_filter)}")

    plan: list[dict] = []

    if "A" in arms_filter:
        a_ready = (MODELS / "tfidf.joblib").is_file() and (MODELS / "cuda_A_20261115" / "cuml_logreg.joblib").is_file()
        print(f"Arm A_cuda_logreg: {'would_run' if a_ready else 'MISSING official CUDA model or tfidf.joblib'}")
        plan.append({"arm": "A_cuda_logreg", "would_run": a_ready, "status": "dry_run"})

    if "B" in arms_filter:
        print(f"DECISION_BACKENDS={','.join(decision_backends)}")
        for bid in decision_backends:
            creds = resolve_decision_creds(bid)
            status = (
                "would_run" if creds["ready"] else f"MISSING:{','.join(creds['missing_keys'])}"
            )
            kn = creds["key_names"]
            print(
                f"  B_{bid}: {status} | "
                f"{kn['base_url']}={'present' if _env_present(kn['base_url']) else 'MISSING'} "
                f"{kn['api_key']}={'present' if _env_present(kn['api_key']) else 'MISSING'} "
                f"{kn['model']}={'present' if _env_present(kn['model']) else 'MISSING'}"
            )
            plan.append(
                {
                    "arm": f"B_{bid}",
                    "would_run": creds["ready"],
                    "missing_keys": creds["missing_keys"],
                    "status": "dry_run",
                }
            )

    if "C" in arms_filter:
        print(f"LLM_BACKENDS={','.join(llm_backends)}")
        for bid in llm_backends:
            creds = resolve_llm_creds(bid)
            status = (
                "would_run" if creds["ready"] else f"MISSING:{','.join(creds['missing_keys'])}"
            )
            kn = creds["key_names"]
            legacy = ""
            if bid == "qwen38_27b":
                legacy = (
                    f" | legacy LLM_BASE_URL={'present' if _env_present('LLM_BASE_URL') else 'MISSING'} "
                    f"LLM_API_KEY={'present' if _env_present('LLM_API_KEY') else 'MISSING'} "
                    f"LLM_MODEL={'present' if _env_present('LLM_MODEL') else 'MISSING'}"
                )
            print(
                f"  C_{bid}: {status} | "
                f"{kn['base_url']}={'present' if _env_present(kn['base_url']) else 'MISSING'} "
                f"{kn['api_key']}={'present' if _env_present(kn['api_key']) else 'MISSING'} "
                f"{kn['model']}={'present' if _env_present(kn['model']) else 'MISSING'}"
                f"{legacy}"
            )
            plan.append(
                {
                    "arm": f"C_{bid}",
                    "would_run": creds["ready"],
                    "missing_keys": creds["missing_keys"],
                    "status": "dry_run",
                }
            )

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "DRY-RUN — no API calls; " + GOLD_NOTE,
            "source_jsonl": GOLD_JSONL.name,
            "split": GOLD_SPLIT.name,
            "taxonomy": LABELS,
            "test_n": test_n,
            "decision_max_n": decision_max,
            "llm_max_n": llm_max,
            "decision_backends": decision_backends,
            "llm_backends": llm_backends,
            "plan": plan,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Wrote dry-run summary", out_json)


def positive_times(value: str) -> int:
    if not isinstance(value, str) or value.strip() != value or not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError("--times must be a positive integer")
    return int(value)


def selected_gpu_targets(arms_filter, decision_backends, llm_backends) -> list[str]:
    targets = []
    if "A" in arms_filter:
        targets.append("A_cuda_logreg")
    if "B" in arms_filter:
        targets.extend("B_" + name for name in decision_backends)
    if "C" in arms_filter:
        targets.extend("C_" + name for name in llm_backends)
    return targets


def default_times_out_path(targets: list[str], dry_run: bool = False) -> Path:
    """Official default: one file per selected target. Multi-target fallback is unused because official runs require exactly one target."""
    suffix = "_dryrun" if dry_run else ""
    if len(targets) == 1:
        return RESULTS / f"eval_times_{targets[0]}{suffix}.json"
    return RESULTS / f"eval_multi_arms{suffix}.json"


ONE_TARGET_MESSAGE = (
    "Require exactly one evaluation target: --arms A, or --arms B --decision-backends <id>, "
    "or --arms C --llm-backends <id>. Official files are not written when the target is missing "
    "or more than one target is named."
)


def require_one_target(arms_filter, decision_backends, llm_backends) -> list[str]:
    targets = selected_gpu_targets(arms_filter, decision_backends, llm_backends)
    if len(targets) != 1:
        named = ", ".join(targets) if targets else "none"
        raise SystemExit(ONE_TARGET_MESSAGE + f" Selected: {named}.")
    return targets


def explicit_cli_targets(args) -> tuple[set[str], list[str], list[str], list[str]]:
    """Select the single official target from CLI flags only, not from env defaults."""
    if args.arms is None or not str(args.arms).strip():
        raise SystemExit(ONE_TARGET_MESSAGE)
    arms_filter = {a.strip().upper() for a in args.arms.split(",") if a.strip()}
    allowed_arms = {"A", "B", "C"}
    bad = arms_filter - allowed_arms
    if bad:
        raise SystemExit(f"Unknown --arms {bad}; use A, B, or C")
    if len(arms_filter) != 1:
        named = ", ".join(sorted(arms_filter)) if arms_filter else "none"
        raise SystemExit(ONE_TARGET_MESSAGE + f" Selected: {named}.")
    arm = next(iter(arms_filter))
    decision_backends: list[str] = []
    llm_backends: list[str] = []
    extra_b = bool(args.decision_backends and str(args.decision_backends).strip())
    extra_c = bool(args.llm_backends and str(args.llm_backends).strip())
    if arm == "A":
        if extra_b or extra_c:
            raise SystemExit(ONE_TARGET_MESSAGE + " Selected: A_cuda_logreg plus named B/C backends.")
    elif arm == "B":
        if extra_c or not extra_b:
            raise SystemExit(ONE_TARGET_MESSAGE)
        decision_backends = parse_backend_list(args.decision_backends, DECISION_BACKEND_IDS, ())
    else:
        if extra_b or not extra_c:
            raise SystemExit(ONE_TARGET_MESSAGE)
        llm_backends = parse_backend_list(args.llm_backends, LLM_BACKEND_IDS, ())
    targets = require_one_target(arms_filter, decision_backends, llm_backends)
    return arms_filter, decision_backends, llm_backends, targets


def run_official_arm_a_session(times: int) -> list[dict]:
    """Lazy official CUDA A entry; CUDA libraries are not imported for --help or B/C tests."""
    from cuda_A_sequential import run_cuda_a_session
    return run_cuda_a_session(times)


def write_times_output(out_json: Path, times: int, runs: list[dict], complete: bool, meta: dict | None = None) -> None:
    provenance = dict(meta or {})
    provenance.update({
        "completed_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "inference_performed": bool(complete and len(runs) == times and all(run.get("inference_performed") and run.get("actual_inference_rows") == provenance.get("required_n", 300) for run in runs)),
        "actual_inference_rows": sum(int(run.get("actual_inference_rows") or 0) for run in runs),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_sha256": hashlib.sha256((Path(__file__).parent / "eval_scoring.py").read_bytes()).hexdigest(),
        "gold_hashes": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (GOLD_JSONL, GOLD_SPLIT)},
    })
    metrics_keys = ("n_correct", "acc_hard", "acc_conf", "mean_top_choice_confidence", "macro_f1", "latency_ms", "errors", "error_rate")
    mean_metrics = None
    if complete and runs and all(run.get("inference_performed") and run.get("actual_inference_rows") == provenance.get("required_n", 300) for run in runs) and len(runs) == times:
        from statistics import mean
        mean_metrics = {key: mean(run["metrics"][key] for run in runs) for key in metrics_keys}
    payload = {
        "schema": "ge2340-eval-times-v1",
        "times": times,
        "complete": bool(complete and mean_metrics is not None),
        "scoring_semantics": SEMANTICS,
        "acc_conf_definition": (
            "Acc_conf = sum of confidence on correct valid rows / N; failed and incorrect contribute 0. "
            "mean_top_choice_confidence is distinct."
        ),
        "n_per_run": provenance.get("required_n"),
        "mean_metrics": mean_metrics,
        "runs": runs,
        "provenance": provenance,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_json)


def run_selected_backends(args, arms_filter, decision_backends, llm_backends, df, df_test, y_true, train_ids, decision_max, llm_max, run_id=None):
    arms: list[dict] = []
    if "A" in arms_filter:
        raise RuntimeError("official Arm A is executed as a whole times-session, not per inner loop")
    if "B" in arms_filter:
        for bid in decision_backends:
            arms.append(run_decision_backend(bid, df_test, decision_max))
    if "C" in arms_filter:
        fewshot = build_fewshot_msgs(df, train_ids)
        for bid in llm_backends:
            arms.append(run_llm_backend(bid, df, train_ids, df_test, llm_max, fewshot_msgs=fewshot))
    return arms


def build_arg_parser() -> argparse.ArgumentParser:
    epilog = """
Backend IDs
  Decision (Arm B): clef, laya, kev, semif, openjev
  LLM (Arm C+):     qwen38_27b, gemma4_26b_a4b, nemotron35_lightning,
                    lfm25_24b_a2b, muse_glimmer_30b

Env vars (names only; set in .env.eval — never echo values)
  Backend URLs and keys are read from the environment. The model itself
  must be named on the CLI (one --arms value, plus one backend id for B or C).
  DECISION_MAX_N      cap for Arm B; <=0 = full test
  LLM_MAX_N           cap for Arm C+; <=0 = full test
  Per decision backend B in {CLEF,LAYA,KEV,SEMIF,OPENJEV}:
    {B}_BASE_URL  {B}_API_KEY  {B}_MODEL
  Per LLM backend M (uppercased id):
    LLM_{M}_BASE_URL  LLM_{M}_API_KEY  LLM_{M}_MODEL
    Legacy LLM_BASE_URL / LLM_API_KEY / LLM_MODEL → qwen38_27b only

Metrics
  Acc_hard = N_correct / N
  Acc_conf = sum(correct valid confidence) / N  (not mean top-choice confidence)
  mean_top_choice_confidence is reported separately
  Also: latency_ms, mean error counts; six-label Macro-F1

  --times N (default 1) runs N genuine full passes over the locked 300-row
  test set and writes one JSON with 300xN raw rows plus arithmetic means
  of the N per-run metrics. Markdown is not written.

  Exactly one target per invocation: --arms A, or --arms B with one
  --decision-backends id, or --arms C with one --llm-backends id.
"""
    parser = argparse.ArgumentParser(
        description=(
            "Eval arms: official CUDA LogReg / SystemOne (clef|laya|kev|semif|openjev) / "
            "multi-LLM (qwen38_27b|gemma4_26b_a4b|nemotron35_lightning|"
            "lfm25_24b_a2b|muse_glimmer_30b)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env.eval",
        help="Optional dotenv file (default: experiments/.env.eval)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default for one target: results/eval_times_<target>.json (dry-run: results/eval_times_<target>_dryrun.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Disabled: result Markdown is no longer written",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data+env; print which backends would run (key names present/MISSING); no API POSTs",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Arm letter: A, B, or C. Exactly one target is required; there is no multi-arm default",
    )
    parser.add_argument(
        "--decision-backends",
        type=str,
        default=None,
        help="Required when --arms B: exactly one decision backend id (not an environment default)",
    )
    parser.add_argument(
        "--llm-backends",
        type=str,
        default=None,
        help="Required when --arms C: exactly one LLM backend id (not an environment default)",
    )
    parser.add_argument(
        "--decision-max-n",
        type=int,
        default=None,
        help="Cap Arm B scoring (overrides DECISION_MAX_N)",
    )
    parser.add_argument(
        "--llm-max-n",
        type=int,
        default=None,
        help="Cap Arm C+ scoring (overrides LLM_MAX_N)",
    )
    parser.add_argument("--times", type=positive_times, default=1, help="Number of genuine evaluation repeats (default 1); one output JSON")
    parser.add_argument("--no-overwrite", action="store_true", help="Refuse existing output paths")
    parser.add_argument("--require-frozen-300", action="store_true", help="Verify locked gold hashes and full 300-item split/caps")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    started_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    invocation_id = str(uuid.uuid4())

    arms_filter, decision_backends, llm_backends, targets = explicit_cli_targets(args)

    print("===", GOLD_NOTE, "===")
    if args.env_file:
        load_dotenv_file(args.env_file)

    if args.require_frozen_300 and not args.dry_run:
        check_requested_configuration(arms_filter, decision_backends, llm_backends)

    df, split = load_data()
    test_ids = split["test"]
    train_ids = split["train"]
    df_test = df.loc[test_ids]
    y_true = df_test["label"].tolist()
    print("test n =", len(df_test))

    decision_max = resolve_max_n(args.decision_max_n, ["DECISION_MAX_N"], len(df_test))
    llm_max = resolve_max_n(args.llm_max_n, ["LLM_MAX_N"], len(df_test))
    os.environ["DECISION_MAX_N"] = str(decision_max)
    os.environ["LLM_MAX_N"] = str(llm_max)
    print(
        f"caps: DECISION_MAX_N={decision_max} LLM_MAX_N={llm_max} (test_n={len(df_test)})"
    )
    print(f"decision_backends={decision_backends}")
    print(f"llm_backends={llm_backends}")

    if args.require_frozen_300:
        for rel, locked in LOCKS.items():
            assert hashlib.sha256((ROOT.parent / rel).read_bytes()).hexdigest() == locked, "Gold hash mismatch"
        assert len(df_test) == len(set(test_ids)) == decision_max == llm_max == 300, "Official repeats require all 300"

    if args.dry_run:
        out_json = args.out_json or default_times_out_path(targets, dry_run=True)
        dry_run_report(
            decision_backends,
            llm_backends,
            len(df_test),
            decision_max,
            llm_max,
            arms_filter,
            out_json,
        )
        print("Dry-run done. No API calls.")
        return

    if args.out_md is not None:
        raise SystemExit("Result Markdown is no longer written; omit --out-md and use the single output JSON")

    out_json = args.out_json or default_times_out_path(targets, dry_run=False)
    if args.no_overwrite and out_json.exists():
        raise SystemExit("Refusing to overwrite a run")

    if not args.require_frozen_300:
        check_requested_configuration(arms_filter, decision_backends, llm_backends)

    times = args.times
    required_n = len(test_ids)
    runs = []
    complete = True
    if "A" in arms_filter:
        session = run_official_arm_a_session(times)
        for result in session:
            arm = result["arm"]
            prov = result["provenance"]
            ok = completed_inference([arm], required_n)
            runs.append({
                "run_id": prov.get("run_id"),
                "invocation_id": prov.get("invocation_id"),
                "started_at": prov.get("started_at"),
                "completed_at": prov.get("completed_at"),
                "inference_performed": ok,
                "actual_inference_rows": len(arm.get("items") or []),
                "metrics": {k: arm.get(k) for k in ("n_correct", "acc_hard", "acc_conf", "mean_top_choice_confidence", "macro_f1", "latency_ms", "errors", "error_rate")},
                "arms": [_arm_record(arm)],
                "provenance": prov,
            })
            if not ok:
                complete = False
        if len(runs) != times:
            complete = False
    else:
        for index in range(times):
            run_started = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
            run_invocation = str(uuid.uuid4())
            arms = run_selected_backends(args, arms_filter, decision_backends, llm_backends, df, df_test, y_true, train_ids, decision_max, llm_max)
            ok = completed_inference(arms, required_n)
            metrics = None
            if arms and ok:
                arm = arms[0]
                metrics = {k: arm.get(k) for k in ("n_correct", "acc_hard", "acc_conf", "mean_top_choice_confidence", "macro_f1", "latency_ms", "errors", "error_rate")}
            runs.append({
                "run_id": "run%d" % (index + 1),
                "invocation_id": run_invocation,
                "started_at": run_started,
                "completed_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
                "inference_performed": ok,
                "actual_inference_rows": sum(len(a.get("items") or []) for a in arms),
                "metrics": metrics,
                "arms": [_arm_record(a) for a in arms],
            })
            if not ok:
                complete = False
                break

    meta = {
        "started_at": started_at,
        "python": sys.executable,
        "decision_backends": decision_backends,
        "llm_backends": llm_backends,
        "arms_filter": sorted(arms_filter),
        "required_n": required_n,
        "times": times,
    }
    write_times_output(out_json, times, runs, complete, meta=meta)
    if not complete or (args.require_frozen_300 and not all(run.get("inference_performed") and run.get("actual_inference_rows") == 300 for run in runs)):
        raise SystemExit("Official run incomplete: every requested repeat requires actual distinct inference rows; partial raw rows were retained in the single JSON; mean_metrics omitted")
    print("Done.", GOLD_NOTE)


if __name__ == "__main__":
    # Parse and reject missing/multi targets with stdlib only so --help, invalid
    # --times, and one-target errors work without joblib/CUDA, and write nothing.
    explicit_cli_targets(build_arg_parser().parse_args())
    _load_runtime_dependencies()
    main()
