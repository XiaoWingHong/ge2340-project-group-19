"""English-only filter for message bodies.

Behavior:
- Prefer langdetect when installed (detect language of body text).
- If langdetect unavailable or confidence/detection fails, use a heuristic:
  keep messages whose Latin-letter ratio is high and CJK character ratio is low.
- Short bodies (< 40 chars after strip) are kept (too little signal to drop).
- Non-English bodies are dropped from the task list.
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

try:
    from langdetect import DetectorFactory, detect, lang_detect_exception

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    _HAS_LANGDETECT = False
    detect = None  # type: ignore
    lang_detect_exception = Exception  # type: ignore


def is_english(text: str, *, min_chars: int = 40) -> bool:
    """Return True if text should be kept as English (or too short to judge)."""
    body = (text or "").strip()
    if len(body) < min_chars:
        return True

    if _HAS_LANGDETECT and detect is not None:
        try:
            # Sample start for speed on long bodies
            sample = body[:4000]
            lang = detect(sample)
            return lang == "en"
        except Exception:
            pass  # fall through to heuristic

    letters = len(_LATIN_RE.findall(body))
    cjk = len(_CJK_RE.findall(body))
    total = max(len(body), 1)
    if cjk / total > 0.08:
        return False
    if letters / total < 0.25:
        return False
    return True
