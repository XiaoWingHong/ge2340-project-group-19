"""PII anonymization for email fields (from / subject / body)."""

from __future__ import annotations

import re

# Email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# CityU-like student IDs BEFORE phones (8-digit 5–7xxxxxxx collide with phone patterns)
_STUDENT_ID_RE = re.compile(
    r"(?i)(\b(?:student[\s\-]?id|sid|stu[\s\-]?id)\s*[:=]?\s*)?\b([5-7]\d{7})\b"
)

# Phones: prefer +852 / 852 / spaced/dashed digit groups
_PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"\+?852[\s\-]?(?:\d[\s\-]?){7,10}\d"  # HK country code
    r"|(?:\d[\s\-]){7,10}\d"  # digits with separators
    r"|(?<![A-Z\[])\b\d{10,11}\b"  # long contiguous (not 8-digit SID)
    r")(?!\w)"
)

_NAME_STOPWORDS = {
    "student",
    "students",
    "class",
    "team",
    "all",
    "sir",
    "madam",
    "colleagues",
    "everyone",
    "staff",
    "faculty",
}

_NAME_AFTER_GREETING_RE = re.compile(
    r"(?i)\b((?:dear|hi|hello|hey)\s+)([A-Z][a-z]{1,20})\b"
)
_NAME_AFTER_SIGNOFF_RE = re.compile(
    r"(?i)\b((?:regards|best|cheers|thanks|thank you)[,!]?\s*\n+\s*)"
    r"([A-Z][a-z]{1,20})\b"
)


def _redact_student_id(match: re.Match[str]) -> str:
    prefix = match.group(1) or ""
    return f"{prefix}[STUDENT_ID]"


def _redact_greeting_name(match: re.Match[str]) -> str:
    prefix, name = match.group(1), match.group(2)
    if name.lower() in _NAME_STOPWORDS:
        return match.group(0)
    return f"{prefix}[NAME]"


def anonymize_text(text: str) -> str:
    """Redact emails, student IDs, phones, light greeting names."""
    if not text:
        return text

    out = _EMAIL_RE.sub("[EMAIL]", text)
    out = _STUDENT_ID_RE.sub(_redact_student_id, out)
    out = _PHONE_RE.sub("[PHONE]", out)
    out = _NAME_AFTER_GREETING_RE.sub(_redact_greeting_name, out)
    out = _NAME_AFTER_SIGNOFF_RE.sub(_redact_greeting_name, out)
    return out


def anonymize_fields(from_: str, subject: str, body: str) -> tuple[str, str, str]:
    return anonymize_text(from_), anonymize_text(subject), anonymize_text(body)
