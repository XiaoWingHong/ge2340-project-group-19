"""Parse .eml files into MailMessage objects."""

from __future__ import annotations

import email
import email.policy
from email.message import EmailMessage, Message
from pathlib import Path

from .html_text import html_to_text
from .message import MailMessage


def _decode_header_value(msg: Message, name: str) -> str:
    raw = msg.get(name, "") or ""
    if isinstance(raw, str):
        return " ".join(raw.split())
    return str(raw)


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to HTML→text."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "") or "").lower()
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                payload = None
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except Exception:
                text = payload.decode("utf-8", errors="replace")
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
    else:
        ctype = msg.get_content_type()
        try:
            if isinstance(msg, EmailMessage):
                text = msg.get_content()
                if not isinstance(text, str):
                    payload = msg.get_payload(decode=True) or b""
                    charset = msg.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
            else:
                payload = msg.get_payload(decode=True) or b""
                charset = msg.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
        except Exception:
            text = str(msg.get_payload() or "")
        if ctype == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        return "\n\n".join(p.strip() for p in plain_parts if p.strip()).strip()
    if html_parts:
        return "\n\n".join(html_to_text(h) for h in html_parts if h.strip()).strip()
    return ""


def parse_eml_bytes(data: bytes, source: str = "") -> MailMessage:
    msg = email.message_from_bytes(data, policy=email.policy.default)
    return MailMessage(
        from_addr=_decode_header_value(msg, "From"),
        subject=_decode_header_value(msg, "Subject"),
        body=_extract_body(msg),
        date=_decode_header_value(msg, "Date"),
        message_id=_decode_header_value(msg, "Message-ID"),
        source=source,
    )


def parse_eml_file(path: Path) -> MailMessage:
    data = path.read_bytes()
    return parse_eml_bytes(data, source=str(path))


def iter_eml_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".eml":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.eml") if p.is_file())
    return []
