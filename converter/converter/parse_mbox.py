"""Parse MBOX files into MailMessage objects."""

from __future__ import annotations

import mailbox
from pathlib import Path

from .parse_eml import parse_eml_bytes
from .message import MailMessage


def iter_mbox_messages(path: Path) -> list[MailMessage]:
    """Read an mbox file; each message is parsed like an .eml."""
    results: list[MailMessage] = []
    mbox = mailbox.mbox(str(path))
    try:
        for i, msg in enumerate(mbox):
            raw = msg.as_bytes()
            results.append(parse_eml_bytes(raw, source=f"{path}#{i}"))
    finally:
        mbox.close()
    return results
