"""Normalized mail message model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class MailMessage:
    from_addr: str
    subject: str
    body: str
    date: str = ""
    message_id: str = ""
    source: str = ""  # path / mailbox name for debugging

    def stable_id(self) -> str:
        mid = (self.message_id or "").strip()
        if mid:
            raw = mid.encode("utf-8", errors="replace")
            return hashlib.sha256(raw).hexdigest()[:16]
        key = f"{self.from_addr}|{self.subject}|{self.date}"
        return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]

    def to_task(self) -> dict[str, Any]:
        return {
            "id": self.stable_id(),
            "data": {
                "from": self.from_addr,
                "subject": self.subject,
                "body": self.body,
            },
        }
