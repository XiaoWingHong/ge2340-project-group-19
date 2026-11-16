"""Careful HTML → plain text (stdlib only)."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        t = tag.lower()
        if t in ("script", "style", "head"):
            self._skip = True
        elif t in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")
        elif t == "td":
            self._parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "head"):
            self._skip = False
        elif t in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def html_to_text(html_src: str) -> str:
    if not html_src or not html_src.strip():
        return ""
    parser = _HTMLToText()
    try:
        parser.feed(html_src)
        parser.close()
    except Exception:
        # Fallback: strip tags naively
        return re.sub(r"<[^>]+>", " ", html_src)
    return parser.get_text()
