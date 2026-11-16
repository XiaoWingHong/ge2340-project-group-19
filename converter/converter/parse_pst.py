"""Parse Outlook .pst files via readpst (LibPST).

PST support requires the external ``readpst`` binary from LibPST
(Debian/Ubuntu package: ``pst-utils``). This module shells out to
``readpst``, extracts messages as separate files, and reuses the .eml
parser.

Install::

    sudo apt install pst-utils   # provides /usr/bin/readpst

Override the binary with the READPST environment variable if needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .message import MailMessage
from .parse_eml import parse_eml_file
from .parse_mbox import iter_mbox_messages

READPST_NOTES = """\
Outlook .pst support uses readpst from LibPST (not a Python library).

Debian/Ubuntu:
  sudo apt install pst-utils

This provides the ``readpst`` binary. The converter calls:

  readpst -r -S -b -8 -o <tempdir> <file.pst>

(-r recursive folders, -S one file per message, -b skip RTF, -8 UTF-8).
Extracted files are then parsed as .eml (or MBOX if a folder file is
in mailbox format).

The READPST environment variable may point at a non-PATH binary.

A single .pst file or a directory of .pst files (including nested
*.pst) is supported.
"""


def find_readpst(explicit: str | Path | None = None) -> Path:
    """Locate the readpst binary or raise a clear install error."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        raise FileNotFoundError(f"readpst not executable: {path}")

    env = os.environ.get("READPST")
    if env:
        path = Path(env).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        raise FileNotFoundError(
            f"READPST is set to {path} but that path is not an executable"
        )

    found = shutil.which("readpst")
    if found:
        return Path(found).resolve()

    raise FileNotFoundError(
        "readpst (LibPST) was not found on PATH. Install the LibPST "
        "CLI, e.g. `sudo apt install pst-utils` on Debian/Ubuntu, or "
        "set the READPST environment variable to the binary. "
        "Run `python -m converter --pst-help` for details."
    )


def iter_pst_paths(input_path: Path) -> list[Path]:
    """Return .pst files from a single file or a directory tree (nested)."""
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() == ".pst":
            return [path]
        raise ValueError(f"Not a .pst file: {path}")
    if path.is_dir():
        found = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pst"
        )
        return found
    raise FileNotFoundError(f"Input not found: {path}")


def parse_pst_file(
    pst_path: Path,
    *,
    readpst_bin: str | Path | None = None,
) -> list[MailMessage]:
    """Extract one .pst with readpst and parse the resulting messages."""
    pst_path = pst_path.expanduser().resolve()
    if not pst_path.is_file():
        raise FileNotFoundError(f"PST not found: {pst_path}")

    binary = find_readpst(readpst_bin)
    messages: list[MailMessage] = []
    with tempfile.TemporaryDirectory(prefix="pst-extract-") as tmp:
        outdir = Path(tmp)
        cmd = [
            str(binary),
            "-r",
            "-S",
            "-b",
            "-8",
            "-o",
            str(outdir),
            str(pst_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to run readpst ({binary}): {exc}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"readpst failed on {pst_path} (exit {proc.returncode})"
                + (f": {err}" if err else "")
            )
        messages.extend(_parse_extracted_tree(outdir, source=str(pst_path)))
    return messages


def _parse_extracted_tree(root: Path, *, source: str) -> list[MailMessage]:
    messages: list[MailMessage] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Skip tiny junk and readpst metadata-looking files
        try:
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        suffix = path.suffix.lower()
        if suffix in {".mbox", ".mbx"}:
            try:
                messages.extend(_with_source(iter_mbox_messages(path), source))
            except Exception:
                continue
            continue
        # .eml or extensionless per-message files from readpst -S
        if suffix in {".eml", ".msg", ""} or suffix not in {
            ".txt",
            ".log",
            ".xml",
            ".html",
            ".htm",
            ".dat",
            ".bin",
            ".db",
        }:
            try:
                msg = parse_eml_file(path)
            except Exception:
                try:
                    messages.extend(_with_source(iter_mbox_messages(path), source))
                    continue
                except Exception:
                    continue
            else:
                messages.append(_replace_source(msg, source))
    return messages


def _replace_source(msg: MailMessage, source: str) -> MailMessage:
    return MailMessage(
        from_addr=msg.from_addr,
        subject=msg.subject,
        body=msg.body,
        date=msg.date,
        message_id=msg.message_id,
        source=source,
    )


def _with_source(messages: list[MailMessage], source: str) -> list[MailMessage]:
    return [_replace_source(m, source) for m in messages]
