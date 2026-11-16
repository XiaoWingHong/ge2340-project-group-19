"""End-to-end: load .pst inputs → parse → anonymize → English filter → empty-drop → dedupe → optional sample → JSON tasks."""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Iterable

from .anonymize import anonymize_fields
from .english import is_english
from .message import MailMessage
from .parse_pst import iter_pst_paths, parse_pst_file


PUBLIC_INPUT_ERROR = (
    "Converter raw input must be a single .pst file or a folder of "
    "*.pst files (nested *.pst are included). "
    ".eml, .olm, and MBOX are not accepted as user input."
)


def discover_and_parse(input_path: Path) -> list[MailMessage]:
    """Load messages from a single .pst or a folder of *.pst files only."""
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() == ".pst":
            return list(parse_pst_file(path))
        raise ValueError(f"{PUBLIC_INPUT_ERROR} Got file: {path}")

    if path.is_dir():
        pst_files = iter_pst_paths(path)
        if not pst_files:
            raise ValueError(f"{PUBLIC_INPUT_ERROR} No .pst files found under {path}")
        messages: list[MailMessage] = []
        for pst in pst_files:
            messages.extend(parse_pst_file(pst))
        return messages

    raise ValueError(f"Not a file or directory: {path}")


def process_messages(
    messages: Iterable[MailMessage],
    *,
    english_only: bool = True,
    anonymize: bool = True,
) -> list[dict]:
    """Anonymize, filter English, drop empty bodies, dedupe → Label Studio tasks."""
    seen: set[str] = set()
    tasks: list[dict] = []

    for msg in messages:
        from_a, subj, body = msg.from_addr, msg.subject, msg.body
        if anonymize:
            from_a, subj, body = anonymize_fields(from_a, subj, body)

        body = (body or "").strip()
        if not body:
            continue

        if english_only and not is_english(body):
            continue

        cleaned = MailMessage(
            from_addr=from_a,
            subject=subj,
            body=body,
            date=msg.date,
            message_id=msg.message_id,
            source=msg.source,
        )
        tid = cleaned.stable_id()
        if tid in seen:
            continue
        seen.add(tid)
        tasks.append(cleaned.to_task())

    return tasks


def sample_tasks(
    tasks: list[dict],
    sample: int | None,
    seed: int = 2340,
) -> list[dict]:
    """Randomly select ``sample`` tasks with a fixed seed.

    If fewer than ``sample`` tasks remain, keep all and warn. Selected
    tasks keep their original relative order. Seed defaults to 2340.
    """
    if sample is None:
        return tasks
    if sample < 1:
        raise ValueError(f"--sample must be a positive integer, got {sample}")
    if len(tasks) <= sample:
        if len(tasks) < sample:
            msg = (
                f"Only {len(tasks)} task(s) remain after parse/anonymize/"
                f"English-filter/empty-drop/dedupe; --sample {sample} requested. "
                "Keeping all remaining tasks."
            )
            print(f"warning: {msg}", file=sys.stderr)
            warnings.warn(msg, UserWarning, stacklevel=2)
        return list(tasks)

    rng = random.Random(seed)
    chosen_idx = sorted(rng.sample(range(len(tasks)), sample))
    return [tasks[i] for i in chosen_idx]


def convert(
    input_path: Path,
    output_path: Path,
    *,
    english_only: bool = True,
    anonymize: bool = True,
    sample: int | None = None,
    seed: int = 2340,
) -> list[dict]:
    messages = discover_and_parse(input_path)
    tasks = process_messages(
        messages, english_only=english_only, anonymize=anonymize
    )
    tasks = sample_tasks(tasks, sample=sample, seed=seed)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return tasks
