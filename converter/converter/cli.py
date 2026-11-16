"""CLI for the mail → Label Studio converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .parse_pst import READPST_NOTES
from .pipeline import convert


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="converter",
        description=(
            "Convert Outlook mail (.pst) into Label Studio import JSON "
            "for email-type classification "
            "(promo | admin | course | personal | news | other). "
            "PST files are read with readpst from LibPST (package pst-utils). "
            "Raw input must be a single .pst file or a folder of *.pst files "
            "(nested *.pst are included). .eml, .olm, and MBOX are not accepted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "PST: pass a single .pst file or a directory containing *.pst "
            "(nested files are included). Requires the readpst binary from "
            "LibPST (`sudo apt install pst-utils` on Debian/Ubuntu). "
            "After parse → anonymize → English-filter → empty-drop → dedupe, "
            "optionally randomly select N tasks with --sample N "
            "(default seed 2340). If fewer than N remain, all are kept and "
            "a warning is printed."
        ),
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Path to a single .pst file, or a folder of *.pst files (nested allowed)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON path (Label Studio tasks array)",
    )
    p.add_argument(
        "--no-english-filter",
        action="store_true",
        help="Keep non-English bodies (default: drop when detectable)",
    )
    p.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip PII anonymization (not recommended)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After the pipeline, randomly select N tasks (e.g. --sample 2000). "
            "If fewer than N remain, keep all and warn. Omitted: keep all."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2340,
        help="RNG seed for --sample (default: 2340)",
    )
    p.add_argument(
        "--pst-help",
        action="store_true",
        help="Print .pst / readpst (LibPST) guidance and exit",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.pst_help:
        print(READPST_NOTES)
        return 0

    if args.input is None or args.output is None:
        parser.error("--input/-i and --output/-o are required (unless --pst-help)")

    try:
        tasks = convert(
            args.input,
            args.output,
            english_only=not args.no_english_filter,
            anonymize=not args.no_anonymize,
            sample=args.sample,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(tasks)} task(s) → {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
