#!/usr/bin/env python3
"""Thin entrypoint: python convert.py --input ... --output ..."""

from converter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
