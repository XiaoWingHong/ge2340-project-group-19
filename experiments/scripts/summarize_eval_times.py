#!/usr/bin/env python3
"""Read-only summary of official ge2340-eval-times-v1 results.

Loads experiments/results/eval_times_*.json and prints the official mean
table plus per-run rows. Does not start models, overwrite result JSON,
write Markdown, or run inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

COLUMNS = (
    "arm",
    "file",
    "n_per_run",
    "n_correct",
    "acc_hard",
    "acc_conf",
    "mean_top_choice_confidence",
    "macro_f1",
    "latency_ms",
    "errors",
)


def load_official_rows(results_dir: Path = RESULTS) -> list[dict]:
    official = sorted(results_dir.glob("eval_times_*.json"))
    if not official:
        raise SystemExit(f"No official eval_times_*.json files under {results_dir}")
    rows: list[dict] = []
    for path in official:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "ge2340-eval-times-v1":
            raise SystemExit(f"{path.name}: unexpected schema {data.get('schema')!r}")
        if data.get("times") != 3 or not data.get("complete"):
            raise SystemExit(f"{path.name}: expected times=3 and complete=true")
        mean = data["mean_metrics"]
        arm = data["runs"][0]["arms"][0]["arm"]
        n_per_run = data.get("n_per_run", 300)
        rows.append(_row(arm, path.name, n_per_run, mean))
        for run in data["runs"]:
            rows.append(_row(arm, run["run_id"], n_per_run, run["metrics"]))
    return rows


def _row(arm: str, file_name: str, n_per_run, metrics: dict) -> dict:
    return {
        "arm": arm,
        "file": file_name,
        "n_per_run": n_per_run,
        "n_correct": metrics["n_correct"],
        "acc_hard": metrics["acc_hard"],
        "acc_conf": metrics["acc_conf"],
        "mean_top_choice_confidence": metrics["mean_top_choice_confidence"],
        "macro_f1": metrics["macro_f1"],
        "latency_ms": metrics["latency_ms"],
        "errors": metrics["errors"],
    }


DISPLAY = {col: col for col in COLUMNS}
DISPLAY["macro_f1"] = "Macro-F1"


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    widths = {col: max(len(DISPLAY[col]), max(len(_fmt(row[col])) for row in rows)) for col in COLUMNS}
    header = "  ".join(DISPLAY[col].ljust(widths[col]) for col in COLUMNS)
    print(header)
    print("  ".join("-" * widths[col] for col in COLUMNS))
    for row in rows:
        print("  ".join(_fmt(row[col]).ljust(widths[col]) for col in COLUMNS))


def main() -> int:
    rows = load_official_rows()
    n_files = len({row["file"] for row in rows if row["file"].endswith(".json")})
    print(n_files, "official files")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
