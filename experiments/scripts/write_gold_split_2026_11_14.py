#!/usr/bin/env python3
"""Write a new 70/15/15 stratified split on the masked human-gold set.

Procedure (documented as required by the gold-train GO):
  - Source: experiments/data/emails_labeled_gold_masked-2026-11-14-v2.jsonl
  - IDs are integer task_id values (1–2000).
  - Stratify on gold `label` (promo|admin|course|personal|news|other).
  - Seed 2340.
  - Two-stage sklearn train_test_split:
      1) hold out 15% as test, stratify=label, random_state=2340
      2) from the remaining 85%, hold out 15/85 (~17.647%) as val so that
         val is 15% of the original set; stratify=label, random_state=2340
    This yields the expected 1400 / 300 / 300 sizes when stratification allows.
  - Does not read or overwrite experiments/data/split.json (old hash-id split).

Output: experiments/data/split_gold-2026-11-14.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GOLD = DATA / "emails_labeled_gold_masked-2026-11-14-v2.jsonl"
OUT = DATA / "split_gold-2026-11-14.json"
LOCKED_SPLIT_SHA256 = "b2e067b9995202df22bb30e5255e575ca0191e57b03a1340df858460493ef835"
SEED = 2340
LABELS = ["promo", "admin", "course", "personal", "news", "other"]


def main() -> None:
    if OUT.is_file():
        raise SystemExit(
            f"Refusing to overwrite the locked split {OUT} "
            f"(sha256 {LOCKED_SPLIT_SHA256})."
        )

    rows = []
    with GOLD.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != 2000:
        raise SystemExit(f"expected 2000 gold rows, got {len(rows)}")

    ids = [int(r["task_id"]) for r in rows]
    labels = [r["label"] for r in rows]
    if sorted(ids) != list(range(1, 2001)):
        raise SystemExit("gold task_id values are not 1..2000")

    # Stage 1: 15% test
    trainval_ids, test_ids, trainval_y, test_y = train_test_split(
        ids, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    # Stage 2: 15% of original = 0.15/0.85 of the remaining 85%
    val_frac_of_remaining = 0.15 / 0.85
    train_ids, val_ids, train_y, val_y = train_test_split(
        trainval_ids,
        trainval_y,
        test_size=val_frac_of_remaining,
        random_state=SEED,
        stratify=trainval_y,
    )

    train_ids = [int(x) for x in train_ids]
    val_ids = [int(x) for x in val_ids]
    test_ids = [int(x) for x in test_ids]

    def counts(ys):
        c = Counter(ys)
        return {lab: c.get(lab, 0) for lab in LABELS}

    payload = {
        "seed": SEED,
        "source": str(GOLD.relative_to(ROOT.parent)) if False else GOLD.name,
        "source_path": str(GOLD),
        "procedure": (
            "sklearn train_test_split, two-stage, stratify=gold label, "
            f"random_state={SEED}. Stage 1: test_size=0.15 → test. "
            "Stage 2: test_size=0.15/0.85 of the remainder → val. "
            "Remainder → train. IDs are integer task_id values."
        ),
        "n": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "label_counts_overall": counts(labels),
        "label_counts": {
            "train": counts(train_y),
            "val": counts(val_y),
            "test": counts(test_y),
        },
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    # fix overall counts
    payload["label_counts_overall"] = counts(labels)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print("n:", payload["n"])
    print("label counts:", json.dumps(payload["label_counts"], indent=2))
    # disjoint / coverage checks
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test)
    assert s_train | s_val | s_test == set(ids)
    assert len(s_train) + len(s_val) + len(s_test) == 2000


if __name__ == "__main__":
    main()
