"""Shared six-class scoring. Failed replies abstain; they never predict other."""
from __future__ import annotations

import math

LABELS = ["promo", "admin", "course", "personal", "news", "other"]
SEMANTICS = "abstention-null-v1"
ACC_CONF_DEFINITION = (
    "Acc_conf = sum_i 1[correct and no error and a valid predicted label] * confidence_i / N, "
    "N = every scored row including failures. Null, error, and incorrect rows contribute 0. "
    "Raw confidence is never rewritten. Missing or invalid confidence on a correct row is a "
    "hard error, not a filled-in zero. Every non-null confidence must be finite and in [0, 1]. "
    "This is not mean top-choice confidence and not acc_hard times a confidence mean."
)


def explicit_error(row: dict) -> str | None:
    for key in ("error", "errors"):
        if row.get(key):
            return str(row[key])
    if row.get("success") is False or row.get("ok") is False:
        return "reply reports failure"
    if str(row.get("status", "")).lower() in {"error", "failed", "failure"}:
        return "reply status=" + str(row["status"])
    return None


def validated_confidence(value):
    """Return a finite [0, 1] float, or None when the field is legitimately absent."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a finite number in [0, 1]")
    confidence = float(value)
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError("confidence must be a finite number in [0, 1]")
    return confidence


def score_records(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        raise ValueError("cannot score empty run")
    cm = [[0 for _ in LABELS] for _ in LABELS]
    abstentions = {lab: 0 for lab in LABELS}
    correct = errors = 0
    weighted_correct = 0.0
    top_choice_total = 0.0
    for row in items:
        gold, pred = row["gold"], row.get("prediction")
        if gold not in LABELS:
            raise ValueError("unknown gold label")
        confidence = validated_confidence(row.get("confidence"))
        failed = bool(explicit_error(row)) or pred not in LABELS
        if failed:
            if pred is not None:
                raise ValueError("failed rows must have prediction=null")
            errors += 1
            abstentions[gold] += 1
        else:
            cm[LABELS.index(gold)][LABELS.index(pred)] += 1
            if pred == gold:
                if confidence is None:
                    raise ValueError("missing or invalid confidence on a correct prediction")
                correct += 1
                weighted_correct += confidence
                top_choice_total += confidence
            elif confidence is not None:
                top_choice_total += confidence
    acc_hard = correct / n
    acc_conf = weighted_correct / n
    if acc_conf < 0 or acc_conf > acc_hard + 1e-12:
        raise ValueError("acc_conf invariant violated: require 0 <= acc_conf <= acc_hard")
    report = {}
    for k, lab in enumerate(LABELS):
        tp = cm[k][k]
        support = sum(cm[k]) + abstentions[lab]
        predicted = sum(r[k] for r in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[lab] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    report["accuracy"] = acc_hard
    for name, weighted in (("macro avg", False), ("weighted avg", True)):
        report[name] = {
            metric: sum(report[lab][metric] * (report[lab]["support"] if weighted else 1) for lab in LABELS) / (n if weighted else len(LABELS))
            for metric in ("precision", "recall", "f1-score")
        }
        report[name]["support"] = n
    return {
        "scoring_semantics": SEMANTICS,
        "acc_conf_definition": ACC_CONF_DEFINITION,
        "n_scored": n, "n_correct": correct,
        "acc_hard": acc_hard, "accuracy": acc_hard,
        "acc_conf": acc_conf,
        "mean_top_choice_confidence": top_choice_total / n,
        "macro_f1": report["macro avg"]["f1-score"],
        "errors": errors, "error_rate": errors / n,
        "classification_report": report, "report": report,
        "confusion_matrix": {"labels": list(LABELS), "matrix": cm, "abstentions_by_gold": abstentions},
    }
