#!/usr/bin/env python3
"""Behavior checks only; mock HTTP, never load a model or perform inference."""
import contextlib
import io
import json
import unittest
import copy
from unittest.mock import patch
import pandas as pd
import requests
import eval_three_arms as e
from eval_scoring import LABELS, score_records, explicit_error


def response(body, status=200):
    r = requests.Response()
    r.status_code = status
    r._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    return r


def frame(n):
    return pd.DataFrame([{"task_id": i, "label": "other", "subject": "test", "body": "test", "from": "test"} for i in range(n)])


def credentials(*args):
    return {"base_url": "http://example.invalid/v1", "api_key": "secret-key", "model": "test", "ready": True, "key_names": {"base_url": "URL", "api_key": "KEY", "model": "MODEL"}}


class SemanticsTests(unittest.TestCase):
    def run_backend(self, family, bodies, backend=None):
        df = frame(len(bodies))
        responses = [b if isinstance(b, Exception) else response(*b) if isinstance(b, tuple) else response(b) for b in bodies]
        with patch.object(e, "resolve_llm_creds", credentials), patch.object(e, "resolve_decision_creds", credentials), patch.object(e, "_endpoint_reachable", return_value=(True, "mock")), patch.object(e.requests, "post", side_effect=responses) as post, contextlib.redirect_stdout(io.StringIO()):
            if family == "C":
                arm = e.run_llm_backend(backend or "muse_glimmer_30b", df, [], df, len(df), fewshot_msgs=[])
            else:
                arm = e.run_decision_backend(backend or "semif", df, len(df))
        return arm, post

    def test_all_300_failed_other_are_wrong(self):
        rows = [{"gold": "other", "prediction": None, "confidence": None, "error": "timeout"} for _ in range(300)]
        m = score_records(rows)
        self.assertEqual((m["n_scored"], m["n_correct"], m["errors"]), (300, 0, 300))
        self.assertEqual((m["acc_hard"], m["acc_conf"], m["macro_f1"]), (0, 0, 0))
        self.assertEqual(m["classification_report"]["other"]["support"], 300)

    def test_six_classes_abstention_false_negatives(self):
        rows = [{"gold": lab, "prediction": lab, "confidence": .9} for lab in LABELS] + [{"gold": lab, "prediction": None, "confidence": None, "error": "bad"} for lab in LABELS]
        m = score_records(rows)
        self.assertEqual(m["acc_hard"], .5)
        self.assertAlmostEqual(m["acc_conf"], .45)
        self.assertAlmostEqual(m["macro_f1"], 2/3)
        for lab in LABELS:
            self.assertEqual(m["classification_report"][lab]["recall"], .5)
        self.assertEqual(len(m["confusion_matrix"]["labels"]), 6)

    def test_missing_confidence_on_correct_row_is_a_blocker(self):
        with self.assertRaises(ValueError):
            score_records([{"gold": "other", "prediction": "other", "confidence": None}])

    def test_missing_confidence_on_incorrect_valid_row_contributes_zero(self):
        m = score_records([{"gold": "other", "prediction": "promo", "confidence": None}])
        self.assertEqual((m["n_correct"], m["errors"], m["acc_hard"], m["acc_conf"]), (0, 0, 0, 0))
        self.assertEqual(m["mean_top_choice_confidence"], 0)

    def test_failed_null_confidence_remains_zero_even_if_gold_is_other(self):
        m = score_records([{"gold": "other", "prediction": None, "confidence": None, "error": "JSON parse error"}])
        self.assertEqual((m["n_correct"], m["errors"], m["acc_hard"], m["acc_conf"]), (0, 1, 0, 0))
        self.assertEqual(m["mean_top_choice_confidence"], 0)

    def test_all_correct_nonuniform_confidence(self):
        rows = [{"gold": lab, "prediction": lab, "confidence": c} for lab, c in zip(LABELS, (1, .5, .2, 0, .8, .9))]
        m = score_records(rows)
        expected = sum((1, .5, .2, 0, .8, .9)) / 6
        self.assertEqual(m["acc_hard"], 1)
        self.assertAlmostEqual(m["acc_conf"], expected)
        self.assertAlmostEqual(m["mean_top_choice_confidence"], expected)
        self.assertLessEqual(m["acc_conf"], m["acc_hard"])

    def test_all_wrong_high_confidence_has_zero_acc_conf(self):
        rows = [{"gold": "other", "prediction": "promo", "confidence": 1} for _ in range(6)]
        m = score_records(rows)
        self.assertEqual((m["acc_hard"], m["acc_conf"], m["errors"]), (0, 0, 0))
        self.assertEqual(m["mean_top_choice_confidence"], 1)
        self.assertNotAlmostEqual(m["acc_conf"], m["acc_hard"] * m["mean_top_choice_confidence"] if False else 1)

    def test_mixed_nonuniform_counts_only_correct_confidence(self):
        rows = [
            {"gold": "promo", "prediction": "promo", "confidence": 0.25},
            {"gold": "admin", "prediction": "course", "confidence": 1.0},
            {"gold": "course", "prediction": "course", "confidence": 0.5},
            {"gold": "personal", "prediction": None, "confidence": None, "error": "timeout"},
            {"gold": "news", "prediction": "news", "confidence": 1.0},
            {"gold": "other", "prediction": "promo", "confidence": 0.9},
        ]
        m = score_records(rows)
        self.assertEqual((m["n_correct"], m["errors"], m["acc_hard"]), (3, 1, 0.5))
        self.assertAlmostEqual(m["acc_conf"], (0.25 + 0.5 + 1.0) / 6)
        self.assertAlmostEqual(m["mean_top_choice_confidence"], (0.25 + 1.0 + 0.5 + 1.0 + 0.9) / 6)
        self.assertLessEqual(m["acc_conf"], m["acc_hard"])
        self.assertNotAlmostEqual(m["acc_conf"], m["acc_hard"] * m["mean_top_choice_confidence"])

    def test_invalid_confidence_values_are_blockers(self):
        correct = {"gold": "other", "prediction": "other", "confidence": 1}
        wrong = {"gold": "other", "prediction": "promo", "confidence": 1}
        for bad in (float("nan"), float("inf"), -0.01, 1.01, "0.5", True):
            with self.assertRaises(ValueError):
                score_records([{**correct, "confidence": bad}])
            with self.assertRaises(ValueError):
                score_records([{**wrong, "confidence": bad}])

    def test_stale_mean_confidence_is_not_acc_conf(self):
        rows = [{"gold": "other", "prediction": "other", "confidence": 1}] * 2 + [{"gold": "other", "prediction": "promo", "confidence": 1}] * 2
        m = score_records(rows)
        self.assertEqual(m["acc_hard"], 0.5)
        self.assertAlmostEqual(m["acc_conf"], 0.5)
        self.assertAlmostEqual(m["mean_top_choice_confidence"], 1)
        self.assertNotAlmostEqual(m["acc_conf"], m["mean_top_choice_confidence"])
        self.assertAlmostEqual(m["acc_hard"] * m["mean_top_choice_confidence"], 0.5)
        # The stale product happens to equal acc_conf here only because every
        # correct confidence is 1; the next assertion uses a case where it would not.
        rows[0] = {"gold": "other", "prediction": "other", "confidence": 0.2}
        m = score_records(rows)
        self.assertAlmostEqual(m["acc_conf"], 0.3)
        self.assertAlmostEqual(m["mean_top_choice_confidence"], 0.8)
        self.assertNotAlmostEqual(m["acc_conf"], m["acc_hard"] * m["mean_top_choice_confidence"])
        self.assertLessEqual(m["acc_conf"], m["acc_hard"])

    def test_legacy_error_with_assigned_label_rejected(self):
        with self.assertRaises(ValueError):
            score_records([{"gold": "other", "prediction": "other", "error": "failed"}])

    def test_parser_optional_single_prefix_only(self):
        raw = '{"label":"other","confidence":0.8}'
        self.assertEqual(e.parse_label_confidence("  to=user " + raw), ("other", .8))
        for text in ("to=self" + raw, "to=user to=user " + raw, "```json " + raw + "```", "other", "[]", raw + " trailing"):
            self.assertEqual(e.parse_label_confidence(text), (None, None))

    def test_json_label_and_explicit_error_is_failure(self):
        self.assertEqual(e.parse_label_confidence('{"label":"other","confidence":1,"error":"failed"}'), (None, None))
        row = {"prediction": "other", "confidence": 1, "raw_reply": '{"label":"other","confidence":1}', "error": "HTTP 500"}
        self.assertEqual(explicit_error(row), "HTTP 500")

    def test_c_mixed_parse_http_and_valid_row(self):
        valid = '{"label":"other","confidence":0.8}'
        invalid = 'to=self other {"label":"other","confidence":1}'
        arm, _ = self.run_backend("C", [{"choices": [{"message": {"content": valid}}]}, {"choices": [{"message": {"content": invalid}}]}, ('{"error":"secret-key"}', 500)])
        self.assertEqual(arm["predictions"], ["other", None, None])
        self.assertEqual(arm["n_correct"], 1)
        self.assertEqual(arm["errors"], 2)
        self.assertAlmostEqual(arm["acc_hard"], 1/3)
        self.assertAlmostEqual(arm["acc_conf"], 0.8/3)
        self.assertAlmostEqual(arm["mean_top_choice_confidence"], 0.8/3)
        self.assertEqual(json.loads(arm["items"][1]["raw_reply"])["choices"][0]["message"]["content"], invalid)
        self.assertEqual(arm["items"][1]["error"], "JSON parse error")
        self.assertEqual(arm["items"][2]["raw_reply"], '{"error":"<REDACTED>"}')
        self.assertNotIn("secret-key", arm["items"][2]["error"])

    def test_c_valid_label_missing_confidence_blocks_scoring(self):
        valid = '{"label":"other"}'
        with self.assertRaises(ValueError):
            self.run_backend("C", [{"choices": [{"message": {"content": valid}}]}])

    def test_c_all_failed_has_zero_scores(self):
        arm, _ = self.run_backend("C", [{"choices": [{"message": {"content": "to=self thinking other"}}]}])
        self.assertEqual((arm["status"], arm["acc_hard"], arm["macro_f1"], arm["errors"]), ("error", 0, 0, 1))

    def test_b_bad_schema_keeps_entire_body(self):
        body = '{"answers":{},"diagnostic":"unparseable"}'
        arm, _ = self.run_backend("B", [body])
        self.assertIsNone(arm["items"][0]["prediction"])
        self.assertEqual(arm["items"][0]["raw_reply"], body)
        self.assertTrue(arm["items"][0]["error"])
        self.assertEqual(arm["acc_hard"], 0)

    def test_b_valid_other_and_explicit_error_never_credit(self):
        valid = {"answers": {"email_type": {"choice": "other", "score": .7}}}
        failed = {**valid, "error": "backend failed"}
        arm, _ = self.run_backend("B", [valid, failed])
        self.assertEqual(arm["predictions"], ["other", None])
        self.assertEqual(arm["n_correct"], 1)
        self.assertAlmostEqual(arm["acc_conf"], .35)

    def test_transport_timeout_has_error_and_latency(self):
        arm, _ = self.run_backend("C", [requests.Timeout("timeout")])
        self.assertIsNone(arm["items"][0]["raw_reply"])
        self.assertEqual(arm["items"][0]["error"], "timeout")
        self.assertGreaterEqual(arm["items"][0]["latency_ms"], 0)

    def test_gemma_only_stop_and_frozen_chat_settings(self):
        body = {"choices": [{"message": {"content": '{"label":"other","confidence":1}'}}]}
        for backend in e.LLM_BACKEND_IDS:
            _, post = self.run_backend("C", [body], backend)
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertIs(payload["enable_thinking"], False)
            self.assertEqual(payload.get("stop"), ["<end_of_turn>"] if backend == "gemma4_26b_a4b" else None)
            self.assertEqual(payload["messages"][0]["content"], e.SYSTEM_PROMPT)



if __name__ == "__main__":
    unittest.main(verbosity=2)
