#!/usr/bin/env python3
"""Offline --times behavior; mock HTTP and CUDA A, never perform inference."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import requests
import eval_three_arms as e
from eval_scoring import score_records


def response(body, status=200):
    r = requests.Response()
    r.status_code = status
    r._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    return r


def frame(n=3):
    return pd.DataFrame([{"task_id": i, "label": "other", "subject": "test", "body": "test", "from": "test"} for i in range(n)])


class TimesTests(unittest.TestCase):
    def parse(self, *argv):
        return e.build_arg_parser().parse_args(list(argv))

    def test_default_times_is_one(self):
        self.assertEqual(self.parse().times, 1)

    def test_invalid_times_rejected(self):
        for value in ("0", "-1", "1.5", "foo", ""):
            with self.assertRaises(SystemExit):
                self.parse("--times", value)

    def test_help_does_not_import_cuda(self):
        original = __import__
        def guarded(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in {"cupy", "cupyx", "cuml", "rmm"}:
                raise AssertionError("CUDA import during --help: " + name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded), contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as raised:
                e.build_arg_parser().parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--times", text)
        self.assertIn("Acc_conf", text)
        self.assertIn("Exactly one target", text)

    def test_times_three_makes_distinct_invocations_and_one_json_without_md(self):
        bodies = [[{"choices": [{"message": {"content": '{"label":"other","confidence":0.5}'}}]}]] * 3
        calls = {"n": 0}
        def post(*args, **kwargs):
            calls["n"] += 1
            return response(bodies[0][0])
        df = frame(1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            with patch.object(e, "resolve_llm_creds", return_value={"base_url": "http://example.invalid/v1", "api_key": "secret-key", "model": "test", "ready": True, "missing_keys": [], "key_names": {"base_url": "URL", "api_key": "KEY", "model": "MODEL"}}), \
                 patch.object(e, "_endpoint_reachable", return_value=(True, "mock")), \
                 patch.object(e.requests, "post", side_effect=post), \
                 patch.object(e, "load_data", return_value=(df, {"train": [], "test": list(df.index)})), \
                 patch.object(e, "load_dotenv_file"), \
                 patch.object(e, "check_requested_configuration"), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["eval_three_arms.py", "--arms", "C", "--llm-backends", "muse_glimmer_30b", "--times", "3", "--llm-max-n", "1", "--out-json", str(out)]):
                    e.main()
            self.assertTrue(out.exists())
            self.assertFalse(out.with_suffix(".md").exists())
            payload = json.loads(out.read_text())
            self.assertEqual(payload["times"], 3)
            self.assertTrue(payload["complete"])
            self.assertEqual(len(payload["runs"]), 3)
            ids = [run["invocation_id"] for run in payload["runs"]]
            self.assertEqual(len(set(ids)), 3)
            self.assertEqual(calls["n"], 3)
            self.assertIsNotNone(payload["mean_metrics"])
            self.assertAlmostEqual(payload["mean_metrics"]["acc_conf"], 0.5)
            self.assertAlmostEqual(payload["mean_metrics"]["acc_hard"], 1)

    def test_missing_config_exits_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            with patch.object(e, "resolve_llm_creds", return_value={"base_url": None, "api_key": None, "model": None, "ready": False, "missing_keys": ["LLM_MUSE_GLIMMER_30B_BASE_URL"], "key_names": {"base_url": "URL", "api_key": "KEY", "model": "MODEL"}}), \
                 patch.object(e, "load_data", return_value=(frame(1), {"train": [], "test": [0]})), \
                 patch.object(e, "load_dotenv_file"), \
                 patch.object(e.requests, "post", side_effect=AssertionError("no live call")), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["eval_three_arms.py", "--arms", "C", "--llm-backends", "muse_glimmer_30b", "--out-json", str(out)]):
                    with self.assertRaises(SystemExit) as raised:
                        e.main()
            self.assertNotEqual(raised.exception.code, 0)

    def test_incomplete_repeat_has_nonzero_exit_and_omits_means(self):
        df = frame(2)
        ok_items = [
            {"task_id": 0, "gold": "other", "prediction": "other", "confidence": 1.0, "latency_ms": 1.0, "raw_reply": "ok", "error": None},
            {"task_id": 1, "gold": "other", "prediction": "other", "confidence": 1.0, "latency_ms": 1.0, "raw_reply": "ok", "error": None},
        ]
        complete_arm = {"arm": "C_muse_glimmer_30b", "backend_id": "muse_glimmer_30b", "status": "ok", "n_scored": 2, "items": ok_items, "n_correct": 2, "acc_hard": 1.0, "acc_conf": 1.0, "mean_top_choice_confidence": 1.0, "macro_f1": 1.0, "latency_ms": 1.0, "errors": 0, "error_rate": 0.0}
        partial_arm = {"arm": "C_muse_glimmer_30b", "backend_id": "muse_glimmer_30b", "status": "partial", "n_scored": 1, "items": ok_items[:1], "n_correct": 1, "acc_hard": 0.5, "acc_conf": 0.5, "mean_top_choice_confidence": 0.5, "macro_f1": 0.0, "latency_ms": 1.0, "errors": 0, "error_rate": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            with patch.object(e, "run_llm_backend", side_effect=[complete_arm, partial_arm]), \
                 patch.object(e, "load_data", return_value=(df, {"train": [], "test": list(df.index)})), \
                 patch.object(e, "load_dotenv_file"), \
                 patch.object(e, "check_requested_configuration"), \
                 patch.object(e, "build_fewshot_msgs", return_value=[]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["eval_three_arms.py", "--arms", "C", "--llm-backends", "muse_glimmer_30b", "--times", "2", "--llm-max-n", "2", "--out-json", str(out)]):
                    with self.assertRaises(SystemExit) as raised:
                        e.main()
            self.assertNotEqual(raised.exception.code, 0)
            self.assertFalse(out.with_suffix(".md").exists())
            payload = json.loads(out.read_text())
            self.assertFalse(payload["complete"])
            self.assertIsNone(payload["mean_metrics"])
            self.assertEqual(len(payload["runs"]), 2)
            self.assertTrue(payload["runs"][0]["inference_performed"])
            self.assertFalse(payload["runs"][1]["inference_performed"])
            self.assertEqual(payload["runs"][0]["arms"][0]["items"][0]["prediction"], "other")
            self.assertEqual(len(payload["runs"][1]["arms"][0]["items"]), 1)
            self.assertFalse(payload["provenance"]["inference_performed"])

    def test_cuda_a_session_is_used_without_real_inference(self):
        def fake_session(times):
            runs = []
            for i in range(times):
                items = [{"task_id": 0, "gold": "other", "prediction": "other", "confidence": 0.4, "latency_ms": 1.0, "raw_reply": None, "error": None, "inference_batch_size": 1, "inference_sequence": 0, "prediction_source": "single-row-native-cuda"}]
                scored = score_records(items)
                arm = {"arm": "A_cuda_logreg", "backend_id": "cuml_native", "status": "ok", "items": items, "n_scored": 1, **scored, "latency_ms": 1.0}
                runs.append({"arm": arm, "provenance": {"run_id": "run%d" % (i + 1), "invocation_id": "synth-%d" % i, "started_at": "t", "completed_at": "t", "inference_performed": True}})
            return runs
        original = __import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"cupy", "cupyx", "cuml", "rmm"}:
                raise AssertionError("CUDA import in mocked A test: " + name)
            return original(name, *args, **kwargs)
        df = frame(1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            with patch.object(e, "run_official_arm_a_session", side_effect=fake_session) as mocked, \
                 patch.object(e, "load_data", return_value=(df, {"train": [], "test": list(df.index)})), \
                 patch.object(e, "load_dotenv_file"), \
                 patch.object(e, "check_requested_configuration"), \
                 patch("builtins.__import__", side_effect=guarded), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["eval_three_arms.py", "--arms", "A", "--times", "2", "--out-json", str(out)]):
                    e.main()
            self.assertEqual(mocked.call_args.args, (2,))
            payload = json.loads(out.read_text())
            self.assertEqual([run["invocation_id"] for run in payload["runs"]], ["synth-0", "synth-1"])
            self.assertTrue(payload["complete"])
            self.assertFalse(out.with_suffix(".md").exists())

    def test_markdown_flag_is_rejected(self):
        with patch("sys.argv", ["eval_three_arms.py", "--arms", "C", "--llm-backends", "muse_glimmer_30b", "--out-md", "x.md"]), \
             patch.object(e, "load_dotenv_file"), \
             patch.object(e, "load_data", return_value=(frame(1), {"train": [], "test": [0]})), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                e.main()
        self.assertNotEqual(raised.exception.code, 0)


    def test_default_out_json_is_per_target(self):
        df = frame(1)
        arm = {"arm": "C_muse_glimmer_30b", "backend_id": "muse_glimmer_30b", "status": "ok", "n_scored": 1,
               "items": [{"task_id": 0, "gold": "other", "prediction": "other", "confidence": 1.0, "latency_ms": 1.0, "raw_reply": "ok", "error": None}],
               "n_correct": 1, "acc_hard": 1.0, "acc_conf": 1.0, "mean_top_choice_confidence": 1.0, "macro_f1": 1.0, "latency_ms": 1.0, "errors": 0, "error_rate": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            with patch.object(e, "RESULTS", models), \
                 patch.object(e, "run_llm_backend", return_value=arm), \
                 patch.object(e, "load_data", return_value=(df, {"train": [], "test": list(df.index)})), \
                 patch.object(e, "load_dotenv_file"), \
                 patch.object(e, "check_requested_configuration"), \
                 patch.object(e, "build_fewshot_msgs", return_value=[]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["eval_three_arms.py", "--arms", "C", "--llm-backends", "muse_glimmer_30b", "--llm-max-n", "1"]):
                    e.main()
            path = models / "eval_times_C_muse_glimmer_30b.json"
            self.assertTrue(path.is_file())
            self.assertFalse((models / "eval_multi_arms.json").exists())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["times"], 1)
            self.assertTrue(payload["complete"])

    def test_missing_or_multiple_targets_write_nothing(self):
        cases = [
            [],
            ["--arms", "A,B"],
            ["--arms", "B"],
            ["--arms", "C"],
            ["--arms", "C", "--llm-backends", "qwen38_27b,muse_glimmer_30b"],
            ["--dry-run"],
            ["--arms", "B", "--dry-run"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as tmp:
                models = Path(tmp)
                with patch.object(e, "RESULTS", models), \
                     patch.object(e, "load_dotenv_file"), \
                     patch.dict(os.environ, {"DECISION_BACKENDS": "", "LLM_BACKENDS": ""}, clear=False), \
                     patch.object(e, "load_data", side_effect=AssertionError("must not load data")), \
                     patch.object(e, "write_times_output", side_effect=AssertionError("must not write official JSON")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", ["eval_three_arms.py"] + argv):
                        with self.assertRaises(SystemExit) as raised:
                            e.main()
                self.assertNotEqual(raised.exception.code, 0)
                self.assertEqual(list(models.glob("*.json")), [])

        # A single env backend list must not stand in for an omitted CLI model.
        for argv in ([], ["--arms", "B"], ["--arms", "C"], ["--arms", "C", "--dry-run"]):
            with self.subTest(env_default=argv), tempfile.TemporaryDirectory() as tmp:
                models = Path(tmp)
                with patch.object(e, "RESULTS", models), \
                     patch.object(e, "load_dotenv_file"), \
                     patch.dict(os.environ, {"DECISION_BACKENDS": "clef", "LLM_BACKENDS": "lfm25_24b_a2b"}, clear=False), \
                     patch.object(e, "load_data", side_effect=AssertionError("must not load data")), \
                     patch.object(e, "write_times_output", side_effect=AssertionError("must not write official JSON")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", ["eval_three_arms.py"] + argv):
                        with self.assertRaises(SystemExit) as raised:
                            e.main()
                self.assertNotEqual(raised.exception.code, 0)
                self.assertEqual(list(models.glob("*.json")), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
