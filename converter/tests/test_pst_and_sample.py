"""Tests for sampling, PST-only public input, and PST folder/file discovery.

Does not touch official gold labeled data. PST extraction is tested
with a mock readpst binary (no real mailbox dump is written).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter.pipeline import convert, process_messages, sample_tasks  # noqa: E402
from converter.parse_pst import (  # noqa: E402
    find_readpst,
    iter_pst_paths,
    parse_pst_file,
)
from converter.message import MailMessage  # noqa: E402


FIXTURES = ROOT / "fixtures"


class SampleTasksTests(unittest.TestCase):
    def test_sample_is_deterministic_with_seed_2340(self):
        tasks = [{"id": str(i), "data": {"body": f"b{i}"}} for i in range(10)]
        a = sample_tasks(tasks, sample=4, seed=2340)
        b = sample_tasks(tasks, sample=4, seed=2340)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        # Original relative order is preserved
        ids = [int(t["id"]) for t in a]
        self.assertEqual(ids, sorted(ids))
        # A different seed can differ
        c = sample_tasks(tasks, sample=4, seed=1)
        self.assertEqual(len(c), 4)

    def test_undersized_pool_keeps_all_and_warns(self):
        tasks = [{"id": "a"}, {"id": "b"}]
        with self.assertWarns(UserWarning) as ctx:
            out = sample_tasks(tasks, sample=2000, seed=2340)
        self.assertEqual(out, tasks)
        self.assertIn("Keeping all", str(ctx.warning))

    def test_sample_none_keeps_all(self):
        tasks = [{"id": "a"}]
        self.assertEqual(sample_tasks(tasks, sample=None, seed=2340), tasks)


class PublicInputBoundaryTests(unittest.TestCase):
    def test_rejects_eml_mbox_olm_and_non_pst_folder(self):
        cases = [
            FIXTURES / "01_promo.eml",
            FIXTURES / "samples.mbox",
            FIXTURES,  # directory of .eml/.mbox, no .pst
        ]
        olm = FIXTURES / "tiny_sample.olm"
        if olm.is_file():
            cases.append(olm)
        for src in cases:
            with self.subTest(src=str(src)), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "tasks.json"
                with self.assertRaises(ValueError) as ctx:
                    convert(src, out)
                msg = str(ctx.exception).lower()
                self.assertIn(".pst", msg)
                self.assertTrue(
                    ".eml" in msg or "mbox" in msg or ".olm" in msg or "not accepted" in msg
                )
                self.assertFalse(out.exists())

    def test_folder_with_pst_and_eml_parses_only_pst(self):
        eml = """From: Announcer <news@example.com>
To: student@example.com
Subject: Welcome week schedule
Message-ID: <boundary-pst-1@example.com>
Date: 1 Jan 2026 00:00:00 +0000

Hello, please find the welcome week schedule attached in spirit only.
This is long enough to pass the English-language filter easily.
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock = tmp_path / "readpst"
            canned = tmp_path / "canned.eml"
            canned.write_text(eml)
            _write_executable(
                mock,
                "#!/bin/bash\n"
                "set -e\n"
                "outdir=\n"
                "while [ $# -gt 0 ]; do\n"
                '  if [ "$1" = "-o" ]; then outdir=$2; shift 2; continue; fi\n'
                "  shift\n"
                "done\n"
                'mkdir -p "$outdir/Inbox"\n'
                f'cp "{canned}" "$outdir/Inbox/0.eml"\n',
            )
            folder = tmp_path / "mixed"
            folder.mkdir()
            (folder / "mail.pst").write_bytes(b"not-a-real-pst")
            (folder / "ignore.eml").write_text("From: x\nSubject: should not be read\n\nbody\n")
            out = tmp_path / "tasks.json"
            os.environ["READPST"] = str(mock)
            try:
                tasks = convert(folder, out, sample=2000, seed=2340)
            finally:
                os.environ.pop("READPST", None)
            self.assertEqual(len(tasks), 1)
            self.assertIn("welcome week", tasks[0]["data"]["subject"].lower())


class PstDiscoveryTests(unittest.TestCase):
    def test_single_file_and_nested_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one.pst"
            nested = root / "sub" / "two.PST"
            nested.parent.mkdir()
            one.write_bytes(b"fake")
            nested.write_bytes(b"fake")
            (root / "note.txt").write_text("x")
            self.assertEqual(iter_pst_paths(one), [one.resolve()])
            found = iter_pst_paths(root)
            self.assertEqual(found, [one.resolve(), nested.resolve()])


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class MockReadpstTests(unittest.TestCase):
    def test_parse_pst_file_with_mock_readpst(self):
        eml = """From: Announcer <news@example.com>
To: student@example.com
Subject: Welcome week schedule
Message-ID: <mock-pst-1@example.com>
Date: 1 Jan 2026 00:00:00 +0000

Hello, please find the welcome week schedule attached in spirit only.
This is long enough to pass the English-language filter easily.
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock = tmp_path / "readpst"
            # The mock copies a canned .eml into the -o directory.
            canned = tmp_path / "canned.eml"
            canned.write_text(eml)
            _write_executable(
                mock,
                "#!/bin/bash\n"
                "set -e\n"
                "outdir=\n"
                "while [ $# -gt 0 ]; do\n"
                '  if [ "$1" = "-o" ]; then outdir=$2; shift 2; continue; fi\n'
                "  shift\n"
                "done\n"
                'mkdir -p "$outdir/Inbox"\n'
                f'cp "{canned}" "$outdir/Inbox/0.eml"\n',
            )
            pst = tmp_path / "mail.pst"
            pst.write_bytes(b"not-a-real-pst")
            messages = parse_pst_file(pst, readpst_bin=mock)
            self.assertEqual(len(messages), 1)
            self.assertIn("welcome week", messages[0].subject.lower())
            self.assertEqual(messages[0].source, str(pst.resolve()))

            out = tmp_path / "tasks.json"
            os.environ["READPST"] = str(mock)
            try:
                # Folder of one .pst
                folder = tmp_path / "psts"
                folder.mkdir()
                (folder / "mail.pst").write_bytes(b"not-a-real-pst")
                # convert() uses parse_pst_file() which calls find_readpst()
                # via READPST when no explicit binary is passed.
                tasks = convert(folder, out, sample=2000, seed=2340)
            finally:
                os.environ.pop("READPST", None)
            # Undersized pool: one English message, warn and keep all
            self.assertEqual(len(tasks), 1)
            self.assertTrue(out.is_file())

    def test_find_readpst_missing_raises(self):
        env = os.environ.pop("READPST", None)
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                find_readpst(explicit=Path("/tmp/definitely-not-readpst-xyz"))
            self.assertIn("not executable", str(ctx.exception).lower() + str(ctx.exception))
        finally:
            if env is not None:
                os.environ["READPST"] = env


if __name__ == "__main__":
    unittest.main()
