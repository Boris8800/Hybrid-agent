"""Regression tests for the --apply parser/guard, parse_enhancement's
clarify heuristic, the response CacheManager, TaskMemory, and the daily token
budget accounting.

Run from hybrid-agent/:
    python -m unittest tests.test_apply_parser -v
"""

import os
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from memory import TaskMemory, TaskRecord  # noqa: E402
from supervise import parse_enhancement  # noqa: E402


class TestParseFencedFiles(unittest.TestCase):
    """FIX 1 regression: bare language tags and comment lines are never paths."""

    def paths(self, text):
        return [p for p, _ in ask._parse_fenced_files(text)]

    def test_bare_json_tag_is_ignored(self):
        self.assertEqual(self.paths('```json\n{"a": 1}\n```'), [])

    def test_path_on_line_before_fence(self):
        self.assertEqual(
            self.paths('package.json\n```json\n{"a": 1}\n```'),
            ["package.json"])

    def test_lang_colon_path_label(self):
        self.assertEqual(
            self.paths('```json:config/app.json\n{"a": 1}\n```'),
            ["config/app.json"])

    def test_lang_space_path_label(self):
        self.assertEqual(
            self.paths('```json src/lib.js\ncode\n```'),
            ["src/lib.js"])

    def test_path_as_first_line_inside_block(self):
        self.assertEqual(
            self.paths('```\nconfig/app.json\n{"a": 1}\n```'),
            ["config/app.json"])

    def test_html_comment_before_fence_is_ignored(self):
        self.assertEqual(
            self.paths('<!-- do not touch -->\n```json\n{"a": 1}\n```'), [])

    def test_bare_tag_after_another_fence_is_ignored(self):
        self.assertEqual(
            self.paths('```py\nx = 1\n```\n```json\n{"a": 1}\n```'), [])

    def test_dedup_parser_still_returns_duplicates_for_apply_guard(self):
        text = ('a.js\n```js\nA\n```\na.js\n```js\nB\n```')
        self.assertEqual(self.paths(text), ["a.js", "a.js"])


class TestApplyFencedFiles(unittest.TestCase):
    """FIX 2 regression: duplicate blocks never overwrite; unsafe paths blocked."""

    def _tmp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_normal_write(self):
        root = self._tmp()
        written, skipped = ask._apply_fenced_files(
            'a.txt\n```txt\nhello\n```', root=root)
        self.assertEqual([rel for rel, _ in written], ["a.txt"])
        self.assertEqual(skipped, [])
        self.assertEqual((Path(root) / "a.txt").read_text(), "hello\n")

    def test_duplicate_block_does_not_overwrite(self):
        root = self._tmp()
        text = ('a.txt\n```txt\nFIRST\n```\na.txt\n```txt\nSECOND\n```')
        written, skipped = ask._apply_fenced_files(text, root=root)
        self.assertEqual(len(written), 1)
        self.assertEqual(skipped, ["a.txt (duplicate block would overwrite)"])
        self.assertEqual((Path(root) / "a.txt").read_text(), "FIRST\n")

    def test_escapes_root_is_blocked(self):
        root = self._tmp()
        _, skipped = ask._apply_fenced_files(
            '../../etc/passwd\n```txt\nx\n```', root=root)
        self.assertEqual(skipped, ["../../etc/passwd (unsafe path)"])

    def test_absolute_path_is_blocked(self):
        root = self._tmp()
        _, skipped = ask._apply_fenced_files(
            '/etc/hostname\n```txt\nx\n```', root=root)
        self.assertEqual(skipped, ["/etc/hostname (unsafe path)"])


class TestClarifyHeuristic(unittest.TestCase):
    """FIX 3 regression: 'no questions needed' never aborts as TASK UNCLEAR."""

    def _enh(self, cq_section):
        raw = (
            "=== ENHANCED PROMPT ===\nDo the thing.\n"
            "=== PLAN ===\nStep 1\n"
            f"=== CLARIFYING QUESTIONS ===\n{cq_section}"
        )
        return parse_enhancement(raw)

    def test_no_clarifying_questions_needed_is_cleared(self):
        enh = self._enh("No clarifying questions needed — the task is clear.")
        self.assertEqual(enh.clarifying_questions, "")

    def test_no_questions_wording_is_cleared(self):
        enh = self._enh("No questions — none needed, already clear.")
        self.assertEqual(enh.clarifying_questions, "")

    def test_real_questions_are_kept(self):
        enh = self._enh("1. Which database?\n2. What auth flow?")
        self.assertIn("Which database", enh.clarifying_questions)

    def test_missing_section_is_empty(self):
        enh = parse_enhancement("=== ENHANCED PROMPT ===\nDo the thing.")
        self.assertEqual(enh.clarifying_questions, "")


class TestCacheManager(unittest.TestCase):
    def _cm(self, tmp, **cache_cfg):
        cfg = {"cache": {"enabled": True, "dir": str(tmp), "ttl_days": 7,
                         "max_entries": 100, **cache_cfg}}
        args = Namespace(no_cache=False, cache_ttl=0, cache_max_size=0)
        return ask.CacheManager(cfg, args)

    def test_set_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._cm(tmp)
            cm.set("gen", "k1", "hello")
            self.assertEqual(cm.get("gen", "k1"), "hello")
            self.assertIsNone(cm.get("gen", "missing"))

    def test_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._cm(tmp, ttl_days=1)
            cm.set("gen", "k1", "hello")
            path = Path(tmp) / "gen" / "k1.json"
            old = time.time() - 2 * 86400
            os.utime(path, (old, old))
            self.assertIsNone(cm.get("gen", "k1"))
            self.assertFalse(path.exists())  # stale entry pruned on read

    def test_max_entries_cap_prunes_oldest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._cm(tmp, max_entries=2)
            base = time.time()
            for i, k in enumerate(("k1", "k2", "k3")):
                cm.set("gen", k, f"text-{k}")
                os.utime(Path(tmp) / "gen" / f"{k}.json", (base + i, base + i))
            self.assertEqual(len(list((Path(tmp) / "gen").glob("*.json"))), 2)
            self.assertIsNone(cm.get("gen", "k1"))   # oldest pruned
            self.assertEqual(cm.get("gen", "k3"), "text-k3")

    def test_disabled_cache_never_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._cm(tmp)
            cm.enabled = False
            cm.set("gen", "k1", "hello")
            self.assertIsNone(cm.get("gen", "k1"))


class TestTaskMemory(unittest.TestCase):
    def test_memory_view_uses_real_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)
            mem.record(TaskRecord(task="fix login bug in auth", ts=1.0,
                                  route="local", verdict="APPROVED", quality=8.0))
            mem.record(TaskRecord(task="add validation to checkout", ts=2.0,
                                  route="deepseek", verdict="REJECTED"))
            view = mem.memory_view("fix login bug")
            self.assertEqual(view.similar_task_success_rate, 1.0)
            self.assertTrue(view.has_history)
            # Novelty has real n-grams to compare against.
            self.assertIn(("fix", "login", "bug"), view.seen_ngrams)

    def test_no_matching_history_means_no_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)
            mem.record(TaskRecord(task="fix login bug in auth", ts=1.0,
                                  route="local", verdict="APPROVED"))
            view = mem.memory_view("build a totally unrelated spaceship")
            self.assertEqual(view.similar_task_success_rate, 0.0)
            self.assertFalse(view.has_history)
            self.assertNotEqual(view.seen_ngrams, frozenset())  # but seen set is real

    def test_cap_bounds_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp, max_records=3)
            for i in range(10):
                mem.record(TaskRecord(task=f"task {i}", ts=i, route="local",
                                      verdict="APPROVED"))
            self.assertEqual(mem.count(), 3)


class TestTokenBudget(unittest.TestCase):
    def test_record_and_daily_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = ask.StatsTracker(path=str(Path(tmp) / "stats.json"))
            stats.record_tokens("deepseek", prompt_tokens=100, completion_tokens=50)
            stats.record_tokens("deepseek", prompt_tokens=10, completion_tokens=0)
            self.assertEqual(stats.daily_api_tokens(), 160)

    def test_budget_exceeded_raises_from_generate_with_retry(self):
        # Cheap path check: _budgeted_cloud raises before any network call.
        class _FakeCloud:
            def generate(self, req):
                raise AssertionError("should not be reached")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"review": {"daily_token_budget": 100}}
            stats = ask.StatsTracker(path=str(Path(tmp) / "stats.json"))
            stats.record_tokens("deepseek", prompt_tokens=200)
            import types
            backup = ask.StatsTracker
            try:
                # Redirect StatsTracker to the temp file so the budget is seen.
                ask.StatsTracker = lambda **kw: stats
                wrapped = ask._budgeted_cloud(_FakeCloud(), cfg)
                with self.assertRaises(ask.BudgetExceeded):
                    wrapped.generate(None)
            finally:
                ask.StatsTracker = backup


if __name__ == "__main__":
    unittest.main()
