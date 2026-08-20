"""Flagship-trio tests: dynamic supervision routing, smart memory
(scored eviction + consolidation + insight injection), and the
review=False local-first supervise path.

Run from hybrid-agent/:
    python -m unittest tests.test_router_memory -v
"""

import sys
import tempfile
import time
import types
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelResponse  # noqa: E402
from memory import TaskMemory, TaskRecord  # noqa: E402
from router.confidence import MemoryView  # noqa: E402
from supervise import supervise  # noqa: E402


def _args(**kw):
    base = {"router": "auto"}
    base.update(kw)
    return Namespace(**base)


def _cfg(**kw):
    cfg = {"router": {"supervision": "auto", "local_threshold": 0.85}}
    cfg.update(kw)
    return cfg


class TestPlanSupervision(unittest.TestCase):
    def test_override_flag_wins(self):
        level, reason = ask._plan_supervision(
            None, _cfg(), _args(router="critical"),
            "fix the typo", "", MemoryView())
        self.assertEqual(level, "critical")
        self.assertEqual(reason, "override:critical")

    def test_config_override(self):
        cfg = _cfg(router={"supervision": "full"})
        level, _ = ask._plan_supervision(
            None, cfg, _args(), "any task", "", MemoryView())
        self.assertEqual(level, "full")

    def test_critical_markers_force_deepseek(self):
        level, reason = ask._plan_supervision(
            None, _cfg(), _args(),
            "Design the security architecture for the auth module", "",
            MemoryView())
        self.assertEqual(level, "critical")
        self.assertIn("critical", reason)

    def test_trivial_markers_skip_review(self):
        level, _ = ask._plan_supervision(
            None, _cfg(), _args(), "fix typo in the readme", "", MemoryView())
        self.assertEqual(level, "local_first")

    def test_default_with_no_history_is_full(self):
        level, reason = ask._plan_supervision(
            None, _cfg(), _args(), "update the checkout flow", "", MemoryView())
        self.assertEqual(level, "full")
        self.assertEqual(reason, "default")

    def test_memory_similarity_alone_can_skip_review(self):
        mem = MemoryView(similar_task_success_rate=0.85,
                         seen_ngrams=frozenset([("a", "b", "c")]))
        level, reason = ask._plan_supervision(
            None, _cfg(), _args(), "fix login bug", "", mem)
        self.assertEqual(level, "local_first")
        self.assertEqual(reason, "memory_similar_tasks")

    def test_router_local_with_history_skips_review(self):
        agent = types.SimpleNamespace(
            route=lambda task, context_chars, memory: ("local", "confidence:0.90"))
        mem = MemoryView(similar_task_success_rate=0.9,
                         seen_ngrams=frozenset([("a", "b", "c")]))
        level, _ = ask._plan_supervision(
            agent, _cfg(), _args(), "update the checkout flow", "", mem)
        self.assertEqual(level, "local_first")

    def test_router_deepseek_archetype_is_critical(self):
        agent = types.SimpleNamespace(
            route=lambda task, context_chars, memory: ("deepseek", "archetype:architecture"))
        mem = MemoryView(similar_task_success_rate=0.9,
                         seen_ngrams=frozenset([("a", "b", "c")]))
        level, _ = ask._plan_supervision(
            agent, _cfg(), _args(), "design the payment system", "", mem)
        self.assertEqual(level, "critical")

    def test_budget_pressure_degrades_to_local(self):
        class _FakeStats:
            def __init__(self):
                self.stats = {"api_tokens": {}}  # unused
            def daily_api_tokens(self):
                return 90

        backup = ask.StatsTracker
        ask.StatsTracker = lambda **kw: _FakeStats()
        try:
            cfg = _cfg(review={"daily_token_budget": 100})
            level, reason = ask._plan_supervision(
                None, cfg, _args(), "update the checkout flow", "", MemoryView())
        finally:
            ask.StatsTracker = backup
        self.assertEqual(level, "local_first")
        self.assertIn("budget_pressure", reason)


class TestScoredEviction(unittest.TestCase):
    def test_frequency_beats_recency_when_evicting(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp, max_records=3)
            now = time.time()

            def rec(task, age_days):
                return {"task": task, "ts": now - age_days * 86400,
                        "route": "local", "verdict": "APPROVED", "quality": 7.0}

            records = [
                rec("hot", 20),   # recurs 3x but old
                rec("hot", 19),
                rec("hot", 18),
                rec("fresh", 0.04),   # new, single-shot
                rec("middling", 2),   # newer than two "hot"s, single-shot
            ]
            kept = [r["task"] for r in mem._evict(records)]
            self.assertEqual(len(kept), 3)
            self.assertEqual(kept.count("hot"), 2)   # frequency preserved
            self.assertIn("fresh", kept)              # recency preserved
            self.assertNotIn("middling", kept)        # one-shot + old -> forgotten

    def test_evict_keeps_order_and_noop_under_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp, max_records=5)
            records = [{"task": f"t{i}", "ts": 0.0, "route": "local",
                        "verdict": "APPROVED", "quality": 0.0} for i in range(3)]
            self.assertEqual(mem._evict(records), records)


class TestConsolidation(unittest.TestCase):
    def _seed(self, mem):
        now = time.time()
        for i in range(12):
            task = (["fix login auth bug", "add auth endpoint validation",
                     "refactor checkout api", "write tests for auth",
                     "update docker config", "fix password reset flow",
                     "add session token handling", "reorganize api routes",
                     "fix typo in docs", "add react login component",
                     "secure the payment endpoint", "rename checkout module"]
                    [i % 12])
            verdict = "APPROVED" if i % 3 != 0 else "FIX_REQUIRED"  # 8 approved / 4 fixed
            mem.record(TaskRecord(task=task, ts=now - i * 3600, route="local",
                                  verdict=verdict, quality=7.5))
        # Last 10: 7 approved -> 0.7; prior 2: 1 approved -> 0.5 -> improving.

    def test_consolidate_produces_insights(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)
            self._seed(mem)
            data = mem.consolidate()
            self.assertEqual(data["records"], 12)
            self.assertAlmostEqual(data["overall_approval"], round(8 / 12, 2), places=2)
            self.assertEqual(data["trend"], "improving")
            self.assertTrue(data["domains"])  # backend/auth buckets detected

    def test_consolidate_requires_min_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)
            mem.record(TaskRecord(task="one off", ts=time.time(), route="local",
                                  verdict="APPROVED"))
            self.assertEqual(mem.consolidate(), {})

    def test_insights_text_injects(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)
            self._seed(mem)
            text = mem.insights_text()
            self.assertIn("TASK MEMORY", text)
            self.assertIn("approval", text)
            self.assertIn("strongest areas", text)
            # Second call is served from the cached insights file.
            self.assertEqual(text, mem.insights_text())

    def test_insights_text_empty_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(TaskMemory(root=tmp).insights_text(), "")


class TestLocalFirstSupervise(unittest.TestCase):
    def test_review_false_skips_deepseek(self):
        calls = {"qwen": 0, "cloud": 0}

        def fake_qwen(req):
            calls["qwen"] += 1
            return ModelResponse(text="```\nfile.py\nprint('ok')\n```",
                                 backend="local")

        class _FakeCloud:
            def generate(self, req):
                calls["cloud"] += 1
                raise AssertionError("cloud must not be called")

        result = supervise(
            local=None, cloud=_FakeCloud(), task="rename a variable",
            qwen_generate=fake_qwen, review=False, status=lambda line: None)
        self.assertEqual(calls["qwen"], 1)
        self.assertEqual(calls["cloud"], 0)
        self.assertEqual(result.reason, "router_local_skip_review")
        self.assertEqual(result.verdicts[0].decision, "APPROVED")
        self.assertIn("file.py", result.final_text)

    def test_review_hint_becomes_quality_score(self):
        result = supervise(
            local=None, cloud=object(), task="t",
            qwen_generate=lambda req: ModelResponse(text="ok", backend="local"),
            review=False, review_quality_hint=8.5, status=lambda line: None)
        self.assertEqual(result.verdicts[0].quality_score, 8.5)


class TestMemoryCLIReport(unittest.TestCase):
    """ask.py --memory / --consolidate report helper (offline)."""

    def _seed(self, tmp):
        mem = TaskMemory(root=tmp)
        now = time.time()
        for i in range(12):
            mem.record(TaskRecord(
                task=f"fix auth bug {i}", ts=now - i * 3600, route="local",
                verdict="APPROVED" if i % 3 else "FIX_REQUIRED"))
        return mem

    def test_report_shows_records_and_insights(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._seed(tmp)
            report = ask._memory_report({"memory": {"root": tmp}})
            self.assertEqual(report["count"], 12)
            self.assertIn("TASK MEMORY", report["insights"])
            # Newest first.
            self.assertEqual(report["records"][0]["task"], "fix auth bug 11")
            self.assertIn("fix auth bug 0", report["records"][-1]["task"])
            self.assertTrue(report["path"].endswith("tasks.json"))

    def test_report_force_consolidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._seed(tmp)
            report = ask._memory_report({"memory": {"root": tmp}}, force=True)
            self.assertIn("TASK MEMORY", report["insights"])
            # The insights file now exists (cached).
            self.assertTrue((Path(tmp) / "insights.json").is_file())

    def test_report_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = ask._memory_report({"memory": {"root": tmp}})
            self.assertEqual(report["count"], 0)
            self.assertEqual(report["records"], [])
            self.assertEqual(report["insights"], "")


if __name__ == "__main__":
    unittest.main()
