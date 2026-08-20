"""Tier 2 tests: review-verdict cache wrapper and parallel file-conflict
detection / serialization.

Run from hybrid-agent/:
    python -m unittest tests.test_tier2 -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelRequest, ModelResponse  # noqa: E402


class _FakeCache:
    """Duck-typed CacheManager interface; never touches the real stats.json."""

    def __init__(self):
        self.store = {}
        self.enabled = True
        self.hits = 0
        self.misses = 0

    def key(self, kind, *parts):
        return kind + "|" + "\x1f".join(parts)

    def get(self, kind, key):
        return self.store.get((kind, key))

    def set(self, kind, key, text, source="deepseek"):
        self.store[(kind, key)] = text

    def record(self, hit):
        if hit:
            self.hits += 1
        else:
            self.misses += 1


class _FakeCloud:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response or ModelResponse(text="VERDICT", backend="deepseek")

    def generate(self, req):
        self.calls += 1
        return self.response


class TestReviewVerdictCache(unittest.TestCase):
    """Cached cloud: identical packages short-circuit the API; verdicts reuse."""

    def test_identical_request_hits_cache_once(self):
        cache = _FakeCache()
        cloud = _FakeCloud()
        wrapped = ask._cached_cloud(cloud, cache)
        req = ModelRequest(system="sys", user="pkg content")
        first = wrapped.generate(req)
        second = wrapped.generate(req)
        self.assertEqual(first.text, "VERDICT")
        self.assertEqual(second.text, "VERDICT")
        self.assertEqual(second.backend, "cache")
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_different_package_does_not_hit(self):
        cache = _FakeCache()
        cloud = _FakeCloud()
        wrapped = ask._cached_cloud(cloud, cache)
        wrapped.generate(ModelRequest(system="s", user="pkg A"))
        wrapped.generate(ModelRequest(system="s", user="pkg B"))
        self.assertEqual(cloud.calls, 2)

    def test_truncated_response_is_not_cached(self):
        cache = _FakeCache()
        cloud = _FakeCloud(response=ModelResponse(text="cut off", truncated=True))
        wrapped = ask._cached_cloud(cloud, cache)
        wrapped.generate(ModelRequest(system="s", user="pkg"))
        wrapped.generate(ModelRequest(system="s", user="pkg"))
        self.assertEqual(cloud.calls, 2)  # nothing cached -> both hit the API
        self.assertEqual(cache.misses, 2)

    def test_disabled_cache_passthrough(self):
        cache = _FakeCache()
        cache.enabled = False
        cloud = _FakeCloud()
        wrapped = ask._cached_cloud(cloud, cache)
        self.assertIs(wrapped, cloud)  # no wrapper at all

    def test_none_cache_passthrough(self):
        cloud = _FakeCloud()
        self.assertIs(ask._cached_cloud(cloud, None), cloud)


class TestParallelConflictDetection(unittest.TestCase):
    def test_detect_overlapping_files(self):
        steps = [
            {"id": 1, "files": ["a.py", "b.py"]},
            {"id": 2, "files": ["b.py", "c.py"]},
            {"id": 3, "files": ["d.py"]},
        ]
        self.assertEqual(ask._detect_step_conflicts(steps), {"b.py"})

    def test_no_conflict_returns_empty(self):
        steps = [{"id": 1, "files": ["a.py"]}, {"id": 2, "files": ["b.py"]}]
        self.assertEqual(ask._detect_step_conflicts(steps), set())

    def test_plan_group_runs_serializes_conflicted_steps(self):
        group = [
            {"id": 1, "files": ["a.py", "shared.py"]},
            {"id": 2, "files": ["shared.py"]},
            {"id": 3, "files": ["c.py"]},
        ]
        ok, conf = ask._plan_group_runs(group)
        self.assertEqual([s["id"] for s in ok], [3])
        self.assertEqual([s["id"] for s in conf], [1, 2])  # id order, not input order

    def test_plan_group_runs_no_conflict_parallelizes_all(self):
        group = [{"id": 1, "files": ["a.py"]}, {"id": 2, "files": ["b.py"]}]
        ok, conf = ask._plan_group_runs(group)
        self.assertEqual([s["id"] for s in ok], [1, 2])
        self.assertEqual(conf, [])

    def test_warn_output_overlaps(self):
        captured = []
        original = ask._status
        ask._status = lambda line: captured.append(line)
        try:
            ask._warn_output_overlaps([
                {"id": 1, "status": "success",
                 "text": "a.py\n```py\nx = 1\n```"},
                {"id": 2, "status": "success",
                 "text": "a.py\n```py\ny = 2\n```"},
                {"id": 3, "status": "failed", "error": "boom"},
            ])
        finally:
            ask._status = original
        self.assertTrue(any("post-hoc conflict" in line and "'a.py'" in line
                            for line in captured))

    def test_warn_output_overlaps_silent_without_overlap(self):
        captured = []
        original = ask._status
        ask._status = lambda line: captured.append(line)
        try:
            ask._warn_output_overlaps([
                {"id": 1, "status": "success", "text": "a.py\n```py\nx\n```"},
                {"id": 2, "status": "success", "text": "b.py\n```py\ny\n```"},
            ])
        finally:
            ask._status = original
        self.assertFalse(any("post-hoc conflict" in line for line in captured))


if __name__ == "__main__":
    unittest.main()
