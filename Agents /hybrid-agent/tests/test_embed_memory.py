"""Tier A+B tests: semantic (embedding) memory recall, per-project memory
scoping, atomic writes, and config-key validation.

Run from hybrid-agent/:
    python -m unittest tests.test_embed_memory -v
"""

import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from embed import EmbeddingClient, cosine  # noqa: E402
from memory import TaskMemory, TaskRecord, memory_root_from_cfg  # noqa: E402
from router.confidence import MemoryView  # noqa: E402


def fake_embed(texts):
    """Deterministic fake embeddings: auth-ish -> [1,0,0], checkout-ish -> [0,1,0]."""
    out = []
    for t in texts:
        low = (t or "").lower()
        if "login" in low or "auth" in low:
            out.append([1.0, 0.0, 0.0])
        elif "checkout" in low or "cart" in low:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return out


class TestCosine(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_empty_and_mismatched_are_zero(self):
        self.assertEqual(cosine([], [1.0]), 0.0)
        self.assertEqual(cosine([1.0], [1.0, 2.0]), 0.0)


class TestEmbeddingClient(unittest.TestCase):
    def test_failure_returns_none_and_disables(self):
        import unittest.mock
        with unittest.mock.patch("embed.urllib.request.urlopen",
                                 side_effect=OSError("conn refused")):
            client = EmbeddingClient("http://localhost:1234/v1")
            self.assertIsNone(client.embed(["hello"]))
            self.assertTrue(client.disabled)
            # Second call short-circuits (no further network attempt).
            self.assertIsNone(client.embed(["again"]))

    def test_make_client_from_cfg_and_disable_flag(self):
        self.assertIsNone(ask_embed_factory({"memory": {"semantic_similarity": False}}))
        client = ask_embed_factory({"backends": {"local": {"base_url": "http://x/v1"}},
                                    "memory": {}})
        self.assertIsNotNone(client)
        self.assertIn("/embeddings", client.url)

    def test_memory_embed_callable_returns_bound_method_or_none(self):
        import embed
        self.assertIsNone(embed.memory_embed_callable(
            {"memory": {"semantic_similarity": False}}))
        fn = embed.memory_embed_callable(
            {"backends": {"local": {"base_url": "http://x/v1"}}, "memory": {}})
        self.assertTrue(callable(fn))
        self.assertIsNone(embed.memory_embed_callable({}))  # no backends


def ask_embed_factory(cfg):
    import embed
    return embed.make_embedding_client(cfg)


class TestSemanticMemoryView(unittest.TestCase):
    def _mem(self, tmp):
        return TaskMemory(root=tmp, embed=fake_embed)

    def test_paraphrased_task_is_recalled_semantically(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = self._mem(tmp)
            mem.record(TaskRecord(task="fix the login bug", ts=1.0, route="local",
                                  verdict="APPROVED", quality=8.0))
            mem.record(TaskRecord(task="update checkout cart flow", ts=2.0,
                                  route="deepseek", verdict="REJECTED"))
            # No trigram overlap with either record — trigram-only would score 0.
            view = mem.memory_view("resolve the authentication issue")
            self.assertEqual(view.similar_task_success_rate, 1.0)
            self.assertTrue(view.has_history)

    def test_unsimilar_task_has_no_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = self._mem(tmp)
            mem.record(TaskRecord(task="fix the login bug", ts=1.0, route="local",
                                  verdict="APPROVED"))
            view = mem.memory_view("deploy the docker image to the cluster")
            self.assertEqual(view.similar_task_success_rate, 0.0)
            self.assertFalse(view.has_history)

    def test_trigram_fallback_when_embed_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = TaskMemory(root=tmp)  # no embed -> trigram path
            mem.record(TaskRecord(task="fix login bug in auth", ts=1.0,
                                  route="local", verdict="APPROVED"))
            view = mem.memory_view("fix login bug")
            self.assertEqual(view.similar_task_success_rate, 1.0)
            self.assertTrue(view.has_history)

    def test_embedding_stored_on_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = self._mem(tmp)
            mem.record(TaskRecord(task="fix login bug", ts=1.0, route="local",
                                  verdict="APPROVED"))
            stored = json.loads((Path(tmp) / "tasks.json").read_text())
            self.assertEqual(stored[0]["embedding"], [1.0, 0.0, 0.0])


class TestMemoryScoping(unittest.TestCase):
    def test_explicit_root_wins(self):
        self.assertEqual(
            memory_root_from_cfg({"memory": {"root": "./custom"}}), "./custom")

    def test_autoscope_returns_memory_prefix(self):
        root = memory_root_from_cfg({})
        self.assertIsInstance(root, str)
        self.assertTrue(root.startswith("memory/"), root)


class TestConfigValidation(unittest.TestCase):
    def test_unknown_key_warns(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            ask._warn_unknown_config({
                "bogus_section": 1,
                "review": {"daily_toke_budget": 999},
            })
        err = buf.getvalue()
        self.assertIn("bogus_section", err)
        self.assertIn("review.daily_toke_budget", err)

    def test_known_keys_are_silent(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            ask._warn_unknown_config({
                "router": {"supervision": "auto", "local_threshold": 0.85},
                "memory": {"root": "./memory", "embedding_threshold": 0.75},
            })
        self.assertEqual(buf.getvalue(), "")


class TestAtomicWrites(unittest.TestCase):
    def test_stats_save_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = ask.StatsTracker(path=str(Path(tmp) / "stats.json"))
            stats.record_review("APPROVED", 8.0)
            data = json.loads((Path(tmp) / "stats.json").read_text())
            self.assertEqual(data["approvals"], 1)
            self.assertEqual(data["deepseek_reviews"], 1)
            self.assertFalse((Path(tmp) / "stats.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
