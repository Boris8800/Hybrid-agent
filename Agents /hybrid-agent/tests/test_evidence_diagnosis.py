"""Evidence & diagnosis tests: UNKNOWN verdicts, evidence-linked approval,
failure fingerprinting, known-solution memory, and loop detection.

Run from hybrid-agent/:
    python -m unittest tests.test_evidence_diagnosis -v
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelResponse  # noqa: E402
from learned_rules import LearnedRules, failure_fingerprint  # noqa: E402
from rules import load_constitution  # noqa: E402
from supervise import parse_verdict, supervise  # noqa: E402


class TestUnknownVerdict(unittest.TestCase):
    def test_approved_without_evidence_is_unknown(self):
        v = parse_verdict("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n8.0\n")
        self.assertEqual(v.decision, "UNKNOWN")

    def test_approved_with_evidence_is_approved(self):
        v = parse_verdict(
            "=== REVIEW DECISION ===\nAPPROVED\n=== EVIDENCE ===\n"
            "tests/test_booking.py::test_min_price passes; booking.service.ts:84\n")
        self.assertEqual(v.decision, "APPROVED")
        self.assertIn("test_min_price", v.evidence)

    def test_unknown_keyword(self):
        v = parse_verdict("=== REVIEW DECISION ===\nUNKNOWN\ninsufficient evidence\n")
        self.assertEqual(v.decision, "UNKNOWN")

    def test_fix_required_unchanged(self):
        v = parse_verdict("=== REVIEW DECISION ===\nFIX_REQUIRED\n")
        self.assertEqual(v.decision, "FIX_REQUIRED")


class TestFailureFingerprint(unittest.TestCase):
    def test_same_failure_same_id(self):
        e1 = "TS2345: error in src/auth.ts:84 - types are incompatible"
        e2 = "TS2345: error in src/auth.ts:84 - types are incompatible"
        self.assertEqual(failure_fingerprint(e1), failure_fingerprint(e2))

    def test_different_failure_different_id(self):
        a = failure_fingerprint("Cannot find module 'x' at src/a.ts:1")
        b = failure_fingerprint("TypeError at src/b.ts:40")
        self.assertNotEqual(a, b)

    def test_id_stable_across_whitespace(self):
        self.assertEqual(failure_fingerprint("TS 2345\n  src/a.ts:1\n   x"),
                         failure_fingerprint("TS2345 src/a.ts:1 x"))


class TestKnownSolution(unittest.TestCase):
    def test_lookup_after_minimum_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr = LearnedRules(root=tmp)
            err = "Cannot find module 'missing' at src/a.ts:5"
            lr.record_success(err, "src/a.ts\n```ts\nimport x\n```")
            self.assertIsNone(lr.lookup(err))  # below min
            lr.record_success(err, "src/a.ts\n```ts\nimport x\n```")
            sol = lr.lookup(err)
            self.assertIsNotNone(sol)
            self.assertIn("import x", sol[0])
            self.assertGreater(sol[1], 0.5)

    def test_different_failure_no_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr = LearnedRules(root=tmp)
            err = "TypeError at src/a.ts:9"
            lr.record_success(err, "src/a.ts\n```ts\nx\n```")
            lr.record_success(err, "src/a.ts\n```ts\nx\n```")
            self.assertIsNone(lr.lookup("TypeError at src/b.ts:99"))


class TestLoopDetection(unittest.TestCase):
    def _run(self, fixes):
        orig_v, orig_f = ask._run_verify_commands, ask._deepseek_fix
        orig_s = ask._git_snapshot
        seq = list(fixes)

        ask._run_verify_commands = lambda *a, **k: ["always failing"]
        ask._git_snapshot = lambda root: None

        def fake_fix(cloud, task, error_text, cache=None, root="."):
            return seq.pop(0) if seq else ""

        ask._deepseek_fix = fake_fix
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                ok, report = ask._run_final_verify(
                    object(), tmp, "t", ["npm test"], lambda line: None,
                    max_iter=3, rules=None)
        finally:
            ask._run_verify_commands, ask._deepseek_fix = orig_v, orig_f
            ask._git_snapshot = orig_s
        return ok, report

    def test_repeated_fix_detected(self):
        # The same file + same fix twice -> LOOP_DETECTED, no third attempt.
        ok, report = self._run(["a.ts\n```ts\nx = 1\n```",
                                "a.ts\n```ts\nx = 1\n```"])
        self.assertFalse(ok)
        self.assertIn("LOOP_DETECTED", report)


_APPROVED_EV = ("=== REVIEW DECISION ===\nAPPROVED\n=== EVIDENCE ===\n"
               "verify output passed\n=== QUALITY SCORE ===\n8.0\n")
_UNKNOWN_EV = "=== REVIEW DECISION ===\nUNKNOWN\nnot enough evidence\n"


class TestUnknownLoopInSupervise(unittest.TestCase):

    def test_unknown_collects_evidence_then_approves(self):
        calls = {"cloud": 0}

        def fake_qwen(req):
            return ModelResponse(text="```\napp.ts\nx\n```", backend="local")

        class _Cloud:
            def generate(self, req):
                calls["cloud"] += 1
                return ModelResponse(
                    text=_APPROVED_EV if calls["cloud"] > 1 else _UNKNOWN_EV,
                    backend="deepseek")

        result = supervise(local=None, cloud=_Cloud(), task="t",
                           qwen_generate=fake_qwen,
                           evidence_provider=lambda: "npm test -> 47 passed",
                           status=lambda line: None)
        self.assertEqual(result.verdicts[-1].decision, "APPROVED")
        self.assertEqual(calls["cloud"], 2)

    def test_unknown_without_evidence_escalates(self):
        def fake_qwen(req):
            return ModelResponse(text="```\napp.ts\nx\n```", backend="local")

        class _Cloud:
            def generate(self, req):
                return ModelResponse(text=_UNKNOWN_EV, backend="deepseek")

        result = supervise(local=None, cloud=_Cloud(), task="t",
                           qwen_generate=fake_qwen,
                           evidence_provider=lambda: None, status=lambda line: None)
        self.assertEqual(result.reason, "unknown_evidence")
        self.assertEqual(result.verdicts[-1].decision, "UNKNOWN")


class TestConstitution(unittest.TestCase):
    def test_load_constitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / ".agent"
            d.mkdir()
            (d / "constitution.md").write_text("1. Never claim success without evidence.")
            text = load_constitution(tmp)
            self.assertIn("AGENT CONSTITUTION", text)
            self.assertIn("Never claim success", text)

    def test_missing_constitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_constitution(tmp), "")


if __name__ == "__main__":
    unittest.main()
