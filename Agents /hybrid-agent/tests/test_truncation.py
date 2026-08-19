"""Truncation-hardening tests for the hybrid-agent supervise flow.

Covers the guarantee: Gemma emits ALL files in one response, truncation is
retried once with a doubled budget, a truncated DeepSeek fallback is never
applied, and FIX_REQUIRED -> APPROVED loops work end to end.

Run from hybrid-agent/:
    python -m unittest tests.test_truncation -v
    # or
    python tests/test_truncation.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.base import ModelResponse  # noqa: E402
from supervise import (  # noqa: E402
    GEMMA_MAX_TOKENS,
    GEMMA_MAX_TOKENS_CAP,
    _cloud_generate_guarded,
    supervise,
)

UNSAFE_REASONS = {"review_failed_no_verdict", "cloud_fallback_truncated"}


def _approved(text="=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n9.0\n"):
    return ModelResponse(text=text)


def _fix_required():
    return ModelResponse(text=(
        "=== REVIEW DECISION ===\nFIX_REQUIRED\n"
        "=== ISSUES FOUND ===\nIssue #1: x\nSeverity: MAJOR\n"
        "=== APPROVAL CONDITIONS ===\nfix it\n"))


class FakeCloud:
    """Records every request; returns the scripted verdict sequence."""

    def __init__(self, verdicts=None):
        self.reqs = []
        self.verdicts = list(verdicts or [_approved()])

    def generate(self, req):
        self.reqs.append(req)
        idx = min(len(self.reqs) - 1, len(self.verdicts) - 1)
        return self.verdicts[idx]


class TruncationRetryTests(unittest.TestCase):
    def test_truncated_then_full_retries_and_approves(self):
        """Truncated at 8192 -> retried at 16384 -> full output -> APPROVED."""
        calls = []

        def gemma(req):
            calls.append(req.max_tokens)
            if len(calls) == 1:
                return ModelResponse(text="partial...", truncated=True)
            return ModelResponse(text="```main.py\nprint('FULL')\n```\n", truncated=False)

        cloud = FakeCloud()
        res = supervise(local=object(), cloud=cloud, task="build 10 files",
                        gemma_generate=gemma, status=lambda l: None)
        self.assertEqual(calls, [GEMMA_MAX_TOKENS, GEMMA_MAX_TOKENS_CAP])
        self.assertTrue(res.final_text.startswith("```main.py"))
        self.assertEqual(res.verdicts[0].decision, "APPROVED")
        self.assertFalse(res.escalated)

    def test_always_truncated_retries_once_then_escalates(self):
        """Never truncating-free: exactly 2 Gemma calls, then escalate; the
        reviewer must never see incomplete code."""
        calls = []

        def gemma(req):
            calls.append(req.max_tokens)
            return ModelResponse(text="cut...", truncated=True)

        cloud = FakeCloud(verdicts=[_approved()])
        res = supervise(local=object(), cloud=cloud, task="big task",
                        gemma_generate=gemma, status=lambda l: None)
        self.assertEqual(calls, [GEMMA_MAX_TOKENS, GEMMA_MAX_TOKENS_CAP])
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_truncation_escalation")
        self.assertEqual(len(res.verdicts), 0, "reviewer must not see truncated code")

    def test_at_cap_still_gets_one_same_budget_retry(self):
        """--max-tokens 16384 (the cap): a truncated response still gets a
        single same-budget retry instead of immediate escalation."""
        calls = []

        def gemma(req):
            calls.append(req.max_tokens)
            if len(calls) == 1:
                return ModelResponse(text="partial", truncated=True)
            return ModelResponse(text="full", truncated=False)

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        gemma_generate=gemma, status=lambda l: None,
                        gemma_max_tokens=GEMMA_MAX_TOKENS_CAP)
        self.assertEqual(calls, [GEMMA_MAX_TOKENS_CAP, GEMMA_MAX_TOKENS_CAP])
        self.assertEqual(res.final_text, "full")

    def test_no_truncation_single_call(self):
        calls = []
        res = supervise(
            local=object(), cloud=FakeCloud(), task="small",
            gemma_generate=lambda req: (calls.append(1)
                                        or ModelResponse(text="```a.py\nx\n```\n")),
            status=lambda l: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res.verdicts[0].decision, "APPROVED")

    def test_fix_required_loop_then_approved(self):
        """FIX_REQUIRED feeds fixes back to Gemma, second iteration APPROVED."""
        gemma_calls = []
        cloud = FakeCloud(verdicts=[_fix_required(), _approved()])

        def gemma(req):
            gemma_calls.append(req.max_tokens)
            return ModelResponse(text="```b.py\ncode\n```\n", truncated=False)

        res = supervise(local=object(), cloud=cloud, task="iter task",
                        gemma_generate=gemma, status=lambda l: None)
        self.assertEqual([v.decision for v in res.verdicts], ["FIX_REQUIRED", "APPROVED"])
        self.assertEqual(len(gemma_calls), 2)
        self.assertEqual(gemma_calls, [GEMMA_MAX_TOKENS, GEMMA_MAX_TOKENS])


class CloudFallbackTruncationTests(unittest.TestCase):
    def test_cloud_fallback_truncated_is_unsafe(self):
        """A DeepSeek fallback truncated even after its same-budget retry must
        surface as cloud_fallback_truncated, which ask.py refuses to apply."""
        calls = []

        def gemma(req):
            return ModelResponse(text="cut...", truncated=True)

        class AlwaysTruncatedCloud:
            def generate(self, req):
                calls.append(req.max_tokens)
                return ModelResponse(text="incomplete", truncated=True)

        res = supervise(local=object(), cloud=AlwaysTruncatedCloud(), task="big",
                        gemma_generate=gemma, status=lambda l: None)
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "cloud_fallback_truncated")
        self.assertIn(res.reason, UNSAFE_REASONS, "ask.py must not --apply this")

    def test_cloud_fallback_retries_once_on_truncation(self):
        """First fallback call truncated, same-budget retry succeeds: the retry
        must happen and the result must NOT be marked unsafe."""
        calls = []

        class TruncatedOnceCloud:
            def generate(self, req):
                calls.append(1)
                if len(calls) == 1:
                    return ModelResponse(text="cut", truncated=True)
                return ModelResponse(text="complete", truncated=False)

        def gemma(req):
            return ModelResponse(text="cut...", truncated=True)

        res = supervise(local=object(), cloud=TruncatedOnceCloud(), task="big",
                        gemma_generate=gemma, status=lambda l: None)
        self.assertEqual(len(calls), 2, "one retry at same budget")
        self.assertEqual(res.final_text, "complete")
        self.assertEqual(res.reason, "local_truncation_escalation")
        self.assertNotIn(res.reason, UNSAFE_REASONS)

    def test_cloud_generate_guarded_plan_path(self):
        """The local-failure escalation (plan path) also honors truncation."""
        calls = []
        cloud = FakeCloud(verdicts=[ModelResponse(text="cut", truncated=True),
                                    ModelResponse(text="plan ok", truncated=False)])
        text, still = _cloud_generate_guarded(cloud, "t", lambda l: None, use_plan=True)
        self.assertEqual(len(calls := cloud.reqs), 2)
        self.assertEqual(text, "plan ok")
        self.assertFalse(still)


class DeepSeekBackendTruncationTests(unittest.TestCase):
    def test_finish_reason_length_sets_truncated(self):
        """DeepSeekBackend must propagate finish_reason == 'length'."""
        from backends.deepseek import DeepSeekBackend

        class FakeChoice:
            def __init__(self, finish_reason):
                self.finish_reason = finish_reason
                self.message = type("M", (), {"content": "partial output"})()

        class FakeCompletion:
            usage = {}

            def __init__(self, finish_reason):
                self.choices = [FakeChoice(finish_reason)]

        class FakeClient:
            def __init__(self, finish_reason):
                self.chat = type("C", (), {
                    "completions": type("Comps", (), {
                        "create": lambda self, **kw: FakeCompletion(finish_reason)
                    })()
                })

        backend = DeepSeekBackend.__new__(DeepSeekBackend)
        backend._client = FakeClient("length")
        backend.model, backend.name = "deepseek-chat", "deepseek"
        backend.max_retries = 0
        backend.backoff_s = [0.0]
        backend.timeout_s = 30.0

        req = type("R", (), {"system": "", "user": "", "max_tokens": 2048,
                             "temperature": 0.2, "timeout_s": 30.0})()
        resp = backend.generate(req)
        self.assertTrue(resp.truncated)

        backend._client = FakeClient("stop")
        resp = backend.generate(req)
        self.assertFalse(resp.truncated)


class LocalLimitsAwarenessTests(unittest.TestCase):
    """DeepSeek must plan within the local model's context/output limits."""

    def test_plan_request_informs_deepseek_of_local_limits(self):
        from supervise import LOCAL_CONTEXT_TOKENS, LOCAL_OUTPUT_TOKENS, _cloud_plan_request
        req = _cloud_plan_request("build a multi-file app")
        self.assertIn("IMPLEMENTER CONSTRAINTS", req.system)
        self.assertIn(str(LOCAL_CONTEXT_TOKENS), req.system)
        self.assertIn(str(LOCAL_OUTPUT_TOKENS), req.system)
        self.assertIn("one response", req.system)

    def test_supervisor_request_notes_local_output_cap(self):
        from supervise import LOCAL_OUTPUT_TOKENS, ReviewPackage, _supervisor_request
        req = _supervisor_request(ReviewPackage(task="t", changes="x"))
        self.assertIn(str(LOCAL_OUTPUT_TOKENS), req.system)
        self.assertIn("output cap", req.system)

    def test_gemma_prompt_warns_about_output_cap(self):
        from supervise import LOCAL_OUTPUT_TOKENS, _gemma_primary_prompt
        req = _gemma_primary_prompt("task")
        self.assertIn(str(LOCAL_OUTPUT_TOKENS), req.user)
        self.assertIn("REMAINING WORK", req.user)


class EnhancementTests(unittest.TestCase):
    def test_parse_enhancement_sections(self):
        from supervise import parse_enhancement
        raw = "=== ENHANCED PROMPT ===\nEnhanced content here\n=== REASONING ===\nReasoning here\n=== PLAN ===\nPlan here"
        result = parse_enhancement(raw)
        self.assertEqual(result.enhanced_prompt, "Enhanced content here")
        self.assertEqual(result.reasoning, "Reasoning here")
        self.assertEqual(result.plan, "Plan here")
        self.assertTrue(result.enhanced)
        self.assertEqual(result.raw, raw)

    def test_parse_enhancement_fallback_raw(self):
        from supervise import parse_enhancement
        raw = "Just a plain prompt without sections"
        result = parse_enhancement(raw)
        self.assertEqual(result.enhanced_prompt, raw.strip())
        # fallback fills enhanced_prompt with the raw text, so it is non-empty.
        self.assertTrue(result.enhanced)

    def test_enhance_request_informs_deepseek_of_local_limits(self):
        from supervise import LOCAL_CONTEXT_TOKENS, LOCAL_OUTPUT_TOKENS, _enhance_request
        req = _enhance_request('build a multi-file app')
        self.assertTrue(req.user.startswith('ORIGINAL USER TASK:'))
        self.assertIn('IMPLEMENTER CONSTRAINTS', req.system)
        self.assertIn(str(LOCAL_CONTEXT_TOKENS), req.system)
        self.assertIn(str(LOCAL_OUTPUT_TOKENS), req.system)
        self.assertIn('=== ENHANCED PROMPT ===', req.system)
        self.assertEqual(req.max_tokens, 1200)

    def test_enhance_request_includes_context(self):
        from supervise import _enhance_request
        req = _enhance_request('task', 'ctx info')
        self.assertIn('ctx info', req.user)


if __name__ == "__main__":
    unittest.main()
