"""Truncation-hardening tests for the hybrid-agent supervise flow.

Covers the guarantee: Qwen emits ALL files in one response, truncation is
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
    QWEN_MAX_TOKENS,
    QWEN_MAX_TOKENS_CAP,
    _cloud_generate_guarded,
    supervise,
)

UNSAFE_REASONS = {"review_failed_no_verdict", "cloud_fallback_truncated"}


def _approved(text="=== REVIEW DECISION ===\nAPPROVED\n=== EVIDENCE ===\nverify command passed\n=== EVIDENCE ===\nverify command passed\n=== QUALITY SCORE ===\n9.0\n"):
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

        def qwen(req):
            calls.append(req.max_tokens)
            if len(calls) == 1:
                return ModelResponse(text="partial...", truncated=True)
            return ModelResponse(text="```main.py\nprint('FULL')\n```\n", truncated=False)

        cloud = FakeCloud()
        res = supervise(local=object(), cloud=cloud, task="build 10 files",
                        qwen_generate=qwen, status=lambda l: None)
        self.assertEqual(calls, [QWEN_MAX_TOKENS, QWEN_MAX_TOKENS_CAP])
        self.assertTrue(res.final_text.startswith("```main.py"))
        self.assertEqual(res.verdicts[0].decision, "APPROVED")
        self.assertFalse(res.escalated)

    def test_always_truncated_retries_once_then_escalates(self):
        """Never truncating-free: exactly 2 Qwen calls, then escalate; the
        reviewer must never see incomplete code."""
        calls = []

        def qwen(req):
            calls.append(req.max_tokens)
            return ModelResponse(text="cut...", truncated=True)

        cloud = FakeCloud(verdicts=[_approved()])
        res = supervise(local=object(), cloud=cloud, task="big task",
                        qwen_generate=qwen, status=lambda l: None)
        self.assertEqual(calls, [QWEN_MAX_TOKENS, QWEN_MAX_TOKENS_CAP])
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_truncation_escalation")
        self.assertEqual(len(res.verdicts), 0, "reviewer must not see truncated code")

    def test_at_cap_still_gets_one_same_budget_retry(self):
        """--max-tokens 16384 (the cap): a truncated response still gets a
        single same-budget retry instead of immediate escalation."""
        calls = []

        def qwen(req):
            calls.append(req.max_tokens)
            if len(calls) == 1:
                return ModelResponse(text="partial", truncated=True)
            return ModelResponse(text="full", truncated=False)

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        qwen_max_tokens=QWEN_MAX_TOKENS_CAP)
        self.assertEqual(calls, [QWEN_MAX_TOKENS_CAP, QWEN_MAX_TOKENS_CAP])
        self.assertEqual(res.final_text, "full")

    def test_no_truncation_single_call(self):
        calls = []
        res = supervise(
            local=object(), cloud=FakeCloud(), task="small",
            qwen_generate=lambda req: (calls.append(1)
                                        or ModelResponse(text="```a.py\nx\n```\n")),
            status=lambda l: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res.verdicts[0].decision, "APPROVED")

    def test_fix_required_loop_then_approved(self):
        """FIX_REQUIRED feeds fixes back to Qwen, second iteration APPROVED."""
        qwen_calls = []
        cloud = FakeCloud(verdicts=[_fix_required(), _approved()])

        def qwen(req):
            qwen_calls.append(req.max_tokens)
            return ModelResponse(text="```b.py\ncode\n```\n", truncated=False)

        res = supervise(local=object(), cloud=cloud, task="iter task",
                        qwen_generate=qwen, status=lambda l: None)
        self.assertEqual([v.decision for v in res.verdicts], ["FIX_REQUIRED", "APPROVED"])
        self.assertEqual(len(qwen_calls), 2)
        self.assertEqual(qwen_calls, [QWEN_MAX_TOKENS, QWEN_MAX_TOKENS])


class CloudFallbackTruncationTests(unittest.TestCase):
    def test_cloud_fallback_truncated_is_unsafe(self):
        """A DeepSeek fallback truncated even after its same-budget retry must
        surface as cloud_fallback_truncated, which ask.py refuses to apply."""
        calls = []

        def qwen(req):
            return ModelResponse(text="cut...", truncated=True)

        class AlwaysTruncatedCloud:
            def generate(self, req):
                calls.append(req.max_tokens)
                return ModelResponse(text="incomplete", truncated=True)

        res = supervise(local=object(), cloud=AlwaysTruncatedCloud(), task="big",
                        qwen_generate=qwen, status=lambda l: None)
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

        def qwen(req):
            return ModelResponse(text="cut...", truncated=True)

        res = supervise(local=object(), cloud=TruncatedOnceCloud(), task="big",
                        qwen_generate=qwen, status=lambda l: None)
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


class LocalContextWindowTruncationTests(unittest.TestCase):
    """LM Studio's /api/v1/chat never reports finish_reason, so truncation
    against the loaded context window must be detected from prompt+output
    token counts (the server-independent signal)."""

    def test_prompt_plus_output_reaching_window_is_truncated(self):
        from backends.local_qwen import _mlx_result
        # 20000 prompt tokens + 13000 output = 33000 >= 32768 -> cut off.
        res = _mlx_result(
            {"output": [{"type": "message", "content": "```main.py\nprint('partial..."}],
             "stats": {"input_tokens": 20000, "total_output_tokens": 13000}},
            max_tokens=10**15, context_window=32768)
        self.assertTrue(res["truncated"])
        self.assertEqual(res["truncate_reason"], "context_window_reached")

    def test_well_under_window_is_not_truncated(self):
        from backends.local_qwen import _mlx_result
        res = _mlx_result(
            {"output": [{"type": "message", "content": "```main.py\nprint('ok')\n```\n"}],
             "stats": {"input_tokens": 2000, "total_output_tokens": 300}},
            max_tokens=10**15, context_window=32768)
        self.assertFalse(res["truncated"])

    def test_window_reached_but_balanced_fence_still_detected(self):
        """Even when the output happens to end on a balanced fence (so the old
        fence heuristic would miss it), the window check must fire."""
        from backends.local_qwen import _mlx_result
        text = "```a.py\nx = 1\n```\n```b.py\ny = 2\n"
        res = _mlx_result(
            {"output": [{"type": "message", "content": text}],
             "stats": {"input_tokens": 30000, "total_output_tokens": 3000}},
            max_tokens=10**15, context_window=32768)
        self.assertTrue(res["truncated"])
        self.assertEqual(res["truncate_reason"], "context_window_reached")

    def test_no_window_knowledge_disables_the_check(self):
        """context_window=0 (unknown) must not false-positive."""
        from backends.local_qwen import _mlx_result
        res = _mlx_result(
            {"output": [{"type": "message", "content": "plain text without fences"}],
             "stats": {"input_tokens": 5000, "total_output_tokens": 5000}},
            max_tokens=10**15, context_window=0)
        self.assertFalse(res["truncated"], "unknown window must not flag")

    def test_backend_default_window_used(self):
        from backends.local_qwen import QwenBackend, DEFAULT_CONTEXT_WINDOW
        backend = QwenBackend.__new__(QwenBackend)
        self.assertEqual(DEFAULT_CONTEXT_WINDOW, 32768,
                         "default window must match the loaded -c 32768")
        self.assertEqual(backend.__class__.__module__, "backends.local_qwen")


class LocalLimitsAwarenessTests(unittest.TestCase):
    """DeepSeek must plan within the local model's context/output limits."""

    def test_plan_request_informs_deepseek_of_local_limits(self):
        from supervise import LOCAL_CONTEXT_TOKENS, LOCAL_OUTPUT_TOKENS, _cloud_plan_request
        req = _cloud_plan_request("build a multi-file app")
        self.assertIn("IMPLEMENTER CONSTRAINTS", req.system)
        self.assertIn(str(LOCAL_CONTEXT_TOKENS), req.system)
        self.assertIn(str(LOCAL_OUTPUT_TOKENS), req.system)
        self.assertIn("NO enforced output cap", req.system)
        self.assertIn("one-response steps", req.system)

    def test_supervisor_request_notes_no_local_output_cap(self):
        from supervise import ReviewPackage, _supervisor_request
        req = _supervisor_request(ReviewPackage(task="t", changes="x"))
        self.assertIn("no engine-imposed output cap", req.system)
        self.assertIn("context window", req.system)

    def test_qwen_prompt_has_no_output_cap(self):
        from supervise import _qwen_primary_prompt
        req = _qwen_primary_prompt("task")
        self.assertIn("NO output token cap", req.user)
        self.assertIn("COMPLETE change", req.user)
        self.assertIn("REMAINING WORK", req.user)


class LocalDiscoveryTests(unittest.TestCase):
    """The engine must plan within the ACTUALLY LOADED local context window,
    discovered from LM Studio for ANY local model — never a hardcoded value."""

    def test_discover_reads_loaded_instance_config(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        payload = {"models": [{
            "key": "some-other-model",
            "max_context_length": 65536,
            "loaded_instances": [{"config": {"context_length": 16384}}],
        }]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            self.assertEqual(discover_context_window("http://x/api/v1", model="some-other-model"),
                             16384, "loaded_instances config wins (real -c window)")

    def test_discover_falls_back_to_max_when_not_loaded(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        payload = {"models": [{"key": "model-b", "max_context_length": 32768,
                               "loaded_instances": []}]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            self.assertEqual(discover_context_window("http://x/api/v1", model="model-b"),
                             32768)

    def test_discover_returns_zero_on_errors(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no server")):
            self.assertEqual(discover_context_window("http://x/api/v1"), 0)
        with mock.patch("urllib.request.urlopen", side_effect=Exception("bad")):
            self.assertEqual(discover_context_window("http://x/api/v1"), 0)

    def test_discover_ignores_wrong_model(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        payload = {"models": [{"key": "other", "max_context_length": 99999,
                               "loaded_instances": []}]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            self.assertEqual(discover_context_window("http://x/api/v1", model="target"),
                             0, "a different model's window must not leak in")

    def test_plan_request_uses_custom_window(self):
        from supervise import _cloud_plan_request
        req = _cloud_plan_request("t", context_window=16384)
        self.assertIn("16384", req.system)
        self.assertNotIn("32768", req.system)

    def test_plan_request_handles_window_larger_than_default(self):
        """A local model loaded with -c 65536 (or 131072) must be planned for
        ITS window, never silently capped at the 32768 default."""
        from supervise import _cloud_plan_request, _enhance_request, _supervisor_request, ReviewPackage
        req = _cloud_plan_request("t", context_window=65536)
        self.assertIn("65536", req.system)
        self.assertNotIn("32768", req.system)
        req2 = _enhance_request("t", context_window=131072)
        self.assertIn("131072", req2.system)
        req3 = _supervisor_request(ReviewPackage(task="t", changes="x"), context_window=65536)
        self.assertIn("65536", req3.system)

    def test_window_larger_than_default_truncation_detection(self):
        """Detection boundary must be the REAL loaded window, even >32768."""
        from backends.local_qwen import _mlx_result
        # 40000 prompt + 30000 output = 70000 >= 65536 -> truncated (would be
        # MISSED if detection were capped at the 32768 default).
        res = _mlx_result(
            {"output": [{"type": "message", "content": "```a.py\nx"}],
             "stats": {"input_tokens": 40000, "total_output_tokens": 30000}},
            max_tokens=10**15, context_window=65536)
        self.assertTrue(res["truncated"])
        self.assertEqual(res["truncate_reason"], "context_window_reached")
        # Under the real window -> not truncated (a 32768 cap would FALSELY flag).
        res2 = _mlx_result(
            {"output": [{"type": "message", "content": "```a.py\nx\n```\n"}],
             "stats": {"input_tokens": 30000, "total_output_tokens": 2000}},
            max_tokens=10**15, context_window=65536)
        self.assertFalse(res2["truncated"])

    def test_discover_reads_large_loaded_window(self):
        """-c 65536 in loaded_instances must be discovered (not capped)."""
        from unittest import mock
        from backends.local_qwen import discover_context_window
        payload = {"models": [{"key": "big-model", "max_context_length": 131072,
                               "loaded_instances": [{"config": {"context_length": 65536}}]}]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            self.assertEqual(discover_context_window("http://x/api/v1", model="big-model"),
                             65536)

    def test_supervisor_request_uses_custom_window(self):
        from supervise import ReviewPackage, _supervisor_request
        req = _supervisor_request(ReviewPackage(task="t", changes="x"), context_window=8192)
        self.assertIn("8192", req.system)

    def test_discovery_wins_over_config_even_when_larger(self):
        """config context_window=32768 must NOT cap a model actually loaded at
        65536 — the server's loaded window is the source of truth."""
        from unittest import mock
        import ask

        class FakeLP:
            base_url = "http://localhost:1234/api/v1"
            model = "big-model"
            api_key = "lm-studio"
            context_window = 32768  # stale config — must lose to discovery

        cfg = {"providers": {"local": [{"name": "qwen", "enabled": True}]},
               "backends": {"local": {"base_url": FakeLP.base_url,
                                      "model": FakeLP.model,
                                      "api_key": "lm-studio"}}}
        payload = {"models": [{"key": "big-model", "max_context_length": 131072,
                               "loaded_instances": [{"config": {"context_length": 65536}}]}]}
        with mock.patch("ask.get_local", return_value=FakeLP()), \
             mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            w = ask._discover_local_context_window(cfg, model="big-model",
                                                   base_url=FakeLP.base_url)
        self.assertEqual(w, 65536,
                         "discovered loaded window must win over config 32768")

    def test_config_used_only_when_discovery_fails(self):
        """When the server is unreachable, the config value is the fallback."""
        from unittest import mock
        import ask

        class FakeLP:
            base_url = "http://localhost:1234/api/v1"
            model = "big-model"
            api_key = "lm-studio"
            context_window = 32768

        cfg = {"providers": {"local": [{"name": "qwen", "enabled": True}]},
               "backends": {"local": {"base_url": FakeLP.base_url,
                                      "model": FakeLP.model,
                                      "api_key": "lm-studio"}}}
        with mock.patch("ask.get_local", return_value=FakeLP()), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            w = ask._discover_local_context_window(cfg, model="big-model",
                                                   base_url=FakeLP.base_url)
        self.assertEqual(w, 32768, "config is the fallback when discovery fails")


class ContextSafetyControllerTests(unittest.TestCase):
    """The CSC must enforce hard budgets and NEVER call the local model with a
    prompt that cannot safely fit — regardless of the loaded window."""

    def test_safe_input_budget_never_eats_the_window(self):
        from context_safety import safe_input_budget
        self.assertEqual(safe_input_budget(65536, 12000, 2000), 51536)
        self.assertEqual(safe_input_budget(32768, 12000, 2000), 18768)
        self.assertEqual(safe_input_budget(0), 0, "unknown window -> no budget")
        self.assertEqual(safe_input_budget(5000, 12000, 2000), 0,
                         "tiny window with big reserve -> 0 (must not call)")

    def test_assess_zone_boundaries(self):
        from context_safety import assess_zone, safe_input_budget
        b = safe_input_budget(65536, 12000, 2000)  # 51536
        self.assertEqual(assess_zone(int(b * 0.69), b), "GREEN")
        self.assertEqual(assess_zone(int(b * 0.75), b), "YELLOW")
        self.assertEqual(assess_zone(int(b * 0.90), b), "ORANGE")
        self.assertEqual(assess_zone(int(b * 0.99), b), "RED")
        self.assertEqual(assess_zone(10**9, b), "RED")

    def test_compact_text_marks_truncation(self):
        from context_safety import compact_text
        short = "abc"
        self.assertEqual(compact_text(short, 100), short)
        out = compact_text("x" * 5000, 100)
        self.assertIn("context budget", out)
        self.assertLessEqual(len(out), 100 * 4)

    def test_summarize_terminal_output_extracts_errors(self):
        from context_safety import summarize_terminal_output
        raw = ("exit 1\nnpm test\nPASS test/a\nFAIL test/b\n"
               "Error: something broke\n    at fn (file.js:10:5)\n"
               "Warning: deprecated\n" + "noise\n" * 500)
        summary = summarize_terminal_output(raw)
        self.assertIn("exit 1", summary)
        self.assertIn("ERRORS:", summary)
        self.assertIn("something broke", summary)
        self.assertIn("STACK:", summary)
        self.assertIn("file.js:10:5", summary)
        self.assertLess(len(summary), len(raw), "must compress, not echo")

    def test_classify_output_states(self):
        from context_safety import classify_output
        from backends.base import ModelResponse
        self.assertEqual(classify_output(ModelResponse(text="done")), "COMPLETED")
        self.assertEqual(classify_output(ModelResponse(
            text="cut", truncated=True, truncate_reason="context_window_reached")),
            "CONTEXT_LIMIT_REACHED")
        self.assertEqual(classify_output(ModelResponse(
            text="cut", truncated=True, truncate_reason="max_tokens_reached")),
            "OUTPUT_LIMIT_REACHED")
        self.assertEqual(classify_output(ModelResponse(
            text="cut", truncated=True, truncate_reason="stream_eof")), "TIMEOUT")
        self.assertEqual(classify_output(ModelResponse(
            text="cut", truncated=True, truncate_reason="unbalanced_code_fence")),
            "OUTPUT_LIMIT_REACHED")

    def test_supervisor_context_block_carries_budget(self):
        from context_safety import supervisor_context_block
        block = supervisor_context_block("qwen2.5-coder-14b-instruct-mlx",
                                         65536, 12000, 2000, 3, 7,
                                         "terminal, filesystem, tests", "GREEN")
        self.assertIn("qwen2.5-coder-14b-instruct-mlx", block)
        self.assertIn("65536", block)
        self.assertIn("51536", block)  # safe input budget
        self.assertIn("12000", block)  # output reserve
        self.assertIn("3/7", block)    # current step
        self.assertIn("GREEN", block)

    def test_qwen_prompt_compacts_source_context_to_budget(self):
        """A huge source context must be compacted, never sent whole."""
        from supervise import _qwen_primary_prompt
        big = "x" * 400_000  # ~100k tokens of context
        req = _qwen_primary_prompt("small task", source_context=big,
                                   context_window=32768, output_reserve_tokens=12000,
                                   safety_margin_tokens=2000)
        self.assertLess(len(req.user), 200_000,
                        "source context must be compacted to the safe budget")
        self.assertIn("compacted", req.user)

    def test_qwen_prompt_summarizes_terminal_output(self):
        from supervise import _qwen_primary_prompt
        big = "noise\n" * 20000 + "Error: boom\n"
        req = _qwen_primary_prompt("small task", terminal_output=big,
                                   context_window=32768)
        self.assertIn("SUMMARIZED", req.user)
        self.assertIn("boom", req.user)
        self.assertLess(len(req.user), len(big))

    def test_supervise_red_never_calls_qwen(self):
        """When the task alone exceeds the safe input budget, Qwen must NOT be
        called — the run escalates to DeepSeek instead."""
        calls = []

        def qwen(req):
            calls.append(req)  # must never happen
            return ModelResponse(text="SHOULD NOT HAPPEN")

        cloud = FakeCloud(verdicts=[_approved()])
        res = supervise(local=object(), cloud=cloud, task="t", qwen_generate=qwen,
                        status=lambda l: None,
                        context_window=512,  # tiny window -> everything is RED
                        output_reserve_tokens=12000, safety_margin_tokens=2000)
        self.assertEqual(calls, [], "Qwen must not be called in RED zone")
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_context_red_escalation")

    def test_supervise_green_calls_qwen_normally(self):
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=FakeCloud(), task="tiny",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res.verdicts[0].decision, "APPROVED")
        self.assertFalse(res.escalated)


class CapabilityDiscoveryTests(unittest.TestCase):
    """Per-model capability discovery: the engine must not assume all local
    models behave identically (window, architecture, max output, vision...)."""

    def test_capabilities_parse_full_payload(self):
        from unittest import mock
        from backends.local_qwen import discover_model_capabilities
        payload = {"models": [{
            "type": "llm", "publisher": "x", "key": "model-x",
            "architecture": "llama", "format": "gguf",
            "quantization": {"name": "q4_k_m"},
            "max_context_length": 131072,
            "capabilities": {"vision": False, "trained_for_tool_use": True},
            "loaded_instances": [{"config": {"context_length": 65536}}],
        }]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            caps = discover_model_capabilities("http://x/api/v1", model="model-x")
        self.assertEqual(caps["model_id"], "model-x")
        self.assertEqual(caps["context_window"], 65536)
        self.assertEqual(caps["max_context_length"], 131072)
        self.assertEqual(caps["architecture"], "llama")
        self.assertEqual(caps["format"], "gguf")
        self.assertEqual(caps["quantization"], "q4_k_m")
        self.assertTrue(caps["tool_use"])
        self.assertFalse(caps["vision"])

    def test_capabilities_safe_defaults_on_error(self):
        from unittest import mock
        from backends.local_qwen import discover_model_capabilities
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            caps = discover_model_capabilities("http://x/api/v1", model="m")
        self.assertEqual(caps["context_window"], 0)
        self.assertEqual(caps["max_output"], 0)
        self.assertEqual(caps["architecture"], "")
        self.assertFalse(caps["vision"])

    def test_discover_context_window_delegates_to_capabilities(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        payload = {"models": [{"key": "m", "max_context_length": 65536,
                               "loaded_instances": [{"config": {"context_length": 32768}}]}]}
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                __import__("json").dumps(payload))
            self.assertEqual(discover_context_window("http://x/api/v1", model="m"), 32768)


class DynamicReserveTests(unittest.TestCase):
    """Output reserve must be dynamic: min(configured, model_max, task need) —
    a tiny task does not sacrifice 12k tokens of context."""

    def test_reserve_capped_by_model_max_output(self):
        from context_safety import dynamic_output_reserve
        self.assertEqual(dynamic_output_reserve(12000, model_max_output=8192), 8192)
        self.assertEqual(dynamic_output_reserve(12000, model_max_output=0), 12000)

    def test_reserve_capped_by_task_need(self):
        from context_safety import dynamic_output_reserve
        self.assertEqual(dynamic_output_reserve(12000, task_required_output=3000), 3000)
        self.assertEqual(dynamic_output_reserve(12000, task_required_output=200), 512,
                         "never below a minimal response")

    def test_reserve_min_of_all(self):
        from context_safety import dynamic_output_reserve
        self.assertEqual(dynamic_output_reserve(12000, model_max_output=8192,
                                                task_required_output=4000), 4000)

    def test_task_required_output_scales_with_files(self):
        from context_safety import task_required_output
        small = task_required_output("fix a typo", ["a.py"])
        big = task_required_output("build an app", ["a.py", "b.py", "c.py", "d.py"])
        self.assertGreater(big, small)
        self.assertGreater(small, 0)

    def test_dynamic_reserve_feeds_supervise_budget(self):
        """A tiny task must yield a LARGER safe input budget (smaller reserve)."""
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=FakeCloud(), task="tiny",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768,
                        output_reserve_tokens=12000, safety_margin_tokens=2000,
                        contract_files=["a.py"])
        self.assertEqual(len(calls), 1)
        self.assertFalse(res.escalated)


class RetryBudgetTests(unittest.TestCase):
    """Recovery loops must be hard-capped — a pathological model cannot burn
    unbounded time/tokens retrying."""

    def test_retry_budget_caps_continuation_calls(self):
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="cut...", truncated=True,
                                 truncate_reason="context_window_reached")

        cloud = FakeCloud(verdicts=[_approved()])
        res = supervise(local=object(), cloud=cloud, task="t", qwen_generate=qwen,
                        status=lambda l: None, context_window=32768,
                        max_recovery_retries=3)
        # 1 primary + exactly 3 recovery retries, then escalation — never more.
        self.assertEqual(len(calls), 4)
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_truncation_escalation")

    def test_retry_budget_zero_means_no_retry(self):
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="cut", truncated=True)

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768, max_recovery_retries=0)
        self.assertEqual(len(calls), 1, "no continuation allowed")

    def test_success_after_retry_within_budget(self):
        calls = []

        def qwen(req):
            calls.append(req)
            if len(calls) == 1:
                return ModelResponse(text="partial", truncated=True)
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768, max_recovery_retries=1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(res.verdicts[0].decision, "APPROVED")
        self.assertFalse(res.escalated)


class CompactionInvariantTests(unittest.TestCase):
    """Compaction must NEVER remove task requirements, BOUND, contract,
    supervisor fixes, or the engine protocol — only bulk context."""

    def test_core_sections_survive_compaction(self):
        from supervise import _qwen_primary_prompt
        task = "TASK TEXT: fix the login flow"
        bound = "BOUND: never touch .env"
        contract = "CONTRACT: must change src/auth.ts"
        fixes = "FIX: revert the bad import"
        big = "x" * 300_000  # forces source compaction
        req = _qwen_primary_prompt(
            task, prior_fixes=fixes, bound_text=bound, contract_text=contract,
            source_context=big, context_window=32768,
            output_reserve_tokens=12000, safety_margin_tokens=2000)
        # All protected invariants present verbatim.
        self.assertIn("TASK TEXT: fix the login flow", req.user)
        self.assertIn("BOUND: never touch .env", req.system)
        self.assertIn("CONTRACT: must change src/auth.ts", req.user)
        self.assertIn("FIX: revert the bad import", req.user)
        self.assertIn("REMAINING WORK", req.user)  # engine protocol survives
        # Bulk context was compacted, not the core.
        self.assertIn("compacted", req.user)
        meta = req.metadata.get("context_safety") or {}
        self.assertIn("source", meta.get("compaction", ""))

    def test_task_never_truncated_by_compactor(self):
        from supervise import _qwen_primary_prompt
        task = "EXACT TASK: " + "important requirement " * 50
        req = _qwen_primary_prompt(
            task, source_context="y" * 500_000, context_window=32768,
            output_reserve_tokens=12000, safety_margin_tokens=2000)
        self.assertIn("EXACT TASK: ", req.user)
        self.assertIn("important requirement", req.user)


class TelemetryTests(unittest.TestCase):
    def test_telemetry_line_has_all_fields(self):
        from context_safety import context_telemetry
        line = context_telemetry(model="m", window=65536, safe_input=51536,
                                 input_tokens=18240, output_reserve=12000,
                                 zone="GREEN", compaction="none",
                                 output_tokens=2840, status="COMPLETED",
                                 step=3, step_total=7, architecture="qwen2",
                                 max_output=0, vision=False, tool_use=False)
        for frag in ["MODEL=m", "WINDOW=65536", "SAFE_INPUT=51536",
                     "INPUT=18240", "OUT_RESERVE=12000", "ZONE=GREEN",
                     "COMPACTION=none", "OUTPUT=2840", "STATUS=COMPLETED",
                     "STEP=3/7", "ARCH=qwen2", "MAX_OUTPUT=0",
                     "VISION=false", "TOOLS=false"]:
            self.assertIn(frag, line)

    def test_supervise_emits_telemetry(self):
        lines = []

        def qwen(req):
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=FakeCloud(), task="tiny",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768, telemetry=lines.append)
        self.assertGreaterEqual(len(lines), 1)
        self.assertIn("STATUS=COMPLETED", lines[-1])
        self.assertIn("ZONE=GREEN", lines[-1])

    def test_telemetry_emitted_on_red(self):
        lines = []
        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=lambda r: ModelResponse(text="x"),
                        status=lambda l: None,
                        context_window=512,  # RED
                        output_reserve_tokens=12000, safety_margin_tokens=2000,
                        telemetry=lines.append)
        self.assertTrue(res.escalated)
        self.assertTrue(any("RED_ESCALATION" in l for l in lines))


class StressTests(unittest.TestCase):
    """Deliberately adversarial inputs: the CSC must survive them without the
    local model EVER receiving an unsafe payload."""

    def test_30k_token_task_is_compacted_not_dropped(self):
        from supervise import _qwen_primary_prompt
        from context_safety import estimate_tokens
        task = ("requirement " * 7500)  # ~30k tokens of task text
        req = _qwen_primary_prompt(task, context_window=32768,
                                   output_reserve_tokens=12000,
                                   safety_margin_tokens=2000)
        # The task core is protected: it stays (compaction happens on bulk
        # context, never the task) — but the total prompt is preflight-checked
        # by supervise(); here we assert the builder never crashes and the
        # engine protocol survives.
        self.assertIn("REMAINING WORK", req.user)
        self.assertGreater(estimate_tokens(req.system + req.user), 0)

    def test_100k_token_source_tree_compacted(self):
        from supervise import _qwen_primary_prompt
        tree = "\n".join(f"--- src/mod{i}.ts ({i} bytes)\n" + "y" * 2000
                         for i in range(50))  # ~100k chars source tree
        req = _qwen_primary_prompt("refactor auth", source_context=tree,
                                   context_window=32768,
                                   output_reserve_tokens=12000,
                                   safety_margin_tokens=2000)
        self.assertIn("compacted", req.user)
        self.assertIn("refactor auth", req.user)  # task intact

    def test_enormous_terminal_output_summarized(self):
        from supervise import _qwen_primary_prompt
        # ~200k chars (~57k tokens) — far beyond the ~18k safe input budget,
        # so the CSC MUST summarize instead of echoing the log into the window.
        log = "noise line\n" * 20000 + "Error: catastrophic failure\n" + "at x.js:1\n"
        req = _qwen_primary_prompt("fix build", terminal_output=log,
                                   context_window=32768)
        self.assertIn("SUMMARIZED", req.user)
        self.assertIn("catastrophic failure", req.user)
        self.assertLess(len(req.user), len(log) // 3)

    def test_repeated_tool_calls_bounded_by_terminal_rounds(self):
        calls = []

        def qwen(req):
            calls.append(req)
            # Model keeps asking to run the same command forever.
            return ModelResponse(text="RUN: npm test\n```a.py\nx\n```\n")

        def tool(cmd):
            return "exit 1\nError: something failed"

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, terminal_tool=tool,
                        max_terminal_rounds=2, status=lambda l: None,
                        context_window=32768)
        self.assertLessEqual(len(calls), 3, "terminal loop must terminate")

    def test_10_iterations_terminate(self):
        calls = []
        cloud = FakeCloud(verdicts=[_fix_required() for _ in range(20)])

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=cloud, task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        max_iterations=10, context_window=32768)
        self.assertLessEqual(len(calls), 10)
        self.assertTrue(res.escalated or res.reason.startswith("deepseek_fallback"))

    def test_window_change_32k_64k_128k(self):
        from unittest import mock
        from backends.local_qwen import discover_context_window
        for window in (32768, 65536, 131072):
            payload = {"models": [{"key": "m", "max_context_length": 131072,
                                   "loaded_instances": [{"config": {"context_length": window}}]}]}
            with mock.patch("urllib.request.urlopen") as m:
                m.return_value.__enter__.return_value.read.return_value.decode.return_value = (
                    __import__("json").dumps(payload))
                self.assertEqual(discover_context_window("http://x/api/v1", model="m"),
                                 window)

    def test_forced_incomplete_output_escalates_not_applies(self):
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="```a.py\ncut mid-file", truncated=True,
                                 truncate_reason="context_window_reached")

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertTrue(res.escalated)
        self.assertNotIn("review_failed", res.reason)
        self.assertNotEqual(res.final_text, "```a.py\ncut mid-file",
                            "incomplete output must never be applied as-is")

    def test_forced_timeout_classified_and_escalated(self):
        calls = []

        def qwen(req):
            calls.append(req)
            return ModelResponse(text="partial...", truncated=True,
                                 truncate_reason="stream_eof")

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_truncation_escalation")

    def test_supervisor_unavailable_applies_without_review_flag(self):
        class DownCloud:
            def generate(self, req):
                raise RuntimeError("supervisor down")

        def qwen(req):
            return ModelResponse(text="```a.py\nx\n```\n")

        res = supervise(local=object(), cloud=DownCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertEqual(res.reason, "review_failed_no_verdict")
        self.assertFalse(res.escalated)

    def test_local_model_unavailable_escalates(self):
        def qwen(req):
            raise RuntimeError("local model crashed")

        res = supervise(local=object(), cloud=FakeCloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertTrue(res.escalated)
        self.assertEqual(res.reason, "local_failure_escalation")


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

    def test_parse_enhancement_clarifying_questions(self):
        from supervise import parse_enhancement
        raw = (
            "=== ENHANCED PROMPT ===\nA clear prompt\n"
            "=== REASONING ===\nwhy\n=== PLAN ===\nplan\n"
            "=== CLARIFYING QUESTIONS ===\nQ1: which file?\nQ2: expected behavior?"
        )
        result = parse_enhancement(raw)
        self.assertIn("Q1: which file?", result.clarifying_questions)
        self.assertIn("Q2: expected behavior?", result.clarifying_questions)

    def test_parse_enhancement_no_clarifying_questions(self):
        from supervise import parse_enhancement
        raw = "=== ENHANCED PROMPT ===\nClear\n=== PLAN ===\nplan"
        result = parse_enhancement(raw)
        self.assertEqual(result.clarifying_questions, "")

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
        self.assertEqual(req.max_tokens, 2400)

    def test_enhance_request_includes_context(self):
        from supervise import _enhance_request
        req = _enhance_request('task', 'ctx info')
        self.assertIn('ctx info', req.user)

    def test_enhance_request_instructs_clarification(self):
        from supervise import _enhance_request
        req = _enhance_request('task')
        self.assertIn('CLARIFYING QUESTIONS', req.system)
        self.assertIn('ambiguous', req.system)
        self.assertIn('OMIT this section', req.system)


class EnhanceTaskClarificationTests(unittest.TestCase):
    """The --enhance clarification loop: DeepSeek flags an unclear task, the
    bridge either asks the user (interactive) or stops for clarification."""

    def setUp(self):
        import io as _io
        import ask
        self.ask = ask
        self._orig_generate = ask._generate_with_retry
        self._orig_stdin = sys.stdin
        self._orig_stdout = sys.stdout
        # _enhance_task prints the enhanced prompt/questions to stdout; silence it.
        sys.stdout = _io.StringIO()

    def tearDown(self):
        self.ask._generate_with_retry = self._orig_generate
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout

    @staticmethod
    def _resp(text):
        from backends.base import ModelResponse
        r = ModelResponse(text=text)
        r.latency_ms = 10.0
        r.token_usage = {}
        return r

    @staticmethod
    def _stdin(isatty, answers=()):
        class FakeStdin:
            def __init__(self, tty, ans):
                self._tty = tty
                self._ans = list(ans)
            def isatty(self):
                return self._tty
            def readline(self):
                return (self._ans.pop(0) if self._ans else "") + "\n"
        return FakeStdin(isatty, answers)

    @staticmethod
    def _args():
        return type("A", (), {"json": False, "cot": False, "proceed": False})()

    def test_non_tty_unclear_stops_for_clarification(self):
        def fake_gen(agent, cfg, args, route, req):
            return self._resp(
                "=== ENHANCED PROMPT ===\nEnhanced\n"
                "=== CLARIFYING QUESTIONS ===\nQ1: which file?")
        self.ask._generate_with_retry = fake_gen
        sys.stdin = self._stdin(False)
        task_for_qwen, enh, clar = self.ask._enhance_task(
            object(), {}, self._args(), "vague task", "")
        self.assertTrue(clar)
        self.assertIn("Q1: which file?", enh.clarifying_questions)

    def test_interactive_answer_re_enhances_and_uses_clarification(self):
        calls = []

        def fake_gen(agent, cfg, args, route, req):
            calls.append(req.user)
            if len(calls) == 1:
                return self._resp(
                    "=== ENHANCED PROMPT ===\nEnhanced v1\n"
                    "=== CLARIFYING QUESTIONS ===\nQ1: which file?")
            return self._resp("=== ENHANCED PROMPT ===\nEnhanced v2\n=== PLAN ===\np")
        self.ask._generate_with_retry = fake_gen
        sys.stdin = self._stdin(True, answers=["use users.py"])
        task_for_qwen, enh, clar = self.ask._enhance_task(
            object(), {}, self._args(), "vague task", "")
        self.assertEqual(len(calls), 2, "clarification must trigger a re-enhance")
        self.assertIn("USER CLARIFICATION", calls[1])
        self.assertIn("use users.py", calls[1])
        self.assertFalse(clar)
        self.assertEqual(task_for_qwen, "Enhanced v2")

    def test_interactive_skip_proceeds_with_enhanced_prompt(self):
        def fake_gen(agent, cfg, args, route, req):
            return self._resp(
                "=== ENHANCED PROMPT ===\nEnhanced\n"
                "=== CLARIFYING QUESTIONS ===\nQ1: scope?")
        self.ask._generate_with_retry = fake_gen
        sys.stdin = self._stdin(True, answers=[""])
        task_for_qwen, enh, clar = self.ask._enhance_task(
            object(), {}, self._args(), "task", "")
        self.assertFalse(clar)
        self.assertEqual(task_for_qwen, "Enhanced")


class SelfEvaluationTests(unittest.TestCase):
    """Self-evaluation: per-session metrics and the weekly comparison report."""

    def test_record_session_tracks_aggregates_and_period(self):
        import json
        import tempfile

        import ask
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        st = ask.StatsTracker(path)
        st.record_session(files_generated=3, iterations=2, truncation=True, quality=9.0)
        st.record_session(files_generated=5, iterations=1, truncation=False, quality=8.0)
        s = st.stats
        self.assertEqual(s["tasks_completed"], 2)
        self.assertEqual(s["files_generated"], 8)
        self.assertEqual(s["total_iterations"], 3)
        self.assertEqual(s["truncation_events"], 1)
        p = s["periods"][ask._iso_week()]
        self.assertEqual(p["files_generated"], 8)
        self.assertEqual(p["tasks_completed"], 2)
        self.assertEqual(p["truncation_events"], 1)

    def test_evaluate_reports_metrics_and_week_comparison(self):
        import json
        import tempfile
        from datetime import date, timedelta

        import ask
        iso = date.today().isocalendar()
        cur = f"{iso[0]}-W{iso[1]:02d}"
        lw = (date.today() - timedelta(days=7)).isocalendar()
        last = f"{lw[0]}-W{lw[1]:02d}"
        stats = {
            "tasks_completed": 5, "files_generated": 12, "total_iterations": 3,
            "truncation_events": 1, "selfeval_quality_sum": 46.0,
            "selfeval_quality_count": 5,
            "periods": {
                cur: {"files_generated": 12, "iterations": 3, "truncation_events": 1,
                      "tasks_completed": 5, "quality_sum": 46.0, "quality_count": 5},
                last: {"files_generated": 10, "iterations": 4, "truncation_events": 2,
                       "tasks_completed": 4, "quality_sum": 34.8, "quality_count": 4},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(stats, f)
            path = f.name
        r = ask.StatsTracker(path).evaluate()
        self.assertEqual(r["files_generated"], 12)
        self.assertEqual(r["iterations"], 3)
        self.assertEqual(r["tasks_completed"], 5)
        self.assertEqual(r["average_quality"], 9.2)
        self.assertEqual(r["truncation_rate"], 20.0)
        self.assertEqual(r["truncation_events"], 1)
        self.assertEqual(r["previous_average"], 8.7)
        self.assertIn("better than last week", r["comparison"])

    def test_evaluate_no_prior_period(self):
        import json
        import tempfile

        import ask
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        r = ask.StatsTracker(path).evaluate()
        self.assertIsNone(r["previous_average"])
        self.assertEqual(r["comparison"], "No prior period to compare against")
        self.assertEqual(r["average_quality"], 0.0)


class ProjectContextTests(unittest.TestCase):
    """Project Context Awareness: scan/deps/architecture/relevant-files."""

    def _fixture(self):
        import tempfile
        import json
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        (d / 'backend').mkdir()
        (d / 'models').mkdir()
        (d / 'tests').mkdir()
        (d / 'backend' / 'package.json').write_text(json.dumps({
            'dependencies': {'express': '^4', 'prisma': '^5', 'pg': '^8'}}))
        (d / 'backend' / 'app.py').write_text(
            'from fastapi import FastAPI\napp = FastAPI()\n'
            '@app.get("/health")\nasync def health() -> dict:\n'
            '    """Return health."""\n    return {"status": "ok"}\n')
        (d / 'models' / 'user.py').write_text(
            'from pydantic import BaseModel\nclass User(BaseModel):\n'
            '    id: int\n    name: str\n')
        (d / 'tests' / 'test_app.py').write_text(
            'def test_health():\n    assert True\n')
        return d

    def test_scan_detects_deps_and_architecture(self):
        from context import ProjectContext
        c = ProjectContext(str(self._fixture()))
        ctx = c.scan()
        self.assertIn('node', ctx['dependencies'])
        self.assertIn('express', ctx['dependencies']['node'])
        arch = ctx['architecture']
        self.assertIn(arch['backend_framework'], ('Express', 'FastAPI'))
        self.assertGreater(ctx['estimate_tokens'], 0)

    def test_find_relevant_files(self):
        from context import ProjectContext
        c = ProjectContext(str(self._fixture()))
        c.scan()
        relevant = c.find_relevant_files('add a user model', limit=3)
        self.assertTrue(any('user' in f for f in relevant))

    def test_recommend_libraries_for_auth(self):
        from context import ProjectContext
        c = ProjectContext(str(self._fixture()))
        c.scan()
        recs = c.recommend_libraries('add jwt auth')
        self.assertTrue(any('jwt' in r for r in recs))

    def test_to_prompt_contains_sections(self):
        from context import ProjectContext
        c = ProjectContext(str(self._fixture()))
        c.scan()
        prompt = c.to_prompt()
        for section in ('## Project Context', '### Structure',
                        '### Dependencies', '### Architecture'):
            self.assertIn(section, prompt)

    def test_scan_never_crashes_on_empty_dir(self):
        import tempfile
        from pathlib import Path
        from context import ProjectContext
        c = ProjectContext(str(Path(tempfile.mkdtemp())))
        ctx = c.scan()
        self.assertEqual(ctx['architecture'].get('type'), 'unknown')


class ChainOfThoughtTests(unittest.TestCase):
    """Chain-of-thought planning parsing."""

    def test_parser_extracts_sections(self):
        from supervise import ChainOfThoughtParser
        text = (
            "=== TASK UNDERSTANDING ===\nCRUD app\n"
            "=== CONSTRAINT ANALYSIS ===\n4096 cap -> 4 steps\n"
            "=== REASONING ===\nmodels first\n"
            "=== ALTERNATIVES ===\nraw sql vs orm\n"
            "=== FINAL PLAN ===\nStep 1: models\nStep 2: routes\n"
        )
        p = ChainOfThoughtParser(text)
        self.assertEqual(p.sections['task_understanding'], 'CRUD app')
        self.assertEqual(p.sections['constraint_analysis'], '4096 cap -> 4 steps')
        self.assertIn('models first', p.sections['reasoning'])
        self.assertIn('raw sql', p.sections['alternatives'])
        self.assertEqual(len(p.get_reasoning_chain()), 5)

    def test_enhance_request_cot_requests_sections(self):
        from supervise import _enhance_request
        req = _enhance_request('task', cot=True)
        self.assertIn('TASK UNDERSTANDING', req.system)
        self.assertIn('CONSTRAINT ANALYSIS', req.system)
        self.assertIn('ALTERNATIVES', req.system)
        self.assertEqual(req.max_tokens, 3200)
        req2 = _enhance_request('task', cot=False)
        self.assertNotIn('TASK UNDERSTANDING', req2.system)
        self.assertEqual(req2.max_tokens, 2400)


class ParallelTests(unittest.TestCase):
    """Parallel step execution: plan parsing + dependency grouping."""

    def test_parse_plan_steps_and_groups(self):
        from parallel import DependencyAnalyzer, parse_plan_steps
        plan = (
            "Step 1: Models\nDependencies: none\n"
            "Step 2: Schemas\nDependencies: none\n"
            "Step 3: Routes\nDependencies: 1, 2\n"
            "Step 4: Tests\nDependencies: 1, 2, 3\n"
        )
        steps = parse_plan_steps(plan)
        self.assertEqual([s['id'] for s in steps], [1, 2, 3, 4])
        groups = DependencyAnalyzer(steps).get_parallel_groups()
        self.assertEqual([[s['id'] for s in g] for g in groups],
                         [[1, 2], [3], [4]])

    def test_parse_plan_steps_fallback_single(self):
        from parallel import parse_plan_steps
        steps = parse_plan_steps("just prose, no steps here")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['name'], 'whole task')

    def test_execute_parallel_runs_all_and_orders(self):
        import threading
        import time
        from parallel import ParallelExecutor

        class FakeGen:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
            def __call__(self, req):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.lock:
                    self.active -= 1
                return type('R', (), {'text': 'out'})()
        steps = [{'id': 1, 'name': 'a', 'description': '', 'files': [], 'dependencies': []},
                 {'id': 2, 'name': 'b', 'description': '', 'files': [], 'dependencies': []}]
        ex = ParallelExecutor(FakeGen(), max_workers=4)
        res = ex.execute_parallel(steps, 'task')
        self.assertEqual([r['status'] for r in res], ['success', 'success'])
        self.assertEqual([r['id'] for r in res], [1, 2])


class FinalVerifyTests(unittest.TestCase):
    """Final error-check stage: run verification, have DeepSeek fix errors."""

    def test_looks_like_error(self):
        from ask import _looks_like_error
        self.assertTrue(_looks_like_error("error: cannot find module 'x'"))
        self.assertTrue(_looks_like_error("TypeScript error: Property 'x' has no"))
        self.assertTrue(_looks_like_error("Traceback (most recent call last)"))
        self.assertFalse(_looks_like_error("Build completed successfully."))

    def test_run_final_verify_passes_when_clean(self):
        from ask import _run_final_verify
        verified, report = _run_final_verify(None, ".", "task", ["echo ok"],
                                             lambda l: None, max_iter=1)
        self.assertTrue(verified)

    def test_run_final_verify_fixes_and_repairs(self):
        import tempfile
        from pathlib import Path
        from backends.base import ModelResponse
        from ask import _run_final_verify

        root = Path(tempfile.mkdtemp())

        class FakeCloud:
            def generate(self, req):
                return ModelResponse(text="```ok.txt\ncreated\n```\n")

        verified, report = _run_final_verify(
            FakeCloud(), str(root), "task", ["test -f ok.txt"],
            lambda l: None, max_iter=1)
        self.assertTrue(verified)
        self.assertTrue((root / "ok.txt").exists())

    def test_run_final_verify_no_cmds_returns_verified(self):
        from ask import _run_final_verify
        verified, report = _run_final_verify(None, ".", "task", [],
                                             lambda l: None)
        self.assertTrue(verified)

    def test_is_safe_verify_cmd(self):
        from ask import _is_safe_verify_cmd
        self.assertTrue(_is_safe_verify_cmd("npm run build"))
        self.assertTrue(_is_safe_verify_cmd("npx tsc --noEmit"))
        self.assertFalse(_is_safe_verify_cmd("rm -rf /"))
        self.assertFalse(_is_safe_verify_cmd("sudo npm test"))
        self.assertFalse(_is_safe_verify_cmd("echo hi; rm -rf /"))

    def test_truncate_error(self):
        from ask import _truncate_error
        short = "x" * 100
        self.assertEqual(_truncate_error(short, limit=200), short)
        long = "x" * 500
        out = _truncate_error(long, limit=200)
        self.assertLess(len(out), len(long))
        self.assertIn("truncated", out)

    def test_run_final_verify_blocks_unsafe_command(self):
        from ask import _run_final_verify
        verified, report = _run_final_verify(None, ".", "task", ["rm -rf /"],
                                             lambda l: None)
        self.assertFalse(verified)
        self.assertIn("blocked", report)

    def test_run_final_verify_rolls_back_fixes_in_git(self):
        import subprocess
        import tempfile
        from pathlib import Path
        from backends.base import ModelResponse
        from ask import _run_final_verify

        root = Path(tempfile.mkdtemp())
        for cmd, args in [
            (["git", "init"], {}),
            (["git", "config", "user.email", "t@t"], {}),
            (["git", "config", "user.name", "t"], {}),
            (["git", "add", "-A"], {}),
            (["git", "commit", "-m", "init", "--allow-empty"], {}),
        ]:
            subprocess.run(cmd, cwd=root, capture_output=True, text=True, **args)

        class FakeCloud:
            def generate(self, req):
                return ModelResponse(text="```ok.txt\ncreated\n```\n")

        # 'test -f never.txt' always fails, so after retries the fix (ok.txt)
        # must be rolled back to the pre-fix snapshot.
        verified, report = _run_final_verify(
            FakeCloud(), str(root), "task", ["test -f never.txt"],
            lambda l: None, max_iter=1)
        self.assertFalse(verified)
        self.assertFalse((root / "ok.txt").exists(),
                         "failed verification must roll back the AI's fixes")

    def test_is_environmental_error(self):
        from ask import _is_environmental_error
        self.assertTrue(_is_environmental_error("sh: cmd: command not found"))
        self.assertTrue(_is_environmental_error("ENOENT: no such file"))
        self.assertTrue(_is_environmental_error("Module not found: 'react'"))
        self.assertFalse(_is_environmental_error("TypeError: x is undefined"))

    def test_run_final_verify_skips_deepseek_on_env_error(self):
        import json
        import tempfile
        from pathlib import Path
        from ask import _run_final_verify

        root = Path(tempfile.mkdtemp())
        (root / "package.json").write_text(json.dumps({
            "scripts": {"build": "echo 'command not found'; exit 1"}}))

        class FakeCloud:
            def __init__(self):
                self.calls = 0
            def generate(self, req):
                self.calls += 1
                return type("R", (), {"text": "```ok.txt\nx\n```"})

        cloud = FakeCloud()
        verified, report = _run_final_verify(
            cloud, str(root), "task", ["npm run build"],
            lambda l: None, max_iter=1)
        self.assertFalse(verified)
        self.assertEqual(cloud.calls, 0, "must NOT call DeepSeek on env errors")

    def test_run_regression_guard(self):
        import tempfile
        from pathlib import Path
        from ask import _run_regression_guard

        root = Path(tempfile.mkdtemp())
        passed, report = _run_regression_guard(str(root), ["echo ok"],
                                               lambda l: None)
        self.assertTrue(passed)
        passed, report = _run_regression_guard(str(root), ["test -f missing"],
                                               lambda l: None)
        self.assertFalse(passed)

    def test_record_verify(self):
        import tempfile
        import ask
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        st = ask.StatsTracker(path)
        st.record_verify({"iterations": 2, "api_calls": 1, "tokens_used": 400,
                          "estimated_cost_usd": 0.001, "status": "PASSED"})
        st.record_verify({"iterations": 3, "api_calls": 2, "tokens_used": 800,
                          "estimated_cost_usd": 0.002, "status": "FAILED"})
        v = st.stats["verify"]
        self.assertEqual(v["runs"], 2)
        self.assertEqual(v["iterations"], 5)
        self.assertEqual(v["api_calls"], 3)
        self.assertEqual(v["tokens_used"], 1200)
        self.assertEqual(v["passed"], 1)
        self.assertEqual(v["failed"], 1)


if __name__ == "__main__":
    unittest.main()
