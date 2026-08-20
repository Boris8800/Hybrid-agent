"""Terminal-tool tests: RUN: block parsing, the supervise terminal loop, and
the ask.py tool factory (allowlist gating).

Run from hybrid-agent/:
    python -m unittest tests.test_terminal_tool -v
"""

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelResponse  # noqa: E402
from supervise import ReviewPackage, _extract_run_blocks, supervise  # noqa: E402

APPROVED_TEXT = (
    "=== REVIEW DECISION ===\nAPPROVED\n\n=== EVIDENCE ===\n"
    "terminal session exit 0; file.py written\n\n=== QUALITY SCORE ===\n8.0\n\n"
    "=== OVERALL ASSESSMENT ===\nSolid change, terminal session confirms it.\n"
)


class TestExtractRunBlocks(unittest.TestCase):
    def test_extracts_and_dedupes(self):
        text = "RUN: npm run build\ncode\nRUN: npx tsc --noEmit\nRUN: npm run build"
        self.assertEqual(_extract_run_blocks(text),
                         ["npm run build", "npx tsc --noEmit"])

    def test_ignores_non_run_lines(self):
        self.assertEqual(_extract_run_blocks("TASK: fix\na = 1\nRUN  x"), [])
        self.assertEqual(_extract_run_blocks(""), [])


class TestSuperviseTerminalLoop(unittest.TestCase):
    def _run(self, qwen_outputs, max_rounds=3):
        qwen_calls = []
        term_calls = []
        pkgs = []

        def fake_qwen(req):
            qwen_calls.append(req)
            text = qwen_outputs[min(len(qwen_calls) - 1, len(qwen_outputs) - 1)]
            return ModelResponse(text=text, backend="local")

        def fake_terminal(cmd):
            term_calls.append(cmd)
            return "exit 0\nbuild ok"

        class _Cloud:
            def generate(self, req):
                return ModelResponse(text=APPROVED_TEXT, backend="deepseek")

        result = supervise(
            local=None, cloud=_Cloud(), task="fix the build",
            qwen_generate=fake_qwen, terminal_tool=fake_terminal,
            max_terminal_rounds=max_rounds, status=lambda line: None,
            package_builder=lambda t, c, i: ReviewPackage(
                task=t, changes=f"iter {i}:\n{c}", verification=""),
        )
        return result, qwen_calls, term_calls

    def test_runs_tool_then_review(self):
        result, qwen_calls, term_calls = self._run([
            "RUN: npm run build\n```\nfile.py\nx=1\n```",
            "```\nfile.py\nx=1\n```",   # no more RUN -> review proceeds
        ])
        self.assertEqual(len(qwen_calls), 2)
        self.assertEqual(term_calls, ["npm run build"])
        self.assertEqual(result.verdicts[0].decision, "APPROVED")

    def test_terminal_session_in_package_verification(self):
        pkgs = []
        def builder(t, c, i):
            p = ReviewPackage(task=t, changes=c, verification="")
            pkgs.append(p)
            return p
        term_calls = []
        def fake_qwen(req):
            if not hasattr(fake_qwen, "n"):
                fake_qwen.n = 1
                return ModelResponse(text="RUN: pytest\n```\nfile.py\nx\n```", backend="local")
            return ModelResponse(text="```\nfile.py\nx\n```", backend="local")
        class _Cloud:
            def generate(self, req):
                return ModelResponse(text=APPROVED_TEXT, backend="deepseek")
        supervise(local=None, cloud=_Cloud(), task="t", qwen_generate=fake_qwen,
                  terminal_tool=lambda c: term_calls.append(c) or "exit 0\n1 passed",
                  status=lambda line: None, package_builder=builder)
        self.assertIn("TERMINAL SESSION", pkgs[0].verification)
        self.assertIn("exit 0", pkgs[0].verification)

    def test_max_rounds_limits_tool_calls(self):
        # Qwen keeps requesting commands; the loop must stop at max rounds.
        result, qwen_calls, term_calls = self._run(
            ["RUN: pytest\n```\nfile.py\nx\n```"], max_rounds=2)
        self.assertEqual(len(term_calls), 2)      # capped
        self.assertEqual(len(qwen_calls), 3)     # initial + 2 re-generations
        self.assertEqual(result.verdicts[0].decision, "APPROVED")

    def test_no_tool_no_extra_calls(self):
        qwen_calls = []
        class _Cloud:
            def generate(self, req):
                return ModelResponse(text=APPROVED_TEXT, backend="deepseek")
        supervise(local=None, cloud=_Cloud(), task="t",
                  qwen_generate=lambda req: qwen_calls.append(req) or ModelResponse(
                      text="RUN: npm test\n```\nf\n```", backend="local"),
                  terminal_tool=None, status=lambda line: None)
        self.assertEqual(len(qwen_calls), 1)  # RUN ignored without the tool


class TestTerminalToolFactory(unittest.TestCase):
    def test_blocked_command_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = ask._make_terminal_tool({"review": {"terminal_timeout": 10}},
                                           Namespace(root=tmp))
            out = tool("rm -rf /")
            self.assertIn("BLOCKED", out)

    def test_allowlisted_command_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = ask._make_terminal_tool({"review": {"terminal_timeout": 10}},
                                           Namespace(root=tmp))
            out = tool("echo hi")
            self.assertIn("exit 0", out)
            self.assertIn("hi", out)

    def test_read_only_inspection_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = ask._make_terminal_tool({"review": {"terminal_timeout": 10}},
                                           Namespace(root=tmp))
            out = tool("pwd")
            self.assertIn("exit 0", out)
            self.assertIn(tmp, out)
            out = tool("ls")
            self.assertIn("exit 0", out)
            # Build/test commands remain allowed through the verify allowlist.
            out = tool("echo hello")
            self.assertNotIn("BLOCKED", out)

    def test_marker_gate_beats_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = ask._make_terminal_tool({"review": {"terminal_timeout": 10}},
                                           Namespace(root=tmp))
            # 'cat' is allowlisted, but the redirect makes it destructive.
            self.assertIn("BLOCKED", tool("cat x > /tmp/evil"))


if __name__ == "__main__":
    unittest.main()
