"""Differential anti-gaming gate + structured guardrails tests.

Run from hybrid-agent/:
    python -m unittest tests.test_differential_guardrails -v
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
import differential  # noqa: E402
from differential import analyze_changes  # noqa: E402
from guardrails import check_guardrails  # noqa: E402

BASE_TEST = ("test('login works', () => {\n"
             "  expect(x).toBe(1);\n"
             "  expect(y).toEqual(2);\n"
             "  assert(x === y);\n"
             "  expect(z).toBe(3);\n"
             "});\n")
BASE_TSCONFIG = '{"compilerOptions": {"strict": true}}\n'


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True).returncode == 0


class DifferentialFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "ws"
        self.repo.mkdir()
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "T")
        (self.repo / "auth.test.ts").write_text(BASE_TEST)
        (self.repo / "tsconfig.json").write_text(BASE_TSCONFIG)
        (self.repo / "config.yml").write_text("review:\n  verify: []\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "base")

    def _violations(self):
        return analyze_changes(str(self.repo))

    def test_clean_diff_no_violations(self):
        self.assertEqual(self._violations(), [])

    def test_skipped_test_is_flagged(self):
        (self.repo / "auth.test.ts").write_text(BASE_TEST + "it.skip('x', () => {});\n")
        v = self._violations()
        self.assertTrue(any("added test weakening" in x and "skip" in x.lower() for x in v))

    def test_removed_assertions_are_flagged(self):
        (self.repo / "auth.test.ts").write_text(
            "test('login works', () => {\n  // assertions removed\n});\n")
        v = self._violations()
        self.assertTrue(any("assertion(s) removed" in x for x in v))

    def test_tsconfig_loosening_flagged(self):
        (self.repo / "tsconfig.json").write_text(
            '{"compilerOptions": {"strict": false, "skipLibCheck": true}}\n')
        v = self._violations()
        self.assertTrue(any("type-check loosened" in x for x in v))

    def test_guard_config_edit_flagged(self):
        (self.repo / "config.yml").write_text("review:\n  verify: ['npm run build']\n")
        v = self._violations()
        self.assertTrue(any("guard/config file modified" in x for x in v))


class TestGuardrails(unittest.TestCase):
    CFG = {"guardrails": {"block": ["drop table"],
                          "approval_required": ["api_key"],
                          "cost_limit": 1000.0}}

    def test_block_content(self):
        decision, reason = check_guardrails("run DROP TABLE users", self.CFG)
        self.assertEqual(decision, "block")
        self.assertIn("drop table", reason)

    def test_approval_required_content(self):
        decision, reason = check_guardrails("rotate the api_key in prod", self.CFG)
        self.assertEqual(decision, "approval_required")
        self.assertIn("api_key", reason)

    def test_cost_gate(self):
        cfg = {"guardrails": {"cost_limit": 0.0001}}
        decision, reason = check_guardrails("a " * 500, cfg)
        self.assertEqual(decision, "approval_required")
        self.assertIn("estimated cost", reason)

    def test_allow(self):
        decision, reason = check_guardrails("fix the checkout button", self.CFG)
        self.assertEqual(decision, "allow")
        self.assertEqual(reason, "")


class TestDifferentialGate(unittest.TestCase):
    def _gate(self, violation_seq, fix_text="auth.test.ts\n```ts\nx\n```"):
        calls = {"fix": 0}
        orig_a = ask.analyze_changes
        orig_f = ask._deepseek_fix
        seq = list(violation_seq)

        def fake_analyze(root, paths=None):
            return seq.pop(0) if seq else []

        def fake_fix(cloud, task, error_text, cache=None, root="."):
            calls["fix"] += 1
            return fix_text

        ask.analyze_changes = fake_analyze
        ask._deepseek_fix = fake_fix
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                ok, report = ask._run_differential_gate(
                    object(), tmp, "t", None, lambda line: None, cache=None,
                    max_iter=2)
        finally:
            ask.analyze_changes = orig_a
            ask._deepseek_fix = orig_f
        return ok, report, calls

    def test_restored_gate_recovers(self):
        ok, report, calls = self._gate([["auth.test.ts: added test weakening"],
                                        []])
        self.assertTrue(ok, report)
        self.assertEqual(calls["fix"], 1)

    def test_persistent_violation_fails(self):
        ok, report, calls = self._gate([["v1"], ["v2"], ["v3"]])
        self.assertFalse(ok)
        self.assertEqual(calls["fix"], 2)  # max_iter rounds

    def test_clean_first_pass(self):
        ok, report, calls = self._gate([])
        self.assertTrue(ok)
        self.assertEqual(calls["fix"], 0)


if __name__ == "__main__":
    unittest.main()
