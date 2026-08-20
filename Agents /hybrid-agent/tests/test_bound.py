"""BOUND (Ouro Loop) tests: runtime enforcement, program phases with
remediation, and the RECALL gate (bound_text re-injected every iteration).

Run from hybrid-agent/:
    python -m unittest tests.test_bound -v
"""

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelResponse  # noqa: E402
from bound import Bound, load_bound  # noqa: E402
from supervise import supervise  # noqa: E402

APPROVED_TEXT = ("=== REVIEW DECISION ===\nAPPROVED\n\n=== QUALITY SCORE ===\n8.0\n\n"
                 "=== OVERALL ASSESSMENT ===\nFine.\n")


class TestBoundModel(unittest.TestCase):
    def test_defaults_when_section_missing(self):
        b = load_bound({})
        self.assertTrue(b.active)
        self.assertIn("**/.env", b.danger_zones)
        self.assertIn("rm -rf", b.never_do)

    def test_config_overrides_defaults(self):
        b = load_bound({"bound": {"danger_zones": ["**/secret/**"],
                                  "never_do": ["git push -f"],
                                  "iron_laws": ["Never X."]}})
        self.assertEqual(b.danger_zones, ["**/secret/**"])
        self.assertEqual(b.iron_laws, ["Never X."])
        self.assertIsNone(b.enforce_path("src/app.ts"))
        self.assertIsNotNone(b.enforce_path("config/secret/token.json"))
        self.assertIsNotNone(b.check_command("git push -f origin main"))

    def test_enforce_path_danger_zones(self):
        b = Bound(danger_zones=["**/.env", "**/.env.*", "**/*.pem", "src/private/**"],
                  never_do=[], iron_laws=[])
        self.assertIsNotNone(b.enforce_path(".env"))
        self.assertIsNotNone(b.enforce_path("config/.env.local"))
        self.assertIsNotNone(b.enforce_path("certs/app.pem"))
        self.assertIsNotNone(b.enforce_path("src/private/keys.txt"))
        self.assertIsNone(b.enforce_path("src/app.ts"))
        self.assertIsNone(b.enforce_path("src/utils/helper.js"))

    def test_check_command_never_do(self):
        b = Bound(danger_zones=[], never_do=["git push --force", "sudo "], iron_laws=[])
        self.assertIsNotNone(b.check_command("git push --force origin main"))
        self.assertIsNotNone(b.check_command("sudo npm install -g x"))
        self.assertIsNone(b.check_command("npm run build"))
        self.assertIsNone(b.check_command("git push origin main"))

    def test_prompt_text_recall_gate(self):
        b = Bound(danger_zones=["**/.env"], never_do=["rm -rf"],
                  iron_laws=["Never touch secrets."])
        text = b.prompt_text()
        self.assertIn("IRON LAWS", text)
        self.assertIn("Never touch secrets.", text)
        self.assertIn("DANGER ZONES", text)
        self.assertIn("NEVER RUN", text)


class TestApplyBoundEnforcement(unittest.TestCase):
    def _tmp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_danger_zone_never_written(self):
        root = self._tmp()
        b = load_bound({})  # includes **/.env
        text = ".env\n```\nSECRET=1\n```\napp.ts\n```ts\nconst x = 1\n```"
        written, skipped = ask._apply_fenced_files(text, root=root, bound=b)
        self.assertEqual([rel for rel, _ in written], ["app.ts"])
        self.assertTrue(any("(bound:" in s for s in skipped))
        self.assertFalse((Path(root) / ".env").exists())
        self.assertEqual((Path(root) / "app.ts").read_text(), "const x = 1\n")

    def test_no_bound_no_change(self):
        root = self._tmp()
        text = ".env\n```\nSECRET=1\n```"
        written, skipped = ask._apply_fenced_files(text, root=root, bound=None)
        self.assertEqual([rel for rel, _ in written], [".env"])
        self.assertEqual((Path(root) / ".env").read_text(), "SECRET=1\n")


class TestTerminalToolBound(unittest.TestCase):
    def test_never_do_blocked_even_if_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = load_bound({})  # includes git push --force? default: yes
            tool = ask._make_terminal_tool({"review": {"terminal_timeout": 10}},
                                           Namespace(root=tmp), bound=b)
            out = tool("git push --force origin main")
            self.assertIn("BOUND", out)
            out = tool("npm run build")
            self.assertNotIn("BLOCKED", out)


class TestProgramPhases(unittest.TestCase):
    def _cfg(self, on_fail):
        return {"program": {"phases": [
            {"name": "build", "gate": ["npm run build"],
             "max_fix_rounds": 2, "on_fail": on_fail},
        ]}}

    def _patch(self, errors_seq, fix_text="app.ts\n```ts\nconst x = 1\n```"):
        import contextlib
        import io
        calls = {"verify": 0, "fix": 0, "apply": 0}
        orig_v = ask._run_verify_commands
        orig_f = ask._deepseek_fix

        def fake_verify(root, cmds, status, timeout_s, **kw):
            calls["verify"] += 1
            err = errors_seq[min(calls["verify"] - 1, len(errors_seq) - 1)]
            return err

        def fake_fix(cloud, task, error_text, cache=None, root="."):
            calls["fix"] += 1
            return fix_text

        ask._run_verify_commands = fake_verify
        ask._deepseek_fix = fake_fix
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                ok, report = ask._run_program_phases(
                    object(), tmp, "t", load_bound({}), self._cfg("retry"),
                    lambda line: None, cache=None)
        finally:
            ask._run_verify_commands = orig_v
            ask._deepseek_fix = orig_f
        return ok, report, calls

    def test_retry_recovers(self):
        ok, report, calls = self._patch([["build error"], []])
        self.assertTrue(ok, report)
        self.assertEqual(calls["verify"], 2)
        self.assertEqual(calls["fix"], 1)

    def test_no_phases_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, report = ask._run_program_phases(object(), tmp, "t", load_bound({}),
                                                 {}, lambda line: None)
        self.assertTrue(ok)

    def test_revert_playbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, report = ask._run_program_phases(
                object(), tmp, "t", load_bound({}),
                self._cfg("revert"), lambda line: None)
        self.assertFalse(ok)
        self.assertIn("reverted", report)

    def test_escalate_playbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, report = ask._run_program_phases(
                object(), tmp, "t", load_bound({}),
                self._cfg("escalate"), lambda line: None)
        self.assertFalse(ok)
        self.assertIn("escalated", report)

    def test_bound_violation_in_fix_reverts(self):
        ok, report, calls = self._patch([["err"]], fix_text=".env\n```\nSECRET=1\n```")
        self.assertFalse(ok)
        self.assertIn("BOUND", report)


class TestRecallGate(unittest.TestCase):
    def test_bound_text_in_every_gemma_prompt(self):
        systems = []

        def fake_gemma(req):
            systems.append(req.system)
            if len(systems) == 1:
                return ModelResponse(text="RUN: npm test\n```\napp.ts\nx\n```", backend="local")
            return ModelResponse(text="```\napp.ts\nx\n```", backend="local")

        class _Cloud:
            def generate(self, req):
                return ModelResponse(text=APPROVED_TEXT, backend="deepseek")

        supervise(local=None, cloud=_Cloud(), task="t", gemma_generate=fake_gemma,
                  terminal_tool=lambda c: "exit 0\nok",
                  bound_text="IRON LAWS — THE BOUND (test)",
                  status=lambda line: None)
        self.assertGreaterEqual(len(systems), 2)
        for s in systems:
            self.assertIn("IRON LAWS — THE BOUND (test)", s)


if __name__ == "__main__":
    unittest.main()
