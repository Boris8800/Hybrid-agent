"""Trust & learning tests: secrets/PII scanner + redaction wrapper, engineering
rules, failure-rule learner, and the dependency manifest gate.

Run from hybrid-agent/:
    python -m unittest tests.test_trust_learning -v
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import ModelRequest, ModelResponse  # noqa: E402
from dependency_gate import detect_new_dependencies, run_dependency_gate  # noqa: E402
from learned_rules import LearnedRules, classify_error  # noqa: E402
import rules  # noqa: E402
from rules import EngineeringRules, load_engineering_rules  # noqa: E402
from scanner import redact_cloud, redact_text, scan_text  # noqa: E402

# Fake secrets built at runtime so GitHub push-protection never sees literal
# key-like strings in the repo (the scanner itself still matches them).
_FAKE_STRIPE = "sk_live_" + "abcdefghijklmnop1234567890"
_FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
_FAKE_JWT = ("eyJhbGciOiJIUzI1NiJ9." +
             "eyJzdWIiOiIxMjM0NTY3ODkwIn0." +
             "abcdefghijklmnopqrstuvwxyz123456")
_FAKE_DB_URL = "postgres://admin:hunter2@db:5432/app"
SECRET_TEXT = (f"{_FAKE_STRIPE} stripe key, "
               f"{_FAKE_AWS} aws, {_FAKE_JWT} jwt, "
               f"{_FAKE_DB_URL} conn, alice@example.com email")


class TestSecretsScanner(unittest.TestCase):
    def test_scan_finds_types(self):
        found = scan_text(SECRET_TEXT)
        for t in ("stripe", "aws_key", "jwt", "conn_string", "email"):
            self.assertIn(t, found, f"{t} not detected")

    def test_redact_masks_and_counts(self):
        res = redact_text(SECRET_TEXT)
        self.assertNotIn("hunter2", res.redacted)
        self.assertNotIn("alice@example.com", res.redacted)
        self.assertNotIn(_FAKE_AWS, res.redacted)
        self.assertIn("<REDACTED:jwt>", res.redacted)
        types = [f["type"] for f in res.findings]
        self.assertIn("email", types)

    def test_types_filter(self):
        res = redact_text(SECRET_TEXT, types=["email"])
        self.assertNotIn("alice@example.com", res.redacted)
        self.assertIn(_FAKE_AWS, res.redacted)  # untouched

    def test_redact_cloud_wrapper(self):
        captured = {}

        class _Cloud:
            def generate(self, req):
                captured["user"] = req.user
                return ModelResponse(text="ok", backend="deepseek")

        cfg = {"secrets_scan": {"mode": "redact"}}
        wrapped = redact_cloud(_Cloud(), cfg)
        req = ModelRequest(system="sys", user="fix the key sk-abc1234567890abcdef")
        wrapped.generate(req)
        self.assertNotIn("sk-abc", captured["user"])
        self.assertIn("<REDACTED:api_key>", captured["user"])

    def test_redact_cloud_block_raises(self):
        class _Cloud:
            def generate(self, req):
                raise AssertionError("must not send")

        cfg = {"secrets_scan": {"mode": "block"}}
        wrapped = redact_cloud(_Cloud(), cfg)
        with self.assertRaises(RuntimeError):
            wrapped.generate(ModelRequest(system="s", user="key sk-abc1234567890abcdef"))

    def test_redact_off_is_passthrough(self):
        cloud = object()
        self.assertIs(redact_cloud(cloud, {"secrets_scan": {"mode": "off"}}), cloud)


class TestEngineeringRules(unittest.TestCase):
    YML = """
architecture:
  frontend: Next.js
  backend: NestJS
  database: PostgreSQL
rules:
  - never access the database from a controller
  - all API responses use DTOs
dependencies:
  block: [left-pad]
  allow: [react]
"""

    @unittest.skipUnless(rules.HAS_YAML, "pyyaml not installed")
    def test_load_and_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / ".agent"
            d.mkdir()
            (d / "engineering-rules.yml").write_text(self.YML)
            r = load_engineering_rules(tmp)
            self.assertTrue(r.active)
            self.assertEqual(r.architecture["backend"], "NestJS")
            self.assertEqual(len(r.rules), 2)
            self.assertIn("left-pad", r.blocked_dependencies)
            self.assertIn("react", r.allowed_dependencies)
            self.assertIn("ENGINEERING RULES", r.prompt_text())

    def test_missing_rules_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = load_engineering_rules(tmp)
            self.assertFalse(r.active)
            self.assertEqual(r.prompt_text(), "")

    def test_dependency_block(self):
        r = EngineeringRules(blocked_dependencies=["left-pad"])
        self.assertIsNotNone(r.check_dependency("left-pad"))
        self.assertIsNone(r.check_dependency("react"))


class TestLearnedRules(unittest.TestCase):
    def test_classify_error(self):
        self.assertEqual(classify_error("Cannot find module './service'"), "missing-module")
        self.assertEqual(classify_error("TS2339: Property does not exist"), "type-error")
        self.assertEqual(classify_error("TypeError: x is not a function"), "runtime-error")
        self.assertEqual(classify_error("weird thing"), "unknown")

    def test_rules_emerge_after_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr = LearnedRules(root=tmp, threshold=3)
            for i in range(3):
                lr.record_failure("backend api",
                                  "Cannot find module './payment.service'",
                                  ["src/checkout/service.ts", "src/checkout/test.ts"])
            rules = lr.rules()
            self.assertEqual(len(rules), 1)
            self.assertIn("backend api", rules[0])
            self.assertIn("missing-module", rules[0])
            self.assertIn("service.ts", rules[0])
            self.assertIn("service.ts", lr.suggested_files())
            self.assertIn("LEARNED RULES", lr.prompt_text())

    def test_below_threshold_no_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            lr = LearnedRules(root=tmp, threshold=3)
            lr.record_failure("x", "Cannot find module", ["a.ts"])
            self.assertEqual(lr.rules(), [])


class TestDependencyGate(unittest.TestCase):
    def _repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "ws"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo)
        subprocess.run(["git", "config", "user.name", "T"], cwd=repo)
        (repo / "package.json").write_text(
            '{\n  "dependencies": {\n    "react": "^18.0.0",\n    "base-dep": "1.0.0"\n  }\n}\n')
        subprocess.run(["git", "add", "-A"], cwd=repo)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo)
        return repo

    def test_detect_new_npm_dep(self):
        repo = self._repo()
        (repo / "package.json").write_text(
            '{\n  "dependencies": {\n    "new-pkg": "^1.0.0",\n    "react": "^18.0.0",\n    "base-dep": "1.0.0"\n  }\n}\n')
        deps = detect_new_dependencies(str(repo))
        self.assertEqual([d["package"] for d in deps], ["new-pkg"])

    def test_gate_requires_approval(self):
        repo = self._repo()
        (repo / "package.json").write_text(
            '{\n  "dependencies": {\n    "new-pkg": "^1.0.0",\n    "react": "^18.0.0",\n    "base-dep": "1.0.0"\n  }\n}\n')
        ok, report, deps = run_dependency_gate(
            str(repo), EngineeringRules(), lambda line: None)
        self.assertFalse(ok)
        self.assertIn("NEW DEPENDENCY DETECTED", report)
        self.assertIn("new-pkg", report)

    def test_gate_allow_passes(self):
        repo = self._repo()
        (repo / "package.json").write_text(
            '{\n  "dependencies": {\n    "react-dom": "^18.0.0",\n    "react": "^18.0.0",\n    "base-dep": "1.0.0"\n  }\n}\n')
        ok, report, deps = run_dependency_gate(
            str(repo), EngineeringRules(allowed_dependencies=["react-dom"]),
            lambda line: None)
        self.assertTrue(ok, report)

    def test_gate_blocked_by_rules(self):
        repo = self._repo()
        (repo / "package.json").write_text(
            '{\n  "dependencies": {\n    "left-pad": "^1.0.0",\n    "react": "^18.0.0",\n    "base-dep": "1.0.0"\n  }\n}\n')
        ok, report, deps = run_dependency_gate(
            str(repo), EngineeringRules(blocked_dependencies=["left-pad"]),
            lambda line: None)
        self.assertFalse(ok)
        self.assertIn("blocked", report)


if __name__ == "__main__":
    unittest.main()
