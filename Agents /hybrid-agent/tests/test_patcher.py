"""AST-aware / diff-first repair tests: unified-diff application (git apply),
BOUND gating of patched files, AST context extraction, and _apply_repair.

Run from hybrid-agent/:
    python -m unittest tests.test_patcher -v
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from bound import load_bound  # noqa: E402
from patcher import (apply_unified_diff, diff_paths, extract_context,
                     extract_error_location, is_diff)  # noqa: E402

DIFF_TEXT = """--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
-def old():
+def new():
"""


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True).returncode == 0


class TestDiffPrimitives(unittest.TestCase):
    def test_is_diff(self):
        self.assertTrue(is_diff(DIFF_TEXT))
        self.assertTrue(is_diff("diff --git a/x b/x\nindex 123..456 100644\n"))
        self.assertFalse(is_diff("app.py\n```py\nx=1\n```"))

    def test_diff_paths(self):
        self.assertEqual(diff_paths(DIFF_TEXT), ["src/app.py"])
        both = "--- a/a.ts\n+++ b/b.ts\n"
        self.assertIn("a.ts", diff_paths(both))
        self.assertIn("b.ts", diff_paths(both))

    def test_error_location(self):
        loc = extract_error_location("Error in src/app.py:42: x is undefined")
        self.assertEqual(loc, ("src/app.py", 42))
        self.assertIsNone(extract_error_location("no location here"))


class TestApplyUnifiedDiff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "ws"
        self.repo.mkdir()
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "T")
        (self.repo / "app.py").write_text("def old():\n    return 1\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "base")

    def _real_diff(self):
        (self.repo / "app.py").write_text("def new():\n    return 2\n")
        diff = subprocess.run(["git", "diff"], cwd=self.repo, capture_output=True,
                              text=True).stdout
        _git(self.repo, "checkout", "-q", "app.py")  # revert to base
        return diff

    def test_applies_real_diff(self):
        diff = self._real_diff()
        res = apply_unified_diff(str(self.repo), diff)
        self.assertEqual(res["bound_violation"], False)
        self.assertTrue(res["applied"])
        self.assertIn("def new()", (self.repo / "app.py").read_text())
        self.assertGreater(res["original_bytes"], 0)

    def test_bound_rejects_patch_to_danger_zone(self):
        bound = load_bound({})  # includes **/.env
        diff = "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-X\n+X2\n"
        res = apply_unified_diff(str(self.repo), diff, bound=bound)
        self.assertTrue(res["bound_violation"])
        self.assertFalse(res["applied"])

    def test_invalid_diff_returns_skipped(self):
        res = apply_unified_diff(str(self.repo), "--- a/x\n+++ b/x\n@@ garbage")
        self.assertFalse(res["applied"])
        self.assertTrue(any("did not apply" in s for s in res["skipped"]))


class TestExtractContext(unittest.TestCase):
    def test_python_ast_locates_function(self):
        src = ("import os\n\n\ndef helper():\n    return 1\n\n\n"
               "def target(x):\n    if x:\n        return helper()\n"
               "    return None\n\n\ndef other():\n    pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mod.py"
            p.write_text(src)
            block = extract_context(str(p), 10)  # inside target()
            self.assertIn("def target(x):", block)
            self.assertNotIn("def helper()", block)
            self.assertNotIn("def other()", block)
            # Inside helper() -> returns the helper block.
            self.assertIn("def helper()", extract_context(str(p), 4))

    def test_ts_scope_scan(self):
        src = ("const a = 1;\n\n"
               "function render() {\n  const b = 2;\n  return <div>{b}</div>;\n}\n\n"
               "const c = 3;\n")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.tsx"
            p.write_text(src)
            block = extract_context(str(p), 4)
            self.assertIn("function render()", block)
            self.assertNotIn("const c = 3", block)

    def test_missing_file(self):
        self.assertEqual(extract_context("/nonexistent/x.py", 1), "")


class TestApplyRepair(unittest.TestCase):
    def test_diff_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ws"
            repo.mkdir()
            _git(repo, "init", "-q", "-b", "main")
            _git(repo, "config", "user.email", "t@t")
            _git(repo, "config", "user.name", "T")
            # A big enough file that a surgical diff saves real tokens.
            body = "\n".join(f"    line_{i} = {i}" for i in range(120))
            (repo / "app.py").write_text("def old():\n" + body + "\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", "base")
            diff = ("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
                    "-def old():\n+def new():\n")
            written, skipped, stats = ask._apply_repair(diff, str(repo))
            self.assertEqual(stats["mode"], "diff")
            self.assertIn("def new()", (repo / "app.py").read_text())
            self.assertGreater(stats.get("saved_tokens_est", 0), 0)

    def test_full_file_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = "app.py\n```py\ndef new():\n    return 2\n```"
            written, skipped, stats = ask._apply_repair(text, tmp)
            self.assertEqual(stats["mode"], "full")
            self.assertEqual([rel for rel, _ in written], ["app.py"])
            self.assertIn("def new()", (Path(tmp) / "app.py").read_text())


if __name__ == "__main__":
    unittest.main()
