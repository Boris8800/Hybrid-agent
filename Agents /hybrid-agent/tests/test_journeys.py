"""Journey verification (Vibe DSL) tests: DSL parsing, the headless runner,
the DeepSeek fix loop, program-phase journeys, and Write-One BOUND protection.

Run from hybrid-agent/:
    python -m unittest tests.test_journeys -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
import journeys  # noqa: E402
from bound import load_bound  # noqa: E402
from journeys import JourneyError, load_journeys, run_journeys  # noqa: E402

VALID_YML = """
app_url: "http://localhost:3000"
journeys:
  - name: "login page renders"
    steps:
      - visit: "/"
      - see_text: "Welcome"
      - see_element: "button[type=submit]"
  - name: "login flow"
    steps:
      - visit: "/login"
      - fill: { selector: "#email", value: "alice@example.com" }
      - click: "button[type=submit]"
      - wait_selector: ".dashboard"
"""


class TestDslParsing(unittest.TestCase):
    @unittest.skipUnless(journeys.HAS_YAML, "pyyaml not installed")
    def test_valid_yaml_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "journeys.yml"
            p.write_text(VALID_YML)
            data = load_journeys(str(p))
            self.assertEqual(len(data["journeys"]), 2)
            self.assertEqual(data["app_url"], "http://localhost:3000")

    def test_missing_file(self):
        with self.assertRaises(JourneyError):
            load_journeys("/nonexistent/journeys.yml")

    def test_unknown_step_rejected(self):
        bad = "journeys:\n  - name: x\n    steps:\n      - explode: true\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "j.yml"
            p.write_text(bad)
            with self.assertRaises(JourneyError):
                load_journeys(str(p))

    def test_empty_journeys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "j.yml"
            p.write_text("app_url: http://x\njourneys: []\n")
            with self.assertRaises(JourneyError):
                load_journeys(str(p))


class TestRunner(unittest.TestCase):
    def test_missing_playwright_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "j.yml"
            p.write_text(VALID_YML)
            orig = journeys.HAS_PLAYWRIGHT
            journeys.HAS_PLAYWRIGHT = False
            try:
                ok, report = run_journeys(str(p))
            finally:
                journeys.HAS_PLAYWRIGHT = orig
            self.assertFalse(ok)
            self.assertIn("JOURNEYS UNAVAILABLE", report)

    @unittest.skipUnless(journeys.HAS_YAML, "pyyaml not installed")
    def test_runner_with_fake_browser(self):
        class _Loc:
            def __init__(self, text=""):
                self.text = text
            def count(self):
                return 1
            def inner_text(self, timeout=0):
                return self.text
            def wait_for(self, state="visible", timeout=0):
                return None
            def click(self, timeout=0):
                return None
            def fill(self, value, timeout=0):
                return None
            @property
            def first(self):
                return self
        class _Page:
            def __init__(self, body_text="Welcome to the app"):
                self.body_text = body_text
            def close(self):
                return None
            def goto(self, url, wait_until="domcontentloaded", timeout=0):
                self.last_url = url
            def locator(self, sel):
                return _Loc(self.body_text if sel == "body" else "body")
            def get_by_text(self, text, exact=False):
                return _Loc()
            def wait_for_timeout(self, ms):
                return None
            def screenshot(self, path):
                return None
        class _Browser:
            def __init__(self):
                self.pages = []
            def new_page(self):
                p = _Page()
                self.pages.append(p)
                return p
            def close(self):
                return None
        class _Chromium:
            def __init__(self):
                self.browser = None
            def launch(self, headless=True, **kw):
                self.browser = _Browser()
                return self.browser
        class _PW:
            def __init__(self):
                self.chromium = _Chromium()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "j.yml"
            p.write_text(VALID_YML)
            orig_pw = getattr(journeys, "sync_playwright", None)
            orig_h = journeys.HAS_PLAYWRIGHT
            journeys.sync_playwright = _PW
            journeys.HAS_PLAYWRIGHT = True
            try:
                ok, report = run_journeys(str(p))
            finally:
                if orig_pw is not None:
                    journeys.sync_playwright = orig_pw
                else:
                    delattr(journeys, "sync_playwright")
                journeys.HAS_PLAYWRIGHT = orig_h
            self.assertTrue(ok, report)
            self.assertIn("PASS", report)

    @unittest.skipUnless(journeys.HAS_YAML, "pyyaml not installed")
    def test_failing_journey_reports_page_snapshot(self):
        class _Loc:
            def __init__(self, text=""):
                self.text = text
            def count(self):
                return 0
            def inner_text(self, timeout=0):
                return self.text
            def wait_for(self, state="visible", timeout=0):
                raise TimeoutError("timed out")
            def click(self, timeout=0):
                return None
            def fill(self, value, timeout=0):
                return None
            @property
            def first(self):
                return self
        class _Page:
            def close(self):
                return None
            def goto(self, url, wait_until="domcontentloaded", timeout=0):
                return None
            def locator(self, sel):
                return _Loc("nothing here")
            def get_by_text(self, text, exact=False):
                return _Loc()
            def wait_for_timeout(self, ms):
                return None
            def screenshot(self, path):
                return None
        class _Browser:
            def new_page(self):
                return _Page()
            def close(self):
                return None
        class _Chromium:
            def launch(self, headless=True, **kw):
                return _Browser()
        class _PW:
            chromium = _Chromium()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "j.yml"
            p.write_text(VALID_YML)
            orig_pw = getattr(journeys, "sync_playwright", None)
            orig_h = journeys.HAS_PLAYWRIGHT
            journeys.sync_playwright = _PW
            journeys.HAS_PLAYWRIGHT = True
            try:
                ok, report = run_journeys(str(p))
            finally:
                if orig_pw is not None:
                    journeys.sync_playwright = orig_pw
                else:
                    delattr(journeys, "sync_playwright")
                journeys.HAS_PLAYWRIGHT = orig_h
            self.assertFalse(ok)
            self.assertIn("FAILED", report)
            self.assertIn("nothing here", report)  # page snapshot fed to the fix loop


class TestJourneyGate(unittest.TestCase):
    def _gate(self, journey_results, fix_text="app.ts\n```ts\nconst x = 1\n```"):
        import contextlib
        import io
        calls = {"fix": 0}
        orig_r = ask.run_journeys
        orig_f = ask._deepseek_fix
        seq = list(journey_results)

        def fake_journeys(file, timeout_s=30, screenshots_dir=None):
            return seq.pop(0) if seq else (True, "no more journeys")

        def fake_fix(cloud, task, error_text, cache=None):
            calls["fix"] += 1
            return fix_text

        ask.run_journeys = fake_journeys
        ask._deepseek_fix = fake_fix
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                ok, report = ask._run_journey_gate(
                    object(), tmp, "t", "j.yml", load_bound({}),
                    lambda line: None, cache=None, max_iter=2)
        finally:
            ask.run_journeys = orig_r
            ask._deepseek_fix = orig_f
        return ok, report, calls

    def test_passes_first_try(self):
        ok, report, calls = self._gate([(True, "all journeys passed")])
        self.assertTrue(ok)
        self.assertEqual(calls["fix"], 0)

    def test_fix_loop_recovers(self):
        ok, report, calls = self._gate([(False, "journey login FAILED"),
                                        (True, "all journeys passed")])
        self.assertTrue(ok)
        self.assertEqual(calls["fix"], 1)

    def test_bound_violation_in_fix(self):
        ok, report, calls = self._gate([(False, "journey login FAILED")],
                                       fix_text="journeys.yml\n```yaml\nsteps: []\n```")
        self.assertFalse(ok)
        self.assertIn("BOUND", report)


class TestJourneysInProgramPhases(unittest.TestCase):
    def test_journey_phase_passes(self):
        orig = ask._run_journey_gate
        ask._run_journey_gate = lambda *a, **k: (True, "journeys passed")
        try:
            import contextlib
            import io
            cfg = {"program": {"phases": [
                {"name": "journeys", "journeys": "journeys.yml", "on_fail": "retry"},
            ]}}
            with tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()):
                ok, report = ask._run_program_phases(
                    object(), tmp, "t", load_bound({}), cfg, lambda line: None)
        finally:
            ask._run_journey_gate = orig
        self.assertTrue(ok, report)


class TestWriteOneBound(unittest.TestCase):
    def test_journeys_yml_is_write_protected(self):
        b = load_bound({})  # defaults include **/journeys.yml
        self.assertIsNotNone(b.enforce_path("journeys.yml"))
        self.assertIsNotNone(b.enforce_path("e2e/journeys.yml"))
        with tempfile.TemporaryDirectory() as tmp:
            text = "journeys.yml\n```yaml\nsteps: []\n```\napp.ts\n```ts\nx\n```"
            written, skipped = ask._apply_fenced_files(text, root=tmp, bound=b)
            self.assertEqual([rel for rel, _ in written], ["app.ts"])
            self.assertTrue(any("(bound:" in s for s in skipped))


if __name__ == "__main__":
    unittest.main()
