"""Parallel-verify tests: dependency groups, race-condition prevention,
deterministic error merge, and allowlist enforcement.

Run from hybrid-agent/:
    python -m unittest tests.test_verify_parallel -v
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402


def noop(*args, **kwargs):
    pass


def _proc(returncode=0, stdout="", stderr=""):
    return type("Proc", (), {
        "returncode": returncode, "stdout": stdout, "stderr": stderr,
    })()


class TestVerifyParallel(unittest.TestCase):
    def test_group2_waits_for_group1(self):
        """Build group must finish before the dependent test group starts."""
        events = []
        start_time = time.time()

        def fake_run(cmd, **kwargs):
            if cmd in ("a", "b"):
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            events.append((cmd, "start", time.time() - start_time))
            time.sleep(0.01)
            events.append((cmd, "end", time.time() - start_time))
            return _proc()

        with patch("ask.subprocess.run", side_effect=fake_run):
            errors = ask._run_verify_commands(
                root=".", cmds=["a", "b", "t"], status=noop,
                timeout_s=5, parallel=True, workers=4,
                verify_groups=[["a", "b"], ["t"]],
            )

        group1_ends = max(e[2] for e in events
                          if e[0] in ("a", "b") and e[1] == "end")
        group2_start = min(e[2] for e in events
                           if e[0] == "t" and e[1] == "start")
        self.assertLess(group1_ends, group2_start,
                        "group 2 started before group 1 finished (race!)")
        self.assertLess(time.time() - start_time, 0.15,
                        "did not run in parallel (wall time too long)")
        self.assertEqual(errors, [])

    def test_sequential_fallback_no_groups(self):
        """No groups configured -> commands run sequentially in order."""
        order = []

        def fake_run(cmd, **kwargs):
            order.append(cmd)
            return _proc()

        with patch("ask.subprocess.run", side_effect=fake_run):
            errors = ask._run_verify_commands(
                root=".", cmds=["a", "b"], status=noop,
                timeout_s=5, parallel=True, workers=4, verify_groups=[],
            )

        self.assertEqual(order, ["a", "b"])
        self.assertEqual(errors, [])

    def test_deterministic_error_order(self):
        """Merged errors keep command order; clean commands add no entries."""
        def fake_run(cmd, **kwargs):
            if cmd == "fail":
                return _proc(returncode=1, stdout="error")
            return _proc()

        with patch("ask.subprocess.run", side_effect=fake_run):
            errors = ask._run_verify_commands(
                root=".", cmds=["ok", "fail"], status=noop,
                timeout_s=5, parallel=True, workers=4,
                verify_groups=[["ok", "fail"]],
            )

        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("$ fail (exit 1)"), errors[0])
        self.assertFalse(any("ok" in e for e in errors))

    def test_allowlist_still_enforced(self):
        """Unsafe commands are blocked before anything runs, parallel or not."""
        verify_stats = {}
        with patch("ask.subprocess.run") as mock_run:
            result, _msg = ask._run_final_verify(
                cloud=None, root=".", task="test", cmds=["rm -rf /"],
                status=noop, verify_stats=verify_stats,
            )
        self.assertFalse(result)
        self.assertEqual(verify_stats["status"], "BLOCKED")
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
