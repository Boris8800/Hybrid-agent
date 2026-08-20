"""gitops tests: git pull/push and deploy helpers.

Run from hybrid-agent/:
    python -m unittest tests.test_gitops -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gitops  # noqa: E402


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True).returncode == 0


class TestGitOps(unittest.TestCase):
    def _repo(self, tmp, name):
        d = Path(tmp) / name
        d.mkdir()
        _git(d, "init", "-q", "-b", "main")
        _git(d, "config", "user.email", "test@example.com")
        _git(d, "config", "user.name", "Test")
        return d

    def test_is_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(tmp, "r")
            self.assertTrue(gitops.is_repo(str(repo)))
            plain = Path(tmp) / "plain"
            plain.mkdir()
            self.assertFalse(gitops.is_repo(str(plain)))

    def test_push_commits_and_reaches_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.git"
            subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
            work = self._repo(tmp, "work")
            _git(work, "remote", "add", "origin", str(remote))
            (work / "a.txt").write_text("hello")
            ok, msg = gitops.git_push(str(work), files=["a.txt"],
                                      message="hybrid-agent: test change")
            self.assertTrue(ok, msg)
            # The commit reached the bare remote.
            log = subprocess.run(["git", "--git-dir", str(remote), "log", "--all", "--oneline"],
                                 capture_output=True, text=True).stdout
            self.assertIn("hybrid-agent: test change", log)

    def test_push_with_no_changes_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self._repo(tmp, "work")
            (work / "a.txt").write_text("x")
            _git(work, "add", "a.txt")
            _git(work, "commit", "-q", "-m", "initial")
            ok, msg = gitops.git_push(str(work), files=["a.txt"])
            self.assertFalse(ok)
            self.assertIn("no changes", msg.lower())

    def test_pull_up_to_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self._repo(tmp, "work")
            (work / "a.txt").write_text("x")
            _git(work, "add", "a.txt")
            _git(work, "commit", "-q", "-m", "initial")
            ok, msg = gitops.git_pull(str(work))
            self.assertTrue(ok)  # no upstream -> fetches, still reports ok

    def test_pull_not_a_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = gitops.git_pull(tmp)
            self.assertFalse(ok)
            self.assertIn("not a git repository", msg)

    def test_deploy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = gitops.run_deploy(tmp, "echo deploy-ok")
            self.assertTrue(ok, msg)
            ok, msg = gitops.run_deploy(tmp, "")
            self.assertFalse(ok)
            self.assertIn("no deploy command", msg)


if __name__ == "__main__":
    unittest.main()
