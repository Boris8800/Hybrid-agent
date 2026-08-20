"""gitops.py — pull / push / deploy helpers for the hybrid engine (stdlib only).

- git_pull: refresh the baseline before the agent works (best-effort).
- git_push: stage + commit + push the engine's own changes (applied files and
  verification fixes) with an identifiable message. Never force-pushes.
- run_deploy: execute a deploy command after the task is verified.

Every function returns (ok: bool, message: str) and never raises.
"""

from __future__ import annotations

import subprocess


def _run(cmd: list[str], cwd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except (FileNotFoundError, OSError):
        return False, "required binary not found"
    out = (proc.stdout or "").strip() + (proc.stderr or "").strip()
    return proc.returncode == 0, out


def _run_shell(command: str, cwd: str, timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(command, cwd=cwd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"deploy timed out after {timeout}s"
    except OSError:
        return False, "could not run deploy command"
    out = (proc.stdout or "").strip() + (proc.stderr or "").strip()
    return proc.returncode == 0, out


def is_repo(root: str) -> bool:
    ok, _ = _run(["git", "rev-parse", "--is-inside-work-tree"], root, timeout=15)
    return ok


def git_pull(root: str, timeout: int = 120) -> tuple[bool, str]:
    """Best-effort 'git pull --ff-only'. A dirty tree or missing upstream is
    reported, never fatal (the engine works on the existing tree)."""
    if not is_repo(root):
        return False, "not a git repository — pull skipped"
    ok, out = _run(["git", "pull", "--ff-only"], root, timeout)
    if not ok and "error: you have unstaged changes" in (out or "").lower():
        return False, "working tree dirty — pull skipped (unstaged changes)"
    if not ok:
        fetch_ok, fetch_out = _run(["git", "fetch"], root, timeout)
        if fetch_ok:
            return True, f"pull not possible ({out.strip()}); fetched remote refs"
    return ok, out or "already up to date"


def git_push(root: str, files: list[str] | None = None,
             message: str = "hybrid-agent: automated change",
             timeout: int = 180) -> tuple[bool, str]:
    """Stage the given files (or -A when none), commit, and push. Never
    force-pushes and never amends."""
    if not is_repo(root):
        return False, "not a git repository — push skipped"
    add_cmd = ["git", "add", "--"] + (files if files else ["-A"])
    ok, out = _run(add_cmd, root, timeout)
    if not ok:
        return False, f"git add failed: {out.strip()}"
    ok, out = _run(["git", "commit", "-m", message], root, timeout)
    if not ok:
        low = (out or "").lower()
        if "nothing to commit" in low or "no changes added" in low:
            return False, "no changes to commit — push skipped"
        return False, f"git commit failed: {out.strip()}"
    ok, out = _run(["git", "push"], root, timeout)
    if not ok and "no upstream branch" in (out or "").lower():
        # First push of a branch with no upstream: set it up automatically.
        ok, out = _run(["git", "push", "-u", "origin", "HEAD"], root, timeout)
    if not ok:
        return False, f"git push failed: {out.strip()}"
    return True, (out.splitlines()[-1] if out else "pushed")


def run_deploy(root: str, command: str, cwd: str | None = None,
               timeout: int = 1800) -> tuple[bool, str]:
    """Run a deploy command (shell) from cwd (default: root)."""
    if not command or not command.strip():
        return False, "no deploy command configured (review.deploy.command)"
    return _run_shell(command, cwd or root, timeout)
