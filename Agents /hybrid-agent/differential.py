"""differential.py — the anti-gaming ratchet (Modonome pattern).

Autonomous fix loops have a predictable failure mode: the agent weakens a gate
to make it go green — skipping tests, removing assertions, loosening type
checks. This module inspects the working-tree diff and rejects changes that
weaken a gate:

- test files: added `it.skip(` / `.only(` / `@pytest.mark.skip` / `xit(` /
  `skip: true`; assertion lines removed without replacement;
- tsconfig: strict checks flipped off (`"strict": false`, `"noImplicitAny":
  false`, `"skipLibCheck": true`, ...);
- guard/config files (config.yml, .github/, jest/vitest/eslint configs):
  any modification needs owner review.

Stdlib only. Returns violation descriptions; the caller decides (the engine
rejects the diff and instructs the fix loop to restore the gate).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_SKIP_PATTERNS = [
    r"\.skip\(", r"\.only\(", r"@pytest\.mark\.skip", r"@unittest\.skip",
    r"\bxit\(", r"\bxdescribe\(", r"\bxtest\(", r"test\.skip", r"describe\.skip",
    r"skip:\s*true",
]
_ASSERTION_KEYWORDS = (
    "expect(", "assert ", "assertEqual", "assertEquals", "assertIn", "assertTrue",
    "assertFalse", "assertIsNone", "assertIsNotNone", "should.", "toEqual",
    "toHaveBeenCalled", "toBe(", "toContain", "assertThat",
)
_TSCONFIG_LOOSENERS = [
    (r'"strict"\s*:\s*false', "strict mode"),
    (r'"noImplicitAny"\s*:\s*false', "noImplicitAny"),
    (r'"strictNullChecks"\s*:\s*false', "strictNullChecks"),
    (r'"skipLibCheck"\s*:\s*true', "skipLibCheck"),
    (r'"strictFunctionTypes"\s*:\s*false', "strictFunctionTypes"),
]
_GUARD_FILES = (
    "config.yml", "config.yaml", ".github/", "jest.config", "vitest.config",
    ".eslintrc", "eslint.config", "bound.yml", "journeys.yml", ".modonome/",
)


def _is_test_file(path: str) -> bool:
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    return ("test" in base or "spec" in base or "__tests__" in low
            or low.startswith("tests/") or "/tests/" in low)


def _is_guard_file(path: str) -> bool:
    return any(g in path.lower() for g in _GUARD_FILES)


def _git_diff(root: str, paths: list[str] | None = None) -> str:
    cmd = ["git", "diff", "-U0"]
    if paths:
        cmd += ["--", *paths]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              timeout=60)
        return proc.stdout if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _parse_hunks(diff: str) -> list[tuple[str, list[str], list[str]]]:
    """Parse a unified diff into [(path, added_lines, removed_lines)]."""
    hunks: list[tuple[str, list[str], list[str]]] = []
    current: str | None = None
    added: list[str] = []
    removed: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            if current is not None:
                hunks.append((current, added, removed))
            current = line[6:].strip()
            added, removed = [], []
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if current is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    if current is not None:
        hunks.append((current, added, removed))
    return hunks


def analyze_changes(root: str, paths: list[str] | None = None) -> list[str]:
    """Return anti-gaming violations in the working-tree diff. [] = clean."""
    violations: list[str] = []
    for path, added, removed in _parse_hunks(_git_diff(root, paths)):
        low_path = path.lower()
        if _is_test_file(low_path):
            for line in added:
                for pat in _SKIP_PATTERNS:
                    if re.search(pat, line):
                        violations.append(
                            f"{path}: added test weakening: {line.strip()[:80]}")
                        break
            added_asserts = sum(1 for l in added
                                if any(k in l for k in _ASSERTION_KEYWORDS))
            removed_asserts = sum(1 for l in removed
                                  if any(k in l for k in _ASSERTION_KEYWORDS))
            if removed_asserts > added_asserts + 2:
                violations.append(
                    f"{path}: {removed_asserts - added_asserts} assertion(s) "
                    "removed without replacement")
        if "tsconfig" in low_path and low_path.endswith(".json"):
            for line in added:
                for pat, label in _TSCONFIG_LOOSENERS:
                    if re.search(pat, line):
                        violations.append(
                            f"{path}: type-check loosened ({label}): {line.strip()[:60]}")
                        break
        if _is_guard_file(low_path):
            violations.append(f"{path}: guard/config file modified — needs owner review")
    return violations
