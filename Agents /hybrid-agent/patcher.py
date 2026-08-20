"""patcher.py — surgical (diff-first) code repair for the hybrid engine.

Instead of regenerating whole files, model fixes are applied as unified diffs
via `git apply` (falling back to `patch -p1`), with every touched file checked
against the BOUND first. Full-file fenced blocks remain the fallback in ask.py
when no diff is present or application fails.

`extract_context()` is the AST-aware part: given a file and a line number it
returns the ENCLOSING function/class/block source (Python via the stdlib `ast`,
TS/JS via a scope scan) so the fix prompt sees the right code, not the whole
file.

Stdlib only.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_DIFF_HEADER = re.compile(r"^(diff --git |--- [ab]/|\+\+\+ [ab]/)", re.M)
_PLUS_PATH = re.compile(r"^\+\+\+ b/(.+)$", re.M)
_MINUS_PATH = re.compile(r"^--- a/(.+)$", re.M)
_ERR_LOCATION = re.compile(r"([\w./\\-]+\.(?:py|pyw|ts|tsx|js|jsx|json|css|html|yml|yaml)):(\d+)")

# TS/JS declaration line starters (heuristic scope scan; best-effort).
_TS_DECL = re.compile(r"\b(function|async function|class|interface|type|const|let|var)\b")


def is_diff(text: str) -> bool:
    return bool(_DIFF_HEADER.search(text or ""))


def diff_paths(text: str) -> list[str]:
    paths: list[str] = []
    for m in _PLUS_PATH.finditer(text or ""):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)
    for m in _MINUS_PATH.finditer(text or ""):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)
    return paths


def apply_unified_diff(root: str, diff_text: str, bound=None,
                       timeout: int = 60) -> dict:
    """Apply a unified diff in `root` via git apply (then patch -p1).

    Every file the diff touches is BOUND-checked first. Returns:
      {"applied": [(path, bytes)], "skipped": [...], "bound_violation": bool,
       "original_bytes": int}
    """
    result = {"applied": [], "skipped": [], "bound_violation": False,
              "original_bytes": 0}
    paths = diff_paths(diff_text)
    for p in paths:
        if bound is not None:
            reason = bound.enforce_path(p)
            if reason:
                result["bound_violation"] = True
                result["skipped"].append(f"{p} (bound: {reason})")
                return result
        fp = Path(root) / p
        if fp.is_file():
            result["original_bytes"] += fp.stat().st_size
    attempts = [(["git", "apply", "-"], False),
                (["patch", "-p1"], False)]
    for cmd, _shell in attempts:
        try:
            if cmd[0] == "git":
                proc = subprocess.run(cmd, cwd=root, input=diff_text,
                                      capture_output=True, text=True, timeout=timeout)
            else:
                proc = subprocess.run(["patch", "-p1"], cwd=root, input=diff_text,
                                      capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            for p in paths:
                fp = Path(root) / p
                if fp.is_file():
                    result["applied"].append((p, fp.stat().st_size))
            return result
    result["skipped"].append("diff did not apply cleanly (git apply / patch failed)")
    return result


def extract_error_location(error_text: str) -> tuple[str, int] | None:
    """Pull (path, line) from the first file:line occurrence in error text."""
    m = _ERR_LOCATION.search(error_text or "")
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _py_block(source: str, line: int, max_lines: int = 80) -> str:
    try:
        import ast as py_ast
        tree = py_ast.parse(source)
    except SyntaxError:
        return ""
    for node in py_ast.walk(tree):
        if isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef,
                             py_ast.ClassDef)):
            lo, hi = node.lineno, node.end_lineno or node.lineno
            if lo <= line <= hi and hi - lo + 1 <= max_lines:
                seg = py_ast.get_source_segment(source, node)
                return seg if seg else ""
    return ""


def _ts_block(source: str, line: int, max_lines: int = 80) -> str:
    lines = source.splitlines()
    idx = line - 1
    if not (0 <= idx < len(lines)):
        return ""
    start = idx
    while start > 0:
        s = lines[start - 1]
        if _TS_DECL.search(s) and "{" in s:
            start = start - 1
            break
        start -= 1
        if idx - start > 60:
            break
    depth = 0
    opened = False
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if "{" in lines[i]:
            opened = True
        if opened and depth <= 0:
            return "\n".join(lines[start:i + 1])[:max_lines * 8]
        if i - start > max_lines * 2:
            break
    return ""


def extract_context(path: str | Path, line: int, max_lines: int = 80) -> str:
    """Return the enclosing block source for `line` in `path` (best-effort).
    Python uses the stdlib ast; TS/JS/others use a scope scan. '' on failure."""
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    if p.suffix in (".py", ".pyw"):
        block = _py_block(source, line, max_lines)
    else:
        block = _ts_block(source, line, max_lines)
    if not block:
        return ""
    return f"{p.name}:{line}\n{block}"
