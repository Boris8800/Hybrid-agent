"""dependencies.py — dependency-aware context retrieval.

Given the contract's "Files likely involved", resolve each file's dependency
graph — direct imports, callers, tests, and exported interfaces/types — and
return a COMPACT context (capped) instead of dumping the whole project.

- Python imports: stdlib `ast` (precise).
- TS/JS imports: regex `from 'x'` / `require('x')` (relative + project alias,
  node_modules skipped).
- Callers: reverse scan for import statements referencing the module.
- Tests: files that import the module and look like tests.

Stdlib only.
"""

from __future__ import annotations

import ast as py_ast
import re
import subprocess
from pathlib import Path

MAX_CONTEXT_CHARS = 5000
_SKIP_DIRS = {"node_modules", ".venv", ".git", "__pycache__", "dist", "build", ".cache"}


def _repo_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts):
            if p.suffix in (".py", ".pyw", ".ts", ".tsx", ".js", ".jsx"):
                out.append(p)
    return out


def _py_imports(source: str) -> list[str]:
    try:
        tree = py_ast.parse(source)
    except SyntaxError:
        return []
    mods: list[str] = []
    for node in py_ast.walk(tree):
        if isinstance(node, py_ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, py_ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


_TS_IMPORT = re.compile(r"(?:import\s+[\w{},*\s]+from\s+['\"]([^'\"]+)['\"]"
                        r"|import\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))")


def _ts_imports(source: str) -> list[str]:
    return [m.group(1) or m.group(2) or m.group(3)
            for m in _TS_IMPORT.finditer(source) if (m.group(1) or m.group(2) or m.group(3))]


def _resolve(module: str, importer: Path, root: Path) -> Path | None:
    """Resolve an import spec to a file in the repo."""
    if module.startswith("."):
        segments = [s for s in module.split("/") if s not in ("", ".")]
        base = importer.parent
        while segments and segments[0] == "..":
            base = base.parent
            segments = segments[1:]
        if not segments:
            return None
        for suffix in (".py", ".ts", ".tsx", ".js", ".jsx"):
            stem = segments[-1]
            cand = (base / ("/".join(segments[:-1] + [stem + suffix])
                            if len(segments) > 1 else stem + suffix))
            if cand.is_file():
                return cand.resolve()
            idx = base / ("/".join(segments + [f"index{suffix}"]))
            if idx.is_file():
                return idx.resolve()
    else:
        parts = module.split(".")
        for suffix in (".py", ".ts", ".tsx", ".js", ".jsx"):
            for base in (root / "/".join(parts[:-1] + [parts[-1] + suffix]),
                         root / "/".join(parts)):
                if base.is_file():
                    return base.resolve()
    return None


def _is_test(p: Path) -> bool:
    low = p.name.lower()
    return "test" in low or "spec" in low


def _exports(file: Path, limit: int = 25) -> list[str]:
    """Exported/defined symbols — interfaces/types/classes/functions."""
    try:
        source = file.read_text(encoding="utf-8")
    except OSError:
        return []
    exports: list[str] = []
    if file.suffix in (".py", ".pyw"):
        try:
            tree = py_ast.parse(source)
        except SyntaxError:
            return []
        for node in py_ast.walk(tree):
            if isinstance(node, (py_ast.FunctionDef, py_ast.ClassDef,
                                 py_ast.AsyncFunctionDef)):
                exports.append(f"def/class {node.name}")
            elif isinstance(node, py_ast.Assign):
                for t in node.targets:
                    if isinstance(t, py_ast.Name):
                        exports.append(t.id)
    else:
        for m in re.finditer(r"\bexport\s+(?:interface|type|class|function|const|let|var)\s+(\w+)",
                             source):
            exports.append(m.group(1))
        for m in re.finditer(r"\b(?:interface|type|class)\s+(\w+)", source):
            if m.group(1) not in exports:
                exports.append(m.group(1))
    return exports[:limit]


def build_dependency_context(root: str, files: list[str]) -> str:
    """Return a compact, dependency-aware context for the involved files."""
    root_p = Path(root).resolve()
    involved = [(Path(f) if Path(f).is_absolute() else root_p / f) for f in (files or [])]
    involved = [p.resolve() for p in involved if p.is_file()]
    if not involved:
        return ""
    repo = _repo_files(root_p)
    seen: dict[Path, str] = {}

    def _add(p: Path, label: str):
        if p not in seen and p.is_file():
            seen[p] = label

    for file in involved:
        _add(file, "INVOLVED")
        try:
            source = file.read_text(encoding="utf-8")
        except OSError:
            continue
        imports = _py_imports(source) if file.suffix in (".py", ".pyw") else _ts_imports(source)
        for imp in imports:
            resolved = _resolve(imp, file, root_p)
            if resolved:
                _add(resolved, "IMPORT")

    # callers: files that import any involved module
    involved_names = {p.stem for p in involved}
    for p in repo:
        if p in seen:
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        imps = _py_imports(src) if p.suffix in (".py", ".pyw") else _ts_imports(src)
        imports_involved = any(name in imp for imp in imps for name in involved_names)
        if _is_test(p) and imports_involved:
            _add(p, "TEST")
        elif imports_involved:
            _add(p, "CALLER")

    lines: list[str] = []
    for p, label in seen.items():
        if p in involved:
            lines.append(f"[INVOLVED] {p.relative_to(root_p)}")
        else:
            lines.append(f"[{label}] {p.relative_to(root_p)}")
    lines.append("")
    for p in involved:
        try:
            body = p.read_text(encoding="utf-8")
        except OSError:
            continue
        exp = _exports(p)
        head = body[:1200]
        lines.append(f"--- {p.relative_to(root_p)} ({len(body)} bytes)"
                     + (f" · exports: {', '.join(exp)}" if exp else ""))
        lines.append(head)
    for p, label in seen.items():
        if p in involved:
            continue
        exp = _exports(p)
        head_lines = []
        try:
            for ln in p.read_text(encoding="utf-8").splitlines()[:40]:
                head_lines.append(ln)
        except OSError:
            continue
        lines.append(f"--- {p.relative_to(root_p)} [{label}]"
                     + (f" · exports: {', '.join(exp)}" if exp else ""))
        lines.append("\n".join(head_lines))
    text = "\n".join(lines)
    return text if len(text) <= MAX_CONTEXT_CHARS else text[:MAX_CONTEXT_CHARS] + "\n…"


def repo_toplevel(root: str) -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else root
    except (OSError, subprocess.SubprocessError):
        return root
