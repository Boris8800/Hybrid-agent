"""learned_rules.py — learning from failures (auto-generated procedural rules).

Memory already records outcomes and consolidates statistics. This layer goes
one step further: when the same (domain, error-category) keeps failing, the
engine records WHAT the successful fixes actually touched and emits a
reusable RULE — e.g. "for backend-API tasks with missing-module failures,
always include controller + service + DTO + tests in the dependency context."

Rules persist to memory/<project>/rules.json and are injected into the task
context, so memory stops merely saying "we solved something similar" and starts
changing how the next task is approached. Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEARN_THRESHOLD = 3          # failures of the same (domain, category) -> rule
MAX_RULES = 12
KNOWN_SOLUTION_MIN = 2       # successful recordings before a known solution is trusted

# Error categories derived from the first chunk of verify/journey output.
_CATEGORIES = [
    (re.compile(r"cannot find module|module not found|no such file|unable to resolve path|missing.*(?:dependency|import|file)", re.I), "missing-module"),
    (re.compile(r"is not exported|does not exist on type|ts\d{4}|type error|expected .* but got", re.I), "type-error"),
    (re.compile(r"is not a function|typeerror|undefined is not", re.I), "runtime-error"),
    (re.compile(r"is not defined|undeclared|no-undef|referenceerror", re.I), "undefined-symbol"),
    (re.compile(r"command not found|enoent|not found: (?:command|module)", re.I), "environment"),
]


def classify_error(text: str) -> str:
    for pattern, label in _CATEGORIES:
        if pattern.search(text or ""):
            return label
    return "unknown"


def failure_fingerprint(error_text: str) -> str:
    """FAILURE_ID: hash(category, file, line, normalized first message). Two
    failures with the same fingerprint are the SAME failure, exactly. All
    whitespace is stripped first so formatting differences don't matter."""
    import hashlib
    clean = re.sub(r"\s+", "", (error_text or ""))
    category = classify_error(clean)
    loc = re.search(r"([\w./\\-]+\.(?:py|pyw|ts|tsx|js|jsx|json|css|html)):(\d+)",
                    clean)
    file_line = f"{loc.group(1)}:{loc.group(2)}" if loc else "?"
    return hashlib.sha1(f"{category}|{file_line}|{clean[:200]}".encode()).hexdigest()[:20]


@dataclass
class FailureRecord:
    domain: str
    category: str
    fixed_files: list = field(default_factory=list)
    ts: float = 0.0


class LearnedRules:
    def __init__(self, root: str | None = None, threshold: int = LEARN_THRESHOLD):
        base = Path(root) if root else Path(__file__).resolve().parent / "memory"
        self.path = Path(base) / "rules.json"
        self.solutions_path = Path(base) / "solutions.json"
        self.threshold = threshold

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save(self, records: list[dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    def record_failure(self, domain: str, error_text: str,
                       fixed_files: list[str]) -> None:
        """Record one failed attempt. When the same (domain, category) repeats
        past the threshold, emit a rule based on the files fixes touched."""
        category = classify_error(error_text)
        records = self._load()
        records.append(asdict(FailureRecord(
            domain=(domain or "general")[:60], category=category,
            fixed_files=[str(f) for f in (fixed_files or [])][:12],
            ts=time.time())))
        records[-1]["fingerprint"] = failure_fingerprint(error_text)
        self._save(records[-400:])

    # --- known-solution memory (exact-failure reuse) ----------------------
    def _load_solutions(self) -> dict:
        if not self.solutions_path.is_file():
            return {}
        try:
            data = json.loads(self.solutions_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_solutions(self, data: dict) -> None:
        try:
            self.solutions_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.solutions_path.with_suffix(self.solutions_path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.solutions_path)
        except OSError:
            pass

    def record_success(self, error_text: str, fix_text: str) -> None:
        """Record a fix that LED TO GREEN verification for an exact failure."""
        fp = failure_fingerprint(error_text)
        solutions = self._load_solutions()
        entry = solutions.get(fp) or {"count": 0, "fix": "", "first": time.time()}
        entry["count"] += 1
        entry["fix"] = fix_text or entry.get("fix", "")
        entry["last"] = time.time()
        solutions[fp] = entry
        self._save_solutions(solutions)

    def lookup(self, error_text: str) -> tuple[str, float] | None:
        """Exact-failure known solution: (fix_text, confidence) when the same
        failure fingerprint was successfully fixed at least KNOWN_SOLUTION_MIN
        times, else None. Confidence = successes / (successes + 1)."""
        fp = failure_fingerprint(error_text)
        entry = self._load_solutions().get(fp)
        if not entry or entry.get("count", 0) < KNOWN_SOLUTION_MIN or not entry.get("fix"):
            return None
        count = int(entry.get("count", 0))
        return entry["fix"], round(count / (count + 1), 2)

    def rules(self) -> list[str]:
        """Emit rules for (domain, category) pairs past the threshold."""
        records = self._load()
        buckets: dict[tuple, list[list[str]]] = {}
        for r in records:
            key = (r.get("domain", "general"), r.get("category", "unknown"))
            buckets.setdefault(key, []).append(r.get("fixed_files", []))
        rules: list[str] = []
        for (domain, category), fixes in buckets.items():
            if len(fixes) < self.threshold:
                continue
            # Files most frequently touched by fixes for this category.
            counts: dict[str, int] = {}
            for fset in fixes:
                for f in fset:
                    counts[f] = counts.get(f, 0) + 1
            top = sorted(counts, key=counts.get, reverse=True)[:4]
            if not top:
                continue
            short = ", ".join(Path(f).name for f in top)
            rules.append(
                f"For {domain} tasks with {category} failures: always include "
                f"these in the dependency context — {short}.")
        return rules[-MAX_RULES:]

    def suggested_files(self) -> list[str]:
        """The files rules say to always retrieve (for dependency context)."""
        files: list[str] = []
        for rule in self.rules():
            for f in re.findall(r"\b[\w./-]+\.(?:ts|tsx|js|jsx|py|pyw)\b", rule):
                if f not in files:
                    files.append(f)
        return files

    def prompt_text(self) -> str:
        rules = self.rules()
        if not rules:
            return ""
        return "LEARNED RULES (from past failures in this project):\n" + \
            "\n".join(f"- {r}" for r in rules)
