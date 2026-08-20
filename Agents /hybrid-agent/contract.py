"""contract.py — the formal Task Contract (machine-readable spine of a task).

DeepSeek converts the user request into a structured contract BEFORE the local
model implements; every stage (Qwen prompt, review package, verification
gates, final auditor) consumes the SAME contract:

  Goal / Must change / Must NOT change / Acceptance criteria /
  Files likely involved / Dependencies / Risk / Verification required /
  Rollback strategy, plus a table of ACCEPTANCE CASES (input -> expected).

Parsed from the === TASK CONTRACT === and === ACCEPTANCE CASES === sections of
the enhance output. Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_RISKS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _bullets(text: str) -> list[str]:
    return [ln.strip().lstrip("-").strip() for ln in (text or "").splitlines()
            if ln.strip() and not ln.strip().startswith(("Goal:", "Must"))][:12]


@dataclass
class TaskContract:
    goal: str = ""
    must_change: list = field(default_factory=list)
    must_not_change: list = field(default_factory=list)
    acceptance_criteria: list = field(default_factory=list)
    files: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    risk: str = ""
    verification_required: list = field(default_factory=list)
    rollback: str = ""
    acceptance_cases: list = field(default_factory=list)  # [{"input": ..., "expected": ...}]

    @property
    def complete(self) -> bool:
        return bool(self.goal.strip())

    @property
    def risky(self) -> bool:
        return self.risk in ("HIGH", "CRITICAL")

    def to_prompt(self) -> str:
        """Render the contract as prompt text for Qwen / the auditor."""
        if not self.complete:
            return ""
        lines = ["TASK CONTRACT:"]
        lines.append(f"Goal: {self.goal}")
        if self.must_change:
            lines.append("Must change:\n  - " + "\n  - ".join(self.must_change[:8]))
        if self.must_not_change:
            lines.append("Must NOT change:\n  - " + "\n  - ".join(self.must_not_change[:8]))
        if self.acceptance_criteria:
            lines.append("Acceptance criteria:\n  - " + "\n  - ".join(self.acceptance_criteria[:8]))
        if self.files:
            lines.append("Files likely involved: " + ", ".join(self.files[:10]))
        if self.dependencies:
            lines.append("Dependencies: " + ", ".join(self.dependencies[:8]))
        if self.risk:
            lines.append(f"Risk: {self.risk}")
        if self.verification_required:
            lines.append("Verification required: " + ", ".join(self.verification_required[:8]))
        if self.rollback:
            lines.append(f"Rollback strategy: {self.rollback}")
        if self.acceptance_cases:
            lines.append("Acceptance cases:")
            for case in self.acceptance_cases[:10]:
                lines.append(f"  - {case.get('input', '?')} -> {case.get('expected', '?')}")
        return "\n".join(lines)


def _extract_section(raw: str, label: str) -> str:
    """Pull the === LABEL === ... section body (same convention as supervise)."""
    m = re.search(rf"===\s*{label}\s*===\s*(.*?)(?=\n\s*===\s*|\Z)", raw or "",
                  re.S | re.I)
    return m.group(1).strip() if m else ""


def _split_labeled(raw: str) -> dict[str, str]:
    """Split a contract body into {Field: rest} by its labeled lines."""
    fields: dict[str, str] = {}
    current: str | None = None
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z /]{2,40}?):\s*(.*)$", line)
        if m:
            current = m.group(1).strip().lower()
            fields[current] = m.group(2)
        elif current and line.strip():
            fields[current] += "\n" + line
    return fields


def parse_contract(raw: str) -> TaskContract:
    """Parse the TASK CONTRACT + ACCEPTANCE CASES sections from enhance output."""
    c = TaskContract()
    body = _extract_section(raw, "TASK CONTRACT")
    if not body:
        return c
    fields = _split_labeled(body)
    c.goal = (fields.get("goal") or "").strip()
    c.must_change = _bullets(fields.get("must change", ""))
    c.must_not_change = _bullets(fields.get("must not change", ""))
    c.acceptance_criteria = _bullets(fields.get("acceptance criteria", ""))
    c.files = [f.strip() for f in re.split(r"[,;]", fields.get("files likely involved", ""))
               if f.strip()]
    c.dependencies = [f.strip() for f in re.split(r"[,;]", fields.get("dependencies", ""))
                      if f.strip()]
    risk = (fields.get("risk") or "").strip().upper()
    c.risk = risk if risk in _RISKS else ""
    c.verification_required = [f.strip().lstrip("-").strip().lower() for f in
                               re.split(r"[,;\n]", fields.get("verification required", ""))
                               if f.strip()]
    c.rollback = (fields.get("rollback strategy") or "").strip()
    cases = _extract_section(raw, "ACCEPTANCE CASES")
    for line in (cases or "").splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        m = re.match(r"(.+?)\s*(?:->|→|:)\s*(.+)$", line)
        if m:
            c.acceptance_cases.append({"input": m.group(1).strip(),
                                       "expected": m.group(2).strip()})
    return c
