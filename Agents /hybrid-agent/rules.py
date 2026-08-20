"""rules.py — project-specific engineering rules (.agent/engineering-rules.yml).

The BOUND is the global, non-negotiable layer. This is the per-project
"engineering constitution": architecture facts, rules the implementer must
follow, and dependency policies. Loaded from --root/.agent/engineering-rules.yml
(or engineering-rules.yml at the root), injected into the model context
alongside the BOUND, and consulted by the dependency gate.

Example:

    architecture:
      frontend: Next.js
      backend: NestJS
      database: PostgreSQL

    rules:
      - never access the database from a controller
      - all API responses use DTOs
      - all mutations require tests
      - never modify generated files

    dependencies:
      block: [left-pad, some-deprecated-pkg]
      allow: [react, express]     # pre-approved; everything else needs approval

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # PyYAML is a venv dep; rules just stay empty without it
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class EngineeringRules:
    architecture: dict = field(default_factory=dict)
    rules: list = field(default_factory=list)
    blocked_dependencies: list = field(default_factory=list)
    allowed_dependencies: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.architecture or self.rules or self.blocked_dependencies)

    def prompt_text(self) -> str:
        if not self.active:
            return ""
        lines = ["ENGINEERING RULES (project constitution — follow these):"]
        if self.architecture:
            lines.append("Architecture: " + ", ".join(
                f"{k}={v}" for k, v in self.architecture.items()))
        for i, rule in enumerate(self.rules, start=1):
            lines.append(f"{i}. {rule}")
        if self.blocked_dependencies:
            lines.append("Blocked dependencies: " + ", ".join(self.blocked_dependencies))
        if self.allowed_dependencies:
            lines.append("Pre-approved dependencies: " + ", ".join(self.allowed_dependencies))
        return "\n".join(lines)

    def check_dependency(self, package: str) -> str | None:
        """Return a reason when a package is blocked, else None."""
        low = package.lower()
        for blocked in self.blocked_dependencies:
            if blocked.lower() == low:
                return f"blocked by engineering rules: {blocked!r}"
        return None


def load_engineering_rules(root: str | Path) -> EngineeringRules:
    """Load .agent/engineering-rules.yml (or root engineering-rules.yml)."""
    candidates = [
        Path(root) / ".agent" / "engineering-rules.yml",
        Path(root) / "engineering-rules.yml",
    ]
    for path in candidates:
        if not path.is_file() or not HAS_YAML:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - bad rules file must never break a run
            return EngineeringRules()
        if not isinstance(data, dict):
            return EngineeringRules()
        deps = data.get("dependencies") or {}
        return EngineeringRules(
            architecture=dict(data.get("architecture") or {}),
            rules=[str(r) for r in (data.get("rules") or []) if str(r).strip()],
            blocked_dependencies=[str(d) for d in (deps.get("block") or []) if str(d).strip()],
            allowed_dependencies=[str(d) for d in (deps.get("allow") or []) if str(d).strip()],
        )
    return EngineeringRules()
