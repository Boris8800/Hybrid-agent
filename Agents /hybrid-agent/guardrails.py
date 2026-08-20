"""guardrails.py — structured task guardrails (BLOCK / APPROVAL_REQUIRED).

Beyond the BOUND (which constrains WHAT the agent does at runtime), these gate
the task BEFORE it runs:

- BLOCK: the task is rejected outright (exit code 2) — e.g. 'drop table'.
- APPROVAL_REQUIRED: the task escalates to a human (exit code 7 non-interactive,
  y/N prompt interactive) — content gates or a cost estimate above cost_limit.

Stdlib only. Reads the `guardrails:` section of config.yml.
"""

from __future__ import annotations

# Rough blended DeepSeek input+output cost per token (input $0.14/M, output
# $0.28/M, mix weighted toward input). Used only as an estimate for the cost
# gate; the default limit is high so it never fires unless configured.
_COST_PER_TOKEN = 0.00000025


def estimate_cost_usd(task: str) -> float:
    return len((task or "").split()) * _COST_PER_TOKEN


def check_guardrails(task: str, cfg: dict) -> tuple[str, str]:
    """Return (decision, reason): 'allow' | 'block' | 'approval_required'."""
    g = cfg.get("guardrails") or {}
    low = (task or "").lower()
    for pat in (g.get("block") or []):
        if str(pat).lower() in low:
            return "block", f"task contains blocked content: {pat!r}"
    for pat in (g.get("approval_required") or []):
        if str(pat).lower() in low:
            return "approval_required", f"task contains approval-required content: {pat!r}"
    try:
        limit = float(g.get("cost_limit", 1000.0))
    except (TypeError, ValueError):
        limit = 1000.0
    est = estimate_cost_usd(task)
    if est > limit:
        return "approval_required", (
            f"estimated cost ${est:.2f} exceeds the configured limit ${limit:.2f}")
    return "allow", ""
