"""scanner.py — secrets / PII scanning and redaction for OUTBOUND cloud traffic.

The BOUND protects what the agent writes to disk; this protects what LEAVES
the machine. Every request sent to DeepSeek (enhance, reviews, fixes, audit)
flows through _redact_cloud, which scans and redacts secrets/PII before the
payload is transmitted:

- api keys (sk-...), AWS access keys, GitHub tokens, JWTs, private keys,
  Stripe live keys, database connection strings, emails, phone numbers.

Modes (config secrets_scan.mode): 'redact' (default, replace with
<REDACTED:type>), 'block' (raise before sending), 'off'.

Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("stripe", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("conn_string", re.compile(r"(?:postgres|mysql|mongodb|redis)://[^\s/]+:[^\s@/]+@")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?:\+\d[\d -]{7,}\d|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b)")),
]

DEFAULT_TYPES = [t for t, _ in _PATTERNS]


@dataclass
class ScanResult:
    redacted: str
    findings: list = field(default_factory=list)  # [{"type": str, "count": int}]


def scan_text(text: str, types: list[str] | None = None) -> list[str]:
    """Return the distinct secret/PII types found in text (no redaction)."""
    types = types or DEFAULT_TYPES
    found: list[str] = []
    for t, pattern in _PATTERNS:
        if t in types and pattern.search(text or ""):
            found.append(t)
    return found


def redact_text(text: str, types: list[str] | None = None) -> ScanResult:
    """Replace every secret/PII match with <REDACTED:<type>>. Returns the
    redacted text plus a per-type finding list."""
    types = types or DEFAULT_TYPES
    result = ScanResult(redacted=text or "", findings=[])
    if not result.redacted:
        return result
    for t, pattern in _PATTERNS:
        if t not in types:
            continue
        if pattern.search(result.redacted):
            count = len(pattern.findall(result.redacted))
            result.redacted = pattern.sub(f"<REDACTED:{t}>", result.redacted)
            result.findings.append({"type": t, "count": count})
    return result


def redact_cloud(cloud, cfg: dict):
    """Wrap a cloud backend so every outbound request is scanned/redacted.
    Composes with the budget/cache/failover chain (wrap OUTERMOST so the cache
    and downstream providers never see raw secrets)."""
    sc = cfg.get("secrets_scan") or {}
    mode = str(sc.get("mode", "redact"))
    if mode == "off":
        return cloud
    types = sc.get("types")

    def generate(self, req):
        user_res = redact_text(req.user, types)
        sys_res = redact_text(req.system, types)
        findings = user_res.findings + sys_res.findings
        if findings:
            kinds = ", ".join(f"{f['type']}x{f['count']}" for f in findings)
            if mode == "block":
                raise RuntimeError(
                    "secrets_scan=block: secret/PII detected in outbound request "
                    f"({kinds}) — refusing to send")
            _status(f"[hybrid] 🔒 redacted {len(findings)} secret/PII finding(s) "
                    f"before cloud send: {kinds}")
            req.user = user_res.redacted
            req.system = sys_res.redacted
        return cloud.generate(req)

    return type("_RedactCloud", (), {"generate": generate})()


# _status is overridden by ask.py at import time (avoids a circular import).
def _status(line: str) -> None:  # pragma: no cover - ask.py replaces this
    pass
