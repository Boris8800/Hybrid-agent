"""bound.py — the BOUND: hard runtime constraints the agent can never bypass.

The Ouro-Loop pattern: a BOUND (danger zones, never-do commands, iron laws)
plus a phased program with verification gates and remediation. The BOUND is
enforced at runtime — files matching a danger zone are never written, commands
matching never_do are never run, and a run that attempts a violation exits
with code 2. The user can disable it entirely with --no-bound, but the agent
itself cannot bypass it.

Stdlib only. Reads the `bound:` section of config.yml.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

DEFAULT_DANGER_ZONES = [
    "**/.env", "**/.env.*", "**/*.pem", "**/*.key", "**/*.p12",
    ".git/**", ".kilo/**", "**/.venv/**", "**/node_modules/**",
    "**/memory/**", "**/.cache/**",
]
DEFAULT_NEVER_DO = [
    "rm -rf", "rm -fr", "git push --force", "git push -f", "git push origin --force",
    "git reset --hard", "git clean -f", "chmod 777", "chmod -R", "sudo ",
    "shutdown", "reboot", "mkfs", ":(){", "base64 -d", "eval ", "curl | sh",
]
DEFAULT_IRON_LAWS = [
    "Never modify or create secrets, credentials, .env files, private keys, or certs.",
    "Never run destructive git operations (force push, hard reset, clean, delete).",
    "Never write files outside the project root or inside declared danger zones.",
    "Never modify the agent's own configuration, memory, or cache.",
    "Never bypass, disable, or weaken the BOUND or the verification gates.",
    "Always leave the project in a building state; revert if you cannot verify.",
]


@dataclass
class Bound:
    danger_zones: list = field(default_factory=list)
    never_do: list = field(default_factory=list)
    iron_laws: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.danger_zones or self.never_do or self.iron_laws)

    # --- runtime enforcement ---------------------------------------------
    def enforce_path(self, rel_path: str) -> str | None:
        """Return a reason string when a relative path hits a danger zone,
        else None (allowed). Backslashes normalized before matching. Zones like
        '**/.env' also match the bare filename at any depth."""
        clean = (rel_path or "").replace("\\", "/")
        if not clean:
            return None
        base = clean.rsplit("/", 1)[-1]
        for zone in self.danger_zones:
            z = zone.replace("**", "*")
            last = zone.rsplit("/", 1)[-1]
            if (fnmatch.fnmatch(clean, z)
                    or fnmatch.fnmatch(clean, z.rstrip("/") + "/*")
                    or (last and last != "**" and fnmatch.fnmatch(base, last))):
                return f"danger zone '{zone}'"
        return None

    def check_command(self, cmd: str) -> str | None:
        """Return a reason string when a command violates never_do, else None."""
        c = (cmd or "").strip().lower()
        if not c:
            return "empty command"
        for pattern in self.never_do:
            if pattern.lower() in c:
                return f"never_do '{pattern}'"
        return None

    # --- prompt (RECALL gate) -------------------------------------------
    def prompt_text(self) -> str:
        """The BOUND as prompt text, re-injected before every step so the
        models cannot 'forget' it (RECALL gate)."""
        if not self.active:
            return ""
        lines = ["IRON LAWS — THE BOUND (hard constraints; you can NEVER bypass these):"]
        for i, law in enumerate(self.iron_laws or [], start=1):
            lines.append(f"{i}. {law}")
        if self.danger_zones:
            lines.append("DANGER ZONES (never read-modify, write, or delete these): "
                         + ", ".join(self.danger_zones))
        if self.never_do:
            lines.append("NEVER RUN: " + ", ".join(self.never_do))
        return "\n".join(lines)


def load_bound(cfg: dict) -> Bound:
    """Build the BOUND from config; sensible defaults fill unset sections."""
    b = cfg.get("bound") or {}
    zones = b.get("danger_zones")
    never = b.get("never_do")
    laws = b.get("iron_laws")
    return Bound(
        danger_zones=list(zones) if isinstance(zones, list) else list(DEFAULT_DANGER_ZONES),
        never_do=list(never) if isinstance(never, list) else list(DEFAULT_NEVER_DO),
        iron_laws=list(laws) if isinstance(laws, list) else list(DEFAULT_IRON_LAWS),
    )
