"""dependency_gate.py — new-dependency protection.

When the agent's applied diff adds a dependency to a manifest (package.json,
requirements.txt, pyproject.toml, go.mod, Cargo.toml), the run pauses:

    NEW DEPENDENCY DETECTED: Package <x> in package.json

Approval is required: interactive y/N, or non-interactive exit/verdict 7.
Engineering rules' blocked_dependencies are rejected outright; pre-approved
packages (rules.allowed_dependencies or --allow-dep) pass silently.

Optional `audit` fetches registry metadata (license/description) best-effort.
Stdlib only.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request

_MANIFESTS = (
    "package.json", "package-lock.json", "requirements.txt",
    "pyproject.toml", "go.mod", "Cargo.toml",
)
_PKG_JSON_ADD = re.compile(r"^\s*\"([A-Za-z0-9@./_-]+)\"\s*:\s*\"[^\"]+\",?\s*$")


def _git_diff_manifests(root: str) -> list[tuple[str, list[str]]]:
    try:
        proc = subprocess.run(["git", "diff", "-U0", "--", *list(_MANIFESTS)],
                              cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    diff = proc.stdout if proc.returncode == 0 else ""
    files: list[tuple[str, list[str]]] = []
    current: str | None = None
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            if current is not None:
                files.append((current, added))
            current = line[6:].strip()
            added = []
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if current is not None and line.startswith("+"):
            added.append(line[1:])
    if current is not None:
        files.append((current, added))
    return [(p, a) for p, a in files if p in _MANIFESTS]


def detect_new_dependencies(root: str) -> list[dict]:
    """Return [{"package": str, "manifest": str, "kind": str}] from the diff."""
    deps: list[dict] = []
    for manifest, added in _git_diff_manifests(root):
        if manifest in ("package.json", "package-lock.json"):
            for line in added:
                m = _PKG_JSON_ADD.match(line)
                if m:
                    deps.append({"package": m.group(1), "manifest": manifest,
                                 "kind": "npm"})
        elif manifest == "requirements.txt":
            for line in added:
                m = re.match(r"^\s*([A-Za-z0-9_.-]+)(?:==|>=|<=|~=|>|<|\s|$)", line)
                if m and not line.strip().startswith(("#", "-r", "-e")):
                    deps.append({"package": m.group(1), "manifest": manifest,
                                 "kind": "pypi"})
        elif manifest == "pyproject.toml":
            for line in added:
                m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*[\"']", line)
                if m:
                    deps.append({"package": m.group(1), "manifest": manifest,
                                 "kind": "pypi"})
        elif manifest == "go.mod":
            for line in added:
                m = re.match(r"^\s*require\s+([\w./-]+)", line)
                if m:
                    deps.append({"package": m.group(1), "manifest": manifest,
                                 "kind": "go"})
        elif manifest == "Cargo.toml":
            for line in added:
                m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=\s*[\"{]", line)
                if m and not m.group(1).startswith(("dependencies", "dev")):
                    deps.append({"package": m.group(1), "manifest": manifest,
                                 "kind": "cargo"})
    # de-duplicate
    seen = set()
    out = []
    for d in deps:
        key = (d["package"], d["manifest"])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def registry_audit(dep: dict) -> str:
    """Best-effort registry metadata (license/description). '' on failure."""
    try:
        if dep["kind"] == "pypi":
            url = f"https://pypi.org/pypi/{dep['package']}/json"
        else:
            url = f"https://registry.npmjs.org/{dep['package']}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        if dep["kind"] == "pypi":
            info = data.get("info", {})
            return f"license={info.get('license') or 'unknown'}; {str(info.get('summary') or '')[:80]}"
        latest = data.get("dist-tags", {}).get("latest", "")
        version = (data.get("versions", {}) or {}).get(latest, {}) or {}
        lic = version.get("license")
        if isinstance(lic, dict):
            lic = lic.get("type")
        return f"latest={latest}; license={lic or 'unknown'}; {str(version.get('description') or '')[:80]}"
    except Exception:  # noqa: BLE001 - audit is best-effort
        return ""


def run_dependency_gate(root: str, rules, status, allow_deps=(),
                        audit: bool = False) -> tuple[bool, str, list[dict]]:
    """Check the applied diff for new dependencies. Returns (ok, report, deps)."""
    deps = detect_new_dependencies(root)
    if not deps:
        return True, "", []
    allowed = {str(a).lower() for a in (rules.allowed_dependencies or [])} | \
              {str(a).lower() for a in (allow_deps or [])}
    report_lines: list[str] = []
    for dep in deps:
        name = dep["package"]
        blocked = rules.check_dependency(name) if rules else None
        if blocked:
            report_lines.append(f"{name} ({dep['manifest']}) — {blocked}")
            status(f"[hybrid] ⛔ DEPENDENCY BLOCKED: {name} — {blocked}")
            continue
        if name.lower() in allowed:
            status(f"[hybrid] ⚙ dependency {name} pre-approved")
            continue
        detail = registry_audit(dep) if audit else ""
        status(f"[hybrid] ⚠ NEW DEPENDENCY DETECTED: {name} in {dep['manifest']}"
               + (f" · {detail}" if detail else ""))
        report_lines.append(f"{name} ({dep['manifest']})"
                            + (f" — {detail}" if detail else ""))
    if not report_lines:
        return True, "", []
    return False, "NEW DEPENDENCY DETECTED (approval required):\n" + \
        "\n".join(report_lines), deps
