"""journeys.py — Vibe-DSL user-journey verification (BotIntern pattern).

A lightweight YAML DSL (journeys.yml) defines user journeys: visit a URL, see
text, see an element, click, fill, wait. The engine runs them in a headless
browser as a verification gate; on failure the report (with a page snapshot)
is handed to DeepSeek for surgical fixes, and the gate re-runs until green.

The agent can edit source code but can NEVER modify the journeys themselves —
journeys.yml is in the BOUND danger zones (Read-Many, Write-One).

Stdlib + the venv's playwright (optional): when playwright is not installed the
runner reports a clear message instead of crashing, so projects without the
feature still work.

Run standalone:  .venv/bin/python journeys.py --file journeys.yml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

try:  # playwright is a venv dep; the feature degrades gracefully without it
    from playwright.sync_api import sync_playwright  # noqa: F401
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:  # PyYAML is a venv dep too
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SUPPORTED_STEPS = {"visit", "see_text", "see_element", "click", "fill",
                   "wait_selector", "wait_text", "wait_ms", "screenshot"}
INSTALL_HINT = ("playwright not installed — run: "
                "./.venv/bin/pip install playwright && "
                "./.venv/bin/python -m playwright install chromium")


# ---------------------------------------------------------------------------
# DSL parsing
# ---------------------------------------------------------------------------

class JourneyError(Exception):
    pass


def load_journeys(path: str | os.PathLike) -> dict:
    """Parse and validate a journeys.yml file. Raises JourneyError on bad shape."""
    if not HAS_YAML:
        raise JourneyError("PyYAML not installed — run ./.venv/bin/pip install pyyaml")
    p = Path(path)
    if not p.is_file():
        raise JourneyError(f"journeys file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise JourneyError(f"invalid YAML in {p}: {exc}")
    if not isinstance(data, dict):
        raise JourneyError("journeys.yml must be a YAML mapping")
    journeys = data.get("journeys")
    if not isinstance(journeys, list) or not journeys:
        raise JourneyError("journeys.yml needs a non-empty 'journeys:' list")
    for j in journeys:
        if not isinstance(j, dict) or not j.get("name") or not isinstance(j.get("steps"), list):
            raise JourneyError("each journey needs 'name:' and a 'steps:' list")
        for step in j["steps"]:
            keys = list(step.keys())
            if len(keys) != 1 or keys[0] not in SUPPORTED_STEPS:
                raise JourneyError(
                    f"journey '{j['name']}': each step must have exactly one of "
                    f"{sorted(SUPPORTED_STEPS)}, got {keys}")
    return data


def _step_command(step: dict) -> str:
    (action, value), = step.items()
    if isinstance(value, dict):
        return f"{action} {json.dumps(value, sort_keys=True)}"
    return f"{action} {value}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _page_snapshot(page, limit: int = 1500) -> str:
    try:
        text = page.locator("body").inner_text(timeout=3000)
    except Exception:  # noqa: BLE001
        text = ""
    text = " ".join((text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _apply_step(page, action: str, value, app_url: str, timeout_ms: int) -> str:
    """Run one step; returns '' on success or an error description."""
    try:
        if action == "visit":
            page.goto(urllib.parse.urljoin(app_url, str(value)),
                      wait_until="domcontentloaded", timeout=timeout_ms)
        elif action == "see_text":
            body = page.locator("body").inner_text(timeout=timeout_ms)
            if str(value) not in body:
                return f"text {value!r} not found on the page"
        elif action == "see_element":
            if page.locator(str(value)).count() == 0:
                return f"element {value!r} not found"
        elif action == "click":
            page.locator(str(value)).first.click(timeout=timeout_ms)
        elif action == "fill":
            page.locator(value["selector"]).fill(str(value["value"]), timeout=timeout_ms)
        elif action == "wait_selector":
            page.locator(str(value)).wait_for(state="visible", timeout=timeout_ms)
        elif action == "wait_text":
            page.get_by_text(str(value), exact=False).wait_for(state="visible",
                                                               timeout=timeout_ms)
        elif action == "wait_ms":
            page.wait_for_timeout(int(value))
        elif action == "screenshot":
            page.screenshot(path=str(value))
        return ""
    except Exception as exc:  # noqa: BLE001 - step failure is reported
        return f"{type(exc).__name__}: {exc}"


def run_journeys(path: str | os.PathLike, base_url: str | None = None,
                 timeout_s: int = 30, screenshots_dir: str | None = None,
                 launch_extra: dict | None = None) -> tuple[bool, str]:
    """Run all journeys headlessly. Returns (ok, report).

    report contains per-journey step results and, on failure, a page snapshot
    so the fix loop has real context. Never raises for browser errors."""
    if not HAS_PLAYWRIGHT:
        return False, f"JOURNEYS UNAVAILABLE: {INSTALL_HINT}"
    try:
        data = load_journeys(path)
    except JourneyError as exc:
        return False, f"JOURNEYS CONFIG ERROR: {exc}"
    app_url = base_url or str(data.get("app_url", "")).rstrip("/")
    if not app_url:
        return False, "JOURNEYS CONFIG ERROR: set app_url in journeys.yml or pass --base-url"
    screens_dir = Path(screenshots_dir) if screenshots_dir else None
    setup = data.get("setup") or []
    browser_name = str(data.get("browser", "chromium")).lower()

    lines: list[str] = []
    ok = True
    try:
        with sync_playwright() as pw:
            browser_cls = getattr(pw, browser_name)
            browser = browser_cls.launch(headless=True, **(launch_extra or {}))
            try:
                for cmd in setup:
                    proc = subprocess.run(cmd, shell=True, capture_output=True,
                                          text=True, timeout=timeout_s * 4)
                    lines.append(f"setup: $ {cmd} -> exit {proc.returncode}")
                for journey in data["journeys"]:
                    name = journey["name"]
                    page = browser.new_page()
                    j_ok = True
                    for i, step in enumerate(journey["steps"], start=1):
                        (action, value), = step.items()
                        err = _apply_step(page, action, value, app_url,
                                          timeout_s * 1000)
                        mark = "OK" if not err else "FAILED"
                        if err:
                            j_ok = False
                            ok = False
                            lines.append(f"journey: {name}\n  step {i} ({_step_command(step)}): "
                                         f"{mark} — {err}\n  page: {_page_snapshot(page)}")
                            if screens_dir and action != "screenshot":
                                shot = screens_dir / f"{name.replace(' ', '_')}.png"
                                try:
                                    shot.parent.mkdir(parents=True, exist_ok=True)
                                    page.screenshot(path=str(shot))
                                    lines[-1] += f"\n  screenshot: {shot}"
                                except Exception:  # noqa: BLE001
                                    pass
                            break
                        lines.append(f"journey: {name}\n  step {i} "
                                     f"({_step_command(step)}): {mark}")
                    page.close()
                    lines.append(f"journey: {name} -> {'PASS' if j_ok else 'FAIL'}")
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - report cleanly
        return False, f"JOURNEYS BROWSER ERROR: {exc}"
    return ok, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="run journeys.yml user journeys headlessly")
    ap.add_argument("--file", default="journeys.yml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    ok, report = run_journeys(args.file, base_url=args.base_url, timeout_s=args.timeout)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
