"""
Bridge CLI for shelling out from a Kilo agent via the bash tool.

Calls the DeepSeek cloud API and the local LM Studio endpoint
(http://localhost:1234/v1) DIRECTLY through the existing backends - no Kilo
providers involved. Unlike agent.py (which truncates to 2000 chars), it returns
FULL, UNTRUNCATED model output on stdout.

Run (from repo root or from inside hybrid-agent/):
    python3 hybrid-agent/ask.py --help
    python3 hybrid-agent/ask.py --models
    python3 hybrid-agent/ask.py --route-only --task "<task>"
    python3 hybrid-agent/ask.py [--local|--deepseek|--auto] --task "<task>"
        [--context "<context>"] [--system "<system>"] [--model "<id>"]
        [--max-tokens N] [--temperature T] [--config PATH] [--json]
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from backends.base import BackendError, ModelRequest, ModelResponse
from backends.local_gemma import GemmaBackend

from agent import HybridAgent, _load_config, apply_env_overrides
from backends.deepseek import _resolve_api_key
from supervise import (GEMMA_MAX_TOKENS, ChainOfThoughtParser, ReviewPackage,
                       SuperviseResult, Verdict, _enhance_request,
                       _supervisor_request, parse_enhancement, parse_verdict,
                       supervise)
from context import ProjectContext
from parallel import (DependencyAnalyzer, ParallelExecutor, parse_plan_steps,
                      summarize)
from memory import TaskMemory, TaskRecord, memory_root_from_cfg
from embed import memory_embed_callable
from gitops import git_pull, git_push, run_deploy
from providers import (backend_for, enabled_online, get_local, get_online,
                       load_providers, resolve_api_key)

# --- Self-heal: loading config.yml requires PyYAML + openai, which exist only
# in the project venv (hybrid-agent/.venv). When invoked with a system python3
# that lacks them, ask.py silently fell back to embedded defaults (10s local
# timeout, unloaded model id "gemma-4-12b") - the root cause of slow and
# truncated local generations. Re-exec with the venv interpreter instead.
# Guarded by env var so the re-exec cannot loop, and by __name__ so importing
# ask.py from tests / other tools (which must never replace the process) is safe.
if __name__ == "__main__" and os.environ.get("HYBRID_REHEALED") != "1":
    try:
        import yaml  # noqa: F401
    except ImportError:
        venv_python = pathlib.Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        if venv_python.is_file():
            try:
                probe = subprocess.run(
                    [str(venv_python), "-c", "import yaml, openai"],
                    capture_output=True, timeout=10,
                )
            except subprocess.SubprocessError:
                probe = None
            if probe is not None and probe.returncode == 0:
                sys.stderr.write(
                    "[hybrid] re-exec'ing with project venv python "
                    "(PyYAML/openai missing in current interpreter)\n"
                )
                os.environ["HYBRID_REHEALED"] = "1"
                os.execv(str(venv_python), [str(venv_python), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])
        sys.stderr.write(
            "[hybrid] warning: PyYAML not available anywhere - using embedded defaults "
            "(local timeout may be too short, expect slow/truncated local calls)\n"
        )

REVIEW_PROMPT_FILE = pathlib.Path(__file__).resolve().parent / "review" / "supervisor.md"


def _strip_confidence_tag(text: str) -> str:
    """Remove a trailing <CONFIDENCE>...</CONFIDENCE> tag a model may emit."""
    import re as _re
    return _re.sub(r"\s*<CONFIDENCE>\s*[0-9.]+</CONFIDENCE>\s*$", "", text).rstrip()


_CONTEXT_FILE = pathlib.Path(__file__).resolve().parent / "context.json"


def get_project_context(task: str) -> str:
    """Return relevant project context from context.json (produced by scan.py).

    If no context file exists, returns a short notice. Otherwise tailors the
    snippet to the task (tests / entry points / general overview).
    """
    if not _CONTEXT_FILE.is_file():
        return "No project context available. Run hybrid-agent/scan.py first."
    try:
        context = json.loads(_CONTEXT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Project context is stale/unreadable. Re-run hybrid-agent/scan.py."

    if "test" in task.lower():
        tests = context.get("structure", {}).get("tests", "not found")
        return f"Tests directory: {tests}"
    if any(w in task.lower() for w in ("main", "app", "entry")):
        return f"Entry points: {context.get('entry_points', [])}"
    return (
        f"Project type: {context.get('structure', {}).get('framework', 'Unknown')}\n"
        f"Dependencies: {list(context.get('dependencies', {}).keys())}"
    )


def _load_supervisor_prompt() -> str:
    """Return the DeepSeek supervisor-review system prompt (review/supervisor.md)."""
    if REVIEW_PROMPT_FILE.is_file():
        return REVIEW_PROMPT_FILE.read_text()
    return (
        "You are a CRITICAL senior reviewer. Review the provided code and return "
        "APPROVED, FIX_REQUIRED, or REJECTED. List every issue with exact location "
        "and a concrete fix code snippet. Distinguish CRITICAL/MAJOR/MINOR/SUGGESTION. "
        "Be definitive. If REJECTED, explain the alternative approach."
    )


def _status(line: str) -> None:
    """Emit a human-readable status line to stderr and flush so it streams to
    the user/agent immediately (before the model call finishes)."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _model_label(cfg: dict, route: str, model_override: str | None) -> str:
    """Human label for the backend being invoked."""
    if route == "local":
        lc = cfg["backends"]["local"]
        return f"local:{model_override or lc['model']} (LM Studio)"
    return "deepseek"


def _task_preview(task: str, limit: int = 60) -> str:
    """Single-line short preview of the task for the status line."""
    text = " ".join(task.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_STATS_FILE = pathlib.Path(__file__).resolve().parent / "stats.json"


def _iso_week() -> str:
    """Return the current ISO week as 'YYYY-Www' (e.g. '2026-W34')."""
    iso = date.today().isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


class ProgressTracker:
    """Minimal progress/percentage indicator for the pipeline phases.

    Tracks weighted phase percentages and per-phase elapsed times. Renders a
    live ``\\r`` bar on stderr when stderr is a TTY, and a plain
    ``[hybrid] progress ...`` line otherwise (so agent stderr parsers keep
    working when piped). Never writes to stdout — stdout carries the model
    output / single JSON object. Strictly additive: existing
    ``[hybrid]`` / ``[supervise]`` status lines are passed through unchanged.
    """

    def __init__(self, weights: dict | None = None):
        self.weights = weights or {
            "enhance": 10, "implement": 45, "apply": 5, "verify": 40,
        }
        self._phases: dict = {}
        self._current: str | None = None
        self._start: float | None = None
        self._done_weight = 0
        self._tty = sys.stderr.isatty()

    def start_phase(self, name: str) -> None:
        self._current = name
        self._start = time.monotonic()

    def end_phase(self) -> None:
        if self._current is None:
            return
        name, self._current = self._current, None
        elapsed = (time.monotonic() - self._start) if self._start else 0.0
        self._done_weight += self.weights.get(name, 0)
        self._phases[name] = {"elapsed_s": round(elapsed, 2), "pct": self._done_weight}
        if not self._tty:
            _status(f"[hybrid] progress phase={name} {self._done_weight}% {elapsed:.2f}s")
        else:
            self.refresh()

    def tick(self, line: str) -> None:
        """Emit a status line, then re-render the bar (chains status callbacks)."""
        _status(line)
        self.refresh()

    def refresh(self) -> None:
        if not self._tty or self._current is None:
            return
        elapsed = (time.monotonic() - self._start) if self._start else 0.0
        pct = min(100, self._done_weight + self.weights.get(self._current, 0))
        filled = int(pct / 100 * 20)
        bar = "#" * filled + "." * (20 - filled)
        sys.stderr.write(f"\r[{self._current} {pct:3d}%] {bar} {elapsed:5.1f}s")
        sys.stderr.flush()

    def phases(self) -> dict:
        return self._phases

    def done(self) -> dict:
        """Close out the bar (newline on TTY, 100% close-out line when piped)."""
        if self._tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            total = round(sum(d["elapsed_s"] for d in self._phases.values()), 2)
            _status(f"[hybrid] progress phase=done 100% {total:.2f}s")
        return self._phases


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Write a file atomically (temp + os.replace) so concurrent sessions can
    never observe or leave a torn file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class CacheManager:
    """Response cache for the hybrid bridge (enhance / generate / fix kinds).

    Stores raw model response text under hybrid-agent/.cache/<kind>/<sha256>.json
    with a TTL and a per-kind entry cap. Reads/writes are exception-safe and
    never raise. Truncated or empty output is never cached. Hit/miss counters
    go to stats.json via StatsTracker.record_cache.
    """

    def __init__(self, cfg: dict, args: argparse.Namespace):
        cache_cfg = cfg.get("cache", {})
        self.enabled = cache_cfg.get("enabled", True) and not args.no_cache
        self.dir = pathlib.Path(__file__).resolve().parent / cache_cfg.get("dir", ".cache")
        self.ttl_days = args.cache_ttl or cache_cfg.get("ttl_days", 7)
        self.max_entries = args.cache_max_size or cache_cfg.get("max_entries", 100)
        if self.enabled:
            self.clean()

    def key(self, kind: str, *parts: str) -> str:
        """Deterministic cache key from the kind and full request material."""
        raw = kind + "|" + "\x1f".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, kind: str, key: str) -> str | None:
        """Return cached text if fresh, else None. Never raises."""
        if not self.enabled:
            return None
        try:
            path = self.dir / kind / f"{key}.json"
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self.ttl_days * 86400:
                path.unlink()
                return None
            return json.loads(path.read_text()).get("text")
        except (OSError, ValueError, KeyError):
            return None

    def set(self, kind: str, key: str, text: str, source: str = "gemma") -> None:
        """Store text (never empty) and prune the oldest entries past the cap."""
        if not self.enabled or not text:
            return
        try:
            kind_dir = self.dir / kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            payload = {"text": text, "source": source, "ts": datetime.now().isoformat()}
            _atomic_write(kind_dir / f"{key}.json", json.dumps(payload))
            entries = sorted(kind_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            while len(entries) > self.max_entries:
                entries[0].unlink()
                entries.pop(0)
        except OSError:
            pass

    def clean(self) -> None:
        """Delete all entries older than the TTL across every kind."""
        if not self.enabled:
            return
        try:
            for kind_dir in self.dir.iterdir():
                if not kind_dir.is_dir():
                    continue
                for entry in kind_dir.glob("*.json"):
                    try:
                        if time.time() - entry.stat().st_mtime > self.ttl_days * 86400:
                            entry.unlink()
                    except OSError:
                        continue
        except OSError:
            pass

    def record(self, hit: bool) -> None:
        """Record a hit/miss counter. No-op when disabled."""
        if not self.enabled:
            return
        StatsTracker().record_cache(hit)


class StatsTracker:
    """Persists 80/20 strategy + self-evaluation metrics to hybrid-agent/stats.json."""

    def __init__(self, path=None):
        self.path = pathlib.Path(path) if path else _STATS_FILE
        self.stats = self._load()

    def _load(self) -> dict:
        data = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        # setdefault so OLD stats.json files (missing the new keys) still work.
        defaults = {
            "deepseek_reviews": 0, "approvals": 0, "rejections": 0,
            "fix_required": 0, "deepseek_fallbacks": 0, "total_quality_score": 0.0,
            "tasks_completed": 0, "files_generated": 0, "total_iterations": 0,
            "truncation_events": 0, "selfeval_quality_sum": 0.0,
            "selfeval_quality_count": 0, "periods": {},
        }
        for key, value in defaults.items():
            data.setdefault(key, value)
        return data

    def _save(self) -> None:
        try:
            _atomic_write(self.path, json.dumps(self.stats, indent=2))
        except OSError:
            pass

    def record_review(self, verdict: str, quality_score: float = 0.0) -> None:
        s = self.stats
        s["deepseek_reviews"] += 1
        s["total_quality_score"] = s.get("total_quality_score", 0.0) + quality_score
        if verdict == "APPROVED":
            s["approvals"] += 1
        elif verdict == "REJECTED":
            s["rejections"] += 1
            s["deepseek_fallbacks"] += 1
        elif verdict == "FIX_REQUIRED":
            s["fix_required"] += 1
        self._save()

    def record_fallback(self) -> None:
        self.stats["deepseek_fallbacks"] += 1
        self._save()

    def record_session(self, files_generated: int = 0, iterations: int = 0,
                       truncation: bool = False, quality: float = 0.0) -> None:
        """Record one completed task/session, plus a per-ISO-week snapshot."""
        s = self.stats
        s["tasks_completed"] += 1
        s["files_generated"] += files_generated
        s["total_iterations"] += iterations
        if truncation:
            s["truncation_events"] += 1
        if quality:
            s["selfeval_quality_sum"] += quality
            s["selfeval_quality_count"] += 1

        week = _iso_week()
        p = s.setdefault("periods", {}).setdefault(week, {
            "files_generated": 0, "iterations": 0, "truncation_events": 0,
            "tasks_completed": 0, "quality_sum": 0.0, "quality_count": 0,
        })
        p["files_generated"] += files_generated
        p["iterations"] += iterations
        if truncation:
            p["truncation_events"] += 1
        p["tasks_completed"] += 1
        if quality:
            p["quality_sum"] += quality
            p["quality_count"] += 1
        self._save()

    def record_verify(self, metrics: dict) -> None:
        """Record final-verify metrics (iterations, API calls, tokens, cost)."""
        s = self.stats
        v = s.setdefault("verify", {
            "runs": 0, "iterations": 0, "api_calls": 0, "tokens_used": 0,
            "estimated_cost_usd": 0.0, "passed": 0, "failed": 0, "last": {},
        })
        v["runs"] += 1
        v["iterations"] += metrics.get("iterations", 0)
        v["api_calls"] += metrics.get("api_calls", 0)
        v["tokens_used"] += metrics.get("tokens_used", 0)
        v["estimated_cost_usd"] = round(
            v["estimated_cost_usd"] + metrics.get("estimated_cost_usd", 0.0), 5)
        status = metrics.get("status", "FAILED")
        if status == "PASSED":
            v["passed"] += 1
        elif status in ("FAILED", "REGRESSION_FAILED", "ENV_ERROR", "BLOCKED"):
            v["failed"] += 1
        v["last"] = metrics
        self._save()

    def record_phases(self, phases: dict) -> None:
        """Persist per-phase timing aggregates (runs, total seconds) to stats.json."""
        if not phases:
            return
        s = self.stats
        pt = s.setdefault("phase_timings", {})
        for name, data in phases.items():
            entry = pt.setdefault(name, {"runs": 0, "total_s": 0.0})
            entry["runs"] += 1
            entry["total_s"] = round(entry["total_s"] + data.get("elapsed_s", 0.0), 2)
        self._save()

    def record_cache(self, hit: bool) -> None:
        """Record a cache hit or miss counter."""
        s = self.stats
        c = s.setdefault("cache", {"hits": 0, "misses": 0})
        c["hits" if hit else "misses"] += 1
        self._save()

    def record_tokens(self, route: str, prompt_tokens: int = 0,
                      completion_tokens: int = 0, estimated: bool = False) -> None:
        """Persist daily token usage per route (API budget accounting)."""
        day = str(date.today())
        s = self.stats.setdefault("api_tokens", {}).setdefault(
            day, {"tokens": 0, "estimated": 0})
        s["tokens"] += int(prompt_tokens or 0) + int(completion_tokens or 0)
        if estimated:
            s["estimated"] += 1
        self._save()

    def daily_api_tokens(self) -> int:
        """Tokens consumed by API routes so far today (0 when never recorded)."""
        return self.stats.get("api_tokens", {}).get(str(date.today()), {}).get("tokens", 0)

    def evaluate(self) -> dict:
        """Self-evaluation report for the current week vs the previous week."""
        s = self.stats
        current_week = _iso_week()
        iso = (date.today() - timedelta(days=7)).isocalendar()
        last_week = f"{iso[0]}-W{iso[1]:02d}"
        p = s.get("periods", {}).get(current_week, {})
        prev = s.get("periods", {}).get(last_week, {})

        files = p.get("files_generated", 0)
        iterations = p.get("iterations", 0)
        tasks = p.get("tasks_completed", 0)
        avg = round(p["quality_sum"] / p["quality_count"], 1) if p.get("quality_count") else 0.0
        trunc = p.get("truncation_events", 0)
        rate = round(trunc / tasks * 100, 1) if tasks else 0.0
        prev_avg = round(prev["quality_sum"] / prev["quality_count"], 1) if prev.get("quality_count") else None

        if prev_avg is None:
            comparison = "No prior period to compare against"
        elif avg > prev_avg:
            comparison = f"This is better than last week ({prev_avg}/10 average)"
        elif avg < prev_avg:
            comparison = f"This is worse than last week ({prev_avg}/10 average)"
        else:
            comparison = f"This matches last week ({prev_avg}/10 average)"

        return {
            "period": current_week,
            "files_generated": files,
            "iterations": iterations,
            "tasks_completed": tasks,
            "average_quality": avg,
            "truncation_rate": rate,
            "truncation_events": trunc,
            "comparison": comparison,
            "previous_average": prev_avg,
        }

    def get_summary(self) -> dict:
        s = self.stats
        total = s["deepseek_reviews"]
        approval_rate = round((s["approvals"] / total) * 100, 1) if total else 0
        avg_quality = round(s["total_quality_score"] / total, 1) if total else 0
        fallback_rate = round((s["deepseek_fallbacks"] / total) * 100, 1) if total else 0
        return {
            "strategy": "80/20 Hybrid",
            "local_usage": "80%",
            "deepseek_usage": "20%",
            "reviews": total,
            "approval_rate": approval_rate,
            "average_quality": avg_quality,
            "fallback_rate": fallback_rate,
            "cost_saved": 80,
        }


def _default_config_path() -> str | None:
    """Locate config.yml / config.yaml next to this script (any cwd)."""
    script_dir = pathlib.Path(__file__).resolve().parent
    for name in ("config.yml", "config.yaml"):
        candidate = script_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def is_kilo_logged_in() -> bool:
    """Check whether a DeepSeek API key is resolvable (env → Kilo auth.json
    → Kilo config) without actually calling the API."""
    return _resolve_api_key("DEEPSEEK_API_KEY") is not None


def _apply_role_overrides(cfg: dict) -> dict:
    """Apply model-agnostic role configuration (see agent.apply_env_overrides)."""
    return apply_env_overrides(cfg)


# Mode-based role enforcement (architecture, not a soft rule).
# Each mode declares which endpoint may implement and whether API supervision
# is available. Enforcement uses the mode, not an env escape hatch.
MODES = {
    "hybrid": {"local_impl": True, "api_impl": False, "api_supervision": True},
    "local":  {"local_impl": True, "api_impl": False, "api_supervision": False},
    "code":   {"local_impl": False, "api_impl": True,  "api_supervision": False},
}


def _resolve_mode(args: argparse.Namespace) -> str:
    """Resolve the agent mode: --mode > $MODE > 'hybrid'."""
    if args.mode:
        return args.mode
    return os.environ.get("MODE", "hybrid")


def _mode_impl_violation(mode: str, route: str) -> str | None:
    """Return a violation message if `route` is not an allowed implementer for `mode`."""
    m = MODES.get(mode)
    if not m:
        return f"unknown mode {mode!r} (use hybrid|local|code)"
    if route == "local" and not m["local_impl"]:
        return f"mode={mode} does not allow LOCAL implementation (use --deepseek, the API implementer)"
    if route == "deepseek" and not m["api_impl"]:
        return f"mode={mode} forbids API implementation (the local model is the implementer; use --local or --supervise)"
    return None


def _load_config_quiet(path: str | None) -> dict:
    """Load config, diverting _load_config's warnings to stderr so stdout
    stays clean (stdout carries the full untruncated model output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = _load_config(path)
    if buf.getvalue():
        sys.stderr.write(buf.getvalue())
    return _apply_role_overrides(cfg)


def _load_cfg(args: argparse.Namespace) -> dict:
    """Load config and apply the startup wiring that depends on it
    (verify-allowlist prefixes, config-key validation)."""
    cfg = _load_config_quiet(args.config or _default_config_path())
    _configure_verify_allowlist(cfg)
    _warn_unknown_config(cfg)
    return cfg


# Every config key the engine understands. Anything else is a typo.
_KNOWN_CONFIG_KEYS = {
    "router": {"local_threshold", "threshold_min", "threshold_max",
               "target_local_rate", "alpha", "supervision", "weights"},
    "backends": {"local", "deepseek"},
    "review": {"max_local_retries", "max_depth_tokens", "max_failure_summary_words",
               "daily_token_budget", "verify", "verify_timeout", "verify_groups",
               "verify_allowlist", "regression", "regression_timeout",
               "terminal_timeout"},
    "cache": {"enabled", "dir", "ttl_days", "max_entries"},
    "circuit_breaker": {"window_size", "local_error_ceiling",
                        "deepseek_error_ceiling", "cooldown_s"},
    "memory": {"root", "max_project_summary_words", "semantic_similarity",
               "embedding_model", "embedding_threshold"},
    "providers": {"online", "local"},
    "deploy": {"command", "cwd", "timeout"},
    "roles": {"implementer", "supervisor"},
}


def _warn_unknown_config(cfg: dict) -> None:
    """Warn (stderr) about config keys the engine does not recognize, so typos
    like 'daily_toke_budget' are caught instead of silently ignored."""
    try:
        for key in cfg:
            if key not in _KNOWN_CONFIG_KEYS:
                print(f"warning: unknown config key '{key}' (typo? ignored by the engine)",
                      file=sys.stderr)
        for section, known in _KNOWN_CONFIG_KEYS.items():
            sub = cfg.get(section)
            if isinstance(sub, dict):
                for key in sub:
                    if key not in known:
                        print(f"warning: unknown config key '{section}.{key}' (typo? ignored)",
                              file=sys.stderr)
    except Exception:  # noqa: BLE001 - validation must never break startup
        pass


def _normalize_tokens(tokens) -> dict:
    """Normalize token usage (openai 1.x dict or openai 3.x pydantic object)
    to a plain JSON-serializable dict."""
    if tokens is None:
        return {}
    if isinstance(tokens, dict):
        return tokens
    dump = getattr(tokens, "model_dump", None)
    if callable(dump):
        return dump()
    data = getattr(tokens, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def _have_openai() -> bool:
    """True when the openai SDK is importable (decides backend vs HTTP path)."""
    try:
        from openai import OpenAI  # noqa: F401
        return True
    except ImportError:
        return False


def _http_generate(base_url: str, api_key: str, model: str, req: ModelRequest,
                   stream: bool = False) -> ModelResponse:
    """Dependency-free OpenAI-compatible chat completion via urllib.

    Used only when the `openai` package isn't installed. With stream=True it
    emits generated tokens live to stderr (so the user sees the local model
    working as it writes), while still returning the full text for stdout."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            body = json.dumps(payload).encode()
            if stream:
                body = json.dumps({**payload, "stream": True}).encode()
            request = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
            )
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=req.timeout_s) as resp:
                if not stream:
                    data = json.loads(resp.read().decode())
                    text = data["choices"][0]["message"]["content"] or ""
                    usage = data.get("usage") or {}
                    truncated = data["choices"][0].get("finish_reason") == "length"
                else:
                    pieces: list[str] = []
                    usage: dict = {}
                    marker = False
                    finish_reason = None
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data_payload = line[len("data:"):].strip()
                        if data_payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_payload)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"] or {}
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                        fr = (chunk.get("choices") or [{}])[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if not delta:
                            continue
                        if not marker:
                            _status("[hybrid]  <<stream>> ")
                            marker = True
                        pieces.append(delta)
                        sys.stderr.write(delta)
                        sys.stderr.flush()
                    if marker:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    text = "".join(pieces)
                    truncated = finish_reason == "length"
            return ModelResponse(
                text=text, raw=text,
                token_usage=usage,
                latency_ms=(time.monotonic() - started) * 1000,
                backend="http-fallback",
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001 - retry then surface
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"http fallback to {url} failed: {last}")


def _generate(agent: HybridAgent, cfg: dict, args: argparse.Namespace,
              route: str, req: ModelRequest, stream: bool = False) -> ModelResponse:
    """Invoke the right backend, or the built-in HTTP fallback when the openai
    package is unavailable.

    When streaming is requested on the LOCAL route, always use the HTTP path
    (not the SDK backend), because that is the only path that emits live
    tokens to stderr — the user's visibility into the local model working."""
    if route == "local" and stream:
        lc = cfg["backends"]["local"]
        base_url = lc["base_url"]
        api_key = lc.get("api_key") or "lm-studio"
        model = args.model or lc["model"]
        return _http_generate(base_url, api_key, model, req, stream=True)

    if _have_openai():
        backend = _select_backend(agent, cfg, route, args.model)
        return backend.generate(req)

    if route == "local":
        lc = cfg["backends"]["local"]
        base_url = lc["base_url"]
        api_key = lc.get("api_key") or "lm-studio"
        model = args.model or lc["model"]
    else:
        dc = cfg["backends"]["deepseek"]
        base_url = dc.get("base_url") or "https://api.deepseek.com"
        api_key = _resolve_api_key(dc["api_key_env"])
        model = dc["model"]
        if not api_key:
            raise RuntimeError(f"missing API key: set {dc['api_key_env']} or install Kilo auth.json")
    return _http_generate(base_url, api_key, model, req,
                          stream=stream and route == "local")


MAX_OUTPUT_TOKENS = 8192
# Local (LM Studio) has no vendor output cap; allow a larger budget there so a
# full multi-file response survives the truncation retry. DeepSeek's API caps
# output at 8192, so the deepseek route keeps MAX_OUTPUT_TOKENS.
LOCAL_MAX_OUTPUT_TOKENS = 16384


class BudgetExceeded(RuntimeError):
    """Raised when the configured daily DeepSeek token budget is exhausted."""


def _budgeted_cloud(cloud, cfg: dict):
    """Wrap the DeepSeek backend so every generate() enforces the daily token
    budget (review.daily_token_budget) and records actual usage. No-op when no
    budget is configured. Covers the supervise loop, parallel review, and the
    verify-fix stage, which call cloud.generate directly."""
    budget = (cfg.get("review") or {}).get("daily_token_budget") or 0
    if not budget:
        return cloud

    class _BudgetedCloud:
        def generate(self, req):
            used = StatsTracker().daily_api_tokens()
            if used >= budget:
                raise BudgetExceeded(
                    f"daily DeepSeek token budget exhausted ({used}/{budget}). "
                    "Raise review.daily_token_budget or retry tomorrow.")
            resp = cloud.generate(req)
            tok = _normalize_tokens(resp.token_usage)
            StatsTracker().record_tokens(
                "deepseek",
                tok.get("prompt_tokens", 0), tok.get("completion_tokens", 0),
                estimated=bool(tok.get("estimated")))
            return resp

    return _BudgetedCloud()


def _cached_cloud(cloud, cache, kind: str = "review"):
    """Wrap the DeepSeek backend so an identical review request short-circuits
    the API. The key is sha256(system + user), so a hit is only possible when
    the EXACT same review package was seen before — the cached verdict is then
    deterministic. Only non-truncated, non-empty responses are stored. No-op
    when the cache is disabled."""
    if cache is None or not getattr(cache, "enabled", False):
        return cloud

    def generate(self, req):
        key = cache.key(kind, req.system, req.user)
        hit = cache.get(kind, key)
        if hit is not None:
            cache.record(True)
            _status(f"[hybrid] 🗃 cache HIT {kind} {key[:8]}")
            return ModelResponse(text=hit, backend="cache", latency_ms=0.0)
        resp = cloud.generate(req)
        cache.record(False)
        if resp.text and not resp.truncated:
            cache.set(kind, key, resp.text, source="deepseek")
        return resp

    return type("_CachedCloud", (), {"generate": generate})()


def _failover_cloud(primary_cloud, cfg: dict, args: argparse.Namespace):
    """Wrap the primary online backend with sequential provider failover: when
    the active provider raises a BackendError, retry the request on the next
    enabled online provider. No-op with a single online provider."""
    online = enabled_online(cfg)
    if len(online) <= 1:
        return primary_cloud

    def generate(self, req):
        last_exc = None
        for i, prov in enumerate(online):
            if i == 0:
                backend = primary_cloud
            else:
                if not resolve_api_key(prov):
                    continue
                backend = backend_for(prov)
            try:
                return backend.generate(req)
            except BackendError as exc:
                last_exc = exc
                _status(f"[hybrid] ⚠ provider '{prov.name}' failed ({exc}); "
                        f"trying next online provider")
        if last_exc is not None:
            raise last_exc
        raise BackendError("no enabled online provider could serve the request")

    return type("_FailoverCloud", (), {"generate": generate})()


def _parallel_cloud(primary_cloud, cfg: dict, args: argparse.Namespace):
    """Turbo mode: fan the SAME request out to every enabled online provider in
    parallel and return the longest non-truncated response. Uses many AIs at
    the same time; multiplies online spend (still capped by the token budget)."""
    online = enabled_online(cfg)
    if len(online) <= 1:
        return primary_cloud

    def generate(self, req):
        workers = [primary_cloud]
        for i, prov in enumerate(online):
            if i == 0:
                continue
            if not resolve_api_key(prov):
                continue
            workers.append(backend_for(prov))
        results = []
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futs = [executor.submit(w.generate, req) for w in workers]
            for f in futs:
                try:
                    results.append(f.result())
                except Exception:  # noqa: BLE001 - a failed provider is tolerated
                    pass
        good = [r for r in results if r and r.text and not r.truncated]
        if not good:
            good = [r for r in results if r and r.text]
        if not good:
            raise BackendError("turbo: all online providers failed")
        best = max(good, key=lambda r: len(r.text))
        _status(f"[hybrid] ⚡ turbo: {len(good)}/{len(workers)} provider(s) "
                f"returned; using {best.backend or 'provider'}")
        return best

    return type("_ParallelCloud", (), {"generate": generate})()


def _detect_step_conflicts(steps: list[dict]) -> set[str]:
    """Plan-declared files claimed by more than one step in the same batch."""
    claims: dict[str, list] = {}
    for step in steps:
        for f in step.get("files", []) or []:
            claims.setdefault(f, []).append(step.get("id"))
    return {f for f, ids in claims.items() if len(ids) > 1}


def _plan_group_runs(group: list[dict]):
    """Split one parallel batch into (parallel_steps, serialized_steps).

    Steps whose plan-declared Files: overlap another step's in the same batch
    cannot run concurrently (divergent edits to one file would clobber each
    other), so they are serialized in ascending step-id order. Returns the
    non-conflicted steps for the parallel pool and the conflicted steps to run
    one at a time."""
    conflicts = _detect_step_conflicts(group)
    if not conflicts:
        return list(group), []
    conf_steps = [s for s in group if set(s.get("files", []) or []) & conflicts]
    ok_steps = [s for s in group if s not in conf_steps]
    return ok_steps, sorted(conf_steps, key=lambda s: s.get("id", 0))


def _warn_output_overlaps(results: list[dict]) -> None:
    """Warn when two steps in the same batch emitted blocks for the same file
    even though the plan's Files: list did not declare it. The --apply guard
    keeps the first block; the user should know the later one was dropped."""
    seen: dict[str, list] = {}
    for r in results:
        if not r or r.get("status") != "success":
            continue
        for path, _ in _parse_fenced_files(r.get("text") or ""):
            seen.setdefault(path, []).append(r.get("id"))
    for path, ids in seen.items():
        if len(ids) > 1:
            _status(f"[hybrid] ⚠ post-hoc conflict: '{path}' written by steps "
                    f"{ids} — apply keeps the first block")


# --- dynamic supervision routing ------------------------------------------

# Task signals that never need a DeepSeek review (local-first candidates).
_LOCAL_SKIP_MARKERS = (
    "typo", "rename", "format", "prettier", "readme", "docstring", "comment",
    "draft", "spelling", "whitespace", "indent", "sort ", "reorder",
)

# Task signals that always deserve DeepSeek involvement (enhance + review).
_CRITICAL_MARKERS = (
    "architecture", "security", "authentication", " auth", "migration",
    "database", "schema", "refactor", "concurrency", "performance",
    "payment", "encryption", "api design", "data model", "deadlock",
    "authorization", "rate limit",
)


def _plan_supervision(agent: HybridAgent | None, cfg: dict, args: argparse.Namespace,
                      task: str, context: str, memory) -> tuple[str, str]:
    """Decide the supervision plan for a task.

    Levels: 'full' (DeepSeek reviews), 'local_first' (skip the DeepSeek review
    — one local pass + apply/verify only), 'critical' (force prompt enhancement
    and the full review loop). Returns (level, reason).

    Precedence: explicit --router / router.supervision config > critical task
    signals > trivial task signals > budget pressure > router confidence +
    memory history > memory-similar tasks > default full.
    """
    override = args.router or (cfg.get("router") or {}).get("supervision") or "auto"
    if override != "auto":
        return override, f"override:{override}"
    low = (task or "").lower()
    if any(m in low for m in _CRITICAL_MARKERS):
        return "critical", "critical_task_signals"
    if any(m in low for m in _LOCAL_SKIP_MARKERS):
        return "local_first", "trivial_task_signals"
    # Budget pressure: near the daily cap, degrade to local-only.
    budget = (cfg.get("review") or {}).get("daily_token_budget") or 0
    if budget:
        used = StatsTracker().daily_api_tokens()
        if used >= budget * 0.8:
            return "local_first", f"budget_pressure ({used}/{budget})"
    if agent is not None:
        try:
            route, reason = agent.route(
                task, context_chars=len(task) + len(context), memory=memory)
        except Exception:  # noqa: BLE001 - routing must never break the run
            route, reason = "", "router_error"
        if route == "deepseek" and reason.startswith("archetype"):
            return "critical", reason
        if route == "local" and memory.has_history:
            return "local_first", reason
    # Memory alone: strong similar-task history of local success is a signal.
    if memory.has_history and memory.similar_task_success_rate >= 0.7:
        return "local_first", "memory_similar_tasks"
    return "full", "default"


def _generate_with_retry(agent: HybridAgent, cfg: dict, args: argparse.Namespace,
                         route: str, req: ModelRequest,
                         stream: bool = False) -> ModelResponse:
    """Generate, and when the output hits the token cap, retry ONCE with a
    doubled budget so truncated files are never returned silently. Long local
    generations (especially thinking-mode models) routinely need more than a
    single 2k budget; this is the guard that makes truncation rare."""
    if route == "deepseek":
        budget = (cfg.get("review") or {}).get("daily_token_budget") or 0
        if budget:
            used = StatsTracker().daily_api_tokens()
            if used >= budget:
                raise BudgetExceeded(
                    f"daily DeepSeek token budget exhausted ({used}/{budget}). "
                    "Raise review.daily_token_budget or retry tomorrow.")
    cap = LOCAL_MAX_OUTPUT_TOKENS if route == "local" else MAX_OUTPUT_TOKENS
    resp = _generate(agent, cfg, args, route, req, stream=stream)
    if resp.truncated and resp.text and req.max_tokens < cap:
        req.max_tokens = min(req.max_tokens * 2, cap)
        req.timeout_s = int(req.timeout_s * 1.5)
        _status(f"[hybrid] ⚠ TRUNCATED at {req.max_tokens // 2} tokens - "
                f"retrying once with {req.max_tokens}")
        resp = _generate(agent, cfg, args, route, req, stream=stream)
        if resp.truncated:
            _status("[hybrid] ⚠ still truncated after retry - output is incomplete, do not apply blindly")
    if route == "deepseek":
        tok = _normalize_tokens(resp.token_usage)
        StatsTracker().record_tokens(
            "deepseek",
            tok.get("prompt_tokens", 0), tok.get("completion_tokens", 0),
            estimated=bool(tok.get("estimated")))
    return resp


def _format_tokens(resp: ModelResponse) -> dict:
    """Token usage for the status line. LM Studio's streaming path does not
    return usage, so fall back to an estimate so the line is never empty."""
    tokens = _normalize_tokens(resp.token_usage)
    if not tokens and resp.text:
        tokens = {"completion_tokens": max(1, len(resp.text) // 4), "estimated": True}
    return tokens


def _supervise_gemma_generate(cfg: dict, model_override: str | None,
                              local_provider: str | None = None):
    """Return a streaming wrapper for the supervise loop's local step.

    Always streams the local model's output live to stderr (via the HTTP path)
    so the user can SEE the local model working in every iteration, then
    returns the full accumulated text for the review package. Uses the selected
    local provider (defaults to the legacy backends.local). gemma4 thinking
    mode can legitimately take 1-2 minutes, so the provider timeout is applied
    to every request.
    """
    lp = get_local(cfg, name=local_provider)
    if lp is not None:
        base_url = lp.base_url
        api_key = lp.api_key or "lm-studio"
        model = model_override or lp.model
        timeout_s = lp.timeout_s
    else:
        lc = cfg["backends"]["local"]
        base_url = lc["base_url"]
        api_key = lc.get("api_key") or "lm-studio"
        model = model_override or lc["model"]
        timeout_s = lc["timeout_s"]

    def _gen(req: ModelRequest) -> ModelResponse:
        req.timeout_s = timeout_s
        return _http_generate(base_url, api_key, model, req, stream=True)

    return _gen


def _make_terminal_tool(cfg: dict, args: argparse.Namespace):
    """Build the RUN: terminal tool for the supervise loop: executes allowlisted
    commands in the task root and returns captured output (never raises)."""
    timeout = int((cfg.get("review") or {}).get("terminal_timeout", 120))

    def run_terminal(cmd: str) -> str:
        c = (cmd or "").strip().lower()
        if not c or any(m in c for m in _DANGEROUS_VERIFY_MARKERS):
            return "⚠ BLOCKED: dangerous command pattern."
        if not (_is_safe_verify_cmd(cmd) or c.startswith(_TERMINAL_TOOL_PREFIXES)):
            return ("⚠ BLOCKED: not an allowlisted terminal command "
                    "(add it to review.verify_allowlist if you trust it).")
        try:
            proc = subprocess.run(cmd, shell=True, cwd=args.root,
                                  capture_output=True, text=True, timeout=timeout)
            out = (proc.stdout or "") + (proc.stderr or "")
            if not out:
                out = f"(exit {proc.returncode}, no output)"
            if len(out) > 4000:
                out = out[:4000] + "\n... [output truncated]"
            return f"exit {proc.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return f"⏱ timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001 - report cleanly
            return f"error: {exc}"

    return run_terminal


def _memory_report(cfg: dict, force: bool = False, cwd: str = ".") -> dict:
    """View or force-consolidate task memory. Fully offline — no model calls.

    Returns {"path", "count", "records": [...newest first...], "insights"}.
    With force=True the consolidation pass runs now (by default it only runs
    lazily when the cached insights are stale and there are >=10 records)."""
    mem = TaskMemory(memory_root_from_cfg(cfg, cwd=cwd))
    if force:
        mem.consolidate()
    return {
        "path": str(mem.path),
        "count": mem.count(),
        "records": [{
            "task": r.get("task", ""), "verdict": r.get("verdict", ""),
            "route": r.get("route", ""), "quality": r.get("quality", 0.0),
            "ts": r.get("ts", 0.0),
        } for r in mem._load()[-15:][::-1]],
        "insights": mem.insights_text(),
    }


def _list_models(base_url: str) -> int:
    """Print one loaded model id per line from the local endpoint. Exit 0 / 1."""
    base_url = base_url.rstrip("/")
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - fallback below keeps it working
        OpenAI = None  # type: ignore

    if OpenAI is not None:
        client = OpenAI(base_url=base_url, api_key="lm-studio")
        try:
            for model in client.models.list().data:
                print(model.id)
            return 0
        except Exception as exc:  # noqa: BLE001 - surface connection failure
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # No openai package: plain urllib GET with a short timeout.
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            payload = json.loads(resp.read().decode())
        for model in payload.get("data", []):
            print(model.get("id"))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface connection failure
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_request(agent: HybridAgent, args: argparse.Namespace,
                   route: str, context: str) -> ModelRequest:
    # Direct CLI bridge: use a clean, neutral system prompt rather than the
    # routing prompt from _local_request()/_cloud_plan_request(), which injects
    # scaffolding like "<CONFIDENCE>0.0-1.0</CONFIDENCE>" meant for the router.
    system = (
        args.system
        if args.system is not None
        else "You are a precise, helpful assistant. Answer the task directly and concisely."
    )
    user = f"TASK: {args.task}"
    if context:
        user += f"\n\nCONTEXT:\n{context}"
    project_context = get_project_context(args.task)
    if project_context and "No project context" not in project_context:
        user += f"\n\nPROJECT CONTEXT:\n{project_context}"
    timeout_s = (
        agent.cfg["backends"]["local"]["timeout_s"]
        if route == "local"
        else agent.cfg["backends"]["deepseek"]["timeout_s"]
    )
    return ModelRequest(
        system=system,
        user=user,
        max_tokens=args.max_tokens or 4096,
        temperature=args.temperature or 0.2,
        timeout_s=timeout_s,
    )


MAX_CLARIFY_ROUNDS = 3


def _enhance_task(agent: HybridAgent, cfg: dict, args: argparse.Namespace,
                  task: str, context: str, cache=None):
    """Run DeepSeek prompt-enhancement with an interactive clarification loop.

    DeepSeek ENHANCES the prompt and plans around the local model's limits. If
    it finds the task ambiguous it emits clarifying questions: interactively we
    ask the user to adjust the prompt (re-enhancing up to MAX_CLARIFY_ROUNDS),
    otherwise we stop and report that clarification is needed.

    Returns (task_for_gemma, enhancement, clar_needed). Raises on generation
    failure so the caller handles it.
    """
    current = task
    enhancement = None
    for round_num in range(MAX_CLARIFY_ROUNDS):
        enh_label = "deepseek (prompt enhancer)"
        _status(f"[hybrid] ▶ {enh_label} working on \"{_task_preview(current)}\" ...")
        req = _enhance_request(current, context, cot=args.cot)
        if cache is not None and cache.enabled:
            k = cache.key("enhance", req.system, req.user)
            hit = cache.get("enhance", k)
            if hit is not None:
                enh_resp = ModelResponse(text=hit, backend="cache", latency_ms=0.0)
                cache.record(True)
                _status(f"[hybrid] 🗃 cache HIT enhance {k[:8]}")
            else:
                enh_resp = _generate_with_retry(agent, cfg, args, "deepseek", req)
                cache.record(False)
                if not enh_resp.truncated and enh_resp.text:
                    cache.set("enhance", k, enh_resp.text, source="deepseek")
        else:
            enh_resp = _generate_with_retry(agent, cfg, args, "deepseek", req)
        enhancement = parse_enhancement(enh_resp.text)
        _status(f"[hybrid] ✓ {enh_label} done · {int(enh_resp.latency_ms)} ms · "
                f"tokens={_format_tokens(enh_resp)}")

        if not args.json:
            print("=== ENHANCED PROMPT ===\n" + enhancement.enhanced_prompt)
            if enhancement.reasoning:
                print("\n=== REASONING ===\n" + enhancement.reasoning)
            if enhancement.plan:
                print("\n=== PLAN ===\n" + enhancement.plan)
            if enhancement.clarifying_questions:
                print("\n=== CLARIFYING QUESTIONS ===\n" + enhancement.clarifying_questions)
            if args.cot:
                cot = ChainOfThoughtParser(enh_resp.text)
                for icon, title, body in cot.get_reasoning_chain():
                    print(f"\n{icon} {title}\n" + body)

        if not enhancement.clarifying_questions:
            break  # prompt is clear
        if not sys.stdin.isatty():
            # Non-interactive: surface the questions and let the caller stop so
            # the user can adjust the prompt and re-run. With --proceed the run
            # continues using the (usually self-contained) enhanced prompt.
            if not args.proceed:
                return (current, enhancement, True)
            _status("[hybrid] ↻ non-interactive + --proceed: continuing with "
                    "the enhanced prompt despite clarifying questions")
            break
        print("\nThe task has ambiguities. Make the prompt clear and concise "
              "by answering DeepSeek's questions.")
        try:
            answer = input("Your clarification (press Enter to proceed as-is): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not answer:
            break
        current = current + "\n\nUSER CLARIFICATION:\n" + answer
        _status(f"[hybrid] ↻ clarification #{round_num + 1} received — re-enhancing")

    task_for_gemma = (
        enhancement.enhanced_prompt
        if enhancement and enhancement.enhanced
        else current
    )
    if not args.json:
        print("\n--- sending enhanced prompt to local model ---\n")
    return (task_for_gemma, enhancement, False)


def _scan_project_context(root: str, task: str = "") -> str:
    """Scan the project and return a rendered context prompt (or '' on failure).

    Emits status lines showing what was detected (architecture, file count, and
    a token estimate) so the user sees the scan happening.
    """
    try:
        _status("[hybrid] 🧠 Scanning project context...")
        ctx = ProjectContext(root or ".")
        ctx.scan()
        if task:
            ctx.find_relevant_files(task)
        prompt = ctx.to_prompt()
        arch = ctx.context.get("architecture", {})
        files = ctx.context.get("structure", {}).get("files", [])
        _status("[hybrid] 📂 Detected: type={} backend={} frontend={} "
                "orm={} db={}".format(
                    arch.get("type", "unknown"),
                    arch.get("backend_framework", "-"),
                    arch.get("frontend_framework", "-"),
                    arch.get("orm", "-"),
                    arch.get("database", "-"),
                ))
        _status(f"[hybrid] 📝 {len(files)} source files, "
                f"deps={sum(len(v) for v in ctx.context.get('dependencies', {}).values())}")
        _status(f"[hybrid] ✅ Context collected ({ctx.context.get('estimate_tokens', 0)} tokens)")
        return prompt
    except Exception as exc:  # noqa: BLE001 - never block the run on a scan failure
        _status(f"[hybrid] ⚠ context scan failed: {exc}")
        return ""


def _run_parallel(agent: HybridAgent, cfg: dict, args: argparse.Namespace,
                  cloud, task: str, enhancement, context: str,
                  review: bool = True) -> SuperviseResult:
    """Parallel --supervise path: split the plan into steps, run independent
    steps in parallel on the local model, then DeepSeek reviews the combined
    output. Returns a SuperviseResult so the existing apply/stats/print tail
    in the --supervise branch works unchanged.
    """
    steps = parse_plan_steps(enhancement.plan if enhancement else "")
    analyzer = DependencyAnalyzer(steps)
    groups = analyzer.get_parallel_groups()
    executor = ParallelExecutor(
        _supervise_gemma_generate(cfg, args.model, local_provider=args.local_provider),
        max_workers=args.parallel_workers,
        max_tokens=args.max_tokens or GEMMA_MAX_TOKENS,
    )
    texts: list[str] = []
    for group in groups:
        if len(group) > 1:
            _status(f"[hybrid] ⚡ Running {len(group)} steps in parallel...")
        ok_steps, conf_steps = _plan_group_runs(group)
        if conf_steps:
            _status("[hybrid] ⚠ file conflict in this batch — serializing "
                    f"steps {[s.get('id') for s in conf_steps]} (their Files: overlap)")
        results: list[dict] = []
        if ok_steps:
            results.extend(executor.execute_parallel(ok_steps, task, context))
        for step in conf_steps:
            _status(f"[hybrid] 🔒 step {step.get('id')} serialized (file conflict)")
            results.extend(executor.execute_parallel([step], task, context))
        if conf_steps:
            by_id = {r.get("id"): r for r in results if r.get("id") is not None}
            results = [by_id.get(s.get("id")) for s in group]
        _status(f"[hybrid] 📝 {summarize(results)}")
        for r in results:
            if r["status"] == "success":
                texts.append(r["text"])
            else:
                _status(f"[hybrid] ⚠ step {r['id']} failed: {r['error']}")
        _warn_output_overlaps(results)
    final_text = "\n\n".join(t for t in texts if t)

    if not review:
        _status("[hybrid] 🔒 local-first plan — skipping DeepSeek review of parallel output")
        return SuperviseResult(
            task=task, final_text=final_text, route="local",
            reason="router_local_skip_review",
            verdicts=[Verdict(decision="APPROVED", quality_score=7.0,
                              assessment="Local-first router plan; no DeepSeek review.")],
            iterations=max(1, len(groups)))

    _status("[hybrid] ▶ deepseek reviewing parallel output ...")
    try:
        pkg = ReviewPackage(
            task=task,
            plan=enhancement.plan if enhancement else "",
            changes=f"Parallel output:\n{final_text}",
            uncertainties=context,
        )
        review_resp = cloud.generate(_supervisor_request(pkg))
    except Exception as exc:  # noqa: BLE001 - report cleanly
        _status(f"[hybrid] ⚠ review failed ({exc}); applying without review")
        return SuperviseResult(task=task, final_text=final_text, route="local",
                               reason="review_failed_no_verdict",
                               iterations=max(1, len(groups)))

    verdict = parse_verdict(review_resp.text)
    _status(f"[hybrid] {verdict.decision} (score {verdict.quality_score:.1f}/10)")
    return SuperviseResult(task=task, final_text=final_text, route="local",
                           reason="parallel", verdicts=[verdict],
                           iterations=max(1, len(groups)))


def _looks_like_error(output: str) -> bool:
    """Heuristic: does the verification output look like it contains errors?"""
    if not output:
        return False
    lowered = output.lower()
    markers = ("error", " failed", "cannot find", "is not assignable",
               "typescript error", "exception", "undefined", "traceback",
               "cannot resolve", "no module", "syntaxerror", "exit code")
    return any(m in lowered for m in markers)


# --- verification safety/cost/performance helpers -----------------------

# Cap on the error text sent to DeepSeek so huge build logs don't waste tokens.
_VERIFY_ERROR_CHARS = 4000

# Allowlisted verification command prefixes (run via shell). Anything else is
# blocked so a mis-set verify: [ ... ] can't run destructive commands.
_SAFE_VERIFY_PREFIXES = (
    "npm run build", "npm run test", "npm run lint", "npm run typecheck",
    "npm run", "npm test",
    "npx tsc", "npx eslint", "npx prettier --check",
    "yarn build", "yarn test", "yarn lint",
    "pnpm run build", "pnpm run test", "pnpm run lint",
    "python -m pytest", "pytest", "python -m unittest", "unittest",
    "make test", "make lint", "make build",
    "go test", "go vet", "go build",
    "cargo test", "cargo build", "cargo clippy",
    "echo", "test -f",
)

# Shell metacharacters / commands that must never appear in a verify command.
_DANGEROUS_VERIFY_MARKERS = (
    "rm ", "rm -rf", "sudo", ">", ">>", "|", "`", "$(", "&&", "||", ";",
    "chmod", "chown", "mkfs", "dd ", "shutdown", "reboot",
)

# Read-only inspection commands the RUN: terminal tool may execute (in addition
# to the build/test verify allowlist). Still marker-gated above.
_TERMINAL_TOOL_PREFIXES = (
    "ls", "cat ", "head ", "tail ", "wc", "find ", "grep ", "rg ",
    "git status", "git log", "git diff", "git show", "git branch", "pwd", "tree",
    "node --version", "npm --version", "python --version", "python3 --version",
    "node -v", "npm -v", "echo ", "test -f", "test -d", "file ", "stat ",
    "which ", "ls -la", "git stash list",
)

# Extra allowlisted prefixes from config review.verify_allowlist. Populated at
# startup by _configure_verify_allowlist (empty = stock allowlist only).
_VERIFY_ALLOWLIST_EXTRA: tuple[str, ...] = ()


def _configure_verify_allowlist(cfg: dict) -> None:
    """Extend the verify allowlist with review.verify_allowlist prefixes from
    config (e.g. ['docker compose build']). Prefixes are lower-cased like the
    hardcoded ones; the dangerous-marker check still applies to every command."""
    global _VERIFY_ALLOWLIST_EXTRA
    extra = (cfg.get("review") or {}).get("verify_allowlist") or []
    _VERIFY_ALLOWLIST_EXTRA = tuple(
        str(p).strip().lower() for p in extra if str(p).strip())


def _is_safe_verify_cmd(cmd: str) -> bool:
    """Only allowlist verification commands; block destructive/compound shell."""
    c = (cmd or "").strip().lower()
    if not c:
        return False
    if any(m in c for m in _DANGEROUS_VERIFY_MARKERS):
        return False
    return c.startswith(_SAFE_VERIFY_PREFIXES) or c.startswith(_VERIFY_ALLOWLIST_EXTRA)


def _truncate_error(text: str, limit: int = _VERIFY_ERROR_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated; first {limit} chars shown]"


def _git_snapshot(root: str) -> str | None:
    """Commit the working tree as a snapshot so fixes can be rolled back."""
    try:
        check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                               cwd=root, capture_output=True, text=True)
        if check.returncode != 0:
            return None
        subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True)
        subprocess.run(["git", "commit", "-m", "hybrid-verify: snapshot before fixes",
                        "--allow-empty"], cwd=root, capture_output=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True).stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


def _git_restore(root: str, snapshot_sha: str, files: list[str]) -> None:
    """Restore only the files DeepSeek fixed to their pre-fix snapshot state.

    checkout restores tracked files to the snapshot; clean removes any fix
    files that were untracked (newly created by the fix) and not in the
    snapshot. Scoped strictly to the fix file paths.
    """
    if not files:
        return
    try:
        subprocess.run(["git", "checkout", snapshot_sha, "--"] + files,
                       cwd=root, capture_output=True)
        subprocess.run(["git", "clean", "-fd", "--"] + files,
                       cwd=root, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def _write_diff_log(root: str, files: list[str]) -> str:
    """Write a unified diff of the fixed files; return log path ('' on failure)."""
    if not files:
        return ""
    try:
        proc = subprocess.run(["git", "diff", "--"] + files, cwd=root,
                              capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        return ""
    diff = proc.stdout or ""
    if not diff:
        return ""
    try:
        log_dir = pathlib.Path(root) / "hybrid-verify"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "fixes.diff"
        log_path.write_text(f"# hybrid-agent verification fixes\n\n{diff}",
                            encoding="utf-8")
        return str(log_path)
    except OSError:
        return ""


def _current_diff(root: str, max_chars: int = 4000) -> str:
    """Render a capped unified diff of the working tree vs HEAD for the review
    package's DIFF section (via review/diff_builder.py). '' when not a git repo
    or nothing changed. Best-effort: never raises."""
    try:
        proc = subprocess.run(["git", "diff"], cwd=root, capture_output=True,
                              text=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout.strip():
            return ""
        from review import diff_builder
        diffs = diff_builder.parse_unified_diff(proc.stdout)
        if not diffs:
            return ""
        rendered = "\n\n".join(d.render() for d in diffs)
        return rendered if len(rendered) <= max_chars else rendered[:max_chars]
    except Exception:  # noqa: BLE001 - diff is a best-effort enrichment
        return ""


# Error patterns that DeepSeek cannot fix (setup/environment) — skip the call.
_ENV_ERROR_MARKERS = (
    "command not found", "enoent", "cannot find module", "module not found",
    "no module named", "no such file or directory", "not found",
    "npm error code enoent", "cannot resolve", "could not find", "is not installed",
)


def _is_environmental_error(text: str) -> bool:
    """True when the error is environmental and not fixable by DeepSeek."""
    lowered = (text or "").lower()
    return any(m in lowered for m in _ENV_ERROR_MARKERS)


def _run_regression_guard(root: str, cmds, status,
                          timeout_s: int = 600) -> tuple[bool, str]:
    """Run the full test suite after verification passes, to catch fixes that
    break tests elsewhere. Regression commands are allowlisted too.
    Returns (passed, report). Does not apply fixes.
    """
    if not cmds:
        return True, "no regression commands configured"
    unsafe = [c for c in cmds if not _is_safe_verify_cmd(c)]
    if unsafe:
        blocked = "; ".join(unsafe[:3])
        status(f"[hybrid] ⛔ BLOCKED unsafe regression command(s): {blocked}")
        return False, f"blocked unsafe regression commands: {blocked}"
    failed: list[str] = []
    for cmd in cmds:
        status(f"[hybrid] 🧪 regression: {cmd}")
        try:
            proc = subprocess.run(cmd, shell=True, cwd=root,
                                  capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            failed.append(f"$ {cmd}\n[timed out after {timeout_s}s]")
            continue
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            failed.append(f"$ {cmd} (exit {proc.returncode})\n{output}")
    if failed:
        status("[hybrid] ⛔ REGRESSION FAILED: the fixes broke tests")
        return False, "regression failed:\n\n" + "\n\n".join(failed)
    status("[hybrid] ✅ regression passed")
    return True, "regression passed"


def _deepseek_fix(cloud, task: str, error_text: str, cache=None):
    """Ask DeepSeek to fix the errors and return the fixed files."""
    system = (
        "You are a senior engineer fixing errors found during verification. "
        "Analyze the errors and the task, then output the COMPLETE corrected "
        "file(s) in fenced code blocks labeled with their paths. Only change "
        "what is needed to fix the errors. No preamble."
    )
    user = (
        f"TASK:\n{task}\n\n"
        f"VERIFICATION ERRORS:\n{_truncate_error(error_text)}\n\n"
        "Fix every error. Output corrected files as path-labeled fenced blocks."
    )
    if cache is not None and cache.enabled:
        key = cache.key("fix", task, _truncate_error(error_text))
        hit = cache.get("fix", key)
        if hit:
            cache.record(True)
            _status(f"[hybrid] 🗃 cache HIT fix {key[:8]}")
            return hit
    resp = cloud.generate(ModelRequest(
        system=system, user=user, max_tokens=8192, temperature=0.1,
    ))
    if cache is not None and cache.enabled:
        cache.record(False)
        if resp.text and not resp.truncated:
            cache.set("fix", key, resp.text, source="deepseek")
    return resp.text


def _run_single_verify(root, cmd: str, timeout_s: int) -> str:
    """Run one verification command; return an error entry or "" when clean."""
    try:
        proc = subprocess.run(cmd, shell=True, cwd=root,
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"$ {cmd}\n[timed out after {timeout_s}s]"
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0 or _looks_like_error(output):
        return f"$ {cmd} (exit {proc.returncode})\n{output}"
    return ""


def _run_verify_commands(root, cmds, status, timeout_s,
                         parallel=False, workers=4, verify_groups=None) -> list[str]:
    """Run verification commands sequentially, or in dependency groups.

    With parallel + verify_groups, commands within a group run concurrently
    (ThreadPoolExecutor, per-command timeout) while groups run in order, so a
    build always completes before the tests that depend on it. Error entries
    are merged in group/command order for deterministic DeepSeek reports.
    """
    if not parallel or not verify_groups:
        errors: list[str] = []
        for cmd in cmds:
            status(f"[hybrid] 🔍 verify: {cmd}")
            entry = _run_single_verify(root, cmd, timeout_s)
            if entry:
                errors.append(entry)
        return errors

    configured = [g for g in verify_groups if g]
    flat = [c for g in configured for c in g]
    leftover = [c for c in cmds if c not in flat]
    groups = configured + ([leftover] if leftover else [])
    if not groups:
        return _run_verify_commands(root, cmds, status, timeout_s)

    errors: list[str] = []
    for i, group in enumerate(groups):
        status(f"[hybrid] 🔍 verify group {i + 1}/{len(groups)} "
               f"(parallel, {len(group)} cmds)")
        results: dict = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(group))) as executor:
            future_to_idx = {}
            for idx, cmd in enumerate(group):
                status(f"[hybrid] 🔍 verify: {cmd}")
                future_to_idx[executor.submit(_run_single_verify, root, cmd, timeout_s)] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001 - report worker failure cleanly
                    results[idx] = f"$ {group[idx]}\n[worker error: {exc}]"
        # Deterministic merge: skip clean (empty) entries, keep group/command order.
        for idx in range(len(group)):
            if results.get(idx):
                errors.append(results[idx])
    return errors


def _run_final_verify(cloud, root: str, task: str, cmds, status,
                      max_iter: int = 2, timeout_s: int = 600,
                      verify_stats: dict | None = None,
                      regression_cmds=None,
                      regression_timeout: int = 600,
                      cache=None,
                      parallel=False, parallel_workers=4,
                      verify_groups=None) -> tuple[bool, str]:
    """Final error-check stage: run verification commands and, on errors, have
    DeepSeek fix them (applied to disk) until clean or max_iter exhausted.

    Safety: commands are allowlisted; the working tree is snapshotted before
    fixes so a failed run rolls back the AI's changes; every fix is diff-logged.
    Cost: error text is truncated; environmental errors skip the DeepSeek call.
    Regression: after verification passes, the regression test suite runs.
    Returns (verified, report). verified True when all commands pass.
    """
    if verify_stats is None:
        verify_stats = {}
    verify_stats.update({"iterations": 0, "api_calls": 0, "tokens_used": 0,
                         "status": "FAILED",
                         "parallel": bool(parallel and verify_groups),
                         "groups": len([g for g in (verify_groups or []) if g]) if parallel else 0})
    api_calls = 0
    tokens_used = 0

    if not cmds:
        verify_stats["status"] = "PASSED"
        return True, "no verification commands configured"

    unsafe = [c for c in cmds if not _is_safe_verify_cmd(c)]
    if unsafe:
        blocked = "; ".join(unsafe[:3])
        status(f"[hybrid] ⛔ BLOCKED unsafe verify command(s): {blocked}")
        verify_stats["status"] = "BLOCKED"
        return False, f"blocked unsafe verification commands: {blocked}"
    safe_cmds = [c for c in cmds if _is_safe_verify_cmd(c)]

    fixed_files: list[str] = []
    snapshot: str | None = None
    for iteration in range(max_iter + 1):
        verify_stats["iterations"] = iteration + 1
        errors = _run_verify_commands(
            root, safe_cmds, status, timeout_s,
            parallel=parallel, workers=parallel_workers, verify_groups=verify_groups)

        if not errors:
            status("[hybrid] ✅ verification passed")
            if fixed_files:
                log = _write_diff_log(root, fixed_files)
                if log:
                    status(f"[hybrid] 📄 fix diff: {log}")
            # Regression guard: run the full test suite to catch broken tests.
            reg_pass, reg_report = _run_regression_guard(
                root, list(regression_cmds or []), status, timeout_s=regression_timeout)
            if reg_pass:
                verify_stats["status"] = "PASSED"
                verify_stats["files"] = list(fixed_files)
                return True, "verification passed"
            verify_stats["status"] = "REGRESSION_FAILED"
            return False, reg_report

        error_text = "\n\n".join(errors)
        # Skip DeepSeek on environmental errors it cannot fix.
        if _is_environmental_error(error_text):
            status("[hybrid] ⛔ environmental error (not fixable by DeepSeek) — skipped")
            verify_stats["status"] = "ENV_ERROR"
            return False, "environmental error:\n\n" + _truncate_error(error_text)

        status(f"[hybrid] ⚠ {len(errors)} error(s) — DeepSeek fixing "
               f"(attempt {iteration + 1}/{max_iter + 1})")
        if snapshot is None:
            snapshot = _git_snapshot(root)
        try:
            fix_text = _deepseek_fix(cloud, task, error_text, cache=cache)
        except Exception as exc:  # noqa: BLE001
            status(f"[hybrid] ⚠ DeepSeek fix failed: {exc}")
            if snapshot:
                _git_restore(root, snapshot, fixed_files)
            return False, error_text
        api_calls += 1
        tokens_used += len(fix_text or "") // 4

        applied, skipped = _apply_fenced_files(fix_text, root)
        for rel, nbytes in applied:
            status(f"[hybrid] ✓ FIXED {rel} ({nbytes} B)")
            fixed_files.append(rel)
        if skipped:
            status(f"[hybrid] ⚠ skipped {len(skipped)} fix block(s): "
                   f"{', '.join(skipped[:3])}")
        if not applied:
            status("[hybrid] ⚠ DeepSeek returned no fixable files")
            if snapshot:
                _git_restore(root, snapshot, fixed_files)
            return False, error_text
        if iteration >= max_iter:
            break

    # Verification still failing after retries: roll back the AI's fixes.
    if snapshot:
        _git_restore(root, snapshot, fixed_files)
        status(f"[hybrid] ↩ rolled back {len(fixed_files)} fix file(s) to snapshot")
    log = _write_diff_log(root, fixed_files)
    if log:
        status(f"[hybrid] 📄 fix diff: {log}")
    verify_stats.update({"api_calls": api_calls, "tokens_used": tokens_used,
                         "estimated_cost_usd": round(tokens_used * 0.000002, 5),
                         "status": "FAILED", "files": list(fixed_files)})
    status("[hybrid] ⛔ verification still failing after max attempts")
    return False, error_text


def _select_backend(agent: HybridAgent, cfg: dict, route: str,
                    model_override: str | None):
    """Return the backend for `route`, applying the local model override and
    the configured local provider.

    For the local route we construct only the local backend directly so that a
    purely-local run never needs an online API key. For deepseek we build via
    agent._backends() (whose key check we have already validated)."""
    if route != "local":
        _, cloud = agent._backends()
        return cloud
    lp = get_local(cfg, name=agent.local_provider)
    if lp is not None:
        return backend_for(lp)
    lc = cfg["backends"]["local"]
    model = model_override or lc["model"]
    return GemmaBackend(
        lc["base_url"], model,
        timeout_s=lc["timeout_s"], max_retries=lc["max_retries"],
    )


# --- --apply support: write path-labeled fenced code blocks to disk ----------

# Extensions accepted as proof a fenced-block label is a file path (not a language).
_APPLY_EXTENSIONS = {
    "py", "js", "jsx", "ts", "tsx", "html", "css", "scss", "json", "jsonc",
    "md", "yml", "yaml", "toml", "ini", "sh", "bash", "zsh", "sql", "txt",
    "svg", "xml", "env", "prisma", "go", "rs", "java", "rb", "php", "vue",
}


def _parse_fenced_files(text: str) -> list[tuple[str, str]]:
    """Parse path-labeled fenced code blocks from model output.

    Accepts ```path, ```path/to/f.ext, ```lang path, and ```lang:path forms,
    plus a clean path on the line just before the fence, or as the first line
    inside the block. Returns [(relative_path, content)] in document order.
    Bare language tags and comment/HTML lines are never treated as paths.
    """
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
    files = []
    for m in pattern.finditer(text):
        label, body = m.group(1).strip(), m.group(2)
        path = None
        if label:
            for token in label.replace(":", " ").split():
                t = token.strip("`").strip()
                if t and not t.startswith("_") and _looks_like_path(t):
                    path = t
                    break
        # fallback: clean path on the line immediately before the fence
        if path is None:
            prefix = text[:m.start()].rstrip("\n")
            if prefix and not prefix.rstrip().endswith("```"):
                prev = prefix.rsplit("\n", 1)[-1].strip()
                if prev and not prev.startswith("```") and _clean_path(prev):
                    path = prev
        # fallback: clean path as the first line inside the block
        if path is None:
            lines = body.lstrip("\n").split("\n", 1)
            if lines and _clean_path(lines[0].strip()):
                path = lines[0].strip()
                body = lines[1] if len(lines) > 1 else ""
        if path:
            files.append((path, body.rstrip("\n") + "\n"))
    return files


_APPLY_LANG = set(_APPLY_EXTENSIONS)
_CLEAN_PATH = re.compile(r"[\w./\\:-]+\Z")
_WIN_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_like_path(t: str) -> bool:
    n = t.replace("\\", "/")
    if "/" in n or "." in n:
        return n.lower() not in _APPLY_LANG and bool(_CLEAN_PATH.match(n))
    return False


def _clean_path(t: str) -> bool:
    n = t.replace("\\", "/")
    return bool(_CLEAN_PATH.match(n)) and ("/" in n or "." in n)


def _apply_fenced_files(text: str, root: str = ".",
                        dry_run: bool = False) -> tuple[list[tuple[str, int]], list[str]]:
    """Write path-labeled fenced blocks from `text` under `root`.

    Returns (written, skipped): `written` is [(relative_path, bytes_written)],
    `skipped` is a list of human-readable reasons. Absolute paths, Windows
    drive/UNC paths, and paths escaping `root` via ".." are never written.
    With dry_run=True nothing touches disk; `written` still lists what would
    be written."""
    root_abs = os.path.abspath(root)
    written, skipped, seen = [], [], set()
    for rel_path, content in _parse_fenced_files(text):
        clean = rel_path.replace("\\", "/")
        # Reject absolute, Windows drive/UNC, and any ".." traversal paths
        # BEFORE normalization.
        if clean.startswith("/") or os.path.isabs(clean) \
                or any(part == ".." for part in clean.split("/")) \
                or _WIN_DRIVE.match(clean) or clean.startswith("\\\\"):
            skipped.append(f"{rel_path} (unsafe path)")
            continue
        target = os.path.normpath(os.path.join(root_abs, clean))
        if target != root_abs and not target.startswith(root_abs + os.sep):
            skipped.append(f"{rel_path} (escapes root)")
            continue
        if clean in seen:
            skipped.append(f"{rel_path} (duplicate block would overwrite)")
            continue
        seen.add(clean)
        if dry_run:
            written.append((os.path.relpath(target, root_abs), len(content.encode("utf-8"))))
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append((os.path.relpath(target, root_abs), len(content.encode("utf-8"))))
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid bridge CLI (local LM Studio + DeepSeek cloud)"
    )
    parser.add_argument("--models", action="store_true",
                        help="list loaded LM Studio model ids")
    parser.add_argument("--route-only", action="store_true",
                        help="print the routing decision without calling any model")
    parser.add_argument("--task", help="coding task description")
    parser.add_argument("--context", default="", help="optional context text")
    parser.add_argument("--review", action="store_true",
                        help="DeepSeek supervisor review of local model output "
                             "(uses review/supervisor.md as the system prompt)")
    parser.add_argument("--code", help="code/output to review (required with --review)")
    parser.add_argument("--system", help="optional system prompt override")
    parser.add_argument("--local", action="store_true", help="force the local backend")
    parser.add_argument("--deepseek", action="store_true", help="force the deepseek backend")
    parser.add_argument("--auto", action="store_true",
                        help="let the router decide (default)")
    parser.add_argument("--model", help="override the local LM Studio model id")
    parser.add_argument("--max-tokens", type=int, help="max output tokens (default 4096; auto-retries once with double budget on truncation)")
    parser.add_argument("--temperature", type=float, help="sampling temperature (default 0.2)")
    parser.add_argument("--config", help="path to an optional YAML config override")
    parser.add_argument("--json", action="store_true",
                        help="emit a single JSON object on stdout")
    parser.add_argument("--stream", action="store_true",
                        help="stream the local model's output live to stderr")
    parser.add_argument("--supervise", action="store_true",
                        help="Gemma-primary / DeepSeek-supervisor loop: Gemma implements, "
                             "DeepSeek reviews a compact package, loop until APPROVED "
                             "(default max 3 iterations)")
    parser.add_argument("--enhance", action="store_true",
                        help="DeepSeek enhances the prompt and plans around the LOCAL "
                             "model's context/output limits BEFORE it implements. The "
                             "improved prompt + reasoning + plan are shown, then sent "
                             "to the local model. Pairs with --supervise (full loop) "
                             "or the local route (single shot). Requires hybrid mode.")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="max supervise loop iterations (default 3)")
    parser.add_argument("--apply", action="store_true",
                        help="with --supervise: write the approved output's path-labeled "
                             "fenced code blocks to disk under --root")
    parser.add_argument("--apply-dry-run", action="store_true",
                        help="with --apply: print what would be written without "
                             "touching the disk")
    parser.add_argument("--root", default=".",
                        help="directory to apply files under (default: current directory)")
    parser.add_argument("--mode", choices=["hybrid", "local", "code"],
                        help="agent mode: hybrid (local impl + API supervision), "
                             "local (local impl, no API), code (API impl, no API supervision). "
                             "Default from $MODE, else hybrid")
    parser.add_argument("--router", choices=["auto", "full", "local_first", "critical"],
                        default="auto",
                        help="supervision plan override: auto (router decides per task), "
                             "full (always DeepSeek review), local_first (skip DeepSeek "
                             "review, local implement + verify only), critical (force "
                             "prompt enhancement + full review loop). Default from "
                             "router.supervision in config, else auto")
    parser.add_argument("--online-provider", default=None,
                        help="name of the online provider to use (see providers: in "
                             "config.yml; default: first enabled online provider)")
    parser.add_argument("--local-provider", default=None,
                        help="name of the local provider to use (see providers: in "
                             "config.yml; default: first enabled local provider)")
    parser.add_argument("--turbo", action="store_true",
                        help="multi-model mode: fan every online call out to all enabled "
                             "online providers in parallel and use the best response "
                             "(multiplies API spend; still capped by the token budget)")
    parser.add_argument("--proceed", action="store_true",
                        help="with --enhance in a non-interactive run: continue with the "
                             "enhanced prompt even when DeepSeek raised clarifying "
                             "questions (otherwise the run aborts with TASK UNCLEAR)")
    parser.add_argument("--pull", action="store_true",
                        help="git pull --ff-only before the task (best-effort; dirty "
                             "tree or missing upstream is reported, never fatal)")
    parser.add_argument("--push", action="store_true",
                        help="after the task is verified: git add + commit + push the "
                             "engine's changes (applied files + verification fixes) with "
                             "an identifiable 'hybrid-agent: ...' message. Never force-pushes")
    parser.add_argument("--deploy", action="store_true",
                        help="after verification passes, run the deploy command from "
                             "deploy.command in config.yml (or --deploy-cmd)")
    parser.add_argument("--deploy-cmd", default=None,
                        help="deploy command override (implies --deploy)")
    parser.add_argument("--terminal-rounds", type=int, default=3,
                        help="max terminal rounds for the RUN: tool in the supervise loop "
                             "(0 disables the terminal tool)")
    parser.add_argument("--memory", action="store_true",
                        help="print the task-memory records and consolidated insights "
                             "(offline, no model calls)")
    parser.add_argument("--consolidate", action="store_true",
                        help="force a memory consolidation pass and print the insights "
                             "(offline; normally runs automatically when stale)")
    parser.add_argument("--stats", action="store_true",
                        help="print the 80/20 strategy summary (hybrid-agent/stats.json)")
    parser.add_argument("--evaluate", action="store_true",
                        help="print the self-evaluation report: files, iterations, "
                             "quality, truncation rate, and comparison vs last week "
                             "(hybrid-agent/stats.json)")
    parser.add_argument("--context-scan", action="store_true",
                        help="scan the project (structure, dependencies, architecture, "
                             "coding standards, code examples) and inject the rendered "
                             "context into the enhancement/supervise flow so DeepSeek "
                             "and Gemma understand the codebase")
    parser.add_argument("--cot", action="store_true",
                        help="chain-of-thought planning: DeepSeek shows TASK "
                             "UNDERSTANDING, CONSTRAINT ANALYSIS, and ALTERNATIVES "
                             "before the plan, so you can verify its reasoning")
    parser.add_argument("--parallel", action="store_true",
                        help="with --supervise --enhance: split the plan into steps, "
                             "run independent steps in parallel via the local model, "
                             "then DeepSeek reviews")
    parser.add_argument("--parallel-workers", type=int, default=4,
                        help="max parallel workers for --parallel (default 4)")
    parser.add_argument("--verify", action="store_true",
                        help="final error-check stage: run the verification commands "
                             "from review.verify (config.yml) after the task is applied; "
                             "on errors DeepSeek analyzes and fixes them before finishing")
    parser.add_argument("--verify-cmd", action="append", default=[],
                        help="add a verification command to run before finishing "
                             "(repeatable, e.g. --verify-cmd 'npm run build'); "
                             "implies --verify")
    parser.add_argument("--verify-max", type=int, default=2,
                        help="max verify-fix iterations before giving up (default 2)")
    parser.add_argument("--verify-parallel", action="store_true",
                        help="run verification commands in parallel within config "
                             "review.verify_groups groups (groups run in order)")
    parser.add_argument("--verify-workers", type=int, default=4,
                        help="max parallel verify workers per group (default 4)")
    parser.add_argument("--verify-timeout", type=int, default=0,
                        help="per-command timeout in seconds for verification "
                             "(default from review.verify_timeout, else 600)")
    parser.add_argument("--regression", action="store_true",
                        help="run the full test suite (review.regression) AFTER "
                             "verification passes, to catch fixes that break tests")
    parser.add_argument("--regression-cmd", action="append", default=[],
                        help="add a regression test command (repeatable); implies --regression")
    parser.add_argument("--regression-timeout", type=int, default=0,
                        help="per-command timeout for regression tests "
                             "(default from review.regression_timeout, else 600)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the response cache entirely")
    parser.add_argument("--cache-max-size", type=int, default=0,
                        help="max entries per cache kind "
                             "(default from config cache.max_entries)")
    parser.add_argument("--cache-ttl", type=int, default=0,
                        help="cache TTL in days (default from config cache.ttl_days)")
    args = parser.parse_args()

    if args.evaluate:
        r = StatsTracker().evaluate()
        print("=== SELF-EVALUATION ===")
        print(f"I generated {r['files_generated']} files in {r['iterations']} iterations")
        print(f"Average quality score: {r['average_quality']}/10")
        print(f"Truncation rate: {r['truncation_rate']}% ({r['truncation_events']}/{r['tasks_completed']} tasks)")
        print(r['comparison'])
        return 0

    if args.stats:
        summary = StatsTracker().get_summary()
        print(json.dumps(summary, indent=2))
        return 0

    if args.models:
        cfg = _load_cfg(args)
        return _list_models(cfg["backends"]["local"]["base_url"])

    if args.memory or args.consolidate:
        cfg = _load_cfg(args)
        report = _memory_report(cfg, force=args.consolidate, cwd=os.getcwd())
        if args.json:
            print(json.dumps(report, ensure_ascii=False))
        else:
            print(f"task memory: {report['count']} record(s) · {report['path']}")
            if report["insights"]:
                print("\n" + report["insights"])
            for r in report["records"]:
                ts = (datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
                      if r["ts"] else "?")
                print(f"  {ts}  {r['verdict']:<12} {r['route']:<9} {r['task'][:70]}")
        return 0

    if args.route_only:
        if not args.task:
            print("error: --route-only requires --task", file=sys.stderr)
            return 2
        cfg = _load_cfg(args)
        agent = HybridAgent(cfg, online_provider=args.online_provider,
                            local_provider=args.local_provider)
        context = args.context or ""
        memory = TaskMemory(memory_root_from_cfg(cfg),
                            embed=memory_embed_callable(cfg)).memory_view(args.task)
        route, reason = agent.route(
            args.task, context_chars=len(args.task) + len(context),
            memory=memory,
        )
        print(f"route\t{route}\t{reason}")
        return 0

    if args.review:
        if not args.task or not args.code:
            print("error: --review requires --task and --code", file=sys.stderr)
            return 2
        cfg = _load_cfg(args)
        mode = _resolve_mode(args)
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --review requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        agent = HybridAgent(cfg, online_provider=args.online_provider,
                            local_provider=args.local_provider)
        api_key_env = cfg["backends"]["deepseek"]["api_key_env"]
        if not _resolve_api_key(api_key_env):
            print("error: No DeepSeek API key found. Please log in to Kilo or set "
                  "DEEPSEEK_API_KEY.", file=sys.stderr)
            return 2
        system = args.system if args.system is not None else _load_supervisor_prompt()
        user = (
            "REVIEW this code following the Supervisor Review Protocol.\n\n"
            f"TASK: {args.task}\n\n"
            f"CODE:\n{args.code}\n"
        )
        if args.context:
            user += f"\nCONTEXT:\n{args.context}\n"
        user += (
            "\nRespond with APPROVED, FIX_REQUIRED, or REJECTED. "
            "Be specific and provide code examples for all fixes."
        )
        req = ModelRequest(
            system=system,
            user=user,
            max_tokens=args.max_tokens or 4096,
            temperature=args.temperature if args.temperature is not None else 0.1,
            timeout_s=cfg["backends"]["deepseek"]["timeout_s"],
        )
        model_label = "deepseek (supervisor review)"
        via = "http-fallback" if not _have_openai() else "backend"
        _status(f"[hybrid] ▶ {model_label} working ... (via={via})")
        try:
            resp = _generate_with_retry(agent, cfg, args, "deepseek", req)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - report any review failure cleanly
            _status(f"[hybrid] ✗ {model_label} failed")
            print(f"error: {exc}", file=sys.stderr)
            return 3
        tokens = _format_tokens(resp)
        if args.json:
            print(json.dumps({"text": resp.text, "backend": resp.backend,
                              "latency_ms": int(resp.latency_ms), "tokens": tokens},
                             ensure_ascii=False))
        else:
            print(resp.text)
        _status(f"[hybrid] ✓ {model_label} done · {int(resp.latency_ms)} ms · tokens={tokens} · via={resp.backend}")
        return 0

    if args.supervise:
        if not args.task:
            print("error: --supervise requires --task", file=sys.stderr)
            return 2
        cfg = _load_cfg(args)
        mode = _resolve_mode(args)
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --supervise requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        agent = HybridAgent(cfg, online_provider=args.online_provider,
                            local_provider=args.local_provider)
        if not _resolve_api_key(cfg["backends"]["deepseek"]["api_key_env"]):
            print("error: --supervise requires a DeepSeek key "
                  "(env DEEPSEEK_API_KEY or Kilo auth.json)", file=sys.stderr)
            return 2
        local, cloud = agent._backends()
        tracker = ProgressTracker()
        cache = CacheManager(cfg, args)
        cloud = _cached_cloud(cloud, cache)
        cloud = _budgeted_cloud(cloud, cfg)
        cloud = (_parallel_cloud(cloud, cfg, args) if args.turbo
                 else _failover_cloud(cloud, cfg, args))

        # --pull: refresh the baseline before the agent works (best-effort).
        if args.pull:
            _status("[hybrid] ⬇ git pull (baseline) ...")
            ok, msg = git_pull(args.root)
            _status(f"[hybrid] {'⬇ pulled' if ok else '⚠ pull skipped'}: {msg}")

        # Optional context (files/diff) goes into the review package via a builder.
        context = args.context or ""
        if args.context_scan:
            scanned = _scan_project_context(args.root, args.task)
            if scanned:
                context = (context + "\n\n" + scanned).strip() if context else scanned

        # Persistent task memory: router signal + consolidated insights injected
        # into the enhance/package context so both models learn from history.
        mem = TaskMemory(memory_root_from_cfg(cfg), embed=memory_embed_callable(cfg))
        memory = mem.memory_view(args.task)
        plan, plan_reason = _plan_supervision(
            agent, cfg, args, args.task, context, memory)
        if plan != "full":
            _status(f"[hybrid] 🧭 router: supervision={plan} ({plan_reason})")
        insights = mem.insights_text()
        if insights:
            context = (context + "\n\n" + insights).strip() if context else insights

        # --enhance: DeepSeek enhances the task and plans around Gemma's limits
        # BEFORE Gemma implements. The improved prompt + reasoning + plan are
        # shown (and DeepSeek asks for clarification if the task is unclear),
        # then the enhanced prompt is what Gemma implements. Critical tasks
        # force enhancement even without an explicit --enhance.
        enhancement = None
        task_for_gemma = args.task
        do_enhance = args.enhance or plan == "critical"
        if do_enhance:
            if plan == "critical" and not args.enhance:
                _status("[hybrid] 🧭 critical plan — forcing prompt enhancement")
            tracker.start_phase("enhance")
            try:
                task_for_gemma, enhancement, clar_needed = _enhance_task(
                    agent, cfg, args, args.task, context, cache=cache)
            except BudgetExceeded as exc:
                _status("[hybrid] ⛔ deepseek (prompt enhancer) skipped: daily token budget exhausted")
                print(f"error: {exc}", file=sys.stderr)
                return 6
            except KeyboardInterrupt:
                return 130
            except Exception as exc:  # noqa: BLE001 - report cleanly
                _status("[hybrid] ✗ deepseek (prompt enhancer) failed")
                print(f"error: {exc}", file=sys.stderr)
                return 3
            if clar_needed:
                _status("[hybrid] ⛔ TASK UNCLEAR: clarify the prompt and re-run.")
                print("CLARIFICATION_NEEDED: the task has ambiguities. Re-run with a "
                      "clearer --task, or run interactively to answer DeepSeek's "
                      "questions.", file=sys.stderr)
                return 4
            tracker.end_phase()

        def _pkg(task: str, code: str, iteration: int) -> ReviewPackage:
            return ReviewPackage(
                task=task,
                plan=enhancement.plan if enhancement else "",
                changes=f"Gemma output (iteration {iteration}):\n{code}",
                uncertainties=context,
                verification="None supplied — caller can pass --context with test/lint results.",
                diff=_current_diff(args.root),
            )

        # Results that must NEVER leave files behind: output was either never
        # independently reviewed (review_failed_no_verdict) or is INCOMPLETE
        # because even the DeepSeek fallback was truncated (cloud_fallback_truncated).
        _UNSAFE_REASONS = {"review_failed_no_verdict", "cloud_fallback_truncated"}

        if args.max_tokens is not None and args.max_tokens < 4096:
            _status(f"[hybrid] ⚠ --max-tokens={args.max_tokens} is low — "
                    "multi-file tasks need 8192+ (truncation will escalate)")

        tracker.start_phase("implement")
        try:
            if args.parallel:
                if not enhancement:
                    _status("[hybrid] ⛔ refusal: --parallel requires --enhance (needs the plan).")
                    print("error: --parallel requires --enhance so steps can be split from the plan.",
                          file=sys.stderr)
                    return 3
                result = _run_parallel(
                    agent, cfg, args, cloud, task_for_gemma, enhancement, context,
                    review=(plan != "local_first"))
            else:
                result = supervise(
                    local, cloud,
                    task=task_for_gemma,
                    package_builder=_pkg,
                    max_iterations=args.max_iterations,
                    gemma_generate=_supervise_gemma_generate(cfg, args.model, local_provider=args.local_provider),
                    status=lambda line: tracker.tick(line),
                    gemma_max_tokens=args.max_tokens or GEMMA_MAX_TOKENS,
                    review=(plan != "local_first"),
                    terminal_tool=_make_terminal_tool(cfg, args),
                    max_terminal_rounds=args.terminal_rounds,
                )
        except BudgetExceeded as exc:
            _status("[hybrid] ⛔ supervise aborted: daily DeepSeek token budget exhausted")
            print(f"error: {exc}", file=sys.stderr)
            return 6
        tracker.end_phase()
        # Record 80/20 metrics.
        stats = StatsTracker()
        for v in result.verdicts:
            stats.record_review(v.decision, v.quality_score)
        if result.escalated:
            stats.record_fallback()

        # With --apply: write the approved output's path-labeled fenced code
        # blocks to disk. NEVER apply unreviewed or incomplete output: those
        # paths return non-zero below and must not leave files behind.
        applied: list[tuple[str, int]] = []
        if args.apply and result.reason not in _UNSAFE_REASONS:
            tracker.start_phase("apply")
            applied, skipped = _apply_fenced_files(result.final_text, args.root,
                                                   dry_run=args.apply_dry_run)
            verb = "DRY-RUN" if args.apply_dry_run else "APPLIED"
            for rel, nbytes in applied:
                _status(f"[hybrid] {verb} {rel} ({nbytes} B)"
                        + (" [dry run]" if args.apply_dry_run else ""))
            if skipped:
                _status(f"[hybrid] ⚠ skipped {len(skipped)} block(s): "
                        f"{', '.join(skipped[:3])}")
            if not applied and not skipped:
                _status("[hybrid] ⚠ --apply: no path-labeled fenced blocks found in output")
            tracker.end_phase()

        # Record the session for self-evaluation (files, iterations, truncation,
        # and the final quality score).
        trunc_reasons = {"local_truncation_escalation", "cloud_fallback_truncated"}
        final_quality = result.verdicts[-1].quality_score if result.verdicts else 0.0
        stats.record_session(
            files_generated=len(applied),
            iterations=result.iterations,
            truncation=result.reason in trunc_reasons,
            quality=final_quality,
        )

        # Persistent task memory for the router's confidence/threshold learning.
        try:
            mem.record(TaskRecord(
                task=args.task,
                ts=time.time(),
                route=result.route,
                verdict=(result.verdicts[-1].decision if result.verdicts
                         else result.reason),
                quality=final_quality,
            ))
        except Exception:  # noqa: BLE001 - memory must never break the run
            pass

        # Final error-check stage: before the task is marked complete, run the
        # verification commands and have DeepSeek fix any errors (looping until
        # clean). Only runs when requested (--verify / --verify-cmd) or when
        # review.verify is configured.
        verified = True
        verify_report = ""
        verify_stats: dict = {}
        verify_cmds = list(args.verify_cmd)
        if args.verify and not verify_cmds:
            verify_cmds = list(cfg.get("review", {}).get("verify", []) or [])
        regression_cmds = list(args.regression_cmd)
        if args.regression and not regression_cmds:
            regression_cmds = list(cfg.get("review", {}).get("regression", []) or [])
        verify_groups = list(cfg.get("review", {}).get("verify_groups", []) or [])
        if args.verify_parallel and not verify_groups:
            _status("[hybrid] ⚠ --verify-parallel set but no review.verify_groups "
                    "configured — falling back to sequential")
        if verify_cmds:
            tracker.start_phase("verify")
            verify_timeout = (args.verify_timeout
                              or cfg.get("review", {}).get("verify_timeout", 600))
            regression_timeout = (args.regression_timeout
                                  or cfg.get("review", {}).get("regression_timeout", 600))
            verified, verify_report = _run_final_verify(
                cloud, args.root, args.task, verify_cmds,
                lambda line: tracker.tick(line),
                max_iter=args.verify_max, timeout_s=verify_timeout,
                verify_stats=verify_stats, regression_cmds=regression_cmds,
                regression_timeout=regression_timeout, cache=cache,
                parallel=args.verify_parallel, parallel_workers=args.verify_workers,
                verify_groups=verify_groups)
            tracker.end_phase()
        if verify_stats:
            stats.record_verify(verify_stats)
        stats.record_phases(tracker.phases())
        if not verified:
            _status("[hybrid] ⛔ FINAL CHECK FAILED: the task is NOT fully verified. "
                    "Fix DEEPSEEK_API_KEY/network, the verify command, or the code.")

        # --push / --deploy: ship the verified changes. Never run when the
        # final check failed or the run never independently reviewed its output.
        push_ok = deploy_ok = None
        push_msg = deploy_msg = ""
        if verified and result.reason not in _UNSAFE_REASONS:
            if args.push:
                push_files = [rel for rel, _ in applied]
                push_files += list(verify_stats.get("files", []))
                _status("[hybrid] ⬆ git push ...")
                push_ok, push_msg = git_push(
                    args.root, files=push_files or None,
                    message=f"hybrid-agent: {args.task[:100]}")
                _status(f"[hybrid] {'⬆ pushed' if push_ok else '⚠ push skipped'}: {push_msg}")
            if args.deploy or args.deploy_cmd:
                deploy_cmd = args.deploy_cmd or (cfg.get("deploy") or {}).get("command", "")
                deploy_cwd = (cfg.get("deploy") or {}).get("cwd") or args.root
                _status("[hybrid] 🚀 deploy ...")
                deploy_ok, deploy_msg = run_deploy(
                    args.root, deploy_cmd, cwd=deploy_cwd,
                    timeout=int((cfg.get("deploy") or {}).get("timeout", 1800)))
                _status(f"[hybrid] {'🚀 deployed' if deploy_ok else '⛔ deploy failed'}: {deploy_msg}")

        tokens = {}
        if args.json:
            payload = {
                "text": result.final_text,
                "route": result.route,
                "reason": result.reason,
                "iterations": result.iterations,
                "verdicts": [v.decision for v in result.verdicts],
                "quality_scores": [v.quality_score for v in result.verdicts],
                "applied_files": [rel for rel, _ in applied],
                "verified": verified,
                "progress": tracker.phases(),
            }
            if args.enhance and enhancement is not None:
                payload["enhanced_prompt"] = enhancement.enhanced_prompt
                payload["reasoning"] = enhancement.reasoning
                payload["plan"] = enhancement.plan
            if push_ok is not None:
                payload["push"] = {"ok": push_ok, "message": push_msg}
            if deploy_ok is not None:
                payload["deploy"] = {"ok": deploy_ok, "message": deploy_msg}
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(_strip_confidence_tag(result.final_text))
            if applied:
                verb = "WOULD APPLY" if args.apply_dry_run else "APPLIED"
                print(f"\n{verb} {len(applied)} file(s) to {os.path.abspath(args.root)}"
                      + (" (dry run)" if args.apply_dry_run else "") + ":")
                for rel, nbytes in applied:
                    print(f"  ✓ {rel} ({nbytes} B)")
            if verify_cmds:
                print("\nFINAL CHECK:", "PASSED" if verified else "FAILED",
                      f"({verify_report})")
            if push_ok is not None:
                print("\nPUSH:", "OK" if push_ok else "SKIPPED", f"({push_msg})")
            if deploy_ok is not None:
                print("DEPLOY:", "OK" if deploy_ok else "FAILED", f"({deploy_msg})")
        if result.reason in _UNSAFE_REASONS:
            # The output above is either UNREVIEWED (DeepSeek never returned a
            # verdict) or INCOMPLETE (DeepSeek fallback truncated even after a
            # same-budget retry). Report the failure and exit non-zero so callers
            # cannot treat the result as supervised, complete, or safe to apply.
            _status("[hybrid] ⛔ SUPERVISOR/OUTPUT FAILURE: output is "
                    "unreviewed or INCOMPLETE (truncated) — NOT safe to apply.")
            if not args.json:
                print(
                    "\nFAILURE: the output above was NOT independently reviewed or is "
                    "INCOMPLETE (truncated). Do not treat it as supervised, complete, "
                    "or safe to apply. Fix DEEPSEEK_API_KEY/network, raise --max-tokens, "
                    "and retry.",
                    file=sys.stderr,
                )
            return 3
        tracker.done()
        _status(f"[hybrid] ✓ supervise done · {result.iterations} iter · reason={result.reason} "
                f"· verdicts={[v.decision for v in result.verdicts]} "
                f"· scores={[v.quality_score for v in result.verdicts]}")
        return 0

    if not args.task:
        parser.print_usage(sys.stderr)
        print("error: --task is required for generation", file=sys.stderr)
        return 2
    if args.local and args.deepseek:
        print("error: --local and --deepseek are mutually exclusive", file=sys.stderr)
        return 2

    cfg = _load_cfg(args)
    agent = HybridAgent(cfg, online_provider=args.online_provider,
                            local_provider=args.local_provider)
    context = args.context or ""
    if args.context_scan:
        scanned = _scan_project_context(args.root, args.task)
        if scanned:
            context = (context + "\n\n" + scanned).strip() if context else scanned
    mode = _resolve_mode(args)
    m = MODES[mode]

    # Route resolution: explicit flags win, otherwise default to the mode's implementer.
    if args.local:
        route, reason = "local", "forced:--local"
    elif args.deepseek:
        route, reason = "deepseek", "forced:--deepseek"
    elif mode in ("hybrid", "local"):
        route, reason = "local", f"mode={mode} implementer"
    else:  # code
        route, reason = "deepseek", "mode=code implementer"

    violation = _mode_impl_violation(mode, route)
    if violation:
        _status(f"[hybrid] ⛔ refusal: {violation} (route={route}).")
        print(f"error: {violation}", file=sys.stderr)
        return 3

    api_key_env = cfg["backends"]["deepseek"]["api_key_env"]
    if route == "deepseek" and not _resolve_api_key(api_key_env):
        print("error: No DeepSeek API key found. Please log in to Kilo or set "
              "DEEPSEEK_API_KEY.", file=sys.stderr)
        return 2

    cache = None if args.no_cache else CacheManager(cfg, args)

    # --enhance (standalone, local route): DeepSeek enhances the prompt and plans
    # around Gemma's limits, shows the improved prompt + reasoning + plan (and
    # asks for clarification if the task is unclear), then the ENHANCED prompt
    # is what the local model implements.
    if args.enhance:
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --enhance requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        if route != "local":
            _status("[hybrid] ⛔ refusal: --enhance needs the local model as implementer.")
            print("error: --enhance is for the local route; use --local or MODE=hybrid/local.", file=sys.stderr)
            return 3
        try:
            task_for_gemma, enhancement, clar_needed = _enhance_task(
                agent, cfg, args, args.task, context, cache=cache)
        except BudgetExceeded as exc:
            _status("[hybrid] ⛔ deepseek (prompt enhancer) skipped: daily token budget exhausted")
            print(f"error: {exc}", file=sys.stderr)
            return 6
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - report cleanly
            _status("[hybrid] ✗ deepseek (prompt enhancer) failed")
            print(f"error: {exc}", file=sys.stderr)
            return 3
        if clar_needed:
            _status("[hybrid] ⛔ TASK UNCLEAR: clarify the prompt and re-run.")
            print("CLARIFICATION_NEEDED: the task has ambiguities. Re-run with a "
                  "clearer --task, or run interactively to answer DeepSeek's "
                  "questions.", file=sys.stderr)
            return 4
        if enhancement.enhanced:
            args.task = task_for_gemma

    req = _build_request(agent, args, route, context)
    cached_text = None
    if cache is not None and cache.enabled:
        key = cache.key("generate", req.system, req.user,
                        str(req.max_tokens), str(req.temperature))
        cached_text = cache.get("generate", key)
        if cached_text is not None:
            cache.record(True)
            _status(f"[hybrid] 🗃 cache HIT generate {key[:8]}")
            resp = ModelResponse(text=cached_text, backend="cache", latency_ms=0.0)
    if cached_text is None:
        model_label = _model_label(cfg, route, args.model)
        via = "http-fallback" if not _have_openai() else "backend"
        streaming = " · streaming" if (args.stream and route == "local") else ""
        _status(f"[hybrid] ▶ {model_label} working on \"{_task_preview(args.task)}\" (route={route}, {reason}, via={via}{streaming})")
        try:
            resp = _generate_with_retry(agent, cfg, args, route, req, stream=args.stream)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - report any generation failure cleanly
            _status(f"[hybrid] ✗ {model_label} failed")
            print(f"error: {exc}", file=sys.stderr)
            return 3
        if cache is not None and cache.enabled:
            cache.record(False)
            if resp.text and not resp.truncated:
                cache.set("generate", key, resp.text, source=route)

    tokens = _format_tokens(resp)
    if resp.truncated:
        _status("[hybrid] ⚠ TRUNCATED: output hit the max_tokens cap — do not apply blindly")
    if args.json:
        payload = {
            "text": resp.text,
            "route": route,
            "reason": reason,
            "backend": resp.backend,
            "latency_ms": int(resp.latency_ms),
            "tokens": tokens,
            "truncated": resp.truncated,
        }
        if args.enhance:
            payload["enhanced_prompt"] = enhancement.enhanced_prompt
            payload["reasoning"] = enhancement.reasoning
            payload["plan"] = enhancement.plan
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(_strip_confidence_tag(resp.text))
    _status(f"[hybrid] ✓ {model_label} done · {int(resp.latency_ms)} ms · tokens={tokens} · via={resp.backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
