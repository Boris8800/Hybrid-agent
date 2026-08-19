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
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

from backends.base import ModelRequest, ModelResponse
from backends.local_gemma import GemmaBackend
from router.confidence import MemoryView

from agent import HybridAgent, _load_config, apply_env_overrides
from backends.deepseek import _resolve_api_key
from supervise import GEMMA_MAX_TOKENS, ReviewPackage, _enhance_request, parse_enhancement, supervise

# --- Self-heal: loading config.yml requires PyYAML + openai, which exist only
# in the project venv (hybrid-agent/.venv). When invoked with a system python3
# that lacks them, ask.py silently fell back to embedded defaults (10s local
# timeout, unloaded model id "gemma-4-12b") - the root cause of slow and
# truncated local generations. Re-exec with the venv interpreter instead.
# Guarded by env var so the re-exec cannot loop.
if os.environ.get("HYBRID_REHEALED") != "1":
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


class StatsTracker:
    """Persists 80/20 strategy metrics to hybrid-agent/stats.json."""

    def __init__(self, path=None):
        self.path = pathlib.Path(path) if path else _STATS_FILE
        self.stats = self._load()

    def _load(self) -> dict:
        if self.path.is_file():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {
            "deepseek_reviews": 0, "approvals": 0, "rejections": 0,
            "fix_required": 0, "deepseek_fallbacks": 0, "total_quality_score": 0.0,
        }

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
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


def _generate_with_retry(agent: HybridAgent, cfg: dict, args: argparse.Namespace,
                         route: str, req: ModelRequest,
                         stream: bool = False) -> ModelResponse:
    """Generate, and when the output hits the token cap, retry ONCE with a
    doubled budget so truncated files are never returned silently. Long local
    generations (especially thinking-mode models) routinely need more than a
    single 2k budget; this is the guard that makes truncation rare."""
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
    return resp


def _format_tokens(resp: ModelResponse) -> dict:
    """Token usage for the status line. LM Studio's streaming path does not
    return usage, so fall back to an estimate so the line is never empty."""
    tokens = _normalize_tokens(resp.token_usage)
    if not tokens and resp.text:
        tokens = {"completion_tokens": max(1, len(resp.text) // 4), "estimated": True}
    return tokens


def _supervise_gemma_generate(cfg: dict, model_override: str | None):
    """Return a streaming wrapper for the supervise loop's local (Gemma) step.

    Always streams Gemma's output live to stderr (via the HTTP path) so the
    user can SEE the local model working in every iteration, then returns the
    full accumulated text for the review package. The local config timeout is
    applied to every request: gemma4 thinking mode can legitimately take 1-2
    minutes, and a short timeout would kill long full-file generations.
    """
    lc = cfg["backends"]["local"]
    base_url = lc["base_url"]
    api_key = lc.get("api_key") or "lm-studio"
    model = model_override or lc["model"]
    timeout_s = lc["timeout_s"]

    def _gen(req: ModelRequest) -> ModelResponse:
        req.timeout_s = timeout_s
        return _http_generate(base_url, api_key, model, req, stream=True)

    return _gen


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


def _select_backend(agent: HybridAgent, cfg: dict, route: str,
                    model_override: str | None):
    """Return the backend for `route`, applying the local model override.

    For the local route we construct only the GemmaBackend directly so that a
    purely-local run never needs DEEPSEEK_API_KEY. For deepseek we build via
    agent._backends() (whose key check we have already validated)."""
    if route != "local":
        _, cloud = agent._backends()
        return cloud
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

    Accepts ```path, ```path/to/f.ext, ```lang path, and ```lang:path forms.
    Returns [(relative_path, content)] in document order. Blocks whose label
    has no plausible path (e.g. bare ```python``` with no filename) are ignored.
    """
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
    files = []
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        body = m.group(2)
        if not label:
            continue
        path = None
        for token in label.replace(":", " ").split():
            t = token.strip("`").strip()
            if not t or t.startswith("_"):
                continue
            # A plausible path: contains a "/" or ends with a known extension.
            if "/" in t or t.lower().rsplit(".", 1)[-1] in _APPLY_EXTENSIONS:
                path = t
                break
        if path:
            files.append((path, body.rstrip("\n") + "\n"))
    return files


def _apply_fenced_files(text: str, root: str = ".") -> tuple[list[tuple[str, int]], list[str]]:
    """Write path-labeled fenced blocks from `text` under `root`.

    Returns (written, skipped): `written` is [(relative_path, bytes_written)],
    `skipped` is a list of human-readable reasons. Absolute paths and paths
    escaping `root` via ".." are never written.
    """
    root_abs = os.path.abspath(root)
    written, skipped = [], []
    for rel_path, content in _parse_fenced_files(text):
        clean = rel_path.replace("\\", "/")
        # Reject absolute paths and any ".." traversal BEFORE normalization.
        if clean.startswith("/") or os.path.isabs(clean) \
                or any(part == ".." for part in clean.split("/")):
            skipped.append(f"{rel_path} (unsafe path)")
            continue
        target = os.path.normpath(os.path.join(root_abs, clean))
        if target != root_abs and not target.startswith(root_abs + os.sep):
            skipped.append(f"{rel_path} (escapes root)")
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
    parser.add_argument("--root", default=".",
                        help="directory to apply files under (default: current directory)")
    parser.add_argument("--mode", choices=["hybrid", "local", "code"],
                        help="agent mode: hybrid (local impl + API supervision), "
                             "local (local impl, no API), code (API impl, no API supervision). "
                             "Default from $MODE, else hybrid")
    parser.add_argument("--stats", action="store_true",
                        help="print the 80/20 strategy summary (hybrid-agent/stats.json)")
    args = parser.parse_args()

    if args.stats:
        summary = StatsTracker().get_summary()
        print(json.dumps(summary, indent=2))
        return 0

    if args.models:
        cfg = _load_config_quiet(args.config or _default_config_path())
        return _list_models(cfg["backends"]["local"]["base_url"])

    if args.route_only:
        if not args.task:
            print("error: --route-only requires --task", file=sys.stderr)
            return 2
        cfg = _load_config_quiet(args.config or _default_config_path())
        agent = HybridAgent(cfg)
        context = args.context or ""
        route, reason = agent.route(
            args.task, context_chars=len(args.task) + len(context),
            memory=MemoryView(),
        )
        print(f"route\t{route}\t{reason}")
        return 0

    if args.review:
        if not args.task or not args.code:
            print("error: --review requires --task and --code", file=sys.stderr)
            return 2
        cfg = _load_config_quiet(args.config or _default_config_path())
        mode = _resolve_mode(args)
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --review requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        agent = HybridAgent(cfg)
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
        cfg = _load_config_quiet(args.config or _default_config_path())
        mode = _resolve_mode(args)
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --supervise requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        agent = HybridAgent(cfg)
        if not _resolve_api_key(cfg["backends"]["deepseek"]["api_key_env"]):
            print("error: --supervise requires a DeepSeek key "
                  "(env DEEPSEEK_API_KEY or Kilo auth.json)", file=sys.stderr)
            return 2
        local, cloud = agent._backends()

        # Optional context (files/diff) goes into the review package via a builder.
        context = args.context or ""

        # --enhance: DeepSeek enhances the task and plans around Gemma's limits
        # BEFORE Gemma implements. The improved prompt + reasoning + plan are
        # shown, then the enhanced prompt is what Gemma implements.
        enhancement = None
        task_for_gemma = args.task
        if args.enhance:
            enh_label = "deepseek (prompt enhancer)"
            _status(f"[hybrid] ▶ {enh_label} working on \"{_task_preview(args.task)}\" ...")
            try:
                enh_resp = _generate_with_retry(
                    agent, cfg, args, "deepseek",
                    _enhance_request(args.task, context),
                )
            except KeyboardInterrupt:
                return 130
            except Exception as exc:  # noqa: BLE001 - report cleanly
                _status(f"[hybrid] ✗ {enh_label} failed")
                print(f"error: {exc}", file=sys.stderr)
                return 3
            enhancement = parse_enhancement(enh_resp.text)
            _status(f"[hybrid] ✓ {enh_label} done · {int(enh_resp.latency_ms)} ms · "
                    f"tokens={_format_tokens(enh_resp)}")
            if not args.json:
                print("=== ENHANCED PROMPT ===\n" + enhancement.enhanced_prompt)
                if enhancement.reasoning:
                    print("\n=== REASONING ===\n" + enhancement.reasoning)
                if enhancement.plan:
                    print("\n=== PLAN ===\n" + enhancement.plan)
                print("\n--- sending enhanced prompt to local model ---\n")
            if enhancement.enhanced:
                task_for_gemma = enhancement.enhanced_prompt

        def _pkg(task: str, code: str, iteration: int) -> ReviewPackage:
            return ReviewPackage(
                task=task,
                plan=enhancement.plan if enhancement else "",
                changes=f"Gemma output (iteration {iteration}):\n{code}",
                uncertainties=context,
                verification="None supplied — caller can pass --context with test/lint results.",
            )

        # Results that must NEVER leave files behind: output was either never
        # independently reviewed (review_failed_no_verdict) or is INCOMPLETE
        # because even the DeepSeek fallback was truncated (cloud_fallback_truncated).
        _UNSAFE_REASONS = {"review_failed_no_verdict", "cloud_fallback_truncated"}

        if args.max_tokens is not None and args.max_tokens < 4096:
            _status(f"[hybrid] ⚠ --max-tokens={args.max_tokens} is low — "
                    "multi-file tasks need 8192+ (truncation will escalate)")

        result = supervise(
            local, cloud,
            task=task_for_gemma,
            package_builder=_pkg,
            max_iterations=args.max_iterations,
            gemma_generate=_supervise_gemma_generate(cfg, args.model),
            status=lambda line: _status(line),
            gemma_max_tokens=args.max_tokens or GEMMA_MAX_TOKENS,
        )
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
            applied, skipped = _apply_fenced_files(result.final_text, args.root)
            for rel, nbytes in applied:
                _status(f"[hybrid] ✓ APPLIED {rel} ({nbytes} B)")
            if skipped:
                _status(f"[hybrid] ⚠ skipped {len(skipped)} block(s): "
                        f"{', '.join(skipped[:3])}")
            if not applied and not skipped:
                _status("[hybrid] ⚠ --apply: no path-labeled fenced blocks found in output")

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
            }
            if args.enhance and enhancement is not None:
                payload["enhanced_prompt"] = enhancement.enhanced_prompt
                payload["reasoning"] = enhancement.reasoning
                payload["plan"] = enhancement.plan
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(_strip_confidence_tag(result.final_text))
            if applied:
                print(f"\nAPPLIED {len(applied)} file(s) to {os.path.abspath(args.root)}:")
                for rel, nbytes in applied:
                    print(f"  ✓ {rel} ({nbytes} B)")
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

    cfg = _load_config_quiet(args.config or _default_config_path())
    agent = HybridAgent(cfg)
    context = args.context or ""
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

    # --enhance (standalone, local route): DeepSeek enhances the prompt and plans
    # around Gemma's limits, shows the improved prompt + reasoning + plan, then
    # the ENHANCED prompt is what the local model implements.
    if args.enhance:
        if not MODES[mode]["api_supervision"]:
            _status(f"[hybrid] ⛔ refusal: mode={mode} has no API supervision (only hybrid enables it).")
            print(f"error: --enhance requires hybrid mode (API supervision). Use MODE=hybrid.", file=sys.stderr)
            return 3
        if route != "local":
            _status("[hybrid] ⛔ refusal: --enhance needs the local model as implementer.")
            print("error: --enhance is for the local route; use --local or MODE=hybrid/local.", file=sys.stderr)
            return 3
        enh_label = "deepseek (prompt enhancer)"
        _status(f"[hybrid] ▶ {enh_label} working on \"{_task_preview(args.task)}\" ...")
        try:
            enh_resp = _generate_with_retry(
                agent, cfg, args, "deepseek", _enhance_request(args.task, context)
            )
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - report cleanly
            _status(f"[hybrid] ✗ {enh_label} failed")
            print(f"error: {exc}", file=sys.stderr)
            return 3
        enhancement = parse_enhancement(enh_resp.text)
        _status(f"[hybrid] ✓ {enh_label} done · {int(enh_resp.latency_ms)} ms · "
                f"tokens={_format_tokens(enh_resp)}")
        if not args.json:
            print("=== ENHANCED PROMPT ===\n" + enhancement.enhanced_prompt)
            if enhancement.reasoning:
                print("\n=== REASONING ===\n" + enhancement.reasoning)
            if enhancement.plan:
                print("\n=== PLAN ===\n" + enhancement.plan)
            print("\n--- sending enhanced prompt to local model ---\n")
        if enhancement.enhanced:
            args.task = enhancement.enhanced_prompt

    req = _build_request(agent, args, route, context)
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
