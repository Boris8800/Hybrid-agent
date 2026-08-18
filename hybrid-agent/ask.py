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
import sys
import time
import urllib.request

from backends.base import ModelRequest, ModelResponse
from backends.local_gemma import GemmaBackend
from router.confidence import MemoryView

from agent import HybridAgent, _load_config
from backends.deepseek import _resolve_api_key
from supervise import ReviewPackage, supervise

LM_STUDIO_BASE = "http://localhost:1234/v1"

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


def _default_config_path() -> str | None:
    """Locate config.yml / config.yaml next to this script (any cwd)."""
    script_dir = pathlib.Path(__file__).resolve().parent
    for name in ("config.yml", "config.yaml"):
        candidate = script_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def _load_config_quiet(path: str | None) -> dict:
    """Load config, diverting _load_config's warnings to stderr so stdout
    stays clean (stdout carries the full untruncated model output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = _load_config(path)
    if buf.getvalue():
        sys.stderr.write(buf.getvalue())
    return cfg


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
                else:
                    pieces: list[str] = []
                    usage: dict = {}
                    marker = False
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
            return ModelResponse(
                text=text, raw=text,
                token_usage=usage,
                latency_ms=(time.monotonic() - started) * 1000,
                backend="http-fallback",
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
        api_key = "lm-studio"
        model = args.model or lc["model"]
        return _http_generate(base_url, api_key, model, req, stream=True)
    if _have_openai():
        backend = _select_backend(agent, cfg, route, args.model)
        return backend.generate(req)
    if route == "local":
        lc = cfg["backends"]["local"]
        base_url = lc["base_url"]
        api_key = "lm-studio"
        model = args.model or lc["model"]
    else:
        dc = cfg["backends"]["deepseek"]
        base_url = "https://api.deepseek.com"
        api_key = _resolve_api_key(dc["api_key_env"])
        model = dc["model"]
        if not api_key:
            raise RuntimeError(f"missing API key: set {dc['api_key_env']} or install Kilo auth.json")
    return _http_generate(base_url, api_key, model, req,
                          stream=stream and route == "local")


def _supervise_gemma_generate(cfg: dict, model_override: str | None):
    """Return a streaming wrapper for the supervise loop's local (Gemma) step.

    Always streams Gemma's output live to stderr (via the HTTP path) so the
    user can SEE the local model working in every iteration, then returns the
    full accumulated text for the review package.
    """
    lc = cfg["backends"]["local"]
    base_url = lc["base_url"]
    model = model_override or lc["model"]

    def _gen(req: ModelRequest) -> ModelResponse:
        return _http_generate(base_url, "lm-studio", model, req, stream=True)

    return _gen


def _list_models() -> int:
    """Print one loaded LM Studio model id per line. Exit 0 / 1."""
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - fallback below keeps it working
        OpenAI = None  # type: ignore

    if OpenAI is not None:
        client = OpenAI(base_url=LM_STUDIO_BASE, api_key="lm-studio")
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
        with urllib.request.urlopen(f"{LM_STUDIO_BASE}/models", timeout=5) as resp:
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
        max_tokens=args.max_tokens or 2048,
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
    parser.add_argument("--max-tokens", type=int, help="max output tokens (default 2048)")
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
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="max supervise loop iterations (default 3)")
    args = parser.parse_args()

    if args.models:
        return _list_models()

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
        agent = HybridAgent(cfg)
        api_key_env = cfg["backends"]["deepseek"]["api_key_env"]
        if not _resolve_api_key(api_key_env):
            print("error: DEEPSEEK_API_KEY is not set", file=sys.stderr)
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
            resp = _generate(agent, cfg, args, "deepseek", req)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - report any review failure cleanly
            _status(f"[hybrid] ✗ {model_label} failed")
            print(f"error: {exc}", file=sys.stderr)
            return 3
        tokens = _normalize_tokens(resp.token_usage)
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
        agent = HybridAgent(cfg)
        if not _resolve_api_key(cfg["backends"]["deepseek"]["api_key_env"]):
            print("error: --supervise requires a DeepSeek key "
                  "(env DEEPSEEK_API_KEY or Kilo auth.json)", file=sys.stderr)
            return 2
        local, cloud = agent._backends()

        # Optional context (files/diff) goes into the review package via a builder.
        context = args.context or ""
        def _pkg(task: str, code: str, iteration: int) -> ReviewPackage:
            return ReviewPackage(
                task=task,
                changes=f"Gemma output (iteration {iteration}):\n{code}",
                uncertainties=context,
                verification="None supplied — caller can pass --context with test/lint results.",
            )

        result = supervise(
            local, cloud,
            task=args.task,
            package_builder=_pkg,
            max_iterations=args.max_iterations,
            gemma_generate=_supervise_gemma_generate(cfg, args.model),
            status=lambda line: _status(line),
        )
        tokens = {}
        if args.json:
            print(json.dumps({
                "text": result.final_text,
                "route": result.route,
                "reason": result.reason,
                "iterations": result.iterations,
                "verdicts": [v.decision for v in result.verdicts],
            }, ensure_ascii=False))
        else:
            print(_strip_confidence_tag(result.final_text))
        _status(f"[hybrid] ✓ supervise done · {result.iterations} iter · reason={result.reason} "
                f"· verdicts={[v.decision for v in result.verdicts]}")
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

    if args.local:
        route, reason = "local", "forced:--local"
    elif args.deepseek:
        route, reason = "deepseek", "forced:--deepseek"
    else:
        route, reason = agent.route(
            args.task, context_chars=len(args.task) + len(context),
            memory=MemoryView(),
        )

    api_key_env = cfg["backends"]["deepseek"]["api_key_env"]
    if route == "deepseek" and not _resolve_api_key(api_key_env):
        print("error: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 2

    req = _build_request(agent, args, route, context)
    model_label = _model_label(cfg, route, args.model)
    via = "http-fallback" if not _have_openai() else "backend"
    streaming = " · streaming" if (args.stream and route == "local") else ""
    _status(f"[hybrid] ▶ {model_label} working on \"{_task_preview(args.task)}\" (route={route}, {reason}, via={via}{streaming})")
    try:
        resp = _generate(agent, cfg, args, route, req, stream=args.stream)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - report any generation failure cleanly
        _status(f"[hybrid] ✗ {model_label} failed")
        print(f"error: {exc}", file=sys.stderr)
        return 3

    tokens = _normalize_tokens(resp.token_usage)
    if args.json:
        payload = {
            "text": resp.text,
            "route": route,
            "reason": reason,
            "backend": resp.backend,
            "latency_ms": int(resp.latency_ms),
            "tokens": tokens,
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(_strip_confidence_tag(resp.text))
    _status(f"[hybrid] ✓ {model_label} done · {int(resp.latency_ms)} ms · tokens={tokens} · via={resp.backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
