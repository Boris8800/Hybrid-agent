"""Local (MLX) backend for the hybrid engine — LM Studio's modern chat API.

Endpoint: POST {base_url}/chat  (base_url = http://localhost:1234/api/v1)
Request:  {"model", "system_prompt", "input", "stream"}
Response: {"output": [{"type": "message", "content": "..."}],
           "stats": {"input_tokens": N, "total_output_tokens": N}}
Streaming: SSE — event: message.delta / data: {"type":"message.delta","content":"..."}
Embeddings stay on the legacy OpenAI-compatible /v1/embeddings endpoint.

Stdlib only (no openai SDK needed for the local backend).
"""

import json
import time
import urllib.request

from backends.base import Backend, BackendError, ModelRequest, ModelResponse

# Default loaded context window (tokens) for the local model. Must match the
# `-c` flag used to load the model (reload-local.sh: -c 32768 for
# qwen2.5-coder-14b-instruct-mlx, whose config.json has max_position_embeddings
# 32768). Used to detect server-side truncation at the context boundary, which
# the /api/v1/chat endpoint never reports via finish_reason. Override per
# provider via config.yml `context_window` for models loaded with other windows.
DEFAULT_CONTEXT_WINDOW = 32768

# Per-model output caps the engine knows about (tokens). The local /api/v1/chat
# endpoint enforces NO output cap itself, so these are advisory for planning
# only; the API supervisor uses them to size steps. Unknown models -> 0 (no cap).
KNOWN_MODEL_MAX_OUTPUT: dict[str, int] = {
    "qwen2.5-coder-14b-instruct-mlx": 0,       # no server-imposed output cap
    "qwen2.5-coder-7b-instruct-mlx": 0,
}


def discover_model_capabilities(base_url: str, api_key: str = "lm-studio",
                                model: str = "", timeout_s: float = 15.0) -> dict:
    """Auto-discover the LOCAL MODEL's actual capabilities from LM Studio's
    /api/v1/models endpoint. Returns a dict with the fields the Context Safety
    Controller needs (all best-effort, 0/"" on unknown):

      model_id            loaded model identifier ('' = unknown)
      context_window      ACTUAL loaded -c window (loaded_instances config)
      max_context_length  the model's maximum supported context
      architecture        e.g. 'qwen2' / 'llama' ('' if unknown)
      format              e.g. 'mlx' / 'gguf' ('' if unknown)
      quantization        e.g. '8bit' ('' if unknown)
      max_output          KNOWN_MODEL_MAX_OUTPUT[model] (0 = no known cap)
      vision              bool — model supports vision/multimodal input
      tool_use            bool — model reports trained_for_tool_use
      tokenizer           '' (LM Studio does not expose one over HTTP)
    """
    caps = {
        "model_id": model,
        "context_window": 0,
        "max_context_length": 0,
        "architecture": "",
        "format": "",
        "quantization": "",
        "max_output": int(KNOWN_MODEL_MAX_OUTPUT.get(model, 0) or 0),
        "vision": False,
        "tool_use": False,
        "tokenizer": "",
    }
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url, headers={"Content-Type": "application/json",
                      "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode(errors="replace"))
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return caps
    if not isinstance(payload, dict):
        return caps
    for m in payload.get("models") or []:
        if not isinstance(m, dict):
            continue
        key = m.get("key") or m.get("id") or ""
        if model and key != model:
            continue
        caps["model_id"] = key or caps["model_id"]
        caps["architecture"] = str(m.get("architecture") or caps["architecture"])
        caps["format"] = str(m.get("format") or caps["format"])
        q = m.get("quantization")
        if isinstance(q, dict):
            caps["quantization"] = str(q.get("name") or caps["quantization"])
        elif q:
            caps["quantization"] = str(q)
        cap = m.get("capabilities")
        if isinstance(cap, dict):
            caps["vision"] = bool(cap.get("vision"))
            caps["tool_use"] = bool(cap.get("trained_for_tool_use"))
        if m.get("max_context_length"):
            caps["max_context_length"] = int(m["max_context_length"])
        # Loaded instance config is the source of truth for the ACTIVE window.
        for inst in m.get("loaded_instances") or []:
            cfg = inst.get("config") if isinstance(inst, dict) else None
            if isinstance(cfg, dict) and cfg.get("context_length"):
                caps["context_window"] = int(cfg["context_length"])
                break
        if caps["context_window"]:
            break
    if not caps["context_window"] and caps["max_context_length"]:
        caps["context_window"] = caps["max_context_length"]
    return caps


def discover_context_window(base_url: str, api_key: str = "lm-studio",
                            model: str = "", timeout_s: float = 15.0) -> int:
    """Auto-discover the LOCAL MODEL's actually loaded context window.

    Queries LM Studio's /api/v1/models endpoint, which reports the real
    `-c` context window in `loaded_instances[].config.context_length` (falling
    back to `max_context_length`). This makes truncation detection correct for
    ANY local model and ANY load setting — no hardcoded number to drift from
    the server. Returns 0 when the endpoint is unreachable or the model is not
    found (caller falls back to config/default).
    """
    return int(discover_model_capabilities(
        base_url, api_key=api_key, model=model, timeout_s=timeout_s
    ).get("context_window") or 0)


def _mlx_body(model: str, system: str, user: str, stream: bool,
              temperature: float, max_tokens: int,
              exclude_keys: tuple = (), extra_body: dict | None = None) -> dict:
    """Build the MLX-format request body. Model-specific quirks (e.g. the
    server rejecting 'max_tokens') are handled config-driven via exclude_keys /
    extra_body, so new endpoints never need code changes."""
    body: dict = {
        "model": model,
        "system_prompt": system,
        "input": user,
        "stream": bool(stream),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    for key in exclude_keys:
        body.pop(key, None)
    if extra_body:
        body.update(extra_body)
    return body


_LENGTH_REASONS = {
    "length", "max_tokens", "truncated", "context_length",
    "context_window", "max_length", "stop_length",
}


def _mlx_result(payload: dict, max_tokens: int, *, saw_end_event: bool = True,
                stream_eof: bool = False, context_window: int = 0) -> dict:
    """Normalize an MLX response; detect truncation for ANY loaded local model.

    LM Studio's /api/v1/chat rejects `max_tokens` (so budgets are advisory) and
    does not include a finish_reason, so truncation is detected from every
    available signal, generically — no model name is assumed:
      - an explicit finish/stop reason field when the server provides one;
      - output tokens reaching the requested budget (servers that accept it);
      - prompt+output reaching the model's loaded context window (the classic
        server-side mid-generation cutoff — the only signal LM Studio gives us
        for this endpoint, and the reason local coding output gets cut off);
      - a stream that ended without the server's message.end/chat.end event
        (premature close = output was cut off);
      - an unbalanced fenced code block, the classic mid-file cut signature.
    """
    stats = payload.get("stats") or {}
    output = payload.get("output") or []
    text = "".join(item.get("content", "") for item in output
                   if isinstance(item, dict) and item.get("type") == "message")
    used = int(stats.get("total_output_tokens", 0) or 0)
    prompt_tokens = int(stats.get("input_tokens", 0) or 0)

    raw_finish = (payload.get("finish_reason") or payload.get("stop_reason")
                  or stats.get("finish_reason") or stats.get("stop_reason"))
    finish = str(raw_finish or "").strip().lower()
    truncated = bool(finish) and finish in _LENGTH_REASONS
    if not truncated and max_tokens and used >= max_tokens:
        truncated = True
    if not truncated and stream_eof and text:
        truncated = True
    if not truncated and text.count("```") % 2 == 1:
        truncated = True
    # LM Studio's /api/v1/chat never reports a finish_reason, so when
    # prompt+output tokens reach the loaded context window the generation was
    # almost certainly cut off mid-output. This is the signal that fires when a
    # coding response is truncated against the server's context limit.
    window_hit = (context_window and prompt_tokens > 0
                  and prompt_tokens + used >= context_window)
    if not truncated and window_hit:
        truncated = True
    # Diagnostic note so the reason is visible in logs / status lines.
    reason = "server_finish_reason"
    if truncated and not finish:
        reason = ("max_tokens_reached" if max_tokens and used >= max_tokens
                  else "stream_eof" if stream_eof
                  else "context_window_reached" if window_hit
                  else "unbalanced_code_fence")
    return {
        "text": text,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": used,
            "total_tokens": prompt_tokens + used,
        },
        "truncated": truncated,
        "truncate_reason": reason if truncated else "",
        "saw_end_event": saw_end_event,
        "raw": json.dumps(payload) if isinstance(payload, dict) else "",
    }


def mlx_chat_request(base_url: str, api_key: str, model: str,
                     system: str, user: str, stream: bool = False,
                     timeout_s: float = 600.0, temperature: float = 0.2,
                     max_tokens: int = 8192, exclude_keys: tuple = (),
                     extra_body: dict | None = None,
                     context_window: int = DEFAULT_CONTEXT_WINDOW) -> dict:
    """POST the MLX-format chat request. Returns a dict with keys
    text, token_usage, truncated, raw. Raises on transport errors."""
    url = base_url.rstrip("/") + "/chat"
    body = _mlx_body(model, system, user, stream, temperature, max_tokens,
                     exclude_keys, extra_body)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        raise BackendError(f"local {model} HTTP {exc.code}: "
                           f"{exc.read().decode(errors='replace')[:300]}",
                           retryable=True)
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"local {model} request failed: {exc}", retryable=True)
    payload = json.loads(raw) if raw.strip() else {}
    return _mlx_result(payload, max_tokens, context_window=context_window)


def mlx_chat_stream(base_url: str, api_key: str, model: str,
                    system: str, user: str, timeout_s: float = 600.0,
                    temperature: float = 0.2, max_tokens: int = 8192,
                    exclude_keys: tuple = (), extra_body: dict | None = None,
                    context_window: int = DEFAULT_CONTEXT_WINDOW) -> dict:
    """Stream the MLX-format chat via SSE, accumulating the full text. Returns
    the same dict as mlx_chat_request."""
    url = base_url.rstrip("/") + "/chat"
    body = _mlx_body(model, system, user, True, temperature, max_tokens,
                     exclude_keys, extra_body)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    parts: list[str] = []
    stats: dict = {}
    saw_end = False
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_line = line[5:].strip()
                if not payload_line or payload_line == "[DONE]":
                    continue
                try:
                    ev = json.loads(payload_line)
                except ValueError:
                    continue
                etype = ev.get("type", "")
                if etype == "message.delta":
                    parts.append(ev.get("content", "") or "")
                elif etype in ("chat.end", "message.end"):
                    saw_end = True
                    stats = ev.get("stats") or stats
    except urllib.error.HTTPError as exc:
        raise BackendError(f"local {model} HTTP {exc.code}: "
                           f"{exc.read().decode(errors='replace')[:300]}",
                           retryable=True)
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"local {model} stream failed: {exc}", retryable=True)
    result = _mlx_result({"output": [{"type": "message", "content": "".join(parts)}],
                          "stats": stats}, max_tokens,
                         saw_end_event=saw_end, stream_eof=not saw_end,
                         context_window=context_window)
    result["latency_ms"] = (time.monotonic() - started) * 1000
    return result


class QwenBackend(Backend):
    name = "local-qwen"

    def __init__(self, base_url: str, model: str, *, api_key: str = "lm-studio",
                 name: str = "local-qwen", timeout_s: float = 600.0,
                 max_retries: int = 3, backoff_s: list[float] | None = None,
                 cold_start_wait_s: float = 30.0, exclude_keys: tuple = (),
                 extra_body: dict | None = None,
                 context_window: int = DEFAULT_CONTEXT_WINDOW):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.name = name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s or [0.5, 1.0, 2.0]
        self.cold_start_wait_s = cold_start_wait_s
        self.exclude_keys = tuple(exclude_keys)
        self.extra_body = extra_body or {}
        self.context_window = context_window

    def generate(self, request: ModelRequest) -> ModelResponse:
        last_error: BackendError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                started = time.monotonic()
                res = mlx_chat_request(
                    self.base_url, self.api_key, self.model,
                    request.system, request.user, stream=False,
                    timeout_s=request.timeout_s,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    exclude_keys=self.exclude_keys, extra_body=self.extra_body,
                    context_window=self.context_window)
                return ModelResponse(
                    text=res["text"], raw=res["raw"],
                    token_usage=res["token_usage"],
                    latency_ms=(time.monotonic() - started) * 1000,
                    backend=self.name, truncated=res["truncated"],
                )
            except BackendError as exc:
                # LM Studio signals "model not loaded yet" with HTTP 503.
                if "HTTP 503" in str(exc) and attempt == 0:
                    time.sleep(self.cold_start_wait_s)
                    continue
                delay = self.backoff_s[attempt] if attempt < len(self.backoff_s) else self.backoff_s[-1]
                time.sleep(delay)
                last_error = exc
        raise BackendError(
            f"local backend failed after {self.max_retries + 1} attempts: {last_error}"
        )
