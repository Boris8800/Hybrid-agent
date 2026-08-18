"""DeepSeek API backend (ARCHITECTURE.md §8.2)."""

import json
import os
import pathlib
import time

from backends.base import Backend, BackendError, ModelRequest, ModelResponse

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def _resolve_api_key(api_key_env: str) -> str:
    """Resolve the DeepSeek API key.

    Precedence:
      1. The named environment variable (e.g. DEEPSEEK_API_KEY).
      2. Kilo's own auth.json (~/.local/share/kilo/auth.json) "deepseek.key".
    Returns "" when neither is available.
    """
    key = os.environ.get(api_key_env) or ""
    if key:
        return key
    try:
        auth_path = pathlib.Path.home() / ".local" / "share" / "kilo" / "auth.json"
        if auth_path.exists():
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            key = (data.get("deepseek") or {}).get("key") or ""
    except (OSError, ValueError):  # unreadable or invalid JSON -> just no key
        key = ""
    return key


class DeepSeekBackend(Backend):
    name = "deepseek"

    def __init__(self, api_key_env: str = "DEEPSEEK_API_KEY", model: str = "deepseek-chat",
                 *, timeout_s: float = 30.0, max_retries: int = 4,
                 backoff_s: list[float] | None = None):
        if OpenAI is None:
            raise RuntimeError("openai>=1.0 is required for the deepseek backend")
        api_key = _resolve_api_key(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing environment variable {api_key_env}")
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s or [1.0, 2.0, 4.0]

    def generate(self, request: ModelRequest) -> ModelResponse:
        last_error: BackendError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                started = time.monotonic()
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    timeout=request.timeout_s,
                )
                text = completion.choices[0].message.content or ""
                return ModelResponse(
                    text=text,
                    raw=text,
                    token_usage=getattr(completion, "usage", {}),
                    latency_ms=(time.monotonic() - started) * 1000,
                    backend=self.name,
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    # Honest Retry-After header when present can be honored here.
                    pass
                delay = self.backoff_s[attempt] if attempt < len(self.backoff_s) else self.backoff_s[-1]
                time.sleep(delay + (attempt * 0.1))  # jitter
                last_error = BackendError(str(exc), retryable=True)

        raise BackendError(
            f"deepseek backend failed after {self.max_retries + 1} attempts: {last_error}"
        )


def parse_json_verdict(text: str) -> dict:
    """Strict JSON extraction with a lenient fallback for ```json fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the outermost JSON object as a last-resort recovery.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"unparseable JSON verdict: {text[:200]!r}")