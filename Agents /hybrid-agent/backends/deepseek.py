"""DeepSeek API backend (ARCHITECTURE.md §8.2)."""

import json
import os
import pathlib
import re
import time

from backends.base import Backend, BackendError, ModelRequest, ModelResponse

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def _extract_deepseek_key(data: dict) -> str:
    """Extract a DeepSeek API key from a parsed Kilo auth/config object.

    Accepts every key layout Kilo has used across versions:
      {"deepseek": {"key": "sk-..."}}
      {"deepseek": {"api_key": "sk-..."}}
      {"providers": {"deepseek": {"apiKey": "sk-..."}}}
      {"kilo": {"access": "..."}}
    Returns "" when no known layout matches.
    """
    sections = (
        ("deepseek",),
        ("providers", "deepseek"),
        ("provider", "deepseek"),
        ("kilo",),
    )
    key_names = ("api_key", "apiKey", "key", "access")
    for section in sections:
        node = data
        for part in section:
            node = (node or {}).get(part)
            if not isinstance(node, dict):
                node = None
                break
        if not isinstance(node, dict):
            continue
        for name in key_names:
            value = node.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


_KILO_AUTH_CANDIDATES = (
    pathlib.Path.home() / ".local" / "share" / "kilo" / "auth.json",
    pathlib.Path.home() / ".config" / "kilo" / "auth.json",
    pathlib.Path.home() / ".kilo" / "auth.json",
)

_KILO_CONFIG_CANDIDATES = (
    pathlib.Path.home() / ".config" / "kilo" / "config.json",
    pathlib.Path.home() / ".config" / "kilo" / "kilo.json",
    pathlib.Path.home() / ".config" / "kilo" / "kilo.jsonc",
    pathlib.Path.home() / ".kilo" / "config.json",
    pathlib.Path.home() / ".kilo" / "kilo.json",
)


def _load_kilo_json(path: pathlib.Path) -> dict | None:
    """Parse a Kilo JSON/JSONC auth or config file; None on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        pass
    # kilo.jsonc tolerates comments: strip block comments and whole-line
    # // comments, then retry. Whole-line matching keeps "https://" URLs
    # inside string values intact.
    try:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"(?m)^\s*//.*$", "", text)
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _resolve_api_key(api_key_env: str) -> str:
    """Resolve the DeepSeek API key.

    Precedence:
      1. The named environment variable (e.g. DEEPSEEK_API_KEY).
      2. Kilo's own auth.json in any known location and key format
         (~/.local/share/kilo/auth.json, ~/.config/kilo/auth.json,
         ~/.kilo/auth.json).
      3. Kilo's config files (providers.deepseek.apiKey).
    Returns "" when none is available.
    """
    key = (os.environ.get(api_key_env) or "").strip()
    if key:
        return key
    for path in _KILO_AUTH_CANDIDATES + _KILO_CONFIG_CANDIDATES:
        data = _load_kilo_json(path)
        if data:
            key = _extract_deepseek_key(data)
            if key:
                return key
    return ""


class DeepSeekBackend(Backend):
    name = "deepseek"

    def __init__(self, api_key_env: str = "DEEPSEEK_API_KEY", model: str = "deepseek-chat",
                 *, base_url: str = "https://api.deepseek.com", api_key: str | None = None,
                 name: str = "deepseek", timeout_s: float = 30.0, max_retries: int = 4,
                 backoff_s: list[float] | None = None):
        if OpenAI is None:
            raise RuntimeError("openai>=1.0 is required for the deepseek backend")
        key = api_key or _resolve_api_key(api_key_env)
        if not key:
            raise RuntimeError(f"missing environment variable {api_key_env}")
        self._client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.base_url = base_url
        self.name = name
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
                    truncated=completion.choices[0].finish_reason == "length",
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