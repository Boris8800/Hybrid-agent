"""LM Studio (OpenAI-compatible) backend with retry logic (ARCHITECTURE.md §8.1)."""

import time
from typing import Any

from backends.base import Backend, BackendError, ModelRequest, ModelResponse

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dev can run router-only without client
    OpenAI = None  # type: ignore


class GemmaBackend(Backend):
    name = "local-gemma"

    def __init__(self, base_url: str, model: str, *, api_key: str = "lm-studio",
                 name: str = "local-gemma", timeout_s: float = 10.0,
                 max_retries: int = 3, backoff_s: list[float] | None = None,
                 cold_start_wait_s: float = 30.0):
        if OpenAI is None:
            raise RuntimeError("openai>=1.0 is required for the local backend")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.name = name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s or [0.5, 1.0, 2.0]
        self.cold_start_wait_s = cold_start_wait_s

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
                )
                text = completion.choices[0].message.content or ""
                truncated = completion.choices[0].finish_reason == "length"
                return ModelResponse(
                    text=text,
                    raw=text,
                    token_usage=getattr(completion, "usage", {}),
                    latency_ms=(time.monotonic() - started) * 1000,
                    backend=self.name,
                    truncated=truncated,
                )
            except Exception as exc:  # noqa: BLE001 - surface as BackendError
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # LM Studio signals "model not loaded yet" with 503.
                if status == 503 and attempt == 0:
                    time.sleep(self.cold_start_wait_s)
                    continue
                delay = self.backoff_s[attempt] if attempt < len(self.backoff_s) else self.backoff_s[-1]
                time.sleep(delay)
                last_error = BackendError(str(exc), retryable=True)

        raise BackendError(
            f"local backend failed after {self.max_retries + 1} attempts: {last_error}"
        )