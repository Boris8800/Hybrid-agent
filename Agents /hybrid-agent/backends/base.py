"""Backend abstraction (ARCHITECTURE.md §10)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelRequest:
    system: str
    user: str
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout_s: float = 30.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    raw: str = ""                 # raw content for diagnostics
    token_usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = ""
    truncated: bool = False       # True when output was cut off mid-generation
    truncate_reason: str = ""     # why: finish_reason / max_tokens / stream_eof / fence


class BackendError(Exception):
    """Base error for transport/model failures."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class Backend(ABC):
    """All model backends implement this interface."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        ...