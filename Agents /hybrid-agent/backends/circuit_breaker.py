"""Rolling failure-rate circuit breaker (ARCHITECTURE.md §8.3)."""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    name: str
    error_ceiling: float = 0.40
    window_size: int = 20
    cooldown_s: float = 60.0
    _outcomes: deque[bool] = field(default_factory=deque)  # True = failure
    _tripped_at: float = 0.0

    def record(self, failed: bool) -> None:
        self._outcomes.append(failed)
        if len(self._outcomes) > self.window_size:
            self._outcomes.popleft()

    @property
    def open(self) -> bool:
        if len(self._outcomes) < self.window_size:
            return False
        rate = sum(self._outcomes) / len(self._outcomes)
        return rate > self.error_ceiling

    def __repr__(self) -> str:
        rate = sum(self._outcomes) / len(self._outcomes) if self._outcomes else 0.0
        return f"<CircuitBreaker {self.name} rate={rate:.2f} open={self.open}>"