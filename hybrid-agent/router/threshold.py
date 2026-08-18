"""Adaptive threshold tuning (ARCHITECTURE.md §7.4)."""

from dataclasses import dataclass


@dataclass
class ThresholdController:
    threshold: float
    threshold_min: float = 0.60
    threshold_max: float = 0.85
    target_local_rate: float = 0.93
    alpha: float = 0.01

    _successes: int = 0
    _total: int = 0

    def record(self, routed_local: bool, success: bool) -> None:
        self._total += 1
        if routed_local and success:
            self._successes += 1

    def update(self) -> float:
        """Call periodically (e.g., every 50 tasks). Moves threshold toward the
        target local containment band (90–95%)."""
        if self._total == 0:
            return self.threshold

        observed = self._successes / self._total
        error = self.target_local_rate - observed
        self.threshold += self.alpha * error
        self.threshold = max(self.threshold_min, min(self.threshold_max, self.threshold))

        # Reset the observation window.
        self._successes = 0
        self._total = 0
        return self.threshold

    def decide(self, confidence: float) -> str:
        return "local" if confidence >= self.threshold else "deepseek"