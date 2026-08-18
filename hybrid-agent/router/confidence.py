"""Confidence scoring for ambiguous tasks (ARCHITECTURE.md §3.3)."""

import math
import re
from dataclasses import dataclass

_AMBIGUITY_WORDS = re.compile(r"\b(maybe|somehow|not sure|perhaps|kind of|roughly)\b", re.I)
_IMPERATIVE = re.compile(r"\b(add|fix|remove|rename|update|create|write|implement|refactor|change|extend)\b", re.I)
_SYMBOL_REF = re.compile(r"[\w./-]+\.(py|js|ts|go|rs|java|rb|php|c|h|cpp|hpp)\b|`[\w_.]+`|#\w+")
_FILE_REF = re.compile(r"[\w./-]+\.(py|js|ts|go|rs|java|rb|php|c|h|cpp|hpp)\b", re.I)


@dataclass(frozen=True)
class Weights:
    clarity: float = 0.30
    specificity: float = 0.25
    size_penalty: float = 0.15
    history_bonus: float = 0.20
    novelty: float = 0.10


@dataclass(frozen=True)
class MemoryView:
    """Minimal read-side of task memory used by the router."""

    similar_task_success_rate: float = 0.0   # 0..1; 0 = no history
    seen_ngrams: set[tuple[str, ...]] = frozenset()

    @property
    def has_history(self) -> bool:
        return self.similar_task_success_rate > 0.0


def _clarity(task: str) -> float:
    score = 0.5
    if _IMPERATIVE.search(task):
        score += 0.25
    if _SYMBOL_REF.search(task):
        score += 0.15
    if len(task.split()) >= 12:          # a detailed instruction is usually clearer
        score += 0.10
    if _AMBIGUITY_WORDS.search(task):
        score -= 0.30
    return max(0.0, min(1.0, score))


def _specificity(task: str) -> float:
    refs = len(_FILE_REF.findall(task))
    if refs >= 2:
        return 1.0
    if refs == 1:
        return 0.8
    return 0.4 if _SYMBOL_REF.search(task) else 0.1


def _size_penalty(context_chars: int, capacity_chars: int = 12_000) -> float:
    """Penalize huge pasted contexts: 1.0 at small sizes → 0.0 at capacity."""
    return max(0.0, 1.0 - min(context_chars, capacity_chars) / capacity_chars)


def _novelty(task: str, memory: MemoryView) -> float:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", task)]
    if len(words) < 3:
        return 0.5
    ngrams = {tuple(words[i : i + 3]) for i in range(len(words) - 2)}
    unseen = ngrams - memory.seen_ngrams
    return min(1.0, len(unseen) / max(1, len(ngrams)))


def score(
    task: str,
    context_chars: int,
    memory: MemoryView,
    weights: Weights = Weights(),
) -> float:
    """Return estimated probability that the local model can handle this task."""
    if not task.strip():
        return 0.0

    clarity = _clarity(task)
    specificity = _specificity(task)
    size_pen = _size_penalty(context_chars)
    history = memory.similar_task_success_rate if memory.has_history else 0.4
    novelty = _novelty(task, memory)

    raw = (
        weights.clarity * clarity
        + weights.specificity * specificity
        + weights.size_penalty * size_pen
        + weights.history_bonus * history
        - weights.novelty * novelty
    )
    return float(max(0.0, min(1.0, math.sqrt(raw * (2 - raw)))))  # smooth to avoid saturation