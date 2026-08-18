"""Deterministic task classification into archetypes (see ARCHITECTURE.md §3)."""

import re
from dataclasses import dataclass

# Priority order matters: DeepSeek archetypes (A1–A5) beat local ones (A6–A10).
DEEPSEEK_ARCHETYPES = {"A1", "A2", "A3", "A4", "A5"}
LOCAL_ARCHETYPES = {"A6", "A7", "A8", "A9", "A10"}

# (archetype, priority, compiled pattern)
_PATTERNS: list[tuple[str, int, re.Pattern[str]]] = [
    ("A1", 1, re.compile(r"design|architecture|scaffold|monorepo|project structure|folder layout", re.I)),
    ("A2", 2, re.compile(r"refactor|migrate|repository pattern|across \d+ files|reorganize", re.I)),
    ("A3", 3, re.compile(r"schema|data model|entity relationship|migration design|database design", re.I)),
    ("A4", 4, re.compile(r"concurren|deadlock|lock-free|race condition|algorithm|complexity|prove|invariant", re.I)),
    ("A5", 5, re.compile(r"why|root cause|intermittent|debug in prod|investigate|crash log", re.I)),
    ("A6", 6, re.compile(r"add validation|fix .*function|update .*handler|change the return type of", re.I)),
    ("A7", 7, re.compile(r"boilerplate|template|generate|scaffold .*file|add getters?/setters?", re.I)),
    ("A8", 8, re.compile(r"tests? for|unit tests|test coverage for", re.I)),
    ("A9", 9, re.compile(r"lint|mypy|type error|compile error|unused import|formatting", re.I)),
    ("A10", 10, re.compile(r"docstring|comment|README|rename variable", re.I)),
]


@dataclass(frozen=True)
class Classification:
    archetypes: list[str]      # all matches, highest priority first
    primary: str               # winning archetype per priority ordering
    route: str                 # "local" | "deepseek" | "ambiguous"

    @property
    def is_deepseek_pinned(self) -> bool:
        return self.primary in DEEPSEEK_ARCHETYPES

    @property
    def is_local_pinned(self) -> bool:
        return self.primary in LOCAL_ARCHETYPES


def classify(task: str) -> Classification:
    """Return the highest-priority archetype match for a task string."""
    matched: list[tuple[str, int]] = []
    for archetype, priority, pattern in _PATTERNS:
        if pattern.search(task):
            matched.append((archetype, priority))

    matched.sort(key=lambda item: item[1])  # lower priority number wins
    archetypes = [a for a, _ in matched]

    if not archetypes:
        return Classification([], "A11", "ambiguous")

    primary = archetypes[0]
    if primary in DEEPSEEK_ARCHETYPES:
        route = "deepseek"
    elif primary in LOCAL_ARCHETYPES:
        route = "local"
    else:
        route = "ambiguous"
    return Classification(archetypes, primary, route)