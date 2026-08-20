"""memory.py — Persistent task memory for the hybrid agent router.

Records one outcome per completed task (route, verdict, quality) and serves
read-side MemoryView objects, so the confidence scorer and the adaptive
threshold learn from REAL history instead of the empty placeholders they
previously saw. Stdlib only. Data lives in memory/tasks.json next to this
file unless the "memory" config section overrides the root. Only the most
recent `max_records` entries are kept to bound file size.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from router.confidence import MemoryView

MAX_RECORDS = 200
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _ngrams(task: str, n: int = 3) -> set[tuple[str, ...]]:
    """Lower-cased word n-grams of a task; the same tokenization the router uses."""
    words = [w.lower() for w in _WORD_RE.findall(task or "")]
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


@dataclass
class TaskRecord:
    """One completed task outcome, persisted for the router's memory."""

    task: str
    ts: float
    route: str
    verdict: str          # "APPROVED" | "FIX_REQUIRED" | "REJECTED" | fallback reason
    quality: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict == "APPROVED"


class TaskMemory:
    """Persistent record of completed tasks; read side for the router."""

    def __init__(self, root: str | None = None, max_records: int = MAX_RECORDS):
        self.root = Path(root) if root else Path(__file__).resolve().parent / "memory"
        self.path = self.root / "tasks.json"
        self.max_records = max_records

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save(self, records: list[dict]) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        except OSError:
            pass

    def record(self, rec: TaskRecord) -> None:
        """Append one outcome and prune to the most recent max_records."""
        records = self._load()
        records.append(asdict(rec))
        self._save(records[-self.max_records:])

    def memory_view(self, task: str) -> MemoryView:
        """Build a MemoryView for `task` from real history.

        similar_task_success_rate is the fraction of APPROVED records among
        tasks sharing at least one trigram with `task` (0.0 when there is no
        matching history). seen_ngrams is the union of every recorded task's
        trigrams, so novelty has something real to compare against.
        """
        records = self._load()
        if not records:
            return MemoryView(seen_ngrams=frozenset())
        current = _ngrams(task)
        similar = [r for r in records if current & _ngrams(r.get("task", ""))]
        rate = 0.0
        if similar:
            rate = sum(1 for r in similar if r.get("verdict") == "APPROVED") / len(similar)
        seen: set = set()
        for r in records:
            seen |= _ngrams(r.get("task", ""))
        return MemoryView(similar_task_success_rate=rate, seen_ngrams=frozenset(seen))

    def count(self) -> int:
        return len(self._load())
