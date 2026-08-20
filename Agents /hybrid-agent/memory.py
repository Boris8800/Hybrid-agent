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
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from embed import DEFAULT_SIMILARITY_THRESHOLD, cosine
from router.confidence import MemoryView

MAX_RECORDS = 200
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _atomic_write(path: Path, text: str) -> None:
    """Write a file atomically (temp + os.replace) so concurrent sessions can
    never observe or leave a torn file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

# Domain keywords used by the consolidation pass to bucket task history.
_DOMAIN_KEYWORDS = {
    "auth": ("authentication", "login", " auth ", "password", "session", "token", "oauth"),
    "backend": ("api", "endpoint", "database", "query", "schema", "server", "route",
                "model", "migration"),
    "frontend": ("react", "component", "ui", "css", "tailwind", "vite", "hook", "modal"),
    "testing": ("test", "pytest", "unittest", "coverage", "jest"),
    "infra": ("docker", "deploy", "ci", "kubernetes", "config", "env", "makefile"),
    "refactor": ("refactor", "cleanup", "reorganize", "restructure"),
}


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
    embedding: list = field(default_factory=list)  # local semantic vector, if any

    @property
    def ok(self) -> bool:
        return self.verdict == "APPROVED"


class TaskMemory:
    """Persistent record of completed tasks; read side for the router."""

    def __init__(self, root: str | None = None, max_records: int = MAX_RECORDS,
                 embed=None, embed_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        # Relative roots resolve against the script dir (like CacheManager) so
        # a bare "./memory" or "memory/<project>" is location-independent.
        if root is None:
            self.root = Path(__file__).resolve().parent / "memory"
        else:
            p = Path(root)
            self.root = p if p.is_absolute() else Path(__file__).resolve().parent / p
        self.path = self.root / "tasks.json"
        self.insights_path = self.root / "insights.json"
        self.max_records = max_records
        self.embed = embed          # embed([texts]) -> list[vectors] | None
        self.embed_threshold = embed_threshold

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
            _atomic_write(self.path, json.dumps(records, indent=2))
        except OSError:
            pass

    def _evict(self, records: list[dict]) -> list[dict]:
        """Score-and-evict: when past max_records, drop the LOWEST-value entries
        (scored by recency + task frequency) instead of a naive FIFO cut. Keeps
        frequently-recurring and recently-relevant tasks, forgets one-off stale
        ones. Preserves record order."""
        if len(records) <= self.max_records:
            return records
        now = time.time()
        counts: dict[str, int] = {}
        for r in records:
            counts[r.get("task", "")] = counts.get(r.get("task", ""), 0) + 1
        max_freq = max(counts.values()) or 1

        def score(r: dict) -> float:
            age_days = max(0.0, (now - float(r.get("ts", now))) / 86400.0)
            recency = 1.0 / (1.0 + age_days)          # 1.0 today -> ~0 in a week
            freq = counts.get(r.get("task", ""), 0) / max_freq
            return 0.7 * recency + 0.3 * freq

        keep_idx = sorted(
            i for i, _ in sorted(enumerate(records), key=lambda t: score(t[1]))[-self.max_records:])
        return [records[i] for i in keep_idx]

    def record(self, rec: TaskRecord) -> None:
        """Append one outcome (embedding computed when available) and evict the
        lowest-value entries past the cap."""
        if not rec.embedding and self.embed is not None and rec.task:
            vec = self.embed([rec.task])
            if vec:
                rec.embedding = vec[0]
        records = self._load()
        records.append(asdict(rec))
        self._save(self._evict(records))

    def memory_view(self, task: str) -> MemoryView:
        """Build a MemoryView for `task` from real history.

        Similarity is trigram overlap PLUS, when embeddings are available,
        cosine similarity of the current task against each record's stored
        embedding (threshold-controlled) — so paraphrased tasks are recalled
        correctly instead of scoring 0. similar_task_success_rate is the
        fraction of APPROVED records among similar tasks.
        """
        records = self._load()
        if not records:
            return MemoryView(seen_ngrams=frozenset())
        current = _ngrams(task)
        similar = [r for r in records if current & _ngrams(r.get("task", ""))]
        if self.embed is not None and task:
            vec = self.embed([task])
            cur_vec = vec[0] if vec else None
            if cur_vec:
                for r in records:
                    rv = r.get("embedding")
                    if (isinstance(rv, list) and len(rv) == len(cur_vec) and rv
                            and cosine(cur_vec, rv) >= self.embed_threshold
                            and r not in similar):
                        similar.append(r)
        rate = 0.0
        if similar:
            rate = sum(1 for r in similar if r.get("verdict") == "APPROVED") / len(similar)
        seen: set = set()
        for r in records:
            seen |= _ngrams(r.get("task", ""))
        return MemoryView(similar_task_success_rate=rate, seen_ngrams=frozenset(seen))

    def count(self) -> int:
        return len(self._load())

    # --- consolidation ---------------------------------------------------

    def consolidate(self) -> dict:
        """Synthesize the recorded history into high-level insights: overall and
        recent approval rates, the trend direction, and the strongest task
        domains. Runs on local data only (no model calls). Returns {} when
        there is not enough history."""
        records = self._load()
        if len(records) < 10:
            return {}
        total = len(records)
        approved = sum(1 for r in records if r.get("verdict") == "APPROVED")
        overall = approved / total
        recent, prior = records[-10:], records[:-10]
        recent_rate = sum(1 for r in recent if r.get("verdict") == "APPROVED") / len(recent)
        prior_rate = (sum(1 for r in prior if r.get("verdict") == "APPROVED") / len(prior)
                      if prior else overall)
        if recent_rate > prior_rate + 0.05:
            trend = "improving"
        elif prior_rate > recent_rate + 0.05:
            trend = "declining"
        else:
            trend = "stable"

        domains: dict[str, dict] = {}
        for r in records:
            task_low = (r.get("task") or "").lower()
            for label, kws in _DOMAIN_KEYWORDS.items():
                if any(k in task_low for k in kws):
                    entry = domains.setdefault(label, {"n": 0, "ok": 0})
                    entry["n"] += 1
                    if r.get("verdict") == "APPROVED":
                        entry["ok"] += 1
        strongest = sorted(domains.items(), key=lambda kv: -kv[1]["n"])[:3]

        insights = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "records": total,
            "overall_approval": round(overall, 2),
            "recent_approval": round(recent_rate, 2),
            "prior_approval": round(prior_rate, 2),
            "trend": trend,
            "domains": {label: {"n": v["n"], "approval": round(v["ok"] / v["n"], 2)}
                        for label, v in strongest},
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.insights_path, json.dumps(insights, indent=2))
        except OSError:
            pass
        return insights

    def insights_text(self, max_age_hours: float = 24.0,
                      min_records: int = 10) -> str:
        """Return a compact, prompt-injectable summary of consolidated memory.
        Lazily re-consolidates when the cached insights are stale and enough
        history exists. Returns '' when there is nothing useful yet."""
        try:
            if self.insights_path.is_file():
                data = json.loads(self.insights_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("generated"):
                    age_hours = (time.time() - datetime.fromisoformat(
                        data["generated"]).timestamp()) / 3600.0
                    if age_hours <= max_age_hours:
                        return self._render_insights(data)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        data = self.consolidate()
        return self._render_insights(data) if data else ""

    def _render_insights(self, data: dict) -> str:
        lines = [f"TASK MEMORY (consolidated from {data.get('records', '?')} tasks):"]
        lines.append(
            f"- approval: {int(data.get('overall_approval', 0) * 100)}% overall, "
            f"recent {int(data.get('recent_approval', 0) * 100)}% "
            f"({data.get('trend', 'stable')})")
        dom = data.get("domains") or {}
        if dom:
            parts = [f"{label} {int(v['approval'] * 100)}% ({v['n']} tasks)"
                     for label, v in dom.items()]
            lines.append(f"- strongest areas: {', '.join(parts)}")
        lines.append("Use this history to judge task difficulty and local-model "
                     "reliability; do not mention this block in your output.")
        return "\n".join(lines)


def memory_root_from_cfg(cfg: dict, cwd: str = ".") -> str | None:
    """Resolve the memory root: an explicit memory.root in config wins;
    otherwise auto-scope per project (git top-level name, else cwd basename) so
    learning is project-relevant and never bleeds across repos."""
    configured = (cfg.get("memory") or {}).get("root")
    if configured:
        return configured
    try:
        proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd,
                              capture_output=True, text=True, timeout=10)
        toplevel = proc.stdout.strip() if proc.returncode == 0 else ""
        name = Path(toplevel).name if toplevel else Path(cwd).resolve().name
    except Exception:  # noqa: BLE001 - fall back to the cwd name
        name = Path(cwd).resolve().name
    return f"memory/{name or 'default'}"
