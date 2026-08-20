"""task_state.py — DURABLE TASK STATE (the contract becomes state, not context).

Makes the Task Contract the central, permanent source of truth. The
conversation is temporary working memory; this module is durable task state
that survives compaction, model switching, Qwen/DeepSeek failure, process
restart, retry, verification, and parallel execution.

Pieces:
  TaskState         durable JSON record (task_id, status, plan, files, tests,
                    supervision, recovery, evidence, final)
  TaskStateMachine  strict legal-transition table — illegal transitions raise
                    IllegalTransition at the CODE level (e.g. FAILED -> DEPLOYED,
                    TRUNCATED -> APPROVED are impossible)
  EvidenceLedger    every important claim gets an evidence object (E-001..):
                    command, exit_code, output_hash, timestamp, files, source
  ModelCapabilities formal capability contract — the orchestrator asks
                    "can this model perform this role?" never "is this Qwen?"
  BatchTransaction  parallel batches have an explicit transaction state
                    (SUCCESS / PARTIAL_FAILURE / FAILED) — 3/4 steps OK is
                    PARTIAL_FAILURE, never silent success
  OperationLog      idempotency: each significant operation has an ID and is
                    executed at most once (npm install, migration, git commit,
                    file generation, deployment)
  GENERATED / REVIEWED / VERIFIED / APPROVED / APPLIED are distinct states —
  generated code != approved code != passing tests != applied files.

Stdlib only. JSON-serializable for restart/resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    # Generation states — distinct facts, never collapsed into one success flag.
    PLANNING = "PLANNING"
    IMPLEMENTING = "IMPLEMENTING"
    GENERATED = "GENERATED"          # Qwen produced code
    REVIEWING = "REVIEWING"          # DeepSeek is reviewing
    FIXING = "FIXING"                # fixes being applied after FIX_REQUIRED
    REVIEWED = "REVIEWED"            # a verdict exists (approved or fix required)
    VERIFYING = "VERIFYING"          # test/lint/verify stage running
    VERIFIED = "VERIFIED"            # tests passed
    REGRESSION_VERIFIED = "REGRESSION_VERIFIED"  # full-suite regression passed
    APPROVED = "APPROVED"            # review + verification + audit accepted
    APPLIED = "APPLIED"              # files written to disk
    DEPLOY_AUTHORIZED = "DEPLOY_AUTHORIZED"  # trust boundary crossed explicitly
    PUSHED = "PUSHED"
    DEPLOYED = "DEPLOYED"
    FAILED = "FAILED"                # terminal
    REJECTED = "REJECTED"            # terminal (supervisor REJECTED the work)
    # Never states: TRUNCATED / INCOMPLETE cannot be APPROVED — the machine
    # simply has no edge from them (see _LEGAL below).


# Legal transition table. Anything not listed raises IllegalTransition.
_LEGAL: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PLANNING: {TaskStatus.IMPLEMENTING, TaskStatus.FAILED, TaskStatus.REJECTED},
    TaskStatus.IMPLEMENTING: {TaskStatus.GENERATED, TaskStatus.FAILED, TaskStatus.REJECTED},
    TaskStatus.GENERATED: {TaskStatus.REVIEWING, TaskStatus.FIXING, TaskStatus.FAILED},
    TaskStatus.REVIEWING: {TaskStatus.REVIEWED, TaskStatus.FAILED},
    TaskStatus.REVIEWED: {TaskStatus.FIXING, TaskStatus.VERIFYING, TaskStatus.REJECTED, TaskStatus.FAILED},
    TaskStatus.FIXING: {TaskStatus.GENERATED, TaskStatus.FAILED},
    TaskStatus.VERIFYING: {TaskStatus.VERIFIED, TaskStatus.FIXING, TaskStatus.FAILED},
    TaskStatus.VERIFIED: {TaskStatus.REGRESSION_VERIFIED, TaskStatus.VERIFYING, TaskStatus.APPROVED, TaskStatus.FAILED},
    TaskStatus.REGRESSION_VERIFIED: {TaskStatus.APPROVED, TaskStatus.VERIFYING, TaskStatus.FAILED},
    TaskStatus.APPROVED: {TaskStatus.APPLIED, TaskStatus.FAILED},
    TaskStatus.APPLIED: {TaskStatus.DEPLOY_AUTHORIZED, TaskStatus.PUSHED, TaskStatus.FAILED},
    TaskStatus.DEPLOY_AUTHORIZED: {TaskStatus.PUSHED, TaskStatus.DEPLOYED, TaskStatus.FAILED},
    TaskStatus.PUSHED: {TaskStatus.DEPLOYED, TaskStatus.FAILED},
    TaskStatus.DEPLOYED: set(),  # terminal success
    TaskStatus.FAILED: set(),    # terminal — no edges out (FAILED -> DEPLOYED impossible)
    TaskStatus.REJECTED: set(),  # terminal — no edges out
}


class IllegalTransition(Exception):
    """Raised when a state transition is not allowed by the machine."""


class TaskStateMachine:
    """Strict state machine: illegal transitions are impossible at code level."""

    def __init__(self, initial: TaskStatus = TaskStatus.PLANNING):
        self.status = initial

    def can_transition(self, new: TaskStatus) -> bool:
        return new in _LEGAL.get(self.status, set())

    def transition(self, new: TaskStatus) -> TaskStatus:
        if new == self.status:
            return self.status  # idempotent no-op
        if not self.can_transition(new):
            raise IllegalTransition(
                f"illegal transition: {self.status.value} -> {new.value} "
                f"(no edge in the state machine)")
        self.status = new
        return self.status


# ---------------------------------------------------------------------------
# Evidence ledger
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    COMMAND = "COMMAND"          # shell command with exit code
    TEST = "TEST"                # test result
    FILE = "FILE"                # file written / changed
    REVIEW = "REVIEW"            # a review verdict
    VERIFY = "VERIFY"            # verification stage result
    AUDIT = "AUDIT"              # adversarial auditor result
    REGRESSION = "REGRESSION"    # full-suite regression
    OTHER = "OTHER"


@dataclass
class Evidence:
    etype: str = EvidenceType.OTHER.value
    command: str = ""
    exit_code: Optional[int] = None
    output_hash: str = ""
    timestamp: float = 0.0
    files_affected: list = field(default_factory=list)
    source: str = ""             # provider / tool / stage that produced it
    summary: str = ""            # human-readable one-liner
    evidence_id: str = ""        # E-001 ... assigned by the ledger

    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = EvidenceLedger.next_id()

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceLedger:
    """Append-only ledger of every important claim. The supervisor receives
    evidence IDs and their facts, so approvals are auditable (E-019: npm test,
    exit 0, hash ..., files src/auth/*)."""

    _counter = 0

    @classmethod
    def next_id(cls) -> str:
        cls._counter += 1
        return f"E-{cls._counter:03d}"

    def __init__(self, entries: Optional[list] = None):
        self.entries: list[Evidence] = [e if isinstance(e, Evidence)
                                        else Evidence(**e) for e in (entries or [])]
        if self.entries:
            # Resume the counter past the highest existing id.
            for e in self.entries:
                m = int(e.evidence_id.split("-")[1]) if "-" in e.evidence_id else 0
                self._counter = max(self._counter, m)

    def add(self, etype: str = EvidenceType.OTHER.value, command: str = "",
            exit_code: Optional[int] = None, output: str = "",
            files_affected: Optional[list] = None, source: str = "",
            summary: str = "") -> Evidence:
        ev = Evidence(
            etype=etype, command=command, exit_code=exit_code,
            output_hash=hashlib.sha256((output or "").encode()).hexdigest()[:16],
            timestamp=time.time(), files_affected=list(files_affected or []),
            source=source, summary=summary,
        )
        self.entries.append(ev)
        return ev

    def add_review(self, verdict: str, score: float, source: str = "deepseek") -> Evidence:
        return self.add(EvidenceType.REVIEW.value, command="review",
                        exit_code=0 if verdict == "APPROVED" else 1,
                        source=source,
                        summary=f"review verdict {verdict} ({score:.1f}/10)")

    def get(self, evidence_id: str) -> Optional[Evidence]:
        for e in self.entries:
            if e.evidence_id == evidence_id:
                return e
        return None

    def ids(self) -> list[str]:
        return [e.evidence_id for e in self.entries]

    def render(self, limit: int = 30) -> str:
        """Compact prompt-safe render: E-019 command exit_code hash files."""
        parts = []
        for e in self.entries[-limit:]:
            bits = [e.evidence_id, e.etype]
            if e.command:
                bits.append(f"cmd:{e.command[:60]}")
            if e.exit_code is not None:
                bits.append(f"exit:{e.exit_code}")
            if e.output_hash:
                bits.append(f"hash:{e.output_hash}")
            if e.files_affected:
                bits.append(f"files:{','.join(e.files_affected[:5])}")
            if e.summary:
                bits.append(e.summary[:80])
            parts.append(" ".join(bits))
        return "\n".join(parts) if parts else "(no evidence recorded)"

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]


# ---------------------------------------------------------------------------
# Plan steps / files / tests (durable)
# ---------------------------------------------------------------------------

@dataclass
class StepState:
    id: int
    name: str = ""
    status: str = "pending"        # pending / running / completed / failed / skipped
    output: str = ""
    evidence_id: str = ""
    attempts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileState:
    path: str
    role: str = "required"         # required / modified / protected
    status: str = "pending"        # pending / applied / skipped
    evidence_id: str = ""


@dataclass
class TestState:
    name: str
    status: str = "pending"        # pending / passed / failed
    evidence_id: str = ""


# ---------------------------------------------------------------------------
# Batch transactions (parallel execution)
# ---------------------------------------------------------------------------

class BatchStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"   # some steps failed — never silent success
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class BatchTransaction:
    batch_id: str
    steps: list = field(default_factory=list)   # [{"id","name","status","error"}]
    status: str = BatchStatus.SUCCESS.value

    def mark_step(self, step_id, ok: bool, error: str = ""):
        for s in self.steps:
            if s.get("id") == step_id:
                s["status"] = "success" if ok else "failed"
                if error:
                    s["error"] = error
        self._recompute()

    def _recompute(self):
        if not self.steps:
            self.status = BatchStatus.SUCCESS.value
            return
        statuses = {s.get("status") for s in self.steps}
        if statuses == {"success"}:
            self.status = BatchStatus.SUCCESS.value
        elif "failed" in statuses and "success" in statuses:
            self.status = BatchStatus.PARTIAL_FAILURE.value
        elif statuses == {"failed"}:
            self.status = BatchStatus.FAILED.value
        # else: pending/mixed-with-pending stays as-is

    def is_success(self) -> bool:
        return self.status == BatchStatus.SUCCESS.value

    def is_partial(self) -> bool:
        return self.status == BatchStatus.PARTIAL_FAILURE.value

    @property
    def failed_step_ids(self) -> list:
        return [s.get("id") for s in self.steps if s.get("status") == "failed"]


# ---------------------------------------------------------------------------
# OperationLog — idempotency
# ---------------------------------------------------------------------------

class OperationLog:
    """Every significant operation (apply, push, deploy, migration, npm
    install) gets an ID and runs at most once: 'if already completed, don't
    execute again'. Prevents restarts/resumes from doubling side effects."""

    def __init__(self, records: Optional[dict] = None):
        self.records: dict[str, dict] = dict(records or {})

    def is_completed(self, op_id: str) -> bool:
        rec = self.records.get(op_id)
        return bool(rec and rec.get("status") == "completed")

    def mark_started(self, op_id: str) -> None:
        self.records[op_id] = {"status": "started", "ts": time.time()}

    def mark_completed(self, op_id: str, result: Any = None) -> None:
        self.records[op_id] = {"status": "completed", "ts": time.time(),
                               "result": result}

    def mark_failed(self, op_id: str, error: str = "") -> None:
        self.records[op_id] = {"status": "failed", "ts": time.time(),
                               "error": error}

    def to_dict(self) -> dict:
        return self.records

    @staticmethod
    def make_id(task_id: str, *parts: str) -> str:
        raw = "-".join([task_id, *parts])
        return f"OP-{hashlib.sha256(raw.encode()).hexdigest()[:10]}"


# ---------------------------------------------------------------------------
# ModelCapabilities — "can this model perform this role?"
# ---------------------------------------------------------------------------

@dataclass
class ModelCapabilities:
    """Formal capability contract. The orchestrator asks
    can_perform_role('implementer', 'long') instead of 'is this Qwen?'.
    All fields are best-effort discovered (0/False/'' when unknown)."""
    model_id: str = ""
    context_window: int = 0
    max_output: int = 0
    tool_use: bool = False
    streaming: bool = True
    vision: bool = False
    structured_output: bool = False
    embeddings: bool = False
    reasoning: bool = False
    diff_generation: bool = False
    architecture: str = ""
    tokenizer: str = ""

    @classmethod
    def from_discovery(cls, caps: dict) -> "ModelCapabilities":
        """Build from backends.local_qwen.discover_model_capabilities() output."""
        return cls(
            model_id=caps.get("model_id", ""),
            context_window=int(caps.get("context_window") or 0),
            max_output=int(caps.get("max_output") or 0),
            tool_use=bool(caps.get("tool_use")),
            vision=bool(caps.get("vision")),
            architecture=caps.get("architecture", ""),
            tokenizer=caps.get("tokenizer", ""),
            # Local MLX endpoint is always stream-capable; structured output
            # is not guaranteed on arbitrary local models.
            streaming=True,
            structured_output=False,
            reasoning="qwen" in str(caps.get("architecture", "")).lower(),
        )

    def can_perform_role(self, role: str, requirement: str = "") -> bool:
        """Capability gate: does this model satisfy the role's minimums?
        Unknown capabilities (0/False) are treated conservatively as NOT
        guaranteed — the orchestrator must not assume."""
        if role == "implementer":
            if requirement == "long":
                return self.context_window >= 32768
            return self.context_window > 0
        if role == "supervisor":
            return True  # API supervisors are assumed capable; local ones need tools
        if role == "local_agent":
            return self.context_window > 0 and self.streaming
        if role == "vision":
            return self.vision
        if role == "structured_output":
            return self.structured_output
        if role == "tool_calling":
            return self.tool_use
        return True  # unknown roles are not gated (default-permissive, documented)

    def describe(self) -> str:
        return (f"ModelCapabilities(model={self.model_id or '?'}, "
                f"window={self.context_window}, max_output={self.max_output}, "
                f"tools={self.tool_use}, stream={self.streaming}, "
                f"vision={self.vision}, structured={self.structured_output}, "
                f"embeddings={self.embeddings}, reasoning={self.reasoning}, "
                f"diff={self.diff_generation})")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# TaskState — the durable task record
# ---------------------------------------------------------------------------

class TaskState:
    """Durable task state: the contract as the central source of truth.

    Survives: compaction, model switching, Qwen/DeepSeek failure, process
    restart (save/load), retry, verification, parallel execution.

    Persisted as JSON at {root}/.hybrid-agent/tasks/{task_id}.json (or the
    configured task_state root).
    """

    def __init__(self, task: str, task_id: str = "", root: str = "",
                 acceptance_criteria: Optional[list] = None,
                 plan: Optional[list] = None,
                 files: Optional[list] = None,
                 tests: Optional[list] = None,
                 supervision_level: str = "full",
                 model_implementer: str = "",
                 model_supervisor: str = "",
                 capabilities: Optional[ModelCapabilities] = None):
        self.task_id = task_id or TaskState.make_task_id(task)
        self.user_request = task
        self.root = root
        self.acceptance_criteria = list(acceptance_criteria or [])
        self.machine = TaskStateMachine()
        self.steps: list[StepState] = [StepState(**s) if isinstance(s, dict) else s
                                       for s in (plan or [])]
        self.files: list[FileState] = [FileState(**f) if isinstance(f, dict) else f
                                       for f in (files or [])]
        self.tests: list[TestState] = [TestState(**t) if isinstance(t, dict) else t
                                       for t in (tests or [])]
        self.supervision_level = supervision_level
        self.iterations = 0
        self.verdicts: list[dict] = []          # [{decision, quality, ts}]
        self.recovery_attempts = 0
        self.recovery_failures = 0
        self.escalations = 0
        self.evidence = EvidenceLedger()
        self.operations = OperationLog()
        self.batches: list[dict] = []           # [{batch_id, status, steps}]
        self.final: dict = {"verified": False, "applied": False,
                            "applied_files": [], "deployed": False}
        self.model_implementer = model_implementer
        self.model_supervisor = model_supervisor
        self.capabilities = capabilities or ModelCapabilities()
        self.updated_at = time.time()
        self.created_at = time.time()
        self.last_error = ""

    # -- identity ----------------------------------------------------------

    @staticmethod
    def make_task_id(task: str) -> str:
        h = hashlib.sha256((task or "").encode()).hexdigest()[:12]
        return f"TASK-{h}"

    # -- state machine passthrough -----------------------------------------

    @property
    def status(self) -> TaskStatus:
        return self.machine.status

    def transition(self, new: TaskStatus) -> TaskStatus:
        s = self.machine.transition(new)
        self.updated_at = time.time()
        return s

    def can_transition(self, new: TaskStatus) -> bool:
        return self.machine.can_transition(new)

    # -- durable facts ------------------------------------------------------

    def record_verdict(self, decision: str, quality: float = 0.0,
                       source: str = "deepseek") -> None:
        self.iterations += 1
        self.verdicts.append({"decision": decision, "quality": quality,
                              "ts": time.time()})
        self.evidence.add_review(decision, quality, source=source)

    def record_recovery(self, ok: bool, escalation: bool = False) -> None:
        self.recovery_attempts += 1
        if not ok:
            self.recovery_failures += 1
        if escalation:
            self.escalations += 1

    def mark_step(self, step_id, ok: bool, output: str = "") -> None:
        for s in self.steps:
            if s.id == step_id:
                s.status = "completed" if ok else "failed"
                if output:
                    s.output = output[:2000]
                s.evidence_id = self.evidence.add(
                    EvidenceType.COMMAND.value if not ok else EvidenceType.OTHER.value,
                    command=f"step {step_id}", exit_code=0 if ok else 1,
                    output=output, summary=f"step {step_id} "
                                           f"{'completed' if ok else 'failed'}").evidence_id

    def mark_file(self, path: str, applied: bool = True) -> None:
        for f in self.files:
            if f.path == path:
                f.status = "applied" if applied else "skipped"
                f.evidence_id = self.evidence.add(
                    EvidenceType.FILE.value, command=f"write {path}",
                    exit_code=0 if applied else 1, files_affected=[path],
                    summary=f"{'applied' if applied else 'skipped'} {path}").evidence_id
                return
        self.files.append(FileState(path=path, status="applied" if applied else "skipped"))

    def record_batch(self, batch: BatchTransaction) -> None:
        self.batches.append({"batch_id": batch.batch_id,
                             "status": batch.status,
                             "steps": list(batch.steps)})

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "root": self.root,
            "acceptance_criteria": self.acceptance_criteria,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "files": [asdict(f) for f in self.files],
            "tests": [asdict(t) for t in self.tests],
            "supervision_level": self.supervision_level,
            "iterations": self.iterations,
            "verdicts": self.verdicts,
            "recovery_attempts": self.recovery_attempts,
            "recovery_failures": self.recovery_failures,
            "escalations": self.escalations,
            "evidence": self.evidence.to_list(),
            "operations": self.operations.to_dict(),
            "batches": self.batches,
            "final": self.final,
            "model_implementer": self.model_implementer,
            "model_supervisor": self.model_supervisor,
            "capabilities": self.capabilities.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        ts = cls(
            task=d.get("user_request", ""),
            task_id=d.get("task_id", ""),
            root=d.get("root", ""),
            acceptance_criteria=d.get("acceptance_criteria"),
            plan=d.get("steps") or d.get("plan"),
            files=d.get("files"),
            tests=d.get("tests"),
            supervision_level=d.get("supervision_level", "full"),
            model_implementer=d.get("model_implementer", ""),
            model_supervisor=d.get("model_supervisor", ""),
        )
        ts.machine.status = TaskStatus(d.get("status", "PLANNING"))
        ts.iterations = int(d.get("iterations", 0))
        ts.verdicts = list(d.get("verdicts", []) or [])
        ts.recovery_attempts = int(d.get("recovery_attempts", 0))
        ts.recovery_failures = int(d.get("recovery_failures", 0))
        ts.escalations = int(d.get("escalations", 0))
        ts.evidence = EvidenceLedger(d.get("evidence", []) or [])
        ts.operations = OperationLog(d.get("operations", {}) or {})
        ts.batches = list(d.get("batches", []) or [])
        ts.final = dict(d.get("final", {}) or {})
        ts.created_at = float(d.get("created_at", time.time()))
        ts.updated_at = float(d.get("updated_at", time.time()))
        ts.last_error = d.get("last_error", "")
        caps = d.get("capabilities") or {}
        ts.capabilities = ModelCapabilities(**{k: v for k, v in caps.items()
                                               if k in ModelCapabilities.__dataclass_fields__})
        return ts

    def state_path(self) -> str:
        base = os.environ.get("HYBRID_TASK_STATE_DIR", "")
        if base:
            root_dir = base
        else:
            root_dir = os.path.join(self.root or ".", ".hybrid-agent", "tasks")
        return os.path.join(root_dir, f"{self.task_id}.json")

    def save(self) -> str:
        path = self.state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, task: str, root: str = "") -> Optional["TaskState"]:
        """Load by task text (same task id) from the durable store."""
        tid = cls.make_task_id(task)
        base = os.environ.get("HYBRID_TASK_STATE_DIR", "")
        if base:
            path = os.path.join(base, f"{tid}.json")
        else:
            path = os.path.join(root or ".", ".hybrid-agent", "tasks", f"{tid}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return cls.from_dict(json.load(fh))
        except Exception:  # noqa: BLE001 - corrupt state must not break runs
            return None

    # -- resume -------------------------------------------------------------

    def resume_summary(self) -> str:
        done = [f"  ✓ {s.name or ('step ' + str(s.id))}" for s in self.steps
                if s.status == "completed"]
        cur = next((f"  → {s.name or ('step ' + str(s.id))}" for s in self.steps
                    if s.status == "pending"), "")
        ev = ", ".join(self.evidence.ids()[-8:]) or "none"
        return (f"TASK {self.task_id} RESUMING\n"
                f"Step: {len([s for s in self.steps if s.status in ('completed','failed')])}"
                f"/{len(self.steps) or '?'}\n"
                f"Previous state: {self.status.value}\n"
                f"Completed:\n" + ("\n".join(done) if done else "  (none)") + "\n"
                f"Current:\n{cur or '  (starting)'}\n"
                f"Previous model: {self.model_implementer or 'unknown'}\n"
                f"Previous context: compacted\n"
                f"Previous evidence: {ev}\n")

    def error(self, msg: str) -> None:
        self.last_error = msg
        self.updated_at = time.time()
