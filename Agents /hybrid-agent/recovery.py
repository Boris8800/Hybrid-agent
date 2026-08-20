"""recovery.py — formal Task Recovery Manager.

Centralizes ALL failure recovery through one policy, instead of leaving it
scattered across supervise.py / verify / parallel.py / gitops.py:

    Failure
       ▼
    FailureClassifier
       ├── transient / model / context / tool / verification
       ├── contract / security / infrastructure / unknown
       ▼
    RecoveryManager
       ├── retry / compact / switch_model / ask_supervisor
       ├── rollback / resume / escalate

The manager is deterministic, bounded (every action has a hard attempt cap),
state-aware (records every attempt in the durable TaskState + OperationLog),
and invariant-preserving: recovery can never manufacture approval. Terminal
facts (FAILED, REJECTED, APPLIED, DEPLOYED) are respected — e.g. a model
failure after DEPLOYED cannot roll back to DEPLOYED = false; the state machine
blocks it.

Invariant guarantee: recovery actions NEVER change task facts. They only
decide WHAT to try next; the state machine still gates every transition, so
recovery cannot produce TRUNCATED→APPROVED, UNVERIFIED→DEPLOY_AUTHORIZED, or
FAILED→DEPLOYED.

Stdlib only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class FailureClass(str, Enum):
    TRANSIENT = "transient"              # network blip, timeout, LM Studio restart
    MODEL = "model"                      # model died, garbage output, bad verdict
    CONTEXT = "context"                  # context RED, window cutoff, compaction failure
    TOOL = "tool"                        # terminal tool failed, command blocked
    VERIFICATION = "verification"        # tests/lint/build failed
    CONTRACT = "contract"                # task unclear, scope violation
    SECURITY = "security"                # bound violation, unsafe command, secrets
    INFRASTRUCTURE = "infrastructure"    # server down, embedding endpoint dead, git broken
    UNKNOWN = "unknown"


# Keywords that map a failure reason/exception to a class. Checks happen in
# order; the first match wins (most specific first).
_CLASSIFIER_RULES: list[tuple[FailureClass, tuple[str, ...]]] = [
    (FailureClass.SECURITY, (
        "bound", "unsafe", "dangerous", "guardrail", "violation",
        "secret", "pii", "never_do", "danger zone",
    )),
    (FailureClass.CONTEXT, (
        "context_red", "context red", "context_window_reached", "context budget",
        "compact", "context_limit", "window", "overflow", "unbalanced_code_fence",
    )),
    (FailureClass.VERIFICATION, (
        "verify", "test failed", "build failed", "lint", "regression",
        "scope_violation", "audit", "journey", "exit 1", "exit_code 1",
        "check failed", "failed after", "still truncated",
    )),
    (FailureClass.TOOL, (
        "terminal tool", "tool blocked", "tool failed", "allowlist", "run:",
        "command not allowlisted", "tool error",
    )),
    (FailureClass.INFRASTRUCTURE, (
        "connection", "unreachable", "dns", "socket", "server down",
        "no models loaded", "endpoint", "http 5", "http 502", "http 503",
        "refused", "git", "network", "timeout",
    )),
    (FailureClass.MODEL, (
        "model", "qwen", "deepseek", "gemma", "generation failed",
        "garbage", "degenerat", "local failed", "cloud", "provider",
        "stream failed", "retry failed", "review failed", "model_error",
    )),
    (FailureClass.CONTRACT, (
        "clarification", "unclear", "ambiguous", "contract", "refusal",
        "requires", "missing", "task",
    )),
    (FailureClass.TRANSIENT, (
        "temporary blip", "retry", "backoff", "interrupted",
    )),
]


def classify_failure(reason: str = "", exception: Any = None) -> FailureClass:
    """Map a failure (reason string and/or exception) to a FailureClass.

    Deterministic keyword matching; unknown failures are UNKNOWN (which the
    RecoveryManager treats conservatively: escalate, never auto-retry into
    something dangerous).
    """
    text = (reason or "") + " " + str(exception or "")
    low = text.lower()
    for cls, keys in _CLASSIFIER_RULES:
        for k in keys:
            if k in low:
                return cls
    return FailureClass.UNKNOWN


# ---------------------------------------------------------------------------
# Recovery actions
# ---------------------------------------------------------------------------

class RecoveryAction(str, Enum):
    RETRY = "retry"                    # same thing again, bounded attempts
    COMPACT = "compact"                # shrink context, then retry
    SWITCH_MODEL = "switch_model"      # use the other model (local <-> cloud)
    ASK_SUPERVISOR = "ask_supervisor"  # ask DeepSeek how to proceed
    ROLLBACK = "rollback"              # undo the step (no-op if nothing applied)
    RESUME = "resume"                  # restart from durable state
    ESCALATE = "escalate"              # stop and report to the operator


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: str = ""
    attempts_used: int = 0
    attempts_max: int = 0
    detail: str = ""

    def __str__(self) -> str:
        return (f"RecoveryDecision({self.action.value}, "
                f"{self.attempts_used}/{self.attempts_max}, {self.reason})")


# Per-class policy: max attempts before escalation + what to try.
@dataclass
class RecoveryPolicy:
    max_attempts: dict = field(default_factory=lambda: {
        FailureClass.TRANSIENT: 3,
        FailureClass.MODEL: 1,
        FailureClass.CONTEXT: 1,
        FailureClass.TOOL: 2,
        FailureClass.VERIFICATION: 1,
        FailureClass.CONTRACT: 0,       # never auto-retry contract problems
        FailureClass.SECURITY: 0,       # never auto-retry security violations
        FailureClass.INFRASTRUCTURE: 2,
        FailureClass.UNKNOWN: 0,        # conservative: escalate
    })
    retry_delay_s: float = 1.0

    def max_for(self, cls: FailureClass) -> int:
        return int(self.max_attempts.get(cls, 0))


class RecoveryManager:
    """One policy for every failure in the system.

    Usage:
        mgr = RecoveryManager()
        decision = mgr.decide("local failed (connection refused)", attempts=2)
        if decision.action == RecoveryAction.RETRY: ...

    The manager NEVER changes task facts — it only decides the next attempt.
    Callers execute the action through the normal state machine, so every
    invariant (no TRUNCATED→APPROVED, no UNVERIFIED→DEPLOY_AUTHORIZED, no
    duplicate operations) stays enforced at the transition level.
    """

    def __init__(self, policy: Optional[RecoveryPolicy] = None,
                 state=None):  # optional durable TaskState for recording
        self.policy = policy or RecoveryPolicy()
        self.state = state
        self._attempts: dict[tuple, int] = {}

    # -- bookkeeping --------------------------------------------------------

    def _attempt_key(self, cls: FailureClass, scope: str) -> tuple:
        return (cls.value, scope)

    def attempts(self, cls: FailureClass, scope: str) -> int:
        return int(self._attempts.get(self._attempt_key(cls, scope), 0))

    def record_attempt(self, cls: FailureClass, scope: str) -> int:
        k = self._attempt_key(cls, scope)
        self._attempts[k] = self._attempts.get(k, 0) + 1
        n = self._attempts[k]
        if self.state is not None:
            try:
                self.state.record_recovery(ok=False)
                self.state.error(f"recovery: {cls.value} failure in {scope} "
                                 f"(attempt {n})")
                self.state.save()
            except Exception:  # noqa: BLE001 - recording must never break recovery
                pass
        return n

    # -- decision -----------------------------------------------------------

    def decide(self, reason: str = "", exception: Any = None,
               scope: str = "task", attempts: int | None = None,
               cls: FailureClass | None = None) -> RecoveryDecision:
        """Decide the next recovery action for a failure.

        Deterministic ladder per class:
          * CONTRACT / SECURITY -> ESCALATE immediately (never auto-retry
            what needs a human or a contract change);
          * UNKNOWN -> ESCALATE (conservative);
          * VERIFICATION -> ASK_SUPERVISOR (DeepSeek decides fix vs rollback);
          * CONTEXT -> COMPACT then RETRY (attempt 1), else ESCALATE;
          * MODEL -> SWITCH_MODEL on attempt 1, else ESCALATE;
          * TRANSIENT / INFRASTRUCTURE / TOOL -> RETRY with bounded attempts
            and backoff, else ESCALATE.
        """
        cls = cls or classify_failure(reason, exception)
        attempts = attempts if attempts is not None else self.attempts(cls, scope)
        max_attempts = self.policy.max_for(cls)

        if cls in (FailureClass.CONTRACT, FailureClass.SECURITY, FailureClass.UNKNOWN):
            return RecoveryDecision(RecoveryAction.ESCALATE, reason,
                                    attempts_used=attempts, attempts_max=max_attempts,
                                    detail=f"{cls.value} failures need operator action")

        if attempts >= max_attempts:
            return RecoveryDecision(RecoveryAction.ESCALATE, reason,
                                    attempts_used=attempts, attempts_max=max_attempts,
                                    detail=f"{cls.value} attempts exhausted")

        if cls == FailureClass.VERIFICATION:
            return RecoveryDecision(RecoveryAction.ASK_SUPERVISOR, reason,
                                    attempts_used=attempts, attempts_max=max_attempts,
                                    detail="verification failures go to the supervisor")

        if cls == FailureClass.CONTEXT:
            if attempts == 0:
                return RecoveryDecision(RecoveryAction.COMPACT, reason,
                                        attempts_used=0, attempts_max=max_attempts,
                                        detail="compact the context and retry once")
            return RecoveryDecision(RecoveryAction.ESCALATE, reason,
                                    attempts_used=attempts, attempts_max=max_attempts,
                                    detail="compaction did not fit the window")

        if cls == FailureClass.MODEL:
            if attempts == 0:
                return RecoveryDecision(RecoveryAction.SWITCH_MODEL, reason,
                                        attempts_used=0, attempts_max=max_attempts,
                                        detail="model failed — try the other model once")
            return RecoveryDecision(RecoveryAction.ESCALATE, reason,
                                    attempts_used=attempts, attempts_max=max_attempts,
                                    detail="both models failed")

        # TRANSIENT / INFRASTRUCTURE / TOOL: bounded retry with backoff.
        return RecoveryDecision(RecoveryAction.RETRY, reason,
                                attempts_used=attempts, attempts_max=max_attempts,
                                detail=f"retry (backoff {self.policy.retry_delay_s}s)")

    def sleep_before(self, decision: RecoveryDecision) -> None:
        """Bounded backoff before executing a RETRY decision."""
        if decision.action == RecoveryAction.RETRY and decision.attempts_used > 0:
            time.sleep(min(self.policy.retry_delay_s * decision.attempts_used, 8.0))

    # -- high-level helpers -------------------------------------------------

    def handle_failure(self, reason: str = "", exception: Any = None,
                       scope: str = "task") -> RecoveryDecision:
        """Record + decide in one call. Callers execute the returned action.

        The first occurrence is decided at attempts=0 (so a MODEL failure
        yields SWITCH_MODEL immediately), then recorded as attempt 1."""
        cls = classify_failure(reason, exception)
        n = self.attempts(cls, scope)
        decision = self.decide(reason, exception, scope=scope, attempts=n, cls=cls)
        self.record_attempt(cls, scope)
        return decision

    def verify_invariants(self) -> list[str]:
        """Self-check: the recovery policy must never manufacture approval.
        Returns a list of violated invariants (empty = safe)."""
        violations = []
        # ESCALATE/ROLLBACK decisions never carry approval semantics.
        for cls in FailureClass:
            d = self.decide("", cls=cls, attempts=self.policy.max_for(cls))
            if d.action in (RecoveryAction.RETRY, RecoveryAction.COMPACT,
                            RecoveryAction.SWITCH_MODEL, RecoveryAction.ASK_SUPERVISOR) \
                    and self.policy.max_for(cls) == 0:
                violations.append(f"{cls.value}: escalation class decided "
                                  f"{d.action.value} with 0 max attempts")
        return violations


# Convenience singleton — the whole system uses one policy.
default_recovery = RecoveryManager()
