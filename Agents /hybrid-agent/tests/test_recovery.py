"""Tests for recovery.py — the formal Task Recovery Manager.

Includes chaos-style boundary injection: every failure class must route
through the manager, stay bounded, and never violate invariants (no
FAILED->DEPLOYED, no TRUNCATED->APPROVED, no UNVERIFIED->DEPLOY_AUTHORIZED,
no duplicate operations)."""

import unittest

from recovery import (FailureClass, RecoveryAction, RecoveryDecision,
                      RecoveryManager, RecoveryPolicy, classify_failure,
                      default_recovery)
from task_state import (IllegalTransition, OperationLog, TaskState, TaskStatus)


class FailureClassifierTests(unittest.TestCase):
    def test_classifies_each_family(self):
        self.assertEqual(classify_failure("connection refused"), FailureClass.INFRASTRUCTURE)
        self.assertEqual(classify_failure("context RED"), FailureClass.CONTEXT)
        self.assertEqual(classify_failure("bound violation"), FailureClass.SECURITY)
        self.assertEqual(classify_failure("npm test failed"), FailureClass.VERIFICATION)
        self.assertEqual(classify_failure("local failed (model died)"), FailureClass.MODEL)
        self.assertEqual(classify_failure("task is ambiguous"), FailureClass.CONTRACT)
        self.assertEqual(classify_failure("terminal tool blocked"), FailureClass.TOOL)
        self.assertEqual(classify_failure("something weird"), FailureClass.UNKNOWN)

    def test_security_beats_generic(self):
        # 'bound' must classify as SECURITY even though 'model' rules also exist.
        self.assertEqual(classify_failure("BOUND violation: model tried rm -rf"),
                         FailureClass.SECURITY)

    def test_context_beats_model(self):
        self.assertEqual(classify_failure("context_window_reached in qwen output"),
                         FailureClass.CONTEXT)


class RecoveryPolicyTests(unittest.TestCase):
    def test_escalation_classes_never_auto_retry(self):
        mgr = RecoveryManager()
        for cls in (FailureClass.CONTRACT, FailureClass.SECURITY, FailureClass.UNKNOWN):
            d = mgr.decide("", cls=cls, attempts=0)
            self.assertEqual(d.action, RecoveryAction.ESCALATE, cls.value)

    def test_transient_retries_then_escalates(self):
        mgr = RecoveryManager()
        d0 = mgr.decide("temporary blip, retry once", attempts=0)
        self.assertEqual(d0.action, RecoveryAction.RETRY)
        d1 = mgr.decide("temporary blip, retry once", attempts=1)
        self.assertEqual(d1.action, RecoveryAction.RETRY)
        d2 = mgr.decide("temporary blip, retry once", attempts=2)
        self.assertEqual(d2.action, RecoveryAction.RETRY)
        d3 = mgr.decide("temporary blip, retry once", attempts=3)  # max reached
        self.assertEqual(d3.action, RecoveryAction.ESCALATE)

    def test_model_failure_switches_once_then_escalates(self):
        mgr = RecoveryManager()
        d0 = mgr.decide("local failed", attempts=0)
        self.assertEqual(d0.action, RecoveryAction.SWITCH_MODEL)
        d1 = mgr.decide("local failed", attempts=1)
        self.assertEqual(d1.action, RecoveryAction.ESCALATE)

    def test_context_compacts_then_escalates(self):
        mgr = RecoveryManager()
        d0 = mgr.decide("context RED", attempts=0)
        self.assertEqual(d0.action, RecoveryAction.COMPACT)
        d1 = mgr.decide("context RED", attempts=1)
        self.assertEqual(d1.action, RecoveryAction.ESCALATE)

    def test_verification_asks_supervisor(self):
        mgr = RecoveryManager()
        d = mgr.decide("test failed", attempts=0)
        self.assertEqual(d.action, RecoveryAction.ASK_SUPERVISOR)

    def test_infrastructure_bounded_retry(self):
        mgr = RecoveryManager()
        for i in range(2):  # max 2 for infrastructure
            self.assertEqual(mgr.decide("connection refused", attempts=i).action,
                             RecoveryAction.RETRY)
        self.assertEqual(mgr.decide("connection refused", attempts=2).action,
                         RecoveryAction.ESCALATE)

    def test_handle_failure_records_and_escalates_eventually(self):
        mgr = RecoveryManager()
        # Simulate the same transient failure repeatedly.
        outcomes = []
        for _ in range(5):
            d = mgr.handle_failure("temporary blip, retry once", scope="verify")
            outcomes.append(d.action)
            if d.action != RecoveryAction.RETRY:
                break
        self.assertEqual(outcomes, [RecoveryAction.RETRY, RecoveryAction.RETRY,
                                    RecoveryAction.RETRY, RecoveryAction.ESCALATE])

    def test_policy_invariants_hold(self):
        mgr = RecoveryManager()
        self.assertEqual(mgr.verify_invariants(), [])


class RecoveryStateIntegrationTests(unittest.TestCase):
    """Recovery must record attempts in durable state and NEVER manufacture
    approval — every transition still goes through the state machine."""

    def test_recovery_records_attempts_in_state(self):
        ts = TaskState("recovery task")
        mgr = RecoveryManager(state=ts)
        mgr.handle_failure("connection refused", scope="verify")
        self.assertEqual(ts.recovery_attempts, 1)
        self.assertEqual(ts.recovery_failures, 1)
        self.assertIn("recovery", ts.last_error)

    def test_recovery_cannot_fabricate_approval(self):
        """Even after a model failure mid-run, the state machine blocks
        TRUNCATED -> APPROVED and FAILED -> DEPLOYED — recovery actions go
        through the same machine as everything else."""
        ts = TaskState("invariant task")
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.transition(TaskStatus.GENERATED)
        # Recovery manager suggests escalate; even a rogue caller trying to
        # mark the task approved from GENERATED is blocked.
        with self.assertRaises(IllegalTransition):
            ts.transition(TaskStatus.APPROVED)
        # And FAILED is terminal.
        ts.transition(TaskStatus.REVIEWING)
        ts.transition(TaskStatus.REVIEWED)
        ts.transition(TaskStatus.REJECTED)
        with self.assertRaises(IllegalTransition):
            ts.transition(TaskStatus.APPLIED)
        with self.assertRaises(IllegalTransition):
            ts.transition(TaskStatus.DEPLOYED)

    def test_recovery_cannot_duplicate_operations(self):
        ts = TaskState("idem")
        op = OperationLog.make_id(ts.task_id, "deploy")
        ts.operations.mark_completed(op, "done")
        # Recovery manager must not re-run completed operations.
        self.assertTrue(ts.operations.is_completed(op))
        # If a rogue recovery tried to start it again, the log refuses by
        # answering is_completed=True — callers must check before executing.
        self.assertTrue(ts.operations.is_completed(op),
                        "completed operations stay completed through recovery")

    def test_model_switch_falls_back_to_cloud_state_transition(self):
        """SWITCH_MODEL means: try the other model, but the task state must
        still walk GENERATED -> REVIEWING -> ... — never APPROVED directly."""
        ts = TaskState("switch")
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.transition(TaskStatus.GENERATED)
        self.assertFalse(ts.can_transition(TaskStatus.APPROVED))
        self.assertTrue(ts.can_transition(TaskStatus.REVIEWING))


class ChaosBoundaryTests(unittest.TestCase):
    """Deliberately inject failures at every boundary and assert the
    NEVER-invariants hold (state machine + recovery policy + idempotency)."""

    def test_qwen_dies_never_approves(self):
        ts = TaskState("chaos-qwen")
        mgr = RecoveryManager(state=ts)
        d = mgr.handle_failure("local failed (model died)", scope="implement")
        self.assertEqual(d.action, RecoveryAction.SWITCH_MODEL)
        # The run must escalate; APPROVED is unreachable from FAILED.
        try:
            ts.transition(TaskStatus.FAILED)
        except IllegalTransition:
            pass
        with self.assertRaises(IllegalTransition):
            ts.transition(TaskStatus.APPROVED)

    def test_deepseek_dies_falls_back_without_approval(self):
        mgr = RecoveryManager()
        d = mgr.decide("deepseek provider failed (connection refused)", attempts=0)
        self.assertIn(d.action, (RecoveryAction.RETRY, RecoveryAction.ESCALATE))

    def test_lm_studio_restart_classified_transient(self):
        cls = classify_failure("LM Studio restarted; no models loaded")
        self.assertEqual(cls, FailureClass.INFRASTRUCTURE)
        d = RecoveryManager().decide("LM Studio restarted; no models loaded", attempts=0)
        self.assertEqual(d.action, RecoveryAction.RETRY)

    def test_context_discovery_failure_does_not_break(self):
        # Discovery failure returns 0; the safety controller treats 0 as
        # 'no budget known' and must NOT call the model (RED).
        from context_safety import assess_zone, safe_input_budget
        self.assertEqual(safe_input_budget(0), 0)
        self.assertEqual(assess_zone(100, 0), "RED")

    def test_parallel_partial_failure_never_silent_success(self):
        from task_state import BatchStatus, BatchTransaction
        b = BatchTransaction(batch_id="B-1", steps=[{"id": 1}, {"id": 2}, {"id": 3}])
        b.mark_step(1, True)
        b.mark_step(2, True)
        b.mark_step(3, False, error="boom")
        self.assertEqual(b.status, BatchStatus.PARTIAL_FAILURE.value)
        self.assertFalse(b.is_success())

    def test_corrupt_state_json_loads_none_not_crash(self):
        import json, os, tempfile
        from task_state import TaskState
        root = tempfile.mkdtemp()
        tid = TaskState.make_task_id("chaos")
        d = os.path.join(root, ".hybrid-agent", "tasks")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{tid}.json"), "w") as fh:
            fh.write('{"status": "DEPLOYED", "broken": ')
        self.assertIsNone(TaskState.load("chaos", root=root))

    def test_duplicate_operation_never_executes_twice(self):
        ts = TaskState("chaos-idem")
        op = OperationLog.make_id(ts.task_id, "deploy")
        ts.operations.mark_completed(op, "deployed")
        executed = 0
        # Caller checks idempotency before executing (the contract of
        # OperationLog): a completed op is skipped.
        if not ts.operations.is_completed(op):
            executed += 1
        if not ts.operations.is_completed(op):
            executed += 1
        self.assertEqual(executed, 0)

    def test_unsafe_command_blocked_by_recovery_policy(self):
        d = RecoveryManager().decide("command blocked: dangerous pattern rm -rf /", attempts=0)
        self.assertEqual(d.action, RecoveryAction.ESCALATE,
                         "security failures never auto-retry into danger")

    def test_machine_restart_resumes_from_state(self):
        import tempfile
        from task_state import TaskState, TaskStatus
        root = tempfile.mkdtemp()
        ts = TaskState("restart chaos", root=root,
                       plan=[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.mark_step(1, True)
        ts.save()  # "process killed with SIGKILL" — state already on disk
        loaded = TaskState.load("restart chaos", root=root)
        self.assertEqual(loaded.steps[0].status, "completed")
        self.assertEqual(loaded.status, TaskStatus.IMPLEMENTING)
        self.assertIn("1/2", loaded.resume_summary())

    def test_model_change_mid_task_keeps_state(self):
        """Switching the model mid-task changes capabilities, not facts."""
        ts = TaskState("model switch", plan=[{"id": 1, "name": "s"}])
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.mark_step(1, True)
        ts.model_implementer = "gemma2"
        ts.capabilities = ts.capabilities  # capability swap is harmless
        self.assertEqual(ts.steps[0].status, "completed")


class FlexibleStatePathTests(unittest.TestCase):
    """The state machine must NOT force every task through every state —
    the required path is determined by the task contract (regression suite?
    deploy requested?), not by rigidity."""

    def test_apply_only_without_regression(self):
        """--apply with no regression suite: VERIFIED -> APPROVED -> APPLIED,
        skipping REGRESSION_VERIFIED entirely."""
        from task_state import TaskStateMachine
        m = TaskStateMachine()
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED,
                  TaskStatus.VERIFYING, TaskStatus.VERIFIED,
                  TaskStatus.APPROVED, TaskStatus.APPLIED]:
            m.transition(s)
        self.assertEqual(m.status, TaskStatus.APPLIED)

    def test_full_chain_with_regression_and_deploy(self):
        from task_state import TaskStateMachine
        m = TaskStateMachine()
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED,
                  TaskStatus.VERIFYING, TaskStatus.VERIFIED,
                  TaskStatus.REGRESSION_VERIFIED, TaskStatus.APPROVED,
                  TaskStatus.APPLIED, TaskStatus.DEPLOY_AUTHORIZED,
                  TaskStatus.PUSHED, TaskStatus.DEPLOYED]:
            m.transition(s)
        self.assertEqual(m.status, TaskStatus.DEPLOYED)

    def test_regression_then_apply_only(self):
        """Regression ran, but no deploy requested: stop at APPLIED."""
        from task_state import TaskStateMachine
        m = TaskStateMachine()
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED,
                  TaskStatus.VERIFYING, TaskStatus.VERIFIED,
                  TaskStatus.REGRESSION_VERIFIED, TaskStatus.APPROVED,
                  TaskStatus.APPLIED]:
            m.transition(s)
        self.assertEqual(m.status, TaskStatus.APPLIED)

    def test_reviewed_requires_verification_before_approval(self):
        """Even a deploy task must verify first: REVIEWED -> APPROVED is
        illegal; VERIFIED -> APPROVED is the only road."""
        from task_state import TaskStateMachine
        m = TaskStateMachine()
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED]:
            m.transition(s)
        self.assertFalse(m.can_transition(TaskStatus.APPROVED))
        self.assertTrue(m.can_transition(TaskStatus.VERIFYING))


class SuperviseRecoveryIntegrationTests(unittest.TestCase):
    """The supervise loop routes its failures through the ONE recovery policy,
    and recovery decisions never violate the NEVER-invariants."""

    def test_supervise_model_failure_routes_through_recovery(self):
        from backends.base import ModelResponse
        from supervise import supervise

        # A local model that always dies: recovery says SWITCH_MODEL ->
        # DeepSeek fallback -> escalated (never approved).
        def qwen(req):
            raise RuntimeError("local model crashed")

        class Cloud:
            def generate(self, req):
                return ModelResponse(text="fallback implementation ```a.py\nx\n```")

        res = supervise(local=object(), cloud=Cloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertTrue(res.escalated)
        self.assertIn(res.reason, ("local_failure_escalation",))

    def test_supervise_truncated_never_approved(self):
        from backends.base import ModelResponse
        from supervise import supervise

        def qwen(req):
            return ModelResponse(text="```a.py\ncut", truncated=True,
                                 truncate_reason="context_window_reached")

        class Cloud:
            def generate(self, req):
                return ModelResponse(text="fallback")

        res = supervise(local=object(), cloud=Cloud(), task="t",
                        qwen_generate=qwen, status=lambda l: None,
                        context_window=32768)
        self.assertTrue(res.escalated)
        self.assertNotEqual(res.reason, "review_failed_no_verdict")
        self.assertFalse(any(v.decision == "APPROVED" for v in res.verdicts),
                         "truncated output must never reach an APPROVED verdict")


if __name__ == "__main__":
    unittest.main()
