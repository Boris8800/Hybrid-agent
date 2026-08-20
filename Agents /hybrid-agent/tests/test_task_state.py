"""Tests for task_state.py — durable task state, strict state machine,
evidence ledger, capability contracts, batch transactions, idempotency,
restart/resume."""

import os
import tempfile
import unittest

from task_state import (BatchStatus, BatchTransaction, Evidence,
                        EvidenceLedger, IllegalTransition, ModelCapabilities,
                        OperationLog, TaskState, TaskStateMachine, TaskStatus)


class StateMachineTests(unittest.TestCase):
    def test_full_legal_chain(self):
        m = TaskStateMachine()
        for s in [TaskStatus.PLANNING, TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED, TaskStatus.VERIFYING,
                  TaskStatus.VERIFIED, TaskStatus.REGRESSION_VERIFIED,
                  TaskStatus.APPROVED, TaskStatus.APPLIED,
                  TaskStatus.DEPLOY_AUTHORIZED, TaskStatus.PUSHED,
                  TaskStatus.DEPLOYED]:
            m.transition(s)
        self.assertEqual(m.status, TaskStatus.DEPLOYED)

    def test_failed_to_deployed_impossible(self):
        m = TaskStateMachine()
        m.transition(TaskStatus.FAILED)
        with self.assertRaises(IllegalTransition):
            m.transition(TaskStatus.DEPLOYED)
        with self.assertRaises(IllegalTransition):
            m.transition(TaskStatus.APPROVED)
        # FAILED is terminal: no edges out at all.
        for s in TaskStatus:
            if s != TaskStatus.FAILED:
                self.assertFalse(m.can_transition(s), f"FAILED->{s} must be illegal")

    def test_generated_to_approved_requires_review_and_verify(self):
        """GENERATED -> APPROVED must be impossible: review and verification
        are distinct facts, never collapsed into one success."""
        m = TaskStateMachine()
        m.transition(TaskStatus.IMPLEMENTING)
        m.transition(TaskStatus.GENERATED)
        self.assertFalse(m.can_transition(TaskStatus.APPROVED))
        with self.assertRaises(IllegalTransition):
            m.transition(TaskStatus.APPROVED)
        m.transition(TaskStatus.REVIEWING)
        m.transition(TaskStatus.REVIEWED)
        self.assertFalse(m.can_transition(TaskStatus.APPROVED),
                         "reviewed but not verified -> approved is illegal")

    def test_approved_before_apply(self):
        m = TaskStateMachine()
        m.transition(TaskStatus.IMPLEMENTING)
        m.transition(TaskStatus.GENERATED)
        m.transition(TaskStatus.REVIEWING)
        m.transition(TaskStatus.REVIEWED)
        m.transition(TaskStatus.VERIFYING)
        m.transition(TaskStatus.VERIFIED)
        m.transition(TaskStatus.APPROVED)
        # APPROVED -> APPLIED is the only forward edge (plus FAILED).
        self.assertTrue(m.can_transition(TaskStatus.APPLIED))
        self.assertFalse(m.can_transition(TaskStatus.DEPLOYED),
                         "cannot deploy before apply")
        m.transition(TaskStatus.APPLIED)
        self.assertTrue(m.can_transition(TaskStatus.DEPLOY_AUTHORIZED))

    def test_rejected_is_terminal(self):
        m = TaskStateMachine()
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED]:
            m.transition(s)
        m.transition(TaskStatus.REJECTED)
        with self.assertRaises(IllegalTransition):
            m.transition(TaskStatus.FIXING)

    def test_idempotent_same_state_noop(self):
        m = TaskStateMachine()
        m.transition(TaskStatus.IMPLEMENTING)
        m.transition(TaskStatus.GENERATED)
        self.assertEqual(m.transition(TaskStatus.GENERATED), TaskStatus.GENERATED)


class EvidenceLedgerTests(unittest.TestCase):
    def test_evidence_fields_and_hash(self):
        led = EvidenceLedger()
        ev = led.add("TEST", command="npm test", exit_code=0,
                     output="12 passing", files_affected=["src/auth.ts"],
                     source="verify", summary="all tests pass")
        self.assertTrue(ev.evidence_id.startswith("E-"))
        self.assertEqual(len(ev.output_hash), 16)
        self.assertEqual(ev.exit_code, 0)
        self.assertEqual(ev.files_affected, ["src/auth.ts"])
        self.assertEqual(led.get(ev.evidence_id), ev)

    def test_ids_increment(self):
        led = EvidenceLedger()
        a = led.add("TEST")
        b = led.add("FILE")
        self.assertNotEqual(a.evidence_id, b.evidence_id)
        self.assertEqual(led.ids(), [a.evidence_id, b.evidence_id])

    def test_render_includes_facts(self):
        led = EvidenceLedger()
        led.add("TEST", command="npm test", exit_code=0, output="ok",
                files_affected=["a.ts"], source="verify", summary="pass")
        rendered = led.render()
        self.assertIn("npm test", rendered)
        self.assertIn("exit:0", rendered)
        self.assertIn("hash:", rendered)
        self.assertIn("a.ts", rendered)

    def test_roundtrip_via_dict(self):
        led = EvidenceLedger()
        led.add("TEST", command="pytest", exit_code=1, output="boom")
        led2 = EvidenceLedger([e.to_dict() for e in led.entries])
        self.assertEqual(led2.entries[0].command, "pytest")
        self.assertEqual(led2.entries[0].exit_code, 1)
        # Counter resumes past loaded ids.
        e3 = led2.add("FILE")
        self.assertNotEqual(e3.evidence_id, led2.entries[0].evidence_id)


class CapabilityContractTests(unittest.TestCase):
    def test_from_discovery(self):
        caps = ModelCapabilities.from_discovery({
            "model_id": "qwen2.5-coder-14b-instruct-mlx",
            "context_window": 32768, "max_output": 0,
            "tool_use": False, "vision": False, "architecture": "qwen2",
        })
        self.assertEqual(caps.context_window, 32768)
        self.assertTrue(caps.streaming)

    def test_can_perform_role(self):
        big = ModelCapabilities(model_id="m", context_window=65536, streaming=True)
        self.assertTrue(big.can_perform_role("implementer"))
        self.assertTrue(big.can_perform_role("implementer", "long"))
        small = ModelCapabilities(model_id="m", context_window=2048)
        self.assertTrue(small.can_perform_role("implementer"))
        self.assertFalse(small.can_perform_role("implementer", "long"),
                         "a 2048-window model cannot do long tasks")
        unknown = ModelCapabilities()
        self.assertFalse(unknown.can_perform_role("implementer"),
                         "unknown capabilities are conservative: not guaranteed")
        self.assertTrue(ModelCapabilities(vision=True).can_perform_role("vision"))
        self.assertFalse(ModelCapabilities().can_perform_role("vision"))
        self.assertTrue(ModelCapabilities(tool_use=True).can_perform_role("tool_calling"))
        self.assertFalse(ModelCapabilities().can_perform_role("tool_calling"))


class BatchTransactionTests(unittest.TestCase):
    def test_success(self):
        b = BatchTransaction(batch_id="B-1", steps=[
            {"id": 1, "status": "pending"}, {"id": 2, "status": "pending"}])
        b.mark_step(1, True)
        b.mark_step(2, True)
        self.assertTrue(b.is_success())

    def test_partial_failure_never_silent_success(self):
        b = BatchTransaction(batch_id="B-1", steps=[
            {"id": 1, "status": "pending"}, {"id": 2, "status": "pending"},
            {"id": 3, "status": "pending"}, {"id": 4, "status": "pending"}])
        for i in (1, 2, 4):
            b.mark_step(i, True)
        b.mark_step(3, False, error="boom")
        self.assertTrue(b.is_partial())
        self.assertEqual(b.status, BatchStatus.PARTIAL_FAILURE.value)
        self.assertEqual(b.failed_step_ids, [3])
        self.assertFalse(b.is_success(), "3/4 steps OK must NOT be success")

    def test_all_failed(self):
        b = BatchTransaction(batch_id="B-1", steps=[{"id": 1}])
        b.mark_step(1, False, error="x")
        self.assertEqual(b.status, BatchStatus.FAILED.value)


class OperationLogTests(unittest.TestCase):
    def test_idempotency(self):
        log = OperationLog()
        op = OperationLog.make_id("TASK-x", "apply")
        self.assertFalse(log.is_completed(op))
        log.mark_started(op)
        log.mark_completed(op, "3 files")
        self.assertTrue(log.is_completed(op))
        self.assertEqual(log.records[op]["result"], "3 files")

    def test_failed_can_retry(self):
        log = OperationLog()
        op = OperationLog.make_id("TASK-x", "push")
        log.mark_failed(op, "network")
        self.assertFalse(log.is_completed(op), "failed ops are retryable")
        log.mark_completed(op)
        self.assertTrue(log.is_completed(op))

    def test_deterministic_ids(self):
        a = OperationLog.make_id("TASK-x", "deploy")
        b = OperationLog.make_id("TASK-x", "deploy")
        self.assertEqual(a, b)


class TaskStatePersistenceTests(unittest.TestCase):
    def _tmp_root(self):
        return tempfile.mkdtemp(prefix="tstate-")

    def test_roundtrip_preserves_all_facts(self):
        root = self._tmp_root()
        ts = TaskState("fix auth", root=root, plan=[{"id": 1, "name": "s1"},
                                                    {"id": 2, "name": "s2"}])
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.transition(TaskStatus.GENERATED)
        ts.mark_step(1, True, "done")
        ts.mark_file("src/auth.ts")
        ts.record_verdict("APPROVED", 8.5)
        ev = ts.evidence.add("TEST", command="npm test", exit_code=0,
                             output="ok", files_affected=["src/auth.ts"],
                             source="verify", summary="pass")
        path = ts.save()
        self.assertTrue(os.path.exists(path))

        loaded = TaskState.load("fix auth", root=root)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.task_id, ts.task_id)
        self.assertEqual(loaded.status, TaskStatus.GENERATED)
        self.assertEqual(loaded.steps[0].status, "completed")
        self.assertEqual(loaded.steps[1].status, "pending")
        self.assertEqual(loaded.files[0].path, "src/auth.ts")
        self.assertEqual(loaded.verdicts[-1]["decision"], "APPROVED")
        self.assertEqual(loaded.evidence.get(ev.evidence_id).exit_code, 0)
        self.assertEqual(loaded.iterations, 1)

    def test_resume_summary_reports_position(self):
        root = self._tmp_root()
        ts = TaskState("big task", root=root,
                       plan=[{"id": 1, "name": "step 1"}, {"id": 2, "name": "step 2"},
                             {"id": 3, "name": "step 3"}, {"id": 4, "name": "step 4"}])
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.mark_step(1, True)
        ts.mark_step(2, True)
        ts.mark_step(3, True)
        ts.evidence.add("TEST", command="t", exit_code=0, output="x")
        summary = ts.resume_summary()
        self.assertIn("TASK ", summary)
        self.assertIn("RESUMING", summary)
        self.assertIn("3/4", summary)
        self.assertIn("step 1", summary)
        self.assertIn("step 2", summary)
        self.assertIn("step 3", summary)
        self.assertIn("→ step 4", summary)
        self.assertIn("E-", summary)

    def test_corrupt_state_file_loads_none(self):
        root = self._tmp_root()
        tid = TaskState.make_task_id("x")
        d = os.path.join(root, ".hybrid-agent", "tasks")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{tid}.json"), "w") as fh:
            fh.write("{not json")
        self.assertIsNone(TaskState.load("x", root=root))

    def test_load_missing_returns_none(self):
        root = self._tmp_root()
        self.assertIsNone(TaskState.load("never existed", root=root))


class TaskStateMachineIntegrationTests(unittest.TestCase):
    def test_truncated_output_cannot_reach_approved(self):
        """The supervisor's truncated->escalate path must not allow the state
        machine to ever mark truncated output APPROVED: the only forward path
        from GENERATED requires REVIEWING->REVIEWED->VERIFYING->VERIFIED."""
        ts = TaskState("t")
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.transition(TaskStatus.GENERATED)
        # No edge GENERATED -> APPROVED exists.
        self.assertFalse(ts.can_transition(TaskStatus.APPROVED))

    def test_state_machine_blocks_deploy_without_authorization(self):
        ts = TaskState("t")
        # Even fully verified+applied, deploy requires DEPLOY_AUTHORIZED.
        for s in [TaskStatus.IMPLEMENTING, TaskStatus.GENERATED,
                  TaskStatus.REVIEWING, TaskStatus.REVIEWED,
                  TaskStatus.VERIFYING, TaskStatus.VERIFIED,
                  TaskStatus.APPROVED, TaskStatus.APPLIED]:
            ts.transition(s)
        self.assertFalse(ts.can_transition(TaskStatus.DEPLOYED),
                         "deploy needs the DEPLOY_AUTHORIZED trust boundary")
        ts.transition(TaskStatus.DEPLOY_AUTHORIZED)
        self.assertTrue(ts.can_transition(TaskStatus.DEPLOYED))

    def test_evidence_ledger_makes_approval_auditable(self):
        """Approval references evidence IDs; each fact has command/exit/hash."""
        ts = TaskState("audit me")
        ts.evidence.add("TEST", command="npm test", exit_code=0,
                        output="12 passing", files_affected=["src/auth.ts"],
                        source="verify", summary="all pass")
        ts.evidence.add("TEST", command="pytest", exit_code=0,
                        output="ok", source="verify")
        rendered = ts.evidence.render()
        self.assertIn("E-", rendered)
        self.assertIn("npm test", rendered)
        self.assertIn("src/auth.ts", rendered)
        # Every evidence entry is addressable by id.
        for e in ts.evidence.entries:
            self.assertIs(ts.evidence.get(e.evidence_id), e)

    def test_operations_make_push_deploy_idempotent(self):
        ts = TaskState("idem", root=tempfile.mkdtemp())
        push = OperationLog.make_id(ts.task_id, "push")
        deploy = OperationLog.make_id(ts.task_id, "deploy")
        self.assertFalse(ts.operations.is_completed(push))
        ts.operations.mark_completed(push, "ok")
        ts.operations.mark_completed(deploy, "ok")
        self.assertTrue(ts.operations.is_completed(push))
        self.assertTrue(ts.operations.is_completed(deploy))
        # After restart (reload), idempotency holds.
        ts.save()
        loaded = TaskState.load("idem", root=ts.root)
        self.assertTrue(loaded.operations.is_completed(push))

    def test_batch_partial_failure_recorded_in_state(self):
        root = tempfile.mkdtemp()
        ts = TaskState("batch", root=root)
        b = BatchTransaction(batch_id="TASK-x-B1", steps=[
            {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])
        for i in (1, 2, 4):
            b.mark_step(i, True)
        b.mark_step(3, False, error="boom")
        ts.record_batch(b)
        ts.save()
        loaded = TaskState.load("batch", root=root)
        self.assertEqual(loaded.batches[0]["status"], "PARTIAL_FAILURE")
        self.assertIn(3, [s["id"] for s in loaded.batches[0]["steps"]
                          if s["status"] == "failed"])

    def test_resume_from_partial_state_keeps_position(self):
        root = tempfile.mkdtemp()
        ts = TaskState("resume me", root=root,
                       plan=[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        ts.transition(TaskStatus.IMPLEMENTING)
        ts.mark_step(1, True)
        ts.save()
        loaded = TaskState.load("resume me", root=root)
        self.assertEqual(loaded.status, TaskStatus.IMPLEMENTING)
        self.assertEqual(loaded.steps[0].status, "completed")
        self.assertEqual(loaded.steps[1].status, "pending")
        self.assertIn("1/2", loaded.resume_summary())

    def test_task_contract_files_become_durable_state(self):
        from contract import parse_contract
        raw = ("=== TASK CONTRACT ===\n"
               "Goal: add auth\n"
               "Files likely involved: src/auth.ts, src/auth.test.ts\n"
               "Acceptance criteria:\n  - login works\n"
               "=== ACCEPTANCE CASES ===\n"
               "valid creds -> session")
        contract = parse_contract(raw)
        root = tempfile.mkdtemp()
        ts = TaskState("contract task", root=root)
        ts.acceptance_criteria = list(contract.acceptance_criteria)
        for f in contract.files:
            ts.files.append(__import__("task_state").FileState(path=f, role="required"))
        ts.save()
        loaded = TaskState.load("contract task", root=root)
        self.assertEqual(len(loaded.files), 2)
        self.assertEqual(loaded.files[0].role, "required")
        self.assertEqual(loaded.acceptance_criteria, ["login works"])


class TurboAdjudicationTests(unittest.TestCase):
    """--turbo must adjudicate verdicts, never pick the longest response."""

    def test_strictest_verdict_wins_over_approval(self):
        import ask
        from backends.base import ModelResponse
        approved = ("=== REVIEW DECISION ===\nAPPROVED\n"
                    "=== QUALITY SCORE ===\n8.0\n=== OVERALL ASSESSMENT ===\nok\n")
        rejected = ("=== REVIEW DECISION ===\nREJECTED\n"
                    "=== QUALITY SCORE ===\n3.0\n=== OVERALL ASSESSMENT ===\n"
                    "security hole\n")
        fix = ("=== REVIEW DECISION ===\nFIX_REQUIRED\n"
               "=== QUALITY SCORE ===\n6.0\n=== OVERALL ASSESSMENT ===\n"
               "needs work\n=== ISSUES FOUND ===\nIssue #1: bug\n")
        winner = ask._adjudicate_turbo([
            ModelResponse(text=approved, backend="A"),
            ModelResponse(text=rejected, backend="B"),
            ModelResponse(text=fix, backend="C"),
        ])
        self.assertEqual(winner.text, rejected,
                         "REJECTED (strictest) must win over longer approvals")

    def test_fix_required_beats_approval(self):
        import ask
        from backends.base import ModelResponse
        approved = ("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n8.0\n"
                    "=== OVERALL ASSESSMENT ===\nok\n")
        fix = ("=== REVIEW DECISION ===\nFIX_REQUIRED\n=== QUALITY SCORE ===\n6.0\n"
               "=== OVERALL ASSESSMENT ===\nneeds work\n=== ISSUES FOUND ===\n"
               "Issue #1: bug\n")
        winner = ask._adjudicate_turbo([
            ModelResponse(text=approved * 3, backend="A"),  # much longer
            ModelResponse(text=fix, backend="B"),
        ])
        self.assertEqual(winner.text, fix,
                         "FIX_REQUIRED wins over a longer APPROVED response")

    def test_all_approved_prefers_higher_confidence(self):
        import ask
        from backends.base import ModelResponse
        low = ("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n6.0\n"
               "=== OVERALL ASSESSMENT ===\nok\n=== EVIDENCE ===\nE-001 basic\n")
        low2 = ("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n6.5\n"
                "=== OVERALL ASSESSMENT ===\nalso fine\n=== EVIDENCE ===\nE-002 basic\n")
        high = ("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n9.0\n"
                "=== OVERALL ASSESSMENT ===\nexcellent with evidence\n"
                "=== EVIDENCE ===\nE-001 npm test exit 0\n")
        winner = ask._adjudicate_turbo([
            ModelResponse(text=low2 * 1, backend="A"),
            ModelResponse(text=high, backend="B"),
        ])
        self.assertEqual(winner.text, high,
                         "among approvals, prefer the higher confidence/evidence")

    def test_truncated_providers_excluded(self):
        import ask
        from backends.base import ModelResponse
        good = ("=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n8.0\n"
                "=== OVERALL ASSESSMENT ===\nok\n")
        winner = ask._adjudicate_turbo([
            ModelResponse(text="cut off", truncated=True, backend="A"),
            ModelResponse(text=good, backend="B"),
        ])
        self.assertEqual(winner.text, good)


if __name__ == "__main__":
    unittest.main()
