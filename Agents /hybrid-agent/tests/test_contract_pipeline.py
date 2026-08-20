"""Task Contract pipeline tests: contract parsing, dependency-aware context
retrieval, the adversarial final auditor, and contract threading.

Run from hybrid-agent/:
    python -m unittest tests.test_contract_pipeline -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
import dependencies  # noqa: E402
from backends.base import ModelResponse  # noqa: E402
from contract import parse_contract  # noqa: E402
from dependencies import build_dependency_context  # noqa: E402
from supervise import supervise  # noqa: E402

CONTRACT_RAW = """=== ENHANCED PROMPT ===
Implement the minimum booking price.

=== TASK CONTRACT ===
Goal: Enforce a £20 minimum booking price.
Must change:
- src/booking/price.ts
Must NOT change:
- payment flow
Acceptance criteria:
- prices below £20 are rejected
Files likely involved: src/booking/price.ts, src/booking/booking.ts
Dependencies: none
Risk: MEDIUM
Verification required:
- unit tests
- browser journey
Rollback strategy: revert src/booking/price.ts only

=== ACCEPTANCE CASES ===
- min price £20 -> accepted
- min price £19.99 -> rejected
- price 0 -> rejected

=== PLAN ===
Step 1: add the guard
"""


class TestContractParse(unittest.TestCase):
    def test_full_contract(self):
        c = parse_contract(CONTRACT_RAW)
        self.assertTrue(c.complete)
        self.assertEqual(c.goal, "Enforce a £20 minimum booking price.")
        self.assertIn("src/booking/price.ts", c.files)
        self.assertIn("payment flow", c.must_not_change)
        self.assertEqual(c.risk, "MEDIUM")
        self.assertIn("unit tests", c.verification_required)
        self.assertIn("browser journey", c.verification_required)
        self.assertEqual(len(c.acceptance_cases), 3)
        self.assertEqual(c.acceptance_cases[1]["input"], "min price £19.99")
        self.assertFalse(c.risky)

    def test_critical_risk(self):
        raw = CONTRACT_RAW.replace("Risk: MEDIUM", "Risk: CRITICAL")
        self.assertTrue(parse_contract(raw).risky)

    def test_no_contract(self):
        c = parse_contract("=== ENHANCED PROMPT ===\njust do it\n")
        self.assertFalse(c.complete)
        self.assertEqual(c.to_prompt(), "")

    def test_to_prompt_renders(self):
        text = parse_contract(CONTRACT_RAW).to_prompt()
        self.assertIn("TASK CONTRACT", text)
        self.assertIn("Goal: Enforce", text)
        self.assertIn("min price £19.99 -> rejected", text)


class TestDependencyContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "src/checkout").mkdir(parents=True)
        (root / "src/app").mkdir(parents=True)
        (root / "src/checkout/checkout.service.ts").write_text(
            "import { paymentService } from './payment.service';\n"
            "export function checkout(cart: any) { return paymentService.pay(cart.total); }\n")
        (root / "src/checkout/payment.service.ts").write_text(
            "export function pay(amount: number) { return { ok: true }; }\n")
        (root / "src/checkout/checkout.service.test.ts").write_text(
            "import { checkout } from './checkout.service';\n"
            "test('works', () => expect(checkout({total: 1})).toBeDefined());\n")
        (root / "src/app/cart.ts").write_text(
            "import { checkout } from '../checkout/checkout.service';\n"
            "export function buy(cart: any) { return checkout(cart); }\n")
        (root / "node_modules/fake").mkdir(parents=True)
        (root / "node_modules/fake/x.ts").write_text("export const x = 1;\n")
        self.root = root

    def test_retrieves_imports_callers_tests(self):
        ctx = build_dependency_context(str(self.root),
                                       ["src/checkout/checkout.service.ts"])
        self.assertIn("[INVOLVED] src/checkout/checkout.service.ts", ctx)
        self.assertIn("payment.service", ctx)          # direct import
        self.assertIn("checkout.service.test.ts", ctx)  # its test
        self.assertIn("cart.ts", ctx)                   # its caller
        self.assertIn("CALLER", ctx)
        self.assertIn("TEST", ctx)
        self.assertNotIn("node_modules", ctx)           # never project junk

    def test_empty_files(self):
        self.assertEqual(build_dependency_context(str(self.root), []), "")
        self.assertEqual(build_dependency_context(str(self.root),
                                                  ["missing/file.ts"]), "")


class TestAuditor(unittest.TestCase):
    def test_parse_audit_pass(self):
        decision, evidence, affected = ask._parse_audit(
            "=== DECISION ===\nPASS\n\n=== EVIDENCE ===\n")
        self.assertEqual(decision, "PASS")

    def test_parse_audit_fail(self):
        decision, evidence, affected = ask._parse_audit(
            "=== DECISION ===\nFAIL\n\n=== EVIDENCE ===\nThe price guard is missing.\n"
            "=== AFFECTED REQUIREMENT ===\nminimum booking price")
        self.assertEqual(decision, "FAIL")
        self.assertIn("price guard", evidence)
        self.assertIn("minimum booking price", affected)

    def test_run_final_audit_fail(self):
        class _Cloud:
            def generate(self, req):
                return ModelResponse(
                    text="=== DECISION ===\nFAIL\n=== EVIDENCE ===\nNo price guard.\n"
                         "=== AFFECTED REQUIREMENT ===\nminimum price",
                    backend="deepseek")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ok, report = ask._run_final_audit(
                _Cloud(), {"review": {}}, ask.argparse.Namespace(root="."),
                parse_contract(CONTRACT_RAW), [("src/booking/price.ts", 10)],
                "tests passed", "add min price")
        self.assertFalse(ok)
        self.assertIn("auditor FAIL", report)
        self.assertIn("price guard", report)

    def test_run_final_audit_pass(self):
        class _Cloud:
            def generate(self, req):
                return ModelResponse(text="=== DECISION ===\nPASS\n=== EVIDENCE ===\n",
                                     backend="deepseek")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ok, report = ask._run_final_audit(
                _Cloud(), {}, ask.argparse.Namespace(root="."),
                parse_contract(CONTRACT_RAW), [], "tests passed", "task")
        self.assertTrue(ok)
        self.assertIn("PASS", report)


class TestContractThreading(unittest.TestCase):
    def test_contract_reaches_qwen(self):
        users = []

        def fake_qwen(req):
            users.append(req.user)
            return ModelResponse(text="```\napp.ts\nx\n```", backend="local")

        class _Cloud:
            def generate(self, req):
                return ModelResponse(
                    text="=== REVIEW DECISION ===\nAPPROVED\n=== QUALITY SCORE ===\n8.0\n"
                         "=== OVERALL ASSESSMENT ===\nOK",
                    backend="deepseek")

        supervise(local=None, cloud=_Cloud(), task="t", qwen_generate=fake_qwen,
                  contract_text="TASK CONTRACT:\nGoal: enforce the minimum price",
                  source_context="RELEVANT SOURCE:\nprice.ts\n", status=lambda line: None)
        self.assertTrue(any("TASK CONTRACT:" in s and "minimum price" in s
                            for s in users))
        self.assertTrue(any("RELEVANT SOURCE" in s for s in users))


if __name__ == "__main__":
    unittest.main()
