"""
Gemma-primary / DeepSeek-supervisor control loop.

Architecture (vs the router in agent.py):
  Instead of deciding "which model does the whole task", Gemma (local) is the
  primary coding agent for almost everything. DeepSeek only intervenes as a
  supervisor/reviewer, and only when the task or verification warrants it.

  Flow:
    TASK
      -> Gemma implements (local backend)
      -> build a COMPACT review package (not the whole conversation)
      -> optional verification hook
      -> DeepSeek supervises (compact package only)
      -> VERDICT:
           APPROVED      -> done
           FIX_REQUIRED  -> feed REQUIRED_FIXES back to Gemma, iterate
           REJECTED      -> stop, report alternative approach
      -> loop until APPROVED or max iterations

  DeepSeek never rewrites code directly; it returns verdicts + required fixes,
  and Gemma implements them. API tokens are spent on judgement, not on doing
  the coding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backends.base import Backend, ModelRequest, ModelResponse


@dataclass
class Verdict:
    decision: str  # "APPROVED" | "FIX_REQUIRED" | "REJECTED"
    assessment: str = ""
    issues: list[dict] = field(default_factory=list)
    approval_conditions: str = ""
    rejection_reason: str = ""
    raw: str = ""

    @property
    def approved(self) -> bool:
        return self.decision == "APPROVED"

    @property
    def rejected(self) -> bool:
        return self.decision == "REJECTED"


@dataclass
class ReviewPackage:
    """The COMPACT context sent to DeepSeek — not the whole conversation."""

    task: str
    plan: str = ""
    files: str = ""
    changes: str = ""
    verification: str = ""
    uncertainties: str = ""
    diff: str = ""

    def to_prompt(self) -> str:
        sections = [
            ("TASK", self.task),
            ("PLAN", self.plan),
            ("FILES", self.files),
            ("CHANGES", self.changes),
            ("VERIFICATION", self.verification),
            ("UNCERTAINTIES", self.uncertainties),
            ("DIFF", self.diff),
        ]
        blocks = [f"{key}:\n{val}".rstrip() for key, val in sections if val.strip()]
        return "\n\n".join(blocks)


def _decision_from(raw: str) -> str:
    """Extract the decision keyword from a review response."""
    m = re.search(r"\b(APPROVED|FIX_REQUIRED|REJECTED)\b", raw.upper())
    return m.group(1) if m else "FIX_REQUIRED"


def parse_verdict(raw: str) -> Verdict:
    decision = _decision_from(raw)
    verdict = Verdict(decision=decision, raw=raw)

    def _section(label: str) -> str:
        m = re.search(rf"===\s*{label}\s*===\s*(.*?)(?:\n\s*===|\Z)", raw, re.S | re.I)
        return m.group(1).strip() if m else ""

    verdict.assessment = _section("OVERALL ASSESSMENT")
    verdict.approval_conditions = _section("APPROVAL CONDITIONS")
    verdict.rejection_reason = _section("REJECTION EXPLANATION")

    issues: list[dict] = []
    for m in re.finditer(
        r"Issue\s*#\d+\s*:\s*(?P<desc>[^\n]*)(?P<rest>(?:\n(?!Issue\s*#)[^\n]*)*)",
        raw, re.I,
    ):
        desc = m.group("desc").strip()
        loc = re.search(r"Location\s*:\s*(.+)", m.group("rest"), re.I)
        sev = re.search(r"Severity\s*:\s*(.+)", m.group("rest"), re.I)
        cat = re.search(r"Category\s*:\s*(.+)", m.group("rest"), re.I)
        fix = re.search(r"Fix\s*:\s*(.+)", m.group("rest"), re.I)
        issues.append({
            "description": desc,
            "location": loc.group(1).strip() if loc else "",
            "severity": sev.group(1).strip() if sev else "MAJOR",
            "category": cat.group(1).strip() if cat else "",
            "fix": fix.group(1).strip() if fix else "",
        })
    verdict.issues = issues
    return verdict


def required_fixes_summary(verdict: Verdict) -> str:
    """Build a short, actionable instruction to hand back to Gemma."""
    if verdict.approved:
        return ""
    lines = [f"DeepSeek supervisor says: {verdict.decision}"]
    for issue in verdict.issues:
        sev = issue.get("severity", "")
        if sev in ("CRITICAL", "MAJOR"):
            lines.append(f"- ({sev}) {issue.get('description','')}")
            if issue.get("fix"):
                lines.append(f"    fix: {issue['fix']}")
    if verdict.approval_conditions and not lines:
        lines.append(f"- {verdict.approval_conditions}")
    return "\n".join(lines)


# --- prompts -------------------------------------------------------------

def _gemma_primary_prompt(task: str, prior_fixes: str = "") -> ModelRequest:
    system = (
        "You are a careful mid-level engineer implementing a change. "
        "Write correct, minimal, well-structured code for the given task. "
        "Output the complete updated file(s) in fenced code blocks labeled with paths."
    )
    user = f"TASK:\n{task}\n"
    if prior_fixes:
        user += f"\nA senior reviewer previously asked you to fix these issues. Apply ALL of them now:\n{prior_fixes}\n"
    user += "\nOutput only the code changes, no preamble."
    return ModelRequest(system=system, user=user, max_tokens=2048, temperature=0.2)


def _supervisor_request(pkg: ReviewPackage, verbose: bool = True) -> ModelRequest:
    system = (
        "You are a CRITICAL senior code reviewer supervising a mid-level engineer. "
        "You receive a compact review package (task, plan, files, changes, "
        "verification, uncertainties, diff) — NOT the whole conversation. "
        "Return a verdict following this exact format:\n\n"
        "=== REVIEW DECISION ===\n"
        "[APPROVED | FIX_REQUIRED | REJECTED]\n\n"
        "=== OVERALL ASSESSMENT ===\n"
        "[2-3 sentences]\n\n"
        "=== ISSUES FOUND ===\n"
        "Issue #1: [description]\n"
        "Location: [file/function]\n"
        "Severity: [CRITICAL|MAJOR|MINOR|SUGGESTION]\n"
        "Category: [Correctness|Design|Performance|Security|Testing|Style]\n"
        "Fix: [exact instruction or code snippet]\n\n"
        "=== APPROVAL CONDITIONS ===\n"
        "[what must be fixed before approval, if FIX_REQUIRED]\n\n"
        "=== REJECTION EXPLANATION ===\n"
        "[only if REJECTED]\n\n"
        "Rules: be specific, show the fix not just describe it, be brutal about "
        "correctness and security, be reasonable about style. Do NOT rewrite the "
        "whole codebase — you are reviewing, Gemma implements."
    )
    user = pkg.to_prompt()
    if verbose:
        user += "\n\nDecide: APPROVED, FIX_REQUIRED, or REJECTED."
    return ModelRequest(system=system, user=user, max_tokens=4096, temperature=0.1)


# --- control loop ---------------------------------------------------------

@dataclass
class SuperviseResult:
    task: str
    final_text: str = ""
    route: str = "local"
    reason: str = "gemma_primary"
    verdicts: list[Verdict] = field(default_factory=list)
    iterations: int = 0
    escalated: bool = False


def supervise(
    local: Backend,
    cloud: Backend,
    task: str,
    package_builder=None,
    max_iterations: int = 3,
    status: callable = None,
    gemma_generate: callable = None,
) -> SuperviseResult:
    """Run Gemma-primary / DeepSeek-supervisor on a task.

    `package_builder(task, gemma_output, iteration) -> ReviewPackage` lets the
    caller supply verification (test/lint/build results, git diff, etc.).

    `gemma_generate(request) -> ModelResponse` overrides how Gemma is invoked.
    Pass a streaming wrapper here so the caller can stream Gemma's output live
    (always showing the local model working). Defaults to `local.generate`.
    """
    if status is None:
        status = lambda line: None  # noqa: E731
    if gemma_generate is None:
        gemma_generate = lambda req: local.generate(req)  # noqa: E731

    result = SuperviseResult(task=task)
    current = task
    prior_fixes = ""

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration

        # 1. Gemma implements (streaming so the user sees it working).
        status(f"[supervise] iter {iteration}: Gemma working...")
        try:
            resp: ModelResponse = gemma_generate(_gemma_primary_prompt(current, prior_fixes))
        except Exception as exc:  # noqa: BLE001
            status(f"[supervise] local failed ({exc}); escalating to DeepSeek")
            plan = cloud.generate(_cloud_plan_request(task))
            result.final_text = plan.text
            result.escalated = True
            result.reason = "local_failure_escalation"
            return result
        code = resp.text

        # 2. Build the compact review package.
        pkg = package_builder(task, code, iteration) if package_builder else _default_package(task, code)

        # 3. DeepSeek supervises.
        status(f"[supervise] iter {iteration}: DeepSeek supervising...")
        try:
            review_resp = cloud.generate(_supervisor_request(pkg))
        except Exception as exc:  # noqa: BLE001
            status(f"[supervise] review failed ({exc}); applying without review")
            result.final_text = code
            result.reason = "review_failed_no_verdict"
            return result

        verdict = parse_verdict(review_resp.text)
        result.verdicts.append(verdict)

        if verdict.approved:
            status(f"[supervise] iter {iteration}: APPROVED")
            result.final_text = code
            return result
        if verdict.rejected:
            status(f"[supervise] iter {iteration}: REJECTED")
            result.final_text = code
            result.reason = "supervisor_rejected"
            return result

        # FIX_REQUIRED: feed fixes back to Gemma.
        prior_fixes = required_fixes_summary(verdict)
        status(f"[supervise] iter {iteration}: FIX_REQUIRED — {len(verdict.issues)} issue(s), looping")
        current = task

    result.final_text = code  # last Gemma output after exhausting iterations
    result.reason = f"max_iterations_{max_iterations}"
    return result


def _default_package(task: str, code: str, iteration: int) -> ReviewPackage:
    return ReviewPackage(
        task=task,
        changes=f"Gemma output (iteration {iteration}):\n{code}",
        uncertainties="None provided — caller can enrich this package with tests/diff.",
    )


def _cloud_plan_request(task: str) -> ModelRequest:
    return ModelRequest(
        system="You are the senior architect for a hybrid coding agent. Produce a step-by-step plan with file-level breakdown (max 400 words).",
        user=f"OBJECTIVE: {task}\n",
        max_tokens=1200,
        temperature=0.1,
    )
