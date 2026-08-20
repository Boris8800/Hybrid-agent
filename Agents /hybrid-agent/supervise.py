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
    decision: str  # "APPROVED" | "FIX_REQUIRED" | "REJECTED" | "UNKNOWN"
    quality_score: float = 0.0  # 0-10 quality score (80/20 strategy)
    assessment: str = ""
    evidence: str = ""          # machine evidence cited for an APPROVED verdict
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


@dataclass
class Enhancement:
    """DeepSeek's enhanced prompt plus the reasoning/plan behind it.

    Produced BEFORE the local model implements, so the API model can clarify
    the task and size a plan to the local model's context/output limits.
    May also carry clarifying questions when the original task is ambiguous.
    """

    enhanced_prompt: str = ""
    reasoning: str = ""
    plan: str = ""
    clarifying_questions: str = ""
    raw: str = ""

    @property
    def enhanced(self) -> bool:
        return bool(self.enhanced_prompt.strip())


def parse_enhancement(raw: str) -> Enhancement:
    """Parse DeepSeek enhancement output into an Enhancement.

    Expects the section markers the enhancement prompt requests:
        === ENHANCED PROMPT ===
        === REASONING ===
        === PLAN ===
    Falls back to using the whole response as the enhanced prompt when no
    ENHANCED PROMPT section is present.
    """
    def _section(label: str) -> str:
        m = re.search(rf"===\s*{label}\s*===\s*(.*?)(?:\n\s*===|\Z)", raw, re.S | re.I)
        return m.group(1).strip() if m else ""

    enh = Enhancement(raw=raw)
    enh.enhanced_prompt = _section("ENHANCED PROMPT")
    enh.reasoning = _section("REASONING")
    enh.plan = _section("PLAN")
    enh.clarifying_questions = _section("CLARIFYING QUESTIONS")
    _cq = enh.clarifying_questions.lower()[:60]
    if enh.clarifying_questions and any(k in _cq for k in (
        "no clarifying", "no questions", "none needed", "already clear",
        "not ambiguous", "no clarification", "task is clear",
        "clear and well-specified", "no ambiguities")):
        enh.clarifying_questions = ""
    if not enh.enhanced_prompt:
        enh.enhanced_prompt = raw.strip()
    return enh


class ChainOfThoughtParser:
    """Parses DeepSeek's chain-of-thought planning sections.

    Reads the same `=== SECTION ===` markers used by the enhancement stage so
    DeepSeek can show WHY it made each planning decision, which the user can
    verify before the local model implements.
    """

    SECTIONS = ('TASK UNDERSTANDING', 'CONSTRAINT ANALYSIS', 'REASONING',
                'ALTERNATIVES', 'FINAL PLAN')

    def __init__(self, text: str):
        self.text = text or ""
        self.sections: dict = {}
        self._parse()

    def _section(self, label: str) -> str:
        m = re.search(rf"===\s*{label}\s*===\s*(.*?)(?:\n\s*===|\Z)",
                      self.text, re.S | re.I)
        return m.group(1).strip() if m else ""

    def _parse(self) -> None:
        for label in self.SECTIONS:
            key = label.lower().replace(" ", "_")
            self.sections[key] = self._section(label)

    def get_reasoning_chain(self) -> list:
        """Flatten the sections into a displayable list of (icon, title, body)."""
        chain = []
        if self.sections.get('task_understanding'):
            chain.append(("📝", "TASK UNDERSTANDING", self.sections['task_understanding']))
        if self.sections.get('constraint_analysis'):
            chain.append(("⚡", "CONSTRAINT ANALYSIS", self.sections['constraint_analysis']))
        if self.sections.get('reasoning'):
            chain.append(("🧩", "REASONING", self.sections['reasoning']))
        if self.sections.get('alternatives'):
            chain.append(("🤔", "ALTERNATIVES", self.sections['alternatives']))
        if self.sections.get('final_plan'):
            chain.append(("📋", "FINAL PLAN", self.sections['final_plan']))
        return chain


def _decision_from(raw: str) -> str:
    """Extract the decision keyword from a review response.

    UNKNOWN is first-class: an APPROVED verdict with NO cited machine evidence
    is downgraded to UNKNOWN, so the supervisor can never approve on vibes.
    """
    if re.search(r"\bUNKNOWN\b", raw.upper()):
        return "UNKNOWN"
    m = re.search(r"\b(APPROVED|FIX_REQUIRED|REJECTED)\b", raw.upper())
    if m and m.group(1) == "APPROVED":
        if not re.search(r"===\s*EVIDENCE\s*===", raw, re.I | re.S) \
                or not re.search(r"===\s*EVIDENCE\s*===.*?\S", raw, re.S | re.I):
            return "UNKNOWN"  # APPROVED without machine evidence
    return m.group(1) if m else "FIX_REQUIRED"


def parse_verdict(raw: str) -> Verdict:
    decision = _decision_from(raw)
    quality_score = 0.0
    m = re.search(r"QUALITY[ _]?SCORE.{0,60}?([0-9]*\.?[0-9]+)", raw, re.I | re.S)
    if m:
        try:
            quality_score = float(m.group(1))
        except ValueError:
            quality_score = 0.0
    verdict = Verdict(decision=decision, quality_score=quality_score, raw=raw)

    def _section(label: str) -> str:
        m = re.search(rf"===\s*{label}\s*===\s*(.*?)(?:\n\s*===|\Z)", raw, re.S | re.I)
        return m.group(1).strip() if m else ""

    verdict.assessment = _section("OVERALL ASSESSMENT")
    verdict.evidence = _section("EVIDENCE")[:2000]
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

# Output budgets: Gemma must emit the FULL multi-file response in one shot, so
# the default budget is generous and truncation is retried once with a doubled
# budget (hard cap GEMMA_MAX_TOKENS_CAP) before any escalation.
GEMMA_MAX_TOKENS = 8192
GEMMA_MAX_TOKENS_CAP = 16384
CLOUD_GEN_TOKENS = 8192

# Local model (qwen) constraints — the DeepSeek supervisor plans within these.
# The local server runs qwen2.5-coder-14b-instruct-mlx with a 32768-token
# context window; a conservative 4096-token output cap is assumed so plans must
# split work into steps that fit in ONE local response.
LOCAL_CONTEXT_TOKENS = 32768
LOCAL_OUTPUT_TOKENS = 4096


def _local_limits_note() -> str:
    """Compact, factual description of the implementer's constraints.

    Injected into the DeepSeek plan request so the supervisor sizes every step
    for one local response instead of assuming an unlimited model.
    """
    return (
        "IMPLEMENTER CONSTRAINTS (local Gemma 12B, must be respected): "
        f"context window {LOCAL_CONTEXT_TOKENS} tokens; hard output cap "
        f"{LOCAL_OUTPUT_TOKENS} tokens per response (higher budgets still "
        "truncate); one response = one step; target under ~3500 output tokens "
        "per step; keep prompt + repo context well under 32768; any task needing "
        "more output must be split into multiple sequential steps with per-step "
        "acceptance criteria; truncated output is retried once then escalated."
    )


def _gemma_primary_prompt(task: str, prior_fixes: str = "",
                          terminal_output: str = "", bound_text: str = "",
                          contract_text: str = "", source_context: str = "",
                          max_tokens: int = GEMMA_MAX_TOKENS) -> ModelRequest:
    system = (
        "You are a careful mid-level engineer implementing a change. "
        "Write correct, minimal, well-structured code for the given task. "
        "Output the complete updated file(s) in fenced code blocks labeled with paths. "
        "You have a terminal tool: to inspect or verify, emit a line like "
        "'RUN: npm run build' and you will receive the command output in the "
        "next turn. Use it to confirm your changes work."
    )
    if bound_text:
        system += "\n\n" + bound_text  # RECALL gate: the BOUND re-injected every iteration
    user = f"TASK:\n{task}\n"
    if contract_text:
        user += f"\n{contract_text}\n"  # the formal Task Contract (same one every stage uses)
    if prior_fixes:
        user += f"\nA senior reviewer previously asked you to fix these issues. Apply ALL of them now:\n{prior_fixes}\n"
    if source_context:
        user += (f"\nRELEVANT SOURCE (dependency-aware context for the files "
                 f"involved):\n{source_context}\n")
    if terminal_output:
        user += (
            f"\nTERMINAL SESSION (output of the commands you asked to run):\n"
            f"{terminal_output}\n"
            "Use this output to fix your changes. You may emit more "
            "'RUN: <command>' lines to inspect or verify, then output the "
            "final code."
        )
    user += (
        f"\nNOTE: your output is capped at {LOCAL_OUTPUT_TOKENS} tokens. If the "
        "change cannot fit in one response, implement the first coherent chunk "
        "and explicitly state what remains in a final 'REMAINING WORK' section."
    )
    user += "\nOutput only the code changes, no preamble."
    return ModelRequest(system=system, user=user, max_tokens=max_tokens, temperature=0.2)


# Terminal tool: model-emitted "RUN: <command>" lines, executed by the engine.
_RUN_RE = re.compile(r"^\s*RUN:\s+(.+?)\s*$", re.M)


def _extract_run_blocks(text: str) -> list[str]:
    """Commands the model asked to run, in order, de-duplicated."""
    cmds: list[str] = []
    for m in _RUN_RE.finditer(text or ""):
        cmd = m.group(1).strip().strip("`").strip()
        if cmd and cmd not in cmds:
            cmds.append(cmd)
    return cmds


def _supervisor_request(pkg: ReviewPackage, verbose: bool = True) -> ModelRequest:
    system = (
        "You are a CRITICAL senior code reviewer supervising a mid-level engineer. "
        "You receive a compact review package (task, plan, files, changes, "
        "verification, uncertainties, diff) — NOT the whole conversation. "
        "Return a verdict following this exact format:\n\n"
        "=== REVIEW DECISION ===\n"
        "[APPROVED | FIX_REQUIRED | REJECTED]\n\n"
        "=== QUALITY SCORE ===\n"
        "[0.0-10.0] — a single number. 8.0-10.0 APPROVED, 5.0-7.9 FIX_REQUIRED, <5.0 REJECTED.\n\n"
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
        "whole codebase — you are reviewing, Gemma implements.\n\n"
        "EVIDENCE RULE: if you return APPROVED you MUST cite the concrete machine "
        "evidence (test names, file:line, command output) under a "
        "'=== EVIDENCE ===' section. If you cannot point to machine evidence for "
        "your verdict, return UNKNOWN instead of guessing.\n\n"
        "Note: the code under review was produced by a local model with a "
        f"{LOCAL_OUTPUT_TOKENS}-token output cap and {LOCAL_CONTEXT_TOKENS}-token "
        "context window. Judge whether the step was appropriately scoped and "
        "whether truncation or over-sized single responses are likely."
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
    gemma_max_tokens: int = GEMMA_MAX_TOKENS,
    review: bool = True,
    review_quality_hint: float = 0.0,
    terminal_tool: callable = None,
    max_terminal_rounds: int = 3,
    bound_text: str = "",
    contract_text: str = "",
    source_context: str = "",
    evidence_provider: callable = None,
) -> SuperviseResult:
    """Run Gemma-primary / DeepSeek-supervisor on a task.

    `package_builder(task, gemma_output, iteration) -> ReviewPackage` lets the
    caller supply verification (test/lint/build results, git diff, etc.).

    `gemma_generate(request) -> ModelResponse` overrides how Gemma is invoked.
    Pass a streaming wrapper here so the caller can stream Gemma's output live
    (always showing the local model working). Defaults to `local.generate`.

    `terminal_tool(command) -> output` executes 'RUN: <command>' blocks the
    model emits and feeds the output back, looping up to max_terminal_rounds.
    None disables the terminal tool.

    With `review=False` the DeepSeek review is skipped entirely (local-first
    routing): one Gemma pass, then a synthetic APPROVED verdict so the
    caller's apply/stats tail works unchanged. Zero API spend.
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
            resp: ModelResponse = gemma_generate(
                _gemma_primary_prompt(current, prior_fixes, bound_text=bound_text,
                    contract_text=contract_text, source_context=source_context,
                    max_tokens=gemma_max_tokens)
            )
        except Exception as exc:  # noqa: BLE001
            status(f"[supervise] local failed ({exc}); escalating to DeepSeek")
            text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=True)
            result.final_text = text
            result.escalated = True
            result.reason = "cloud_fallback_truncated" if still_truncated else "local_failure_escalation"
            return result

        # Truncated local output must never reach the reviewer or be applied
        # as if complete. Retry ONCE with a doubled budget (long multi-file
        # generations routinely need more than a single budget); only escalate
        # when the retry is still cut off.
        if getattr(resp, "truncated", False):
            retry_budget = min(gemma_max_tokens * 2, GEMMA_MAX_TOKENS_CAP)
            if retry_budget >= gemma_max_tokens:
                status(f"[supervise] iter {iteration}: Gemma output TRUNCATED at "
                       f"{gemma_max_tokens} tokens - retrying once with {retry_budget}")
                try:
                    resp = gemma_generate(
                        _gemma_primary_prompt(current, prior_fixes, bound_text=bound_text,
                    contract_text=contract_text, source_context=source_context,
                    max_tokens=retry_budget)
                    )
                except Exception as exc:  # noqa: BLE001
                    status(f"[supervise] local retry failed ({exc}); escalating to DeepSeek")
                    text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False)
                    result.final_text = text
                    result.escalated = True
                    result.reason = "cloud_fallback_truncated" if still_truncated else "local_truncation_escalation"
                    return result
            if getattr(resp, "truncated", False):
                status("[supervise] Gemma still TRUNCATED after retry; escalating to DeepSeek")
                text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False)
                result.final_text = text
                result.escalated = True
                result.reason = "cloud_fallback_truncated" if still_truncated else "local_truncation_escalation"
                return result
        code = resp.text

        # Terminal tool: execute the 'RUN: <command>' blocks the model emitted
        # and feed the output back so it can iterate against a real shell
        # (inspect errors, run checks, then emit the final code).
        terminal_feedback = ""
        terminal_rounds = 0
        while terminal_tool is not None and max_terminal_rounds > 0:
            cmds = _extract_run_blocks(code)
            if not cmds or terminal_rounds >= max_terminal_rounds:
                break
            terminal_rounds += 1
            status(f"[supervise] iter {iteration}: terminal round {terminal_rounds} — "
                   f"running: {', '.join(c[:44] for c in cmds)}")
            outputs = [f"$ {cmd}\n{terminal_tool(cmd)}" for cmd in cmds]
            terminal_feedback = "\n\n".join(outputs)
            try:
                resp = gemma_generate(_gemma_primary_prompt(
                    current, prior_fixes, terminal_output=terminal_feedback,
                    bound_text=bound_text, contract_text=contract_text,
                    source_context=source_context, max_tokens=gemma_max_tokens))
            except Exception as exc:  # noqa: BLE001 - keep the last good output
                status(f"[supervise] terminal re-generate failed ({exc}); continuing")
                break
            code = resp.text

        # Local-first routing: skip the DeepSeek review entirely. One Gemma
        # pass plus a synthetic APPROVED verdict keeps the apply/stats tail
        # unchanged. review_quality_hint carries the router confidence (0-10).
        if not review:
            result.final_text = code
            result.reason = "router_local_skip_review"
            result.verdicts.append(Verdict(
                decision="APPROVED",
                quality_score=review_quality_hint or 7.0,
                assessment="Router decided this task needs no DeepSeek review "
                           "(local-first supervision plan).",
            ))
            status(f"[supervise] iter {iteration}: local-first — DeepSeek review skipped")
            return result

        # 2. Build the compact review package.
        pkg = package_builder(task, code, iteration) if package_builder else _default_package(task, code)
        if terminal_feedback:
            pkg.verification = ((pkg.verification + "\n\n" if pkg.verification else "")
                                + "TERMINAL SESSION:\n" + terminal_feedback)

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

        if verdict.decision == "UNKNOWN":
            # First-class UNKNOWN: collect machine evidence once and re-review.
            evidence = evidence_provider() if evidence_provider else None
            if evidence:
                status(f"[supervise] iter {iteration}: UNKNOWN — collecting evidence")
                pkg.verification = ((pkg.verification + "\n\n" if pkg.verification else "")
                                    + "MACHINE EVIDENCE:\n" + evidence[:3000])
                try:
                    review_resp = cloud.generate(_supervisor_request(pkg))
                except Exception as exc:  # noqa: BLE001
                    status(f"[supervise] evidence re-review failed ({exc})")
                    evidence = None
                if evidence:
                    verdict = parse_verdict(review_resp.text)
                    result.verdicts.append(verdict)
                    status(f"[supervise] iter {iteration}: re-review -> {verdict.decision}")
            if verdict.decision == "UNKNOWN":
                result.final_text = code
                result.reason = "unknown_evidence"
                status("[supervise] ⛔ UNKNOWN: not enough machine evidence — "
                       "requiring human review")
                return result

        if verdict.approved:
            status(f"[supervise] iter {iteration}: APPROVED (score {verdict.quality_score:.1f}/10)")
            result.final_text = code
            return result
        if verdict.rejected:
            status(f"[supervise] iter {iteration}: REJECTED (score {verdict.quality_score:.1f}/10) — falling back to DeepSeek")
            text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False)
            result.final_text = text
            result.escalated = True
            result.reason = "cloud_fallback_truncated" if still_truncated else "deepseek_fallback_rejected"
            return result

        # FIX_REQUIRED: feed fixes back to Gemma.
        prior_fixes = required_fixes_summary(verdict)
        status(f"[supervise] iter {iteration}: FIX_REQUIRED — {len(verdict.issues)} issue(s), looping")
        current = task

    # Retries exhausted with FIX_REQUIRED: fall back to DeepSeek (80/20 fallback).
    status(f"[supervise] max iterations ({max_iterations}) reached; falling back to DeepSeek")
    text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False)
    result.final_text = text
    result.escalated = True
    result.reason = "cloud_fallback_truncated" if still_truncated else f"deepseek_fallback_max_iterations_{max_iterations}"
    return result


def _default_package(task: str, code: str, iteration: int = 0) -> ReviewPackage:
    return ReviewPackage(
        task=task,
        changes=f"Gemma output (iteration {iteration}):\n{code}",
        uncertainties="None provided — caller can enrich this package with tests/diff.",
    )


def _cloud_generate_guarded(cloud: Backend, task: str, status: callable,
                            use_plan: bool) -> tuple[str, bool]:
    """Generate the DeepSeek fallback, retrying ONCE at the same budget when the
    output is truncated (the API caps output at 8192, so the budget cannot grow).

    Returns (text, still_truncated). When still_truncated is True the caller
    must NOT let the output be applied as if complete.
    """
    request = _cloud_plan_request(task) if use_plan else _cloud_generate_request(task)
    response = cloud.generate(request)
    if getattr(response, "truncated", False):
        status("[supervise] cloud output truncated; retrying once at same budget")
        response = cloud.generate(request)
    if getattr(response, "truncated", False):
        status("[supervise] cloud output STILL truncated — INCOMPLETE, must NOT be applied")
    return response.text, bool(getattr(response, "truncated", False))


def _enhance_request(task: str, context: str = "", cot: bool = False) -> ModelRequest:
    """Request DeepSeek to ENHANCE the task and plan it for the local implementer.

    Runs BEFORE the local model implements: DeepSeek clarifies the user's prompt
    into a self-contained implementation prompt AND produces a step-by-step plan
    where each step fits in ONE local response (context window / output cap),
    so the local model never receives an oversized prompt that guarantees
    truncation. With cot=True it also shows its step-by-step reasoning
    (TASK UNDERSTANDING / CONSTRAINT ANALYSIS / ALTERNATIVES sections).
    """
    cot_extra = (
        "\n\nAdditionally produce these chain-of-thought sections so the user "
        "can verify your reasoning:\n"
        "=== TASK UNDERSTANDING ===\n"
        "<what the task really asks for>\n"
        "=== CONSTRAINT ANALYSIS ===\n"
        "<how you size each step to the local model's limits>\n"
        "=== ALTERNATIVES ===\n"
        "<other approaches you considered and why you chose this one>\n"
        if cot else ""
    )
    return ModelRequest(
        system=(
            "You are the senior architect for a hybrid coding agent. Your job:\n"
            "(a) ENHANCE the user's prompt into a clear, self-contained, "
            "unambiguous implementation prompt ready to send to the local "
            "implementer. Do not invent requirements that conflict with the task.\n"
            "(b) Produce a REASONING section explaining what you improved and why.\n"
            "(c) Produce a PLAN section giving a step-by-step implementation plan "
            "where EACH STEP fits in ONE local model response.\n\n"
            "Format your response with these exact section markers:\n"
            "=== ENHANCED PROMPT ===\n"
            "<the enhanced prompt>\n"
            "=== TASK CONTRACT ===\n"
            "ALWAYS include this machine-readable contract; every stage follows "
            "it. Use exactly these labeled fields (lists as '- item' lines):\n"
            "Goal: <one-sentence goal>\n"
            "Must change:\n- <file or behavior that MUST change>\n"
            "Must NOT change:\n- <things that must stay untouched, if any>\n"
            "Acceptance criteria:\n- <verifiable criteria>\n"
            "Files likely involved:\n- <path>, <path>\n"
            "Dependencies:\n- <apis/modules the change depends on, if known>\n"
            "Risk: <LOW | MEDIUM | HIGH | CRITICAL>\n"
            "Verification required:\n- <build | unit tests | integration tests | "
            "browser journey, as appropriate>\n"
            "Rollback strategy: <how to undo this change safely>\n"
            "=== ACCEPTANCE CASES ===\n"
            "A concrete input->expected table for the acceptance criteria "
            "(e.g. 'min price £20 -> accepted'), one per line:\n"
            "- <input> -> <expected>\n"
            "=== REASONING ===\n"
            "<what you improved and why>\n"
            "=== PLAN ===\n"
            "<step-by-step plan>\n\n"
            "=== CLARIFYING QUESTIONS ===\n"
            "ONLY include this section if the ORIGINAL USER TASK is ambiguous, "
            "under-specified, or unclear. List 1-5 concise, concrete questions or "
            "answer-options (e.g., exact target files, expected behavior, edge cases, "
            "scope). If the task is already clear, OMIT this section entirely.\n"
            f"{cot_extra}\n"
            f"{_local_limits_note()}"
        ),
        user=f"ORIGINAL USER TASK:\n{task}\n"
             + (f"\nCONTEXT:\n{context}\n" if context else ""),
        max_tokens=2400 if not cot else 3200,
        temperature=0.1,
    )


def _cloud_plan_request(task: str) -> ModelRequest:
    """Request DeepSeek to plan the task sized to the local implementer.

    The plan must fit the work into steps that the local model can produce in
    ONE response (its context window and output cap), each with a file-level
    breakdown and acceptance criteria — otherwise the plan guarantees
    truncation and the loop burns tokens on retries and escalation.
    """
    return ModelRequest(
        system=(
            "You are the senior architect for a hybrid coding agent. Produce a "
            "step-by-step plan with file-level breakdown (max 400 words). "
            "Each step must fit in ONE local model response (target under ~3500 "
            "output tokens), with per-step acceptance criteria. Split any task "
            "needing more output into multiple sequential steps.\n\n"
            f"{_local_limits_note()}"
        ),
        user=f"OBJECTIVE: {task}\n",
        max_tokens=1200,
        temperature=0.1,
    )


def _cloud_generate_request(task: str) -> ModelRequest:
    """Request DeepSeek to generate the full implementation (80/20 fallback path)."""
    return ModelRequest(
        system=(
            "You are a senior engineer. Given the task, produce the COMPLETE "
            "implementation — all files, in fenced code blocks labeled with paths. "
            "No preamble."
        ),
        user=f"TASK:\n{task}\n",
        max_tokens=CLOUD_GEN_TOKENS,
        temperature=0.1,
    )
