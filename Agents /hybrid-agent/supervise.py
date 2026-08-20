"""
Qwen-primary / DeepSeek-supervisor control loop.

Architecture (vs the router in agent.py):
  Instead of deciding "which model does the whole task", Qwen (local) is the
  primary coding agent for almost everything. DeepSeek only intervenes as a
  supervisor/reviewer, and only when the task or verification warrants it.

  Flow:
    TASK
      -> Qwen implements (local backend)
      -> build a COMPACT review package (not the whole conversation)
      -> optional verification hook
      -> DeepSeek supervises (compact package only)
      -> VERDICT:
           APPROVED      -> done
           FIX_REQUIRED  -> feed REQUIRED_FIXES back to Qwen, iterate
           REJECTED      -> stop, report alternative approach
      -> loop until APPROVED or max iterations

  DeepSeek never rewrites code directly; it returns verdicts + required fixes,
  and Qwen implements them. API tokens are spent on judgement, not on doing
  the coding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backends.base import Backend, ModelRequest, ModelResponse
from context_safety import (
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    assess_zone,
    classify_output,
    compact_text,
    context_telemetry,
    dynamic_output_reserve,
    estimate_tokens,
    safe_input_budget,
    summarize_terminal_output,
    supervisor_context_block,
    task_required_output,
)
from recovery import (RecoveryAction, RecoveryManager, classify_failure,
                      default_recovery)


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
    """Build a short, actionable instruction to hand back to Qwen."""
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

# Output budgets: the LOCAL model has NO engine-imposed cap. LM Studio's
# /api/v1/chat rejects max_tokens outright, so the local model generates until
# its own EOS or the server's context window — these constants are advisory
# only (kept astronomically large so no retry/escalation path ever treats the
# local budget as the limiting factor). Truncation (server-side context cutoff)
# is detected and recovered with a continuation prompt, not a budget bump.
QWEN_MAX_TOKENS = 1_000_000_000_000_000
QWEN_MAX_TOKENS_CAP = 2_000_000_000_000_000
CLOUD_GEN_TOKENS = 8192

# Local model constraints — advisory planning guidance for the DeepSeek
# supervisor (steps sized for one local response), NOT an enforced cap.
# DEFAULT values apply when the real window cannot be discovered; ask.py
# overrides these with the ACTUAL loaded window (via LM Studio /api/v1/models
# `loaded_instances[].config.context_length`) so planning always matches the
# server, whatever local model is loaded.
LOCAL_CONTEXT_TOKENS = 32768
LOCAL_OUTPUT_TOKENS = 4096


def _local_limits_note(context_window: int = LOCAL_CONTEXT_TOKENS,
                       output_tokens: int = LOCAL_OUTPUT_TOKENS,
                       output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                       safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> str:
    """Compact, factual description of the implementer's constraints.

    Injected into the DeepSeek plan request so the supervisor sizes every step
    for one local response instead of assuming an unlimited model.
    `context_window` must be the ACTUALLY LOADED window of the local model —
    a smaller window than planned means server-side mid-generation cutoffs.
    Includes the SAFE INPUT budget (window - output reserve - safety margin):
    the real usable space the supervisor must plan within.
    """
    safe_input = safe_input_budget(context_window, output_reserve_tokens,
                                   safety_margin_tokens)
    return (
        "IMPLEMENTER CONSTRAINTS (local model, must be respected): "
        f"context window {context_window} tokens; SAFE INPUT BUDGET "
        f"{safe_input} tokens (window - {output_reserve_tokens} output reserve "
        f"- {safety_margin_tokens} safety margin — never fill the window); "
        f"a step of about {output_tokens} output tokens is a safe planning "
        "target (there is NO enforced output cap — the engine detects "
        "truncation and continues the response, so an occasional overflow is "
        "recovered, but complete one-response steps are preferred); keep "
        f"prompt + repo context well under {safe_input}; any task needing "
        "more output must be split into multiple sequential steps with "
        "per-step acceptance criteria; truncated output is continued once "
        "then escalated."
    )


def _qwen_primary_prompt(task: str, prior_fixes: str = "",
                          terminal_output: str = "", bound_text: str = "",
                          contract_text: str = "", source_context: str = "",
                          continuation_from: str = "",
                          max_tokens: int = QWEN_MAX_TOKENS,
                          context_window: int = LOCAL_CONTEXT_TOKENS,
                          output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                          safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> ModelRequest:
    system = (
        "You are a careful mid-level engineer implementing a change. "
        "Write correct, minimal, well-structured code for the given task. "
        "Output the complete updated file(s) in fenced code blocks labeled with "
        "their file paths, e.g. ```utils.py / ```src/main.ts / ```README.md — "
        "NEVER a bare language tag like ```python or ```markdown, because the "
        "engine uses the fence label to know where to write each file. "
        "You have a terminal tool: to inspect or verify, emit a line like "
        "'RUN: npm run build' and you will receive the command output in the "
        "next turn. Use it to confirm your changes work."
    )
    if bound_text:
        system += "\n\n" + bound_text  # RECALL gate: the BOUND re-injected every iteration

    # Context Safety Controller: compute the hard input budget (window minus
    # output reserve minus safety margin) and compact OPTIONAL sections to fit.
    # NEVER let the assembled prompt fill the window — the model needs room to
    # produce its response.
    #
    # PROTECTED CORE (invariants, NEVER dropped or compacted):
    #   - TASK (current task requirements)
    #   - BOUND text (constraints + security requirements)
    #   - CONTRACT (files being modified, acceptance criteria)
    #   - prior_fixes (supervisor instructions)
    #   - the NOTE/tail (engine protocol)
    # COMPACTABLE (bulk context, trimmed in order of least value):
    #   - source context (biggest chunk) -> terminal output (summarized)
    # If even the protected core exceeds the safe input budget, the prompt is
    # returned INTACT and the preflight RED zone escalates — the task is never
    # truncated by the compactor.
    sys_tokens = estimate_tokens(system)

    # Fixed trailing text (NOTE + continuation tail) is measured FIRST so the
    # compaction below reserves room for it — otherwise the final NOTE would
    # silently overflow the budget after the sections were sized.
    note = (
        "\nNOTE: you have NO output token cap — output the COMPLETE change in "
        "this one response. Do not stop early or split the work voluntarily; "
        "write every file fully. If your response is truncated the engine "
        "detects it and asks you to continue exactly where you stopped, so "
        "never abbreviate. Only if the change genuinely cannot fit, end with a "
        "final 'REMAINING WORK' line listing exactly what is left."
    )
    tail = "\nOutput only the code changes, no preamble."
    if continuation_from:
        # Truncation recovery: the model is stateless across calls, so feed it
        # the tail of the cut-off output and demand a pure continuation. Works
        # for ANY local model regardless of whether max_tokens was honoured.
        note = note.replace(
            "\nNOTE: you have NO output token cap — output the COMPLETE change in "
            "this one response. Do not stop early or split the work voluntarily; "
            "write every file fully.",
            "\nNOTE: continue the response you were writing. Output only the "
            "missing remainder.")
        tail += (
            "\n\nA previous attempt was truncated mid-output. It ended with:\n"
            f"--- TRUNCATED TAIL ---\n{continuation_from}\n--- END ---\n"
            "Continue EXACTLY from where that output stopped. Do not repeat "
            "anything already written above. Output only the missing remainder."
        )
    fixed_overhead = estimate_tokens(note) + estimate_tokens(tail)
    user_budget = max(0, safe_input_budget(context_window, output_reserve_tokens,
                                           safety_margin_tokens) - sys_tokens)

    def _fits(user_text: str) -> bool:
        return estimate_tokens(user_text) + fixed_overhead <= user_budget

    # PROTECTED CORE — never compacted (task, contract, bound, fixes).
    core = f"TASK:\n{task}\n"
    if contract_text:
        core += f"\n{contract_text}\n"  # the formal Task Contract (same one every stage uses)
    if prior_fixes:
        core += f"\nA senior reviewer previously asked you to fix these issues. Apply ALL of them now:\n{prior_fixes}\n"
    user = core

    # COMPACTABLE sections — trimmed in order of least value, never the core.
    compacted = []
    src = ""
    if source_context and not continuation_from:
        # Dependency-aware context is only useful when starting fresh. On a
        # truncation-continuation retry the prompt is already near the context
        # window (that is why it was cut off), so re-injecting the source
        # context would push the continuation over the edge again and guarantee
        # a second truncation. The tail alone anchors the model.
        src = (f"\nRELEVANT SOURCE (dependency-aware context for the files "
               f"involved):\n{source_context}\n")
        if not _fits(user + src):
            # Compact the biggest chunk first: source context. Leave slack
            # (the marker + label add a few tokens beyond max_tokens).
            src = (f"\nRELEVANT SOURCE (dependency-aware context, compacted):\n"
                   f"{compact_text(source_context, max(256, user_budget - estimate_tokens(user) - fixed_overhead - 32))}\n")
            compacted.append("source")
        user += src
    if terminal_output:
        term = (
            f"\nTERMINAL SESSION (output of the commands you asked to run):\n"
            f"{terminal_output}\n"
            "Use this output to fix your changes. You may emit more "
            "'RUN: <command>' lines to inspect or verify, then output the "
            "final code."
        )
        if not _fits(user + term):
            # Summarize/compact tool output so it never floods the window.
            summary = summarize_terminal_output(terminal_output)
            term = (
                f"\nTERMINAL SESSION (SUMMARIZED output — full text kept "
                f"separately):\n{summary}\n"
                "Use this output to fix your changes. You may emit more "
                "'RUN: <command>' lines to inspect or verify, then output the "
                "final code."
            )
            compacted.append("terminal")
        user += term
    user += note + tail

    # Hard cap: if even the PROTECTED CORE (without any compactable section)
    # exceeds the budget, return it intact — the preflight RED zone must
    # escalate rather than truncate the task/constraints. This preserves the
    # invariant: compaction NEVER removes task requirements, BOUND, contract,
    # supervisor fixes, or the engine protocol.
    if estimate_tokens(user) > user_budget and src:
        user = core + note + tail
        compacted.append("source-dropped")
    req = ModelRequest(system=system, user=user, max_tokens=max_tokens, temperature=0.2)
    req.metadata["context_safety"] = {
        "compaction": ",".join(compacted) if compacted else "none",
        "input_tokens": estimate_tokens(system) + estimate_tokens(user),
        "safe_budget": max(0, safe_input_budget(context_window, output_reserve_tokens,
                                                safety_margin_tokens)),
    }
    return req


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


def _supervisor_request(pkg: ReviewPackage, verbose: bool = True,
                        context_window: int = LOCAL_CONTEXT_TOKENS,
                        local_model: str = "local model",
                        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                        safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
                        step: int = 1, step_total: int = 1,
                        tools: str = "terminal, filesystem, tests",
                        zone: str = "GREEN") -> ModelRequest:
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
        "whole codebase — you are reviewing, Qwen implements.\n\n"
        "EVIDENCE RULE: if you return APPROVED you MUST cite the concrete machine "
        "evidence (test names, file:line, command output) under a "
        "'=== EVIDENCE ===' section. If you cannot point to machine evidence for "
        "your verdict, return UNKNOWN instead of guessing.\n\n"
        "Note: the code under review was produced by a local model with "
        "no engine-imposed output cap and a "
        f"{context_window}-token context window (truncation is detected "
        "and recovered via continuation, not by capping the model). Judge "
        "whether the step was appropriately scoped and whether responses "
        "look cut off or over-sized for a single pass.\n\n"
        + supervisor_context_block(
            local_model, context_window, output_reserve_tokens,
            safety_margin_tokens, step, step_total, tools, zone)
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
    reason: str = "qwen_primary"
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
    qwen_generate: callable = None,
    qwen_max_tokens: int = QWEN_MAX_TOKENS,
    review: bool = True,
    review_quality_hint: float = 0.0,
    terminal_tool: callable = None,
    max_terminal_rounds: int = 3,
    bound_text: str = "",
    contract_text: str = "",
    source_context: str = "",
    evidence_provider: callable = None,
    context_window: int = LOCAL_CONTEXT_TOKENS,
    output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
    local_model: str = "local model",
    tools: str = "terminal, filesystem, tests",
    model_max_output: int = 0,
    contract_files: list | None = None,
    max_recovery_retries: int = 1,
    telemetry: callable = None,
    architecture: str = "",
    vision: bool = False,
    tool_use: bool = False,
    recovery: RecoveryManager | None = None,
) -> SuperviseResult:
    """Run Qwen-primary / DeepSeek-supervisor on a task.

    `package_builder(task, qwen_output, iteration) -> ReviewPackage` lets the
    caller supply verification (test/lint/build results, git diff, etc.).

    `qwen_generate(request) -> ModelResponse` overrides how Qwen is invoked.
    Pass a streaming wrapper here so the caller can stream Qwen's output live
    (always showing the local model working). Defaults to `local.generate`.

    `terminal_tool(command) -> output` executes 'RUN: <command>' blocks the
    model emits and feeds the output back, looping up to max_terminal_rounds.
    None disables the terminal tool.

    With `review=False` the DeepSeek review is skipped entirely (local-first
    routing): one Qwen pass, then a synthetic APPROVED verdict so the
    caller's apply/stats tail works unchanged. Zero API spend.

    CONTEXT SAFETY: before EVERY local-model call the prompt is measured
    against the safe input budget (context_window - output_reserve_tokens -
    safety_margin_tokens) and preflighted into a zone. GREEN proceeds;
    YELLOW/ORANGE trigger compaction (source context and terminal output are
    the first sections dropped/summarized); RED means the prompt cannot be
    made safe — the local model is NOT called and the task escalates to
    DeepSeek instead of guaranteeing a truncated generation.

    DYNAMIC OUTPUT RESERVE: when `model_max_output` (per-model cap) and/or
    `contract_files` (task need) are supplied, the effective reserve becomes
    min(configured, model_max_output, task_required_output) so a tiny task
    does not sacrifice 12k tokens of context.

    RETRY BUDGET: recovery loops (continuation after truncated output) are
    hard-capped at `max_recovery_retries` per iteration (default 1: one
    continuation, then supervisor escalation) — a pathological model cannot
    burn unbounded time/tokens recovering.

    `telemetry(line)` receives one structured context_telemetry() string after
    each local-model call (and on RED) for logs/dashboard.
    """
    if status is None:
        status = lambda line: None  # noqa: E731
    if qwen_generate is None:
        qwen_generate = lambda req: local.generate(req)  # noqa: E731
    # Centralized recovery: every failure in this loop routes through ONE
    # policy (classify -> bounded retry/compact/switch/escalate). Pass a
    # RecoveryManager bound to the durable TaskState to record attempts.
    recovery_mgr = recovery or default_recovery

    result = SuperviseResult(task=task)
    current = task
    prior_fixes = ""
    # Dynamic output reserve: never waste context on a reserve bigger than the
    # model can produce or the task needs.
    eff_reserve = dynamic_output_reserve(
        output_reserve_tokens,
        model_max_output=model_max_output,
        task_required_output=task_required_output(task, contract_files))
    safe_budget = safe_input_budget(context_window, eff_reserve,
                                    safety_margin_tokens)
    zone = "GREEN"

    def _telemetry(**kw) -> None:
        if telemetry is not None:
            try:
                telemetry(context_telemetry(
                    model=local_model, window=context_window,
                    safe_input=safe_budget, output_reserve=eff_reserve,
                    step=iteration, step_total=max_iterations,
                    max_output=model_max_output,
                    architecture=architecture, vision=vision, tool_use=tool_use,
                    **kw))
            except Exception:  # noqa: BLE001 - telemetry must never break runs
                pass

    def _preflight(req: ModelRequest, tag: str) -> ModelRequest:
        """Context Safety Controller: measure + preflight before calling Qwen.
        YELLOW/ORANGE compact in place (already done by the prompt builder);
        RED is caught by the caller, which must NOT call the model."""
        nonlocal zone
        used = estimate_tokens(req.system) + estimate_tokens(req.user)
        zone = assess_zone(used, safe_budget)
        status(f"[supervise] iter {iteration}: context {zone} "
               f"({used}/{safe_budget} tokens) {tag}")
        return req

    def _safe_reason(still_truncated: bool, base: str) -> str:
        return "cloud_fallback_truncated" if still_truncated else base

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration

        # 1. CONTEXT SAFETY CONTROLLER: build + preflight the payload BEFORE
        #    calling Qwen. The prompt builder already compacts to the safe
        #    input budget; RED here means even the compacted task cannot fit.
        req = _qwen_primary_prompt(
            current, prior_fixes, bound_text=bound_text,
            contract_text=contract_text, source_context=source_context,
            max_tokens=qwen_max_tokens, context_window=context_window,
            output_reserve_tokens=eff_reserve,
            safety_margin_tokens=safety_margin_tokens)
        _preflight(req, "-> Qwen")

        if zone == "RED":
            status(f"[supervise] iter {iteration}: ⛔ context RED — the task "
                   f"cannot fit the local model's safe budget ({safe_budget} "
                   f"tokens); NOT calling Qwen; escalating to DeepSeek")
            _telemetry(input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
                       compaction="red", status="RED_ESCALATION")
            # Centralized recovery: context failures -> compact (already done
            # by the prompt builder) then escalate; NEVER call Qwen in RED.
            decision = recovery_mgr.decide("context RED", scope=f"iter{iteration}")
            status(f"[supervise] recovery decision: {decision.action.value} "
                   f"({decision.detail})")
            text, still_truncated = _cloud_generate_guarded(
                cloud, task, status, use_plan=True, context_window=context_window,
                output_reserve_tokens=eff_reserve,
                safety_margin_tokens=safety_margin_tokens)
            result.final_text = text
            result.escalated = True
            result.reason = _safe_reason(still_truncated, "local_context_red_escalation")
            return result

        try:
            resp: ModelResponse = qwen_generate(req)
        except Exception as exc:  # noqa: BLE001
            # Centralized recovery: model failures -> switch model (DeepSeek
            # fallback) on the first occurrence, escalate afterwards.
            decision = recovery_mgr.handle_failure(
                f"local failed ({exc})", scope=f"iter{iteration}")
            status(f"[supervise] local failed ({exc}); recovery: "
                   f"{decision.action.value} ({decision.detail})")
            _telemetry(input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
                       compaction="none", status="MODEL_ERROR")
            text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=True,
                                  context_window=context_window,
                                  output_reserve_tokens=eff_reserve,
                                  safety_margin_tokens=safety_margin_tokens)
            result.final_text = text
            result.escalated = True
            result.reason = _safe_reason(still_truncated, "local_failure_escalation")
            return result

        # Distinguish WHY the local output is incomplete (never silently treat
        # a cutoff as success): OUTPUT_LIMIT_REACHED (model hit its own stop
        # budget), CONTEXT_LIMIT_REACHED (server window cutoff), TIMEOUT
        # (stream died), COMPLETED (clean).
        out_state = classify_output(resp)
        recovery_retries = 0
        while out_state != "COMPLETED" and recovery_retries < max_recovery_retries:
            # RETRY BUDGET: hard cap on continuation recovery loops — a
            # pathological model cannot burn unbounded time/tokens retrying.
            recovery_retries += 1
            tail = (resp.text or "")[-1200:]
            retry_budget = min(qwen_max_tokens * 2, QWEN_MAX_TOKENS_CAP)
            status(f"[supervise] iter {iteration}: local output {out_state} "
                   f"({getattr(resp, 'truncate_reason', 'no-reason')}) - "
                   f"retry {recovery_retries}/{max_recovery_retries} with a "
                   f"continuation prompt")
            try:
                resp = qwen_generate(_qwen_primary_prompt(
                    current, prior_fixes, bound_text=bound_text,
                    contract_text=contract_text, source_context=source_context,
                    continuation_from=tail, max_tokens=retry_budget,
                    context_window=context_window,
                    output_reserve_tokens=eff_reserve,
                    safety_margin_tokens=safety_margin_tokens))
            except Exception as exc:  # noqa: BLE001
                # Centralized recovery for the continuation retry.
                decision = recovery_mgr.handle_failure(
                    f"local retry failed ({exc})", scope=f"iter{iteration}-retry")
                status(f"[supervise] local retry failed ({exc}); recovery: "
                       f"{decision.action.value} ({decision.detail})")
                _telemetry(input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
                           compaction="none", status="MODEL_ERROR")
                text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False,
                                  context_window=context_window,
                                  output_reserve_tokens=eff_reserve,
                                  safety_margin_tokens=safety_margin_tokens)
                result.final_text = text
                result.escalated = True
                result.reason = _safe_reason(still_truncated, "local_truncation_escalation")
                return result
            out_state = classify_output(resp)
        if out_state != "COMPLETED":
            status(f"[supervise] Qwen still {out_state} after "
                   f"{recovery_retries}/{max_recovery_retries} retries "
                   "(retry budget exhausted); escalating to DeepSeek")
            _telemetry(input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
                       compaction="none", status=out_state)
            text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False,
                              context_window=context_window,
                              output_reserve_tokens=eff_reserve,
                              safety_margin_tokens=safety_margin_tokens)
            result.final_text = text
            result.escalated = True
            result.reason = _safe_reason(still_truncated, "local_truncation_escalation")
            return result
        # Telemetry: one structured line per completed local call.
        compaction = (req.metadata.get("context_safety") or {}).get("compaction", "none")
        _telemetry(
            input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
            compaction=compaction,
            output_tokens=len(resp.text or "") // 4,
            status="COMPLETED")
        code = resp.text

        # Terminal tool: execute the 'RUN: <command>' blocks the model emitted
        # and feed the output back so it can iterate against a real shell
        # (inspect errors, run checks, then emit the final code). Tool output
        # is summarized by the prompt builder so a 20k-token log never floods
        # the local model's window (full text stays in the review package).
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
                resp = qwen_generate(_qwen_primary_prompt(
                    current, prior_fixes, terminal_output=terminal_feedback,
                    bound_text=bound_text, contract_text=contract_text,
                    source_context=source_context, max_tokens=qwen_max_tokens,
                    context_window=context_window,
                    output_reserve_tokens=eff_reserve,
                    safety_margin_tokens=safety_margin_tokens))
            except Exception as exc:  # noqa: BLE001 - keep the last good output
                status(f"[supervise] terminal re-generate failed ({exc}); continuing")
                break
            code = resp.text

        # Local-first routing: skip the DeepSeek review entirely. One Qwen
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

        # 3. DeepSeek supervises — with the LOCAL MODEL CONTEXT block so it
        #    plans/reviews within the real safe input budget, current step,
        #    available tools, and the observed context zone.
        status(f"[supervise] iter {iteration}: DeepSeek supervising...")
        try:
            review_resp = cloud.generate(
                _supervisor_request(
                    pkg, context_window=context_window, local_model=local_model,
                    output_reserve_tokens=eff_reserve,
                    safety_margin_tokens=safety_margin_tokens,
                    step=iteration, step_total=max_iterations,
                    tools=tools, zone=zone))
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
                    review_resp = cloud.generate(
                        _supervisor_request(
                            pkg, context_window=context_window, local_model=local_model,
                            output_reserve_tokens=eff_reserve,
                            safety_margin_tokens=safety_margin_tokens,
                            step=iteration, step_total=max_iterations,
                            tools=tools, zone=zone))
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
            text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False,
                                  context_window=context_window,
                                  output_reserve_tokens=eff_reserve,
                                  safety_margin_tokens=safety_margin_tokens)
            result.final_text = text
            result.escalated = True
            result.reason = "cloud_fallback_truncated" if still_truncated else "deepseek_fallback_rejected"
            return result

        # FIX_REQUIRED: feed fixes back to Qwen.
        prior_fixes = required_fixes_summary(verdict)
        status(f"[supervise] iter {iteration}: FIX_REQUIRED — {len(verdict.issues)} issue(s), looping")
        current = task

    # Retries exhausted with FIX_REQUIRED: fall back to DeepSeek (80/20 fallback).
    status(f"[supervise] max iterations ({max_iterations}) reached; falling back to DeepSeek")
    text, still_truncated = _cloud_generate_guarded(cloud, task, status, use_plan=False,
                                  context_window=context_window,
                                  output_reserve_tokens=eff_reserve,
                                  safety_margin_tokens=safety_margin_tokens)
    result.final_text = text
    result.escalated = True
    result.reason = "cloud_fallback_truncated" if still_truncated else f"deepseek_fallback_max_iterations_{max_iterations}"
    return result


def _default_package(task: str, code: str, iteration: int = 0) -> ReviewPackage:
    return ReviewPackage(
        task=task,
        changes=f"Qwen output (iteration {iteration}):\n{code}",
        uncertainties="None provided — caller can enrich this package with tests/diff.",
    )


def _cloud_generate_guarded(cloud: Backend, task: str, status: callable,
                            use_plan: bool,
                            context_window: int = LOCAL_CONTEXT_TOKENS,
                            output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                            safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> tuple[str, bool]:
    """Generate the DeepSeek fallback, retrying ONCE at the same budget when the
    output is truncated (the API caps output at 8192, so the budget cannot grow).

    Returns (text, still_truncated). When still_truncated is True the caller
    must NOT let the output be applied as if complete.
    """
    request = (_cloud_plan_request(task, context_window=context_window,
                                   output_reserve_tokens=output_reserve_tokens,
                                   safety_margin_tokens=safety_margin_tokens)
               if use_plan else _cloud_generate_request(task))
    response = cloud.generate(request)
    if getattr(response, "truncated", False):
        status("[supervise] cloud output truncated; retrying once at same budget")
        response = cloud.generate(request)
    if getattr(response, "truncated", False):
        status("[supervise] cloud output STILL truncated — INCOMPLETE, must NOT be applied")
    return response.text, bool(getattr(response, "truncated", False))


def _enhance_request(task: str, context: str = "", cot: bool = False,
                     context_window: int = LOCAL_CONTEXT_TOKENS,
                     output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                     safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> ModelRequest:
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
             f"{_local_limits_note(context_window, output_reserve_tokens=output_reserve_tokens, safety_margin_tokens=safety_margin_tokens)}"
         ),
        user=f"ORIGINAL USER TASK:\n{task}\n"
             + (f"\nCONTEXT:\n{context}\n" if context else ""),
        max_tokens=2400 if not cot else 3200,
        temperature=0.1,
    )


def _cloud_plan_request(task: str, context_window: int = LOCAL_CONTEXT_TOKENS,
                        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                        safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> ModelRequest:
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
            f"{_local_limits_note(context_window, output_reserve_tokens=output_reserve_tokens, safety_margin_tokens=safety_margin_tokens)}"
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
