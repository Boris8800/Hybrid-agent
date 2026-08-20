"""Context Safety Controller (CSC) — prevents the local model from EVER
reaching an unusable context state.

Budget math (never 100% of the window for input):

    discovered_window
        - output_reserve_tokens    (space the model needs to answer)
        - safety_margin_tokens     (hard headroom)
    = safe_input_budget            (max tokens the prompt may consume)

Preflight zones (checked BEFORE every local-model call):

    GREEN   < 70% of budget  -> proceed
    YELLOW  70-85%           -> compress / reduce context
    ORANGE  85-95%           -> aggressive compaction
    RED     > 95%            -> do NOT call the model; compact, then escalate

Also provides: tool-output summarization (so huge command logs never flow back
into the window), output-state classification (COMPLETED / OUTPUT_LIMIT_REACHED
/ CONTEXT_LIMIT_REACHED / TIMEOUT / MODEL_ERROR), a machine-readable context
block for the API supervisor, and structured telemetry for debugging.

Token estimation is deliberately CONSERVATIVE (fewer chars per token than
prose): code and structured text tokenize denser than natural language, so
under-estimating risks overflow. The 3.5 chars/token default over-estimates
tokens vs the classic 4 chars/token, giving the safety controller more margin.
"""

from __future__ import annotations

import os
import re

# Zone thresholds as fractions of the safe input budget.
ZONE_GREEN = 0.70    # < 70%  -> proceed
ZONE_YELLOW = 0.85   # 70-85% -> compress / reduce context
ZONE_ORANGE = 0.95   # 85-95% -> aggressive compaction
ZONE_RED = 1.0       # > 95%  -> do NOT call the model

# Defaults (config.yml review.context_safety overrides these).
DEFAULT_OUTPUT_RESERVE_TOKENS = 12000
DEFAULT_SAFETY_MARGIN_TOKENS = 2000
# Conservative chars-per-token estimate: code/structured text tokenize denser
# than prose, so 3.5 (not 4) gives the controller more safety margin.
DEFAULT_CHARS_PER_TOKEN = float(os.environ.get("HYBRID_CHARS_PER_TOKEN", "3.5"))

# Compaction must NEVER drop these: task requirements, BOUND constraints,
# contract (files being modified / security requirements), supervisor fixes,
# failing-test feedback. Only the bulk context (source files, tool output) is
# compactable. These markers document that invariant in prompts.
PROTECTED_SECTION_HINT = " (protected — never dropped by compaction)"


def estimate_tokens(text: str, chars_per_token: float | None = None) -> int:
    """Conservative token estimate: characters / chars_per_token.

    Default 3.5 chars/token over-estimates tokens vs the prose standard of 4,
    because code and structured text (paths, brackets, symbols) tokenize denser.
    Conservative = safe: the budget controller would rather compact more than
    risk a prompt that silently overflows the model's real window.
    """
    cpt = chars_per_token or DEFAULT_CHARS_PER_TOKEN
    if cpt <= 0:
        cpt = 3.5
    return max(0, int(len(text or "") / cpt))


def safe_input_budget(context_window: int,
                      output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
                      safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS) -> int:
    """Max tokens the INPUT may consume: window - output reserve - margin.
    Never allows input to eat the whole window. Returns 0 for unknown windows
    (callers treat 0 as 'no budget known' and must not call the model)."""
    if context_window <= 0:
        return 0
    return max(0, context_window - output_reserve_tokens - safety_margin_tokens)


def dynamic_output_reserve(configured_reserve: int,
                           model_max_output: int = 0,
                           task_required_output: int = 0) -> int:
    """Size the output reserve to the ACTUAL need, never wasting context:

        output_reserve = min(configured_reserve,
                             model_max_output,        # per-model cap
                             task_required_output)    # what this task needs

    A tiny task does not sacrifice 12k tokens of window; a model with a hard
    output cap is never over-reserved; the configured value bounds everything.
    """
    reserve = max(0, int(configured_reserve or 0))
    if model_max_output and model_max_output > 0:
        reserve = min(reserve, int(model_max_output))
    if task_required_output and task_required_output > 0:
        reserve = min(reserve, int(task_required_output))
    # Never reserve less than a minimal response (one small file).
    return max(512, reserve)


def task_required_output(task: str, contract_files: list[str] | None = None,
                         scale: float = 4.0) -> int:
    """Estimate how many output tokens a task realistically needs, so a small
    task doesn't burn 12k of context on reserve. Rough heuristic: task tokens
    times `scale`, plus per-file overhead for the contract's involved files.
    Callers clamp this with dynamic_output_reserve()."""
    base = estimate_tokens(task or "")
    files = list(contract_files or [])
    per_file = 300  # a small file is ~300 tokens of output
    return int(base * scale) + len(files) * per_file


def assess_zone(input_tokens: int, budget: int) -> str:
    """GREEN / YELLOW / ORANGE / RED for the current input size vs the budget."""
    if budget <= 0:
        return "RED"
    ratio = input_tokens / budget
    if ratio < ZONE_GREEN:
        return "GREEN"
    if ratio < ZONE_YELLOW:
        return "YELLOW"
    if ratio < ZONE_ORANGE:
        return "ORANGE"
    return "RED"


def compact_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit max_tokens, keeping the head (imports, signatures,
    first lines) plus a marker. Never silently drops the tail without notice."""
    if not text or max_tokens <= 0:
        return text if text else ""
    budget_chars = int(max_tokens * DEFAULT_CHARS_PER_TOKEN)
    if len(text) <= budget_chars:
        return text
    keep = max(0, budget_chars - 64)  # leave room for the marker
    return text[:keep] + f"\n... [context budget: kept {keep}/{len(text)} chars]"


def summarize_terminal_output(raw: str, max_tokens: int = 1500) -> str:
    """Compress command output for re-injection into the local model: exit
    code, error lines, stack traces, changed files, warnings, and a short head/
    tail — NOT the raw 4000+ char dump. Keeps the original available elsewhere
    (the review package) without flooding the model's window."""
    if not raw:
        return raw or ""
    lines = raw.splitlines()
    exit_code = next((ln for ln in lines if ln.startswith("exit ")), "")
    errors, traces, changed, warnings = [], [], [], []
    for ln in lines:
        low = ln.lower()
        if re.search(r"\b(error|exception|failed|fatal|cannot|could not)\b", low):
            errors.append(ln)
        elif "traceback" in low or re.match(r"\s+at ", ln) or re.match(r'file "', ln):
            traces.append(ln)
        elif re.match(r"^(changed|modified|created|deleted|updated)\s", low):
            changed.append(ln)
        elif "warning" in low:
            warnings.append(ln)
    head_tail = lines[:15] + (["..."] if len(lines) > 30 else []) + lines[-10:]
    parts = [exit_code] if exit_code else []
    if errors:
        parts.append("ERRORS:\n" + "\n".join(errors[:15]))
    if traces:
        parts.append("STACK:\n" + "\n".join(traces[:10]))
    if changed:
        parts.append("CHANGED:\n" + "\n".join(changed[:10]))
    if warnings:
        parts.append("WARNINGS:\n" + "\n".join(warnings[:10]))
    parts.append("OUTPUT:\n" + "\n".join(head_tail))
    return compact_text("\n\n".join(parts), max_tokens)


def classify_output(resp) -> str:
    """COMPLETED / OUTPUT_LIMIT_REACHED / CONTEXT_LIMIT_REACHED / TIMEOUT /
    MODEL_ERROR for a ModelResponse. Lets the caller distinguish a clean
    finish from every failure mode instead of treating all as 'truncated'."""
    truncated = bool(getattr(resp, "truncated", False))
    if not truncated:
        return "COMPLETED"
    reason = (getattr(resp, "truncate_reason", "") or "").lower()
    if "context" in reason:
        return "CONTEXT_LIMIT_REACHED"
    if "eof" in reason or "stream" in reason:
        return "TIMEOUT"
    if "fence" in reason or "finish" in reason or "max_tokens" in reason:
        return "OUTPUT_LIMIT_REACHED"
    return "OUTPUT_LIMIT_REACHED"


def supervisor_context_block(local_model: str, context_window: int,
                             output_reserve_tokens: int,
                             safety_margin_tokens: int,
                             step: int, step_total: int,
                             tools: str, zone: str) -> str:
    """Machine-readable context block for the DeepSeek supervisor so it plans
    within the REAL usable context — not just 'the window is N'."""
    budget = safe_input_budget(context_window, output_reserve_tokens,
                               safety_margin_tokens)
    return (
        "LOCAL MODEL CONTEXT (hard numbers — plan within them):\n"
        f"- local model: {local_model}\n"
        f"- actual context window: {context_window} tokens\n"
        f"- safe input budget: {budget} tokens "
        f"(window - {output_reserve_tokens} output reserve "
        f"- {safety_margin_tokens} safety margin)\n"
        f"- max output reserve: {output_reserve_tokens} tokens\n"
        f"- current step: {step}/{step_total}\n"
        f"- available tools: {tools}\n"
        f"- context status: {zone}\n"
    )


def context_telemetry(*, model: str = "", window: int = 0, safe_input: int = 0,
                      input_tokens: int = 0, output_reserve: int = 0,
                      zone: str = "GREEN", compaction: str = "none",
                      output_tokens: int = 0, status: str = "",
                      step: int = 0, step_total: int = 0,
                      architecture: str = "", max_output: int = 0,
                      vision: bool = False, tool_use: bool = False) -> str:
    """One-line structured telemetry for logs/dashboard — makes debugging the
    context pipeline dramatically easier:

      MODEL=qwen2.5-coder-14b-instruct-mlx ARCH=qwen2 WINDOW=32768
      SAFE_INPUT=18768 INPUT=18240 OUT_RESERVE=12000 ZONE=GREEN
      COMPACTION=none OUTPUT=2840 STATUS=COMPLETED STEP=3/7
      MAX_OUTPUT=0 VISION=false TOOLS=true
    """
    parts = [
        f"MODEL={model or '?'}",
        f"ARCH={architecture or '?'}",
        f"WINDOW={window}",
        f"SAFE_INPUT={safe_input}",
        f"INPUT={input_tokens}",
        f"OUT_RESERVE={output_reserve}",
        f"ZONE={zone}",
        f"COMPACTION={compaction}",
        f"OUTPUT={output_tokens}",
        f"STATUS={status or '?'}",
        f"STEP={step}/{step_total}",
        f"MAX_OUTPUT={max_output}",
        f"VISION={'true' if vision else 'false'}",
        f"TOOLS={'true' if tool_use else 'false'}",
    ]
    return " | ".join(parts)
