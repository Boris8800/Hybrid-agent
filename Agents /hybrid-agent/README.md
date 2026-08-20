# hybrid-agent

A production-grade hybrid coding agent bridge: **Gemma (local) implements, DeepSeek (cloud) supervises**, with a routing layer, prompt enhancement, parallel step execution, cache, token budgeting, persistent memory, and a hardened file-application engine.

The bridge is a self-contained CLI (`ask.py`) that talks directly to two OpenAI-compatible endpoints — a local LM Studio model and the DeepSeek cloud API. It does **not** use the Kilo provider system, so it can run standalone or be driven by an orchestrator.

---

## Table of Contents

- [How it works](#how-it-works)
- [Modes](#modes)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [The supervise loop](#the-supervise-loop)
- [Prompt enhancement](#prompt-enhancement)
- [Parallel execution](#parallel-execution)
- [Verification stage](#verification-stage)
- [File application engine](#file-application-engine)
- [Caching](#caching)
- [Token budget](#token-budget)
- [Memory & routing](#memory--routing)
- [Project context](#project-context)
- [Configuration](#configuration)
- [Telemetry](#telemetry)
- [Exit codes](#exit-codes)
- [Testing & CI](#testing--ci)
- [Project layout](#project-layout)
- [Agent definitions](#agent-definitions)

---

## How it works

```
                ┌─────────────────────────────────────────────────────────┐
 TASK ─────────▶│  supervise loop (Gemma-primary / DeepSeek-supervisor)   │
                │                                                         │
                │  optional: DeepSeek ENHANCE + clarify + plan (--enhance)│
                │  Gemma implements  ──▶  compact review package          │
                │  DeepSeek verdict:  APPROVED / FIX_REQUIRED / REJECTED  │
                │  FIX_REQUIRED ──▶ fixes fed back to Gemma, iterate      │
                │                                                         │
                │  optional: --apply  writes path-labeled fenced blocks   │
                │  optional: --verify runs build/tests until clean        │
                └─────────────────────────────────────────────────────────┘
```

DeepSeek never rewrites code directly in the supervise loop — it returns verdicts
and required fixes, and the local model implements them. **API tokens are spent on
judgement, not on doing the coding.**

- **Local implementer** — `google/gemma-4-12b-qat` via LM Studio at `http://localhost:1234/v1`. Fast, free, no API key. Best for mechanical, well-specified edits.
- **API supervisor** — `deepseek-chat`, requires `DEEPSEEK_API_KEY` (or a key under `deepseek.key` in Kilo's `auth.json`). Best for architecture, design review, and ambiguous debugging.
- **Router** — archetype pinning, confidence scoring, adaptive threshold, and circuit breakers decide local-vs-API routing (`--route-only` previews the decision).

## Modes

Enforced roles, not soft hints (controlled by `--mode`, `$MODE`, default `hybrid`):

| Mode | Implementer | API supervision | Typical use |
|------|-------------|-----------------|-------------|
| `hybrid` | local | ✅ | Full Gemma-primary / DeepSeek-supervisor loop |
| `local` | local | ❌ | Purely local, no API key needed |
| `code` | API | ❌ | API implements directly, no supervision |

## Requirements

- Python 3.11+ (uses `str | None` and modern stdlib only — no runtime framework).
- A running **LM Studio** (or any OpenAI-compatible local endpoint) for local routing.
- `DEEPSEEK_API_KEY` for API supervision (or Kilo `auth.json` with a `deepseek.key`).
- PyYAML + openai inside `hybrid-agent/.venv` (auto-healed — see [Setup](#setup)).

## Setup

```bash
# 1. Create the bridge venv (system python3 cannot pip-install globally on macOS)
python3 -m venv "Agents /hybrid-agent/.venv"
"Agents /hybrid-agent/.venv/bin/pip" install openai pyyaml

# 2. Install the pre-commit hook (git hooks are not versioned)
"Agents /scripts/install-hooks.sh"

# 3. Validate the agent definitions
"Agents /scripts/validate-agents.sh"   # -> "All agent files valid."

# 4. Start using it
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --help
```

> **Self-heal:** when invoked with a system `python3` that lacks PyYAML/openai, `ask.py`
> automatically re-executes itself with the project venv interpreter (guarded against
> loops and never fired when the module is imported). A plain `python3 ask.py …` works.

## Quick start

```bash
# List local models
python3 ask.py --models

# Single-shot local generation (no API key needed)
python3 ask.py --local --task "Add input validation to validate_email" --stream

# Single-shot DeepSeek
python3 ask.py --deepseek --task "Draft a migration plan for auth to refresh tokens"

# Full supervise loop (Gemma implements, DeepSeek reviews)
python3 ask.py --supervise --task "<task>" --max-iterations 4

# Supervise + DeepSeek prompt enhancement + apply files to disk
python3 ask.py --supervise --enhance --task "<task>" --apply

# Preview the routing decision
python3 ask.py --route-only --task "Investigate the intermittent crash in checkout"
```

Run from `Agents /hybrid-agent/` (the directory containing `ask.py`), or use the
full path form shown above.

## CLI reference

### Routing & execution

| Flag | Description |
|------|-------------|
| `--task TEXT` | The coding task description (required for most paths). |
| `--context TEXT` | Extra context appended to every request. |
| `--local` / `--deepseek` / `--auto` | Force the route; default is `--auto` (router decision). |
| `--model ID` | Override the local LM Studio model id. |
| `--max-tokens N` | Max output tokens (default 4096; truncation retries once at double budget). |
| `--temperature F` | Sampling temperature (default 0.2). |
| `--stream` | Stream the local model's output live to stderr. |
| `--json` | Emit a single JSON object on stdout (for tooling). |
| `--config PATH` | Optional YAML config override. |
| `--mode hybrid\|local\|code` | Role enforcement mode (see [Modes](#modes)). |
| `--route-only` | Print the routing decision only, without calling any model. |
| `--models` | List loaded local model ids. |

### Supervision pipeline

| Flag | Description |
|------|-------------|
| `--supervise` | Gemma-primary / DeepSeek-supervisor loop until APPROVED (default max 3 iterations). |
| `--max-iterations N` | Max supervise iterations (default 3). |
| `--review` | Single-shot DeepSeek supervisor review of `--code` (uses `review/supervisor.md`). |
| `--code TEXT` | Code/output to review (required with `--review`). |
| `--system TEXT` | Override the system prompt. |
| `--enhance` | DeepSeek enhances the prompt, plans around Gemma's limits, and asks clarifying questions before Gemma implements. |
| `--cot` | Chain-of-thought planning: TASK UNDERSTANDING / CONSTRAINT ANALYSIS / ALTERNATIVES before the plan. |
| `--parallel` | With `--supervise --enhance`: split the plan into steps and run independent steps in parallel on the local model. |
| `--parallel-workers N` | Max parallel workers per batch (default 4). |

### File application

| Flag | Description |
|------|-------------|
| `--apply` | Write the approved output's path-labeled fenced code blocks to disk under `--root`. |
| `--apply-dry-run` | With `--apply`: print what would be written without touching the disk. |
| `--root DIR` | Directory to apply files under (default `.`). |

### Verification

| Flag | Description |
|------|-------------|
| `--verify` | Run the final error-check stage (build/lint/tests) before the task is marked complete. |
| `--verify-cmd CMD` | Add a verification command (repeatable; implies `--verify`). |
| `--verify-max N` | Max verify-fix iterations (default 2). |
| `--verify-timeout S` | Per-command timeout (default `review.verify_timeout`, 600s). |
| `--verify-parallel` | Run verification commands in parallel within `review.verify_groups` groups (groups run in order). |
| `--verify-workers N` | Max parallel verify workers per group (default 4). |
| `--regression` | Run the full test suite after verification passes (`review.regression`). |
| `--regression-cmd CMD` | Add a regression command (repeatable; implies `--regression`). |
| `--regression-timeout S` | Per-command timeout for regression (default `review.regression_timeout`, 600s). |

### Cache & telemetry

| Flag | Description |
|------|-------------|
| `--no-cache` | Bypass the response cache entirely. |
| `--cache-max-size N` | Max entries per cache kind (default `cache.max_entries`, 100). |
| `--cache-ttl N` | Cache TTL in days (default `cache.ttl_days`, 7). |
| `--stats` | Print the 80/20 strategy summary (`stats.json`). |
| `--evaluate` | Print the self-evaluation report: files, iterations, quality, truncation rate, comparison vs the previous week. |
| `--context-scan` | Scan the project (structure, dependencies, architecture, examples) and inject it into the enhancement/supervise flow. |

## The supervise loop

1. **Gemma implements** the task (streamed live). Truncated output is retried once at
   double budget; if still cut off, the run escalates to DeepSeek rather than
   pretending the output is complete.
2. **A compact review package** is built — task, plan, Gemma output, uncertainties,
   a **live git diff** (via `review/diff_builder.py`), and any verification context.
   DeepSeek never sees the whole conversation, only this package.
3. **DeepSeek returns a verdict**: `APPROVED`, `FIX_REQUIRED`, or `REJECTED`, with a
   quality score, issues, and exact fixes.
4. **FIX_REQUIRED** feeds the fixes back to Gemma and loops; **REJECTED** falls back
   to a DeepSeek implementation; **APPROVED** finishes and (with `--apply`) writes
   files to disk.
5. The outcome (route, verdict, quality) is recorded to **persistent task memory**
   and stats, so the router and the 80/20 metrics learn from real history.

**Verdict caching:** identical review packages (same task + output + diff) are served
from the response cache — no API spend on a repeat review.

## Prompt enhancement

`--enhance` sends the raw task to DeepSeek **before** Gemma implements:

- returns an **enhanced prompt**, **reasoning**, and a **plan** sized to Gemma's
  context/output limits;
- if the task is ambiguous, asks **clarifying questions** — interactively it waits for
  answers; non-interactively it exits with code `4` (`TASK UNCLEAR`) so you can re-run
  with a clearer `--task`. A section that says "no questions needed" is treated as clear.
- `--cot` adds explicit TASK UNDERSTANDING / CONSTRAINT ANALYSIS / ALTERNATIVES
  sections so you can verify the reasoning before Gemma implements.

## Parallel execution

`--parallel` (with `--enhance`) parses the plan into steps, groups them by dependency,
and runs each batch concurrently on the local model, then DeepSeek reviews the merged
output.

**File-conflict protection** — steps whose plan-declared `Files:` overlap within the
same batch are **serialized** (run one at a time, in step-id order) instead of racing;
the rest still run in parallel. After execution, a post-hoc scan of every step's fenced
output warns about any overlapping files the plan failed to declare, so a dropped block
is never silent.

## Verification stage

`--verify` (or `review.verify` in config) runs build/lint/typecheck commands before a
task is marked complete. On errors, DeepSeek analyzes the output, produces fixes that
are applied and re-verified, looping until clean or `--verify-max` (default 2).

Safety and cost protections:

- **Command allowlist** — commands must match a safe prefix (`npm run build`, `npx tsc`,
  `python -m pytest`, …). Destructive/compound shell (`rm`, `sudo`, `|`, `>`, `&&`, `;`,
  backticks, `$(`, …) is always blocked. Extend the list per-project with
  `review.verify_allowlist` in config.
- **Parallel groups** — `review.verify_groups` lets you run independent commands in
  parallel while keeping dependencies ordered (e.g. build finishes before tests).
- **Error truncation** — only the first ~4000 chars of error output reach DeepSeek.
- **Automatic rollback** — the working tree is git-snapshotted before fixes; if
  verification still fails, the AI's fix files are restored.
- **Diff logging** — every fix is written as a unified diff to `hybrid-verify/fixes.diff`.
- **Timeouts** — per-command `--verify-timeout` (default 600s).
- **Regression guard** — after verification passes, `--regression` runs the full test
  suite (`review.regression`) to catch fixes that break tests elsewhere.
- **Environmental-error skip** — `command not found`, `ENOENT`, `Module not found`, etc.
  are recognized and DeepSeek is **not** called (nothing it can fix).
- **Cost tracking** — iterations, API calls, tokens, and estimated cost are recorded in
  `stats.json` under `verify`.

## File application engine

`--apply` writes path-labeled fenced code blocks from the approved output to disk under
`--root`. The parser (`_parse_fenced_files`) accepts the natural model formats:

- ```` ```path ````, ```` ```path/to/f.ext ````, ```` ```lang path ````, ```` ```lang:path ````
- a clean path on the line **just before** the fence, or as the **first line inside** the block.

Bare language tags (`json`, `tsx`, `html`) and comment/HTML lines are **never** treated
as paths — a block labeled ```` ```json ```` with no path is ignored.

Safety guards:

- absolute paths, `..` traversal, Windows drive (`C:\…`) and UNC (`\\server\…`) paths are rejected;
- paths escaping `--root` are rejected;
- **duplicate blocks** for the same file are skipped — the first block wins and a
  `(duplicate block would overwrite)` warning is emitted;
- `--apply-dry-run` prints exactly what would be written without touching the disk.

## Caching

`CacheManager` stores successful (non-truncated) model responses under
`hybrid-agent/.cache/<kind>/` with a TTL and a per-kind entry cap.

| Kind | What it caches |
|------|----------------|
| `enhance` | DeepSeek prompt-enhancement output (keyed by request). |
| `fix` | DeepSeek verification-fix output (keyed by task + error text). |
| `generate` | Single-shot generation responses. |
| `review` | DeepSeek review verdicts — keyed by `sha256(system + user)` of the exact review package. |

Controls: `cache.enabled / dir / ttl_days / max_entries` in config, or
`--no-cache`, `--cache-ttl`, `--cache-max-size`. Hit/miss counters are recorded in
`stats.json` under `cache`.

## Token budget

`review.daily_token_budget` (default 200,000) caps the daily DeepSeek spend. When
exhausted, a `BudgetExceeded` error stops the run with exit code `6` and a clear
message. Every API route is covered — the supervise loop, parallel review, verification
fixes, enhancement, and single-shot calls — and actual usage is persisted to
`stats.json` under `api_tokens` (daily totals, per route).

## Memory & routing

`memory.py` persists one outcome per completed task in `memory/tasks.json` (kept to the
most recent 200 records): task text, route, verdict, and quality score.

- The router's confidence scorer reads a real `MemoryView` — `similar_task_success_rate`
  is the share of APPROVED tasks sharing a trigram with the current task, and novelty is
  computed against genuinely seen n-grams.
- The **adaptive threshold** (`router/threshold.py`) is updated from real outcomes only
  once a 50-sample observation window exists (`maybe_update`), so it learns without
  being moved by noise.
- Circuit breakers independently trip the local or API backend after error-rate ceilings
  are exceeded, with a cooldown.

`--route-only` previews the decision: e.g. `route deepseek confidence:0.78` or
`route local archetype:refactor`.

## Project context

`--context-scan` (or `scan.py --project-root .`) builds a compact project index —
file structure, dependencies, architecture, entry points — and injects it into the
enhancement/supervise flow so both models understand the codebase before implementing.
`manage-context.sh` shows status and refreshes the cached `context.json`.

## Configuration

`config.yml` (next to `ask.py`) is the source of truth; `--config` overrides it.
Unknown keys are ignored by the loader, and every section has a safe default.

| Section | Key settings |
|---------|--------------|
| `router` | `local_threshold`, `threshold_min/max`, `target_local_rate`, `alpha`, `weights` |
| `backends.local` | `base_url`, `api_key`, `model`, `timeout_s`, `max_retries`, `backoff_s`, `cold_start_wait_s` |
| `backends.deepseek` | `api_key_env`, `base_url`, `model`, `timeout_s`, `max_retries`, `backoff_s` |
| `review` | `verify`, `verify_timeout`, `verify_groups`, `verify_allowlist`, `regression`, `regression_timeout`, `daily_token_budget`, `max_depth_tokens`, `max_failure_summary_words` |
| `cache` | `enabled`, `dir`, `ttl_days`, `max_entries` |
| `circuit_breaker` | `window_size`, `local_error_ceiling`, `deepseek_error_ceiling`, `cooldown_s` |
| `memory` | `root`, `max_project_summary_words` |
| `roles` | `implementer`, `supervisor` (architecture — do not change) |

**Environment overrides** (roles stay architecture; only models/endpoints change):

| Env var | Effect |
|---------|--------|
| `LOCAL_MODEL` / `LOCAL_BASE_URL` / `LOCAL_API_KEY` | Local backend model/endpoint/key |
| `API_MODEL` / `API_BASE_URL` / `API_KEY_ENV` | API supervisor model/endpoint/key env |
| `MODE` | Role mode (`hybrid`/`local`/`code`) |

## Telemetry

`stats.json` (git-ignored) tracks the 80/20 strategy and self-evaluation:

- review counts (`deepseek_reviews`, `approvals`, `rejections`, `fix_required`, `deepseek_fallbacks`);
- sessions (`tasks_completed`, `files_generated`, `total_iterations`, `truncation_events`, quality);
- weekly snapshots under `periods` and phase timings under `phase_timings`;
- `verify` metrics (iterations, API calls, tokens, estimated cost, pass/fail);
- `cache` hit/miss counters and `api_tokens` daily usage for the budget.

`--stats` prints the summary; `--evaluate` prints the week-over-week self-evaluation
report.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `2` | Usage error, missing task, or missing DeepSeek key. |
| `3` | Mode violation or generation failure (no verdict / truncated fallback). |
| `4` | TASK UNCLEAR — clarifying questions were raised non-interactively. |
| `6` | Daily DeepSeek token budget exhausted. |
| `130` | Interrupted (Ctrl-C). |

## Testing & CI

```bash
cd "Agents /hybrid-agent"
./.venv/bin/python -m unittest discover -s tests -v    # 95 tests
```

Coverage includes: the fenced-file parser, the apply overwrite/unsafe-path guards,
dry-run, the clarify heuristic, `CacheManager` (TTL/cap/disabled), the verdict cache,
`TaskMemory`, token-budget accounting, parallel-verify groups, truncation retries, and
parallel step conflict detection/serialization.

`.github/workflows/ci.yml` runs the full suite on Python 3.11 and 3.12 for every push and
pull request, plus a syntax check of every engine module and the `.kilo/agent/*.md`
frontmatter check. The same `validate-agents.sh` runs as a local pre-commit gate.

## Project layout

```
repo root/
├── .kilo/                       # Agent definitions, commands, Kilo config
├── .github/workflows/ci.yml     # CI (tests, syntax, frontmatter) — runs on push/PR
├── Agents /scripts/             # validate-agents.sh, install-hooks.sh, hooks/
└── Agents /hybrid-agent/
    ├── ask.py                   # Bridge CLI (routing, supervise, verify, apply, cache)
    ├── agent.py                 # HybridAgent: backends, router, adaptive threshold
    ├── supervise.py             # Gemma-primary / DeepSeek-supervisor control loop
    ├── parallel.py              # Plan parsing, dependency groups, parallel executor
    ├── memory.py                # Persistent task memory (router learning)
    ├── context.py / scan.py     # Project context scanner
    ├── backends/                # Gemma/DeepSeek clients + circuit breaker
    ├── router/                  # Archetypes, confidence scoring, threshold
    ├── review/                  # diff_builder + supervisor.md (review protocol)
    ├── tests/                   # 95 unit tests
    ├── config.yml               # Runtime configuration
    └── README.md                # This file
```

## Agent definitions

Agent prompts live in `.kilo/agent/*.md` (validated by `kilo config check` and
`scripts/validate-agents.sh`). A squiggle from the VS Code extension's whole-file YAML
parser is a known false positive — see `.kilo/known-false-positives`; `kilo config check`
is the authoritative gate.
