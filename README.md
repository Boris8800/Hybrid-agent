# hybrid-agent

A production-grade hybrid coding agent bridge: **Qwen (local) implements, DeepSeek (cloud) supervises**, with a routing layer, prompt enhancement, parallel step execution, cache, token budgeting, persistent memory, and a hardened file-application engine.

The bridge is a self-contained CLI (`ask.py`) that talks directly to two OpenAI-compatible endpoints — a local LM Studio model and the DeepSeek cloud API. It does **not** use the Kilo provider system, so it can run standalone or be driven by an orchestrator.

---

## Table of Contents

- [How it works](#how-it-works)
- [Modes](#modes)
- [Requirements](#requirements)
- [Setup](#setup)
- [Web dashboard](#web-dashboard)
- [Providers & multi-model](#providers--multi-model)
- [Git & deploy](#git--deploy-pull--push--deploy)
- [Terminal tool (RUN:)](#terminal-tool-run)
- [Bounded autonomy (BOUND)](#bounded-autonomy-bound)
- [Journey verification (Vibe DSL)](#journey-verification-vibe-dsl)
- [Anti-gaming ratchet & guardrails](#anti-gaming-ratchet--guardrails)
- [Surgical AST-aware repairs](#surgical-ast-aware-repairs)
- [Trust & learning layer](#trust--learning-layer)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [The supervise loop](#the-supervise-loop)
- [Context Safety Controller](#context-safety-controller)
- [Durable task state](#durable-task-state-state-machine--evidence-ledger)
- [Task Recovery Manager](#task-recovery-manager)
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
 TASK ─────────▶│  supervise loop (Qwen-primary / DeepSeek-supervisor)   │
                │                                                         │
                │  optional: DeepSeek ENHANCE + clarify + plan (--enhance)│
                │  Qwen implements  ──▶  compact review package          │
                │  DeepSeek verdict:  APPROVED / FIX_REQUIRED / REJECTED  │
                │  FIX_REQUIRED ──▶ fixes fed back to Qwen, iterate      │
                │                                                         │
                │  optional: --apply  writes path-labeled fenced blocks   │
                │  optional: --verify runs build/tests until clean        │
                └─────────────────────────────────────────────────────────┘
```

DeepSeek never rewrites code directly in the supervise loop — it returns verdicts
and required fixes, and the local model implements them. **API tokens are spent on
judgement, not on doing the coding.**

- **Local implementer** — `qwen2.5-coder-14b-instruct-mlx` via the local MLX-style API at `http://localhost:1234/api/v1` (embeddings via the legacy `/v1/embeddings`). Fast, free, no API key. Best for mechanical, well-specified edits.
- **API supervisor** — `deepseek-chat`, requires `DEEPSEEK_API_KEY` (or a key under `deepseek.key` in Kilo's `auth.json`). Best for architecture, design review, and ambiguous debugging.
- **Router** — archetype pinning, confidence scoring, adaptive threshold, and circuit breakers decide local-vs-API routing (`--route-only` previews the decision).

## Modes

Enforced roles, not soft hints (controlled by `--mode`, `$MODE`, default `hybrid`):

| Mode | Implementer | API supervision | Typical use |
|------|-------------|-----------------|-------------|
| `hybrid` | local | ✅ | Full Qwen-primary / DeepSeek-supervisor loop |
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

## Web dashboard

A single-user local control panel for the engine:

```bash
cd "Agents /hybrid-agent"
./.venv/bin/pip install flask flask-socketio cryptography   # one-time
./.venv/bin/python web_dashboard.py --port 8660              # -> http://127.0.0.1:8660
# `python3 web_dashboard.py …` also works — it self-heals into the venv.
```

- **Providers & API keys** — manage all 2+2 providers: enter/test/delete keys
  (stored in macOS Keychain, Fernet-encrypted file elsewhere — keys are never
  returned to the browser, only a masked hint), edit endpoint/model, toggle
  enabled, and list models loaded on local endpoints.
- **Submit & queue tasks** — choose mode, provider, and flags (enhance, verify,
  regression, apply, parallel, cot, context-scan, turbo), queue multiple tasks
  (one runs at a time), cancel the running task (graceful SIGINT).
- **Live output** — real-time progress bar (parses `[hybrid] progress` lines),
  phase/percent/elapsed, filtered log view with download.
- **Stats & charts** — task counts, approval rate, cache hit rate, verify cost,
  phase timing pie, cache donut, daily API-token bar, review breakdown.
- **Config editor** — view/edit `config.yml` with YAML validation and a backup
  before saving.
- **Diff viewer** — the latest `hybrid-verify/fixes.diff`.
- **Auth (optional)** — export `DASHBOARD_TOKEN=…`; API calls require
  `Authorization: Bearer …`.

## Providers & multi-model

The engine talks to **any OpenAI-compatible endpoint** in either role, with a
fleet of **2 online + 2 local providers** configured out of the box:

| Kind | Default | Endpoint | Model |
|------|---------|----------|-------|
| online | `deepseek` | `api.deepseek.com` | `deepseek-chat` |
| online | `groq` | `api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| local | `qwen` | `localhost:1234/api/v1` | `qwen2.5-coder-14b-instruct-mlx` |
| local | `local-2` | `localhost:1234/api/v1` | (pick any loaded model) |

- **API keys** resolve in order: environment variable → Kilo `auth.json` → the
  dashboard's encrypted secrets store. Manage them in the web UI.
- `--online-provider NAME` / `--local-provider NAME` select which provider is
  primary for a run.
- **Failover** — if the active online provider fails, the request automatically
  retries on the next enabled online provider.
- **`--turbo` (multi-AI)** — fans every online call out to all enabled online
  providers **in parallel** and returns the longest non-truncated response.
  Uses many AIs at the same time; multiplies API spend (still capped by
  `review.daily_token_budget`).

## Git & deploy (pull / push / deploy)

The engine can manage the whole ship-and-ship-it loop:

- **`--pull`** — `git pull --ff-only` before the agent starts, so it works on
  the freshest baseline. Best-effort: a dirty tree or missing upstream is
  reported, never fatal.
- **`--push`** — after verification passes, stage exactly the engine's own
  changes (applied files + verification fixes), commit with an identifiable
  `hybrid-agent: <task>` message, and push. Never force-pushes; the first push
  of a new branch auto-sets upstream. Idempotent via the OperationLog: a
  completed push is skipped on resume.
- **`--deploy`** / `--deploy-cmd` — run the deploy command from `deploy.command`
  (config.yml) or the flag override, from `deploy.cwd` (default: the project
  root). **Trust boundary:** deployment requires the durable task state to be
  `DEPLOY_AUTHORIZED` (the full GENERATED → REVIEWED → VERIFIED → APPROVED →
  APPLIED → DEPLOY_AUTHORIZED chain). Passing tests alone does **not**
  authorize deployment — a `FAILED`/`REJECTED` state or a missing boundary
  step is refused at the code level. Idempotent via the OperationLog.
- **`--resume`** — resume a previous run of the same task from its durable
  state: prints the `TASK <id> RESUMING` summary (step N/M, previous state,
  completed steps, previous evidence), skips already-completed operations,
  and continues from the last recorded position. A non-terminal prior state
  is auto-continued even without `--resume`.

Push and deploy only run when the task is **verified** and the output was
independently reviewed — never on `review_failed_no_verdict` or truncated
fallbacks. Results (ok/message) are reported in the status lines, the printed
summary, and the `--json` payload under `push` / `deploy`.

```bash
# Pull latest, fix, verify, commit+push, then deploy — in one shot:
python3 ask.py --supervise --enhance --task "fix the checkout bug" \
  --verify --verify-cmd "npm run build" --apply --pull --push --deploy
```

## Terminal tool (RUN:)

The models get a **real terminal tool** in the supervise loop: Qwen can emit
`RUN: <command>` lines in its output, the engine executes them, and the output
is fed back for another iteration (up to `--terminal-rounds`, default 3).

```
RUN: npm run build        ← the model inspects/verifies
<engine runs it, returns exit code + output>
<model fixes code, emits RUN again or finishes>
```

- **Safety**: every command is gated by the dangerous-shell-marker block
  (`rm`, `sudo`, `|`, `>`, `;`, `&&`, … are always rejected) plus the
  allowlist — build/test commands (from the verify allowlist) and read-only
  inspection commands (`ls`, `cat`, `head`, `git status`, `git log`, `pwd`,
  `find`, `grep`, …).
- `review.terminal_timeout` (default 120s) caps each command; output is
  truncated to 4000 chars.
- The terminal session is appended to the review package's `VERIFICATION`
  section, so DeepSeek reviews with full knowledge of what ran.
- `--terminal-rounds 0` disables the tool; it's enabled by default in the
  supervise loop.

## Bounded autonomy (BOUND)

The **Ouro Loop pattern**: hard constraints enforced at runtime, so the agent
can run unattended without breaking things.

**The BOUND** (`bound:` in config.yml) has three parts:

- **DANGER ZONES** — glob patterns the agent can never write (`**/.env`,
  `**/*.pem`, `**/*.key`, `.git/**`, the engine dir, memory/cache, …).
- **NEVER DO** — command patterns that never run (`rm -rf`, `git push --force`,
  `git reset --hard`, `chmod 777`, `sudo`, …).
- **IRON LAWS** — human rules injected into every prompt.

**Runtime enforcement (the agent cannot bypass it):**
- Files matching a danger zone are **never written** by `--apply` — the run
  reports `BOUND_VIOLATION` and exits with **code 2**.
- `never_do` commands are blocked in the `RUN:` terminal tool.
- Verification-fix outputs are BOUND-checked too; a violation reverts the fixes.
- `--no-bound` is the user-level escape hatch — the agent has no path to it.

**RECALL gate** — the BOUND is re-injected into the enhance context, the review
package, and **every** Qwen iteration, so the models can't "forget" it as the
conversation grows.

**Program phases** (`program:` in config.yml) — structured Build → Verify →
Self-Fix cycles inside the verify stage, each with its own gate and a
remediation playbook:

```yaml
program:
  phases:
    - name: build
      gate: ["npm run build"]
      max_fix_rounds: 2
      on_fail: retry        # retry | revert | escalate
    - name: test
      gate: ["npm test"]
      on_fail: escalate
```

- **`retry`** — DeepSeek fixes the gate errors (BOUND-enforced), re-verify.
- **`revert`** — git-restore the phase's changes and fail the program.
- **`escalate`** — stop and flag the run for human review.

Phases run when `program.phases` is configured; otherwise the standard
`--verify` flow runs unchanged.

## Journey verification (Vibe DSL)

User-journey verification in a headless browser — not just build/lint, but
actual user-facing behavior (BotIntern pattern).

A `journeys.yml` at the project root defines user journeys:

```yaml
app_url: "http://localhost:3000"
setup: ["npm run dev"]                     # optional app startup commands
journeys:
  - name: "login page renders"
    steps:
      - visit: "/"
      - see_text: "Sign in"
      - see_element: "button[type=submit]"
  - name: "login flow works"
    steps:
      - visit: "/login"
      - fill: { selector: "#email", value: "alice@example.com" }
      - fill: { selector: "#password", value: "password123" }
      - click: "button[type=submit]"
      - wait_selector: ".dashboard"
```

Step types: `visit`, `see_text`, `see_element`, `click`, `fill`,
`wait_selector`, `wait_text`, `wait_ms`, `screenshot`.

- **Run as a gate** — `--journeys journeys.yml` (or a `journeys:` program phase)
  runs them headlessly (Playwright/Chromium, in the venv). On failure the
  report — including a **page snapshot** of the failing state and a screenshot
  in `hybrid-verify/screenshots/` — goes to DeepSeek, which generates surgical
  fixes; the journeys re-run until green or `--verify-max` rounds.
- **Write-One safety** — `journeys.yml` is in the BOUND danger zones: the agent
  can fix source code but can **never modify the test suite** (no cheating).
- Standalone: `./.venv/bin/python journeys.py --file journeys.yml`.
- Without Playwright/PyYAML installed the runner reports a clear message
  instead of crashing: `./.venv/bin/pip install playwright pyyaml && ./.venv/bin/python -m playwright install chromium`.

## Anti-gaming ratchet & guardrails

**Anti-gaming ratchet (Modonome pattern)** — after the agent's changes are
applied, a differential gate inspects the working-tree diff and **rejects any
change that weakens a gate**:

- test files: added `it.skip(` / `.only(` / `@pytest.mark.skip` / `xit(` /
  `skip: true` → flagged; assertion lines removed without replacement → flagged;
- `tsconfig*.json`: strict checks flipped off (`"strict": false`,
  `"skipLibCheck": true`, …) → flagged;
- guard/config files (`config.yml`, `.github/`, jest/vitest/eslint configs,
  `journeys.yml`): any modification → flagged as needing owner review.

On violations, DeepSeek is instructed to **RESTORE the gate, never weaken it**,
re-applies (BOUND-enforced), and re-checks — bounded by `--verify-max`. The run
is only "green" when the diff is clean.

**Structured guardrails** (`guardrails:` in config.yml) gate the task *before*
it runs:

- **`block`** — content patterns that reject the task outright (exit 2), e.g.
  `drop table`.
- **`approval_required`** — content patterns that escalate to a human: y/N
  prompt interactively, **exit code 7** non-interactively. Composes with the
  program's `escalate` playbook.
- **`cost_limit`** — a rough pre-run cost estimate above the limit escalates to
  approval.

## Surgical AST-aware repairs

Fixes are **diff-first**, not full-file rewrites:

- The fix prompt asks DeepSeek for a **unified diff** (`--- a/…` / `+++ b/…`).
  `_apply_repair` applies it with `git apply` (falling back to `patch -p1`);
  full-file fenced blocks remain the fallback when no diff is present or it
  won't apply cleanly.
- Every file a diff touches is **BOUND-checked** first — a patch targeting a
  danger zone is rejected outright. The differential ratchet still runs over
  the applied diff afterwards.
- **AST-aware context**: the failing `file:line` from the error text is
  resolved to the **enclosing function/class/block** — Python via the stdlib
  `ast`, TS/JS via a scope scan — and only that block is sent as source
  context. The model fixes a narrow, contract-aware region, not the whole file.
- **Token accounting**: each surgical diff logs the estimated savings (patch
  size vs. the full-file rewrite it replaces). Measured on a 300-line file:
  a 1-line fix cost **53 tokens as a diff vs ~1822 as a full rewrite (97%
  saved)**.

## Trust & learning layer

Four systems that make the engine safer and smarter over time:

**Secrets/PII scanning** — the BOUND protects what gets *written*; this
protects what *leaves the machine*. Every request to the API models is scanned
and redacted before transmission: API keys, AWS keys, GitHub tokens, JWTs,
private keys, Stripe keys, connection strings, emails, phones.

- `secrets_scan.mode`: `redact` (default, `<REDACTED:type>`) · `block` (refuse
  to send, exit) · `off`. `--secrets-scan` overrides per run.

**Engineering rules** — a per-project constitution at
`.agent/engineering-rules.yml` (architecture facts, rules, dependency policies)
injected alongside the BOUND and consulted by the dependency gate.

**Learning from failures** — when the same (domain, error-category) keeps
failing, the engine records what the successful fixes actually touched and
emits a reusable rule into `memory/<project>/rules.json` — e.g. *"for backend
API tasks with missing-module failures: always include service + test in the
dependency context."* Rules are injected into the next task's context and
extend the dependency graph retrieval.

**Dependency gate** — when the applied diff adds a package to a manifest
(`package.json`, `requirements.txt`, `pyproject.toml`, `go.mod`, `Cargo.toml`),
the run stops pending approval. Engineering-rule blocklists reject outright;
`dependency_gate.allow` / `--allow-dep` pre-approve; `--dep-audit` looks up
license/description from npm/PyPI.

## Evidence & diagnosis

**Evidence-based approval** — `APPROVED` requires cited **machine evidence**
(`=== EVIDENCE ===`: test names, file:line, command output). An approval
without evidence is downgraded to **`UNKNOWN`** — a first-class verdict, never
a guess. `UNKNOWN` triggers an evidence-collection pass (the configured verify
commands run, their output is attached, and the package is re-reviewed once);
if still `UNKNOWN`, the run escalates for human review and is never applied.

**Loop detection** — every fix loop (verify, journeys, differential, program
phases) fingerprints each round's applied files + fix text; the same fix
appearing twice stops the run: `LOOP_DETECTED`, human review required.

**Failure fingerprinting** — every failure gets a stable `FAILURE_ID`
(hash of error category + file:line + normalized message). The same failure
recorded as successfully fixed ≥ 2 times becomes a **known solution** — the
engine reuses it (`KNOWN SOLUTION FOUND, confidence 96%`) without an API call.

**Agent Constitution** — an optional `.agent/constitution.md` (or root
`constitution.md`) is injected into every stage alongside the BOUND and
engineering rules.

**Scope metric** — after apply, changed files are compared against the
contract's "Files likely involved"; changes well outside scope report
`SCOPE_VIOLATION` in the verification output.

## Quick start

```bash
PY="Agents /hybrid-agent/.venv/bin/python"   # self-heals if run as: python3 .../ask.py
ASK="Agents /hybrid-agent/ask.py"

# List local models
$PY $ASK --models

# Single-shot local generation (no API key needed)
$PY $ASK --local --task "Add input validation to validate_email" --stream

# Single-shot DeepSeek
$PY $ASK --deepseek --task "Draft a migration plan for auth to refresh tokens"

# Full supervise loop (Qwen implements, DeepSeek reviews)
$PY $ASK --supervise --task "<task>" --max-iterations 4

# Supervise + DeepSeek prompt enhancement + apply files to disk
$PY $ASK --supervise --enhance --task "<task>" --apply

# Preview the routing decision
$PY $ASK --route-only --task "Investigate the intermittent crash in checkout"
```

Every command above runs from the repo root. You can also `cd "Agents /hybrid-agent"` and
use plain `python3 ask.py …` (the CLI self-heals into the venv interpreter).

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
| `--router auto\|full\|local_first\|critical` | Supervision-plan override (see [Dynamic supervision routing](#dynamic-supervision-routing)). |
| `--online-provider NAME` / `--local-provider NAME` | Select which provider is primary (see [Providers & multi-model](#providers--multi-model)). |
| `--turbo` | Fan every online call out to all enabled online providers in parallel; best response wins. |
| `--route-only` | Print the routing decision only, without calling any model. |
| `--models` | List loaded local model ids. |

### Supervision pipeline

| Flag | Description |
|------|-------------|
| `--supervise` | Qwen-primary / DeepSeek-supervisor loop until APPROVED (default max 3 iterations). |
| `--max-iterations N` | Max supervise iterations (default 3). |
| `--review` | Single-shot DeepSeek supervisor review of `--code` (uses `review/supervisor.md`). |
| `--code TEXT` | Code/output to review (required with `--review`). |
| `--system TEXT` | Override the system prompt. |
| `--enhance` | DeepSeek enhances the prompt, plans around Qwen's limits, and asks clarifying questions before Qwen implements. |
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
| `--pull` | `git pull --ff-only` before the task (best-effort). |
| `--push` | After verification: git add + commit + push the engine's changes (idempotent). |
| `--deploy` / `--deploy-cmd CMD` | After verification AND the DEPLOY_AUTHORIZED trust boundary: run the deploy command (idempotent). |
| `--resume` | Resume the same task from its durable state (step N/M, completed ops skipped). |
| `--terminal-rounds N` | Max terminal rounds for the `RUN:` tool (0 disables). |
| `--no-bound` | Disable BOUND runtime enforcement (user-level escape hatch). |
| `--journeys FILE` | Verify user journeys headlessly (Vibe DSL); DeepSeek fixes and re-runs until green. |
| `--journeys-timeout S` | Per-step timeout for journey verification (default 30). |
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

1. **Qwen implements** the task (streamed live), gated by the [Context Safety
   Controller](#context-safety-controller): the prompt is budgeted and preflighted
   before the call. Incomplete output (`OUTPUT_LIMIT_REACHED` /
   `CONTEXT_LIMIT_REACHED` / `TIMEOUT`) is continued once at double budget
   (hard retry cap); if still cut off, the run escalates to DeepSeek rather
   than pretending the output is complete.
2. **A compact review package** is built — task, plan, Qwen output, uncertainties,
   a **live git diff** (via `review/diff_builder.py`), and any verification context.
   DeepSeek never sees the whole conversation, only this package.
3. **DeepSeek returns a verdict**: `APPROVED`, `FIX_REQUIRED`, or `REJECTED`, with a
   quality score, issues, and exact fixes.
4. **FIX_REQUIRED** feeds the fixes back to Qwen and loops; **REJECTED** falls back
   to a DeepSeek implementation; **APPROVED** finishes and (with `--apply`) writes
   files to disk.
5. The outcome (route, verdict, quality) is recorded to **persistent task memory**
   and stats, so the router and the 80/20 metrics learn from real history.

**Verdict caching:** identical review packages (same task + output + diff) are served
from the response cache — no API spend on a repeat review.

## Context Safety Controller

The orchestration layer — never the local model — decides whether the context is
safe. Before **every** local-model call, the prompt is measured against hard
budgets derived from the model's **actually loaded** context window, so the
local model can never reach an unusable context state. Truncation is a
*prevented* state, not a failure detected after the fact.

**Per-model capability discovery.** At startup the engine queries LM Studio's
`/api/v1/models` and reads the real loaded window
(`loaded_instances[].config.context_length`), max context, architecture,
format, quantization, max output, vision and tool-use support. Discovery wins
over config: a stale `context_window:` value in `config.yml` can never cap a
model loaded with a larger window (`-c 65536`, `131072`, …). Config is used
only when the server is unreachable; the engine default is 32768. Models are
never assumed to behave alike.

**Budget math.** The safe input budget never uses 100% of the window:

```
safe_input_budget = discovered_window − output_reserve − safety_margin
```

Defaults (configurable via `review.context_safety`): reserve 12000, margin
2000. The reserve is **dynamic** — `min(configured_reserve, model_max_output,
task_required_output)` — so a tiny task does not sacrifice 12k tokens of
window (a 2-file task runs with reserve ≈ 512 → safe budget ≈ 30256 on a
32768 window).

**Preflight zones** (checked before the model is called):

| Zone | Usage | Action |
|------|-------|--------|
| GREEN | < 70% | proceed |
| YELLOW | 70–85% | compress / reduce context |
| ORANGE | 85–95% | aggressive compaction |
| RED | > 95% | **do NOT call the model** — escalate to DeepSeek (`local_context_red_escalation`) |

**Compaction invariants.** Compaction never removes the protected core: task
requirements, BOUND constraints, contract (files being modified, security
requirements), supervisor fixes, and the engine protocol. Only bulk context is
compactable — source context first, then terminal output, which is summarized
(exit code / errors / stack traces / changed files / warnings) instead of
echoing raw 20k-token logs back into the window. If even the protected core
exceeds the budget, the prompt is returned intact and the RED preflight
escalates; the task is never truncated by the compactor.

**Retry budget.** Recovery loops after incomplete local output are hard-capped
(`max_recovery_retries`, default 1: one continuation, then escalation), so a
pathological model cannot burn unbounded time/tokens. Output states are
classified distinctly: `COMPLETED` / `OUTPUT_LIMIT_REACHED` /
`CONTEXT_LIMIT_REACHED` / `TIMEOUT` / `MODEL_ERROR` — an incomplete response is
never marked successful or applied.

**Supervisor awareness.** Every DeepSeek review/plan/enhance prompt carries a
`LOCAL MODEL CONTEXT` block (model, actual window, safe input budget, output
reserve, step N/M, tools, zone) so the API supervisor plans within the real
usable context.

**Telemetry.** One structured line per local call:

```
MODEL=qwen2.5-coder-14b-instruct-mlx | ARCH=qwen2 | WINDOW=32768 | SAFE_INPUT=30256 | INPUT=553 | OUT_RESERVE=512 | ZONE=GREEN | COMPACTION=none | OUTPUT=28 | STATUS=COMPLETED | STEP=1/3 | MAX_OUTPUT=0 | VISION=false | TOOLS=false
```

Token estimation is conservative (3.5 chars/token — code tokenizes denser than
prose; override with `HYBRID_CHARS_PER_TOKEN`).

## Durable task state (state machine + evidence ledger)

The Task Contract is the **permanent source of truth**, not just another context
component. The conversation is temporary working memory; `task_state.py` is
durable task state that survives context compaction, model switching, Qwen or
DeepSeek failure, process restart, retry, verification, and parallel execution.
Persisted as JSON at `<root>/.hybrid-agent/tasks/TASK-<hash>.json` after every
stage (`HYBRID_TASK_STATE_DIR` overrides the location).

**Strict state machine.** Illegal transitions are impossible at code level —
there is no edge where they can happen:

```
PLANNING → IMPLEMENTING → GENERATED → REVIEWING → REVIEWED
  → FIXING (loops back to GENERATED) | VERIFYING → VERIFIED
  → REGRESSION_VERIFIED → APPROVED → APPLIED → DEPLOY_AUTHORIZED
  → PUSHED → DEPLOYED          FAILED / REJECTED are terminal
```

- `FAILED → DEPLOYED`, `TRUNCATED → APPROVED`, `GENERATED → APPROVED`
  (approval without review) and `REVIEWED → APPROVED` (approval without
  verification) raise `IllegalTransition`.
- **Generated ≠ approved ≠ verified ≠ applied ≠ deployed** — each is a
  distinct state, never collapsed into one `success=True`.

**Evidence ledger.** Every important claim gets an evidence object
(`E-001`, `E-002`, …): type, command, exit code, output hash, timestamp, files
affected, source, summary. Verify commands, applied files, review verdicts and
batch outcomes are all recorded, and the ledger renders compactly so approvals
are auditable (`E-019 cmd:npm test exit:0 hash:… files:src/auth.ts`).

**Model capability contracts.** `ModelCapabilities` formalizes what a model can
do (context window, max output, tool use, streaming, vision, structured output,
embeddings, reasoning, diff generation). The orchestrator asks
`can_perform_role("implementer", "long")` — never "is this Qwen?". Unknown
capabilities are treated conservatively (not guaranteed).

**Batch transactions.** Parallel batches get an explicit transaction state:
`SUCCESS` / `PARTIAL_FAILURE` / `FAILED`. 3/4 steps OK is `PARTIAL_FAILURE` and
the supervisor decides retry vs rollback — never silent success.

**Idempotency.** Apply, push, and deploy are tracked in an `OperationLog`
(`OP-<hash>`); a completed operation is never executed again — restart/resume
cannot double `npm install`, migrations, commits, or deployments.

**Restart/resume.** `--resume` (or an automatic continue when a non-terminal
state exists) reports `TASK <id> RESUMING · step 3/4 · state IMPLEMENTING ·
completed step 1, step 2 · evidence E-014…E-021` and continues from the last
recorded position without resending the conversation.

**Deploy trust boundary.** Passing tests does **not** authorize deployment.
`--deploy` requires the durable state to have walked the full chain to
`DEPLOY_AUTHORIZED` (GENERATED → REVIEWED → VERIFIED → APPROVED → APPLIED →
DEPLOY_AUTHORIZED); anything less, including a `FAILED`/`REJECTED` terminal
state, is refused at the code level.

**Turbo consensus.** `--turbo` no longer picks the longest response — that is
not a quality criterion for supervision. Each provider's output is normalized
to a structured opinion (VERDICT / CONFIDENCE / CRITICAL_ISSUES / REQUIRED_FIXES)
and adjudicated deterministically with zero extra API spend: any non-APPROVED
verdict beats APPROVED, the strictest (REJECTED > FIX_REQUIRED > UNKNOWN) wins,
and among approvals the highest confidence/evidence wins.

## Task Recovery Manager

All failure recovery is centralized in `recovery.py` — one policy instead of
scattered logic in supervise.py / verify / parallel.py / gitops.py:

```
Failure → FailureClassifier (transient / model / context / tool /
          verification / contract / security / infrastructure / unknown)
       → RecoveryManager (retry / compact / switch_model /
          ask_supervisor / rollback / resume / escalate)
```

Deterministic, bounded, and state-aware:

| Failure class | Recovery decision |
|---------------|-------------------|
| transient / infrastructure / tool | bounded retry with backoff (3/2/2), then escalate |
| model | switch model once (local ↔ DeepSeek), then escalate |
| context | compact once (the CSC already shrank the prompt), then escalate |
| verification | ask the supervisor (fix vs rollback is a review decision) |
| contract / security / unknown | **escalate immediately** — never auto-retry what needs a human |

Every attempt is recorded in the durable TaskState (recovery_attempts,
recovery_failures, escalations, last_error) via a RecoveryManager bound to the
state. Crucially, the manager **never changes task facts** — it only decides
the next action, and every action still executes through the state machine, so
recovery cannot manufacture approval: `TRUNCATED → APPROVED`,
`UNVERIFIED → DEPLOY_AUTHORIZED`, `FAILED → DEPLOYED`, and duplicate operations
remain impossible at the transition level regardless of what failed.

**Flexible, contract-driven paths.** The machine is not rigid: a task with
`--apply` and no regression suite walks `GENERATED → REVIEWED → VERIFIED →
APPROVED → APPLIED` (skipping REGRESSION_VERIFIED); a deploy task walks the
full chain through `DEPLOY_AUTHORIZED → DEPLOYED`. What is *never* flexible:
approval without review, approval without verification, deploy without the
authorization boundary.

**Chaos testing.** The suite (`tests/test_recovery.py`) injects failures at
every boundary — Qwen dies, DeepSeek dies, LM Studio restarts, context
discovery fails, parallel workers crash, state JSON corrupts, SIGKILL-style
restart, duplicate operations, model switches mid-task, unsafe commands — and
asserts the NEVER-invariants hold under every injection.

## Dynamic supervision routing

Before a task enters the pipeline, the router decides the **supervision plan** —
the per-task "autonomy schedule":

| Level | Behavior |
|-------|----------|
| `full` | Current behavior: DeepSeek reviews every iteration. |
| `local_first` | **Skip the DeepSeek review entirely** — one Qwen pass, then apply/verify. Zero API spend (unless `--verify` finds errors DeepSeek must fix). |
| `critical` | Force prompt enhancement + the full review loop, even without `--enhance`. |

`_plan_supervision` picks a level by precedence:

1. explicit `--router` / `router.supervision` config override;
2. **critical task signals** (`architecture`, `security`, `auth`, `migration`, `schema`, `concurrency`, …) → `critical`;
3. **trivial task signals** (`typo`, `rename`, `readme`, `docstring`, `draft`, `format`, …) → `local_first`;
4. **budget pressure** — daily API usage ≥ 80% of `review.daily_token_budget` degrades to `local_first`;
5. router confidence: DeepSeek-archetype-pinned tasks → `critical`; local-pinned tasks with memory history → `local_first`;
6. **memory similarity** — strong (≥ 0.7) local success on similar past tasks → `local_first`;
7. default → `full`.

`local_first` results carry a synthetic APPROVED verdict (`reason="router_local_skip_review"`) so the apply/stats tail works unchanged. Safety bias: **critical always wins** when a task is ambiguous, and the `--verify` stage remains a real error-driven safety net.

## Prompt enhancement

`--enhance` sends the raw task to DeepSeek **before** Qwen implements:

- returns an **enhanced prompt**, **reasoning**, and a **plan** sized to Qwen's
  context/output limits;
- if the task is ambiguous, asks **clarifying questions** — interactively it waits for
  answers; non-interactively it exits with code `4` (`TASK UNCLEAR`) so you can re-run
  with a clearer `--task`. A section that says "no questions needed" is treated as clear.
- `--cot` adds explicit TASK UNDERSTANDING / CONSTRAINT ANALYSIS / ALTERNATIVES
  sections so you can verify the reasoning before Qwen implements.

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

`memory.py` persists one outcome per completed task in `memory/tasks.json`:

- **Semantic similarity** — every task is embedded via the local LM Studio
  `/v1/embeddings` endpoint (768-dim, free, offline). Similar-task recall is
  **cosine-based** (threshold `memory.embedding_threshold`, default 0.60,
  calibrated on the real model), so paraphrased tasks like "fix the login bug"
  and "resolve the authentication issue" are matched correctly — something
  word-trigram matching could not do. When the embedding endpoint is down or
  disabled, the engine transparently falls back to trigrams.
- **Per-project scoping** — memory auto-scopes to `memory/<project>/` (from the
  git top-level name), so learning never bleeds across repositories; an explicit
  `memory.root` in config overrides this.
- **Scored eviction** — entries past the cap are evicted by recency + task
  frequency (0.7/0.3) instead of FIFO.
- **Consolidation pass** — with ≥ 10 records, `consolidate()` synthesizes
  approval rates, trend direction, and strongest task domains into
  `memory/insights.json`; `insights_text()` injects a compact summary into the
  enhancement/review context. Local-only. Runs automatically when the cached
  insights are stale (24h); `--memory` prints the records + insights and
  `--consolidate` forces a refresh — both fully offline.
- **Atomic writes** — all memory, stats, and cache files are written via
  temp + `os.replace`, so concurrent Agent Manager sessions can never tear them.
- **MemoryView** — `similar_task_success_rate` (semantic + trigram) and `seen_ngrams`
  feed the confidence scorer and the supervision plan.

The **adaptive threshold** (`router/threshold.py`) is updated from real outcomes
only once a 50-sample observation window exists (`maybe_update`), and circuit
breakers independently trip the local or API backend after error-rate ceilings
are exceeded. The router's decision feeds the [supervision plan](#dynamic-supervision-routing).

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
| `router` | `local_threshold`, `threshold_min/max`, `target_local_rate`, `alpha`, `supervision` (auto/full/local_first/critical), `weights` |
| `backends.local` | `base_url`, `api_key`, `model`, `timeout_s`, `max_retries`, `backoff_s`, `cold_start_wait_s` |
| `backends.deepseek` | `api_key_env`, `base_url`, `model`, `timeout_s`, `max_retries`, `backoff_s` |
| `review` | `verify`, `verify_timeout`, `verify_groups`, `verify_allowlist`, `regression`, `regression_timeout`, `daily_token_budget`, `max_depth_tokens`, `max_failure_summary_words`, `terminal_timeout`, `context_safety` (`output_reserve_tokens`, `safety_margin_tokens`) |
| `cache` | `enabled`, `dir`, `ttl_days`, `max_entries` |
| `circuit_breaker` | `window_size`, `local_error_ceiling`, `deepseek_error_ceiling`, `cooldown_s` |
| `memory` | `root`, `max_project_summary_words`, `semantic_similarity`, `embedding_model`, `embedding_threshold` |
| `providers` | `online`, `local` (lists of `{name, base_url, model, api_key_env, api_key, enabled, timeout_s, max_retries}`) |
| `deploy` | `command`, `cwd`, `timeout` |
| `bound` | `danger_zones`, `never_do`, `iron_laws` |
| `program` | `phases` (list of `{name, gate | journeys, max_fix_rounds, on_fail}`) |
| `journey` | `file`, `browser`, `timeout_s`, `screenshots_dir` |
| `guardrails` | `block`, `approval_required`, `cost_limit` |
| `secrets_scan` | `mode` (redact/block/off), `types` |
| `dependency_gate` | `allow`, `audit` |
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

Per-call **context telemetry** is emitted live during runs (see
[Context Safety Controller](#context-safety-controller)): model, architecture,
window, safe input budget, input tokens, output reserve, zone, compaction,
output tokens, status, step, max output, vision and tool-use — one structured
line per local-model call so context-pipeline issues are easy to debug.
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
| `7` | GUARDRAIL APPROVAL_REQUIRED — the task needs human approval. |
| `130` | Interrupted (Ctrl-C). |

## Testing & CI

```bash
cd "Agents /hybrid-agent"
./.venv/bin/python -m unittest discover -s tests -v    # 256 tests
```

Coverage includes: the fenced-file parser, the apply overwrite/unsafe-path guards,
dry-run, the clarify heuristic, `CacheManager` (TTL/cap/disabled), the verdict cache,
`TaskMemory` (semantic recall, scored eviction, consolidation, insights), the
supervision router (trivial/critical signals, budget pressure, overrides, memory
similarity), the local-first `review=False` supervise path, token-budget accounting,
embedding clients and cosine, per-project memory scoping, config-key validation,
provider registry (2+2, overrides, key resolution), failover + turbo cloud wrappers,
the encrypted secrets store, the dashboard API (stats/providers/config/auth),
gitops pull/push/deploy (commit-to-bare-remote, no-changes skip, upstream
auto-set, deploy commands), parallel-verify groups, truncation retries, and
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
    ├── supervise.py             # Qwen-primary / DeepSeek-supervisor control loop
    ├── parallel.py              # Plan parsing, dependency groups, parallel executor
    ├── memory.py                # Persistent task memory (semantic recall, consolidation)
    ├── embed.py                 # Local embeddings client (/v1/embeddings) + cosine
    ├── providers.py             # Provider registry (2 online + 2 local, failover-ready)
    ├── gitops.py                # git pull / push / deploy helpers
    ├── bound.py                 # BOUND: danger zones, never_do, iron laws (Ouro Loop)
    ├── journeys.py              # Vibe-DSL user-journey verification (headless browser)
    ├── differential.py          # Anti-gaming ratchet (rejects weakened gates)
    ├── guardrails.py            # BLOCK / APPROVAL_REQUIRED task guardrails
    ├── patcher.py               # Surgical diff-first repairs + AST context extraction
    ├── contract.py              # Task Contract + acceptance cases
    ├── dependencies.py          # Dependency-aware context retrieval
    ├── scanner.py               # Secrets/PII redaction for outbound traffic
    ├── rules.py                 # .agent/engineering-rules.yml loader
    ├── learned_rules.py         # Failure-rule learner (procedural memory)
    ├── dependency_gate.py       # New-dependency approval gate
    ├── web_dashboard.py         # Web control panel (Flask + SocketIO, port 8660)
    ├── dashboard/               # Dashboard UI + encrypted secrets store
    ├── context.py / scan.py     # Project context scanner
    ├── backends/                # Qwen/DeepSeek clients + circuit breaker
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
