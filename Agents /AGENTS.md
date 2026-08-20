# AGENTS.md — Workspace Knowledge

Operational conventions and known pitfalls for this workspace (`/Users/user/Desktop/VS`). This is a Kilo/hybrid-agent setup that manages the project at `vvvxvvxvvx/`.

## Local AI / LM Studio

- Local model: `qwen2.5-coder-14b-instruct-mlx` via the MLX-style chat API at `http://localhost:1234/api/v1` (POST {base}/chat with `system_prompt`/`input`; embeddings stay on the legacy `http://localhost:1234/v1/embeddings`). Run the bridge with `python3 hybrid-agent/ask.py` — it self-heals into `hybrid-agent/.venv` for PyYAML/openai, so `config.yml` (600s local timeout) is always loaded.
- The hybrid agent runs the 80/20 law: local model generates first, DeepSeek reviews (needs `DEEPSEEK_API_KEY`; when unset the Kilo brain is the review gate).
- Local inference is slow (~10-15 tok/s). The bridge sends NO `max_tokens` — the `/api/v1/chat` endpoint rejects it (`Unrecognized key(s) in object: 'max_tokens'` → empty response). `exclude_keys` handles this.
- **Truncation safety is model-agnostic and window-agnostic:** at startup the bridge queries LM Studio `/api/v1/models` and reads the ACTUAL loaded context window (`loaded_instances[].config.context_length`). That number drives (a) server-side truncation detection at the real boundary and (b) the DeepSeek planning prompts ("IMPLEMENTER CONSTRAINTS"). Any local model — whatever `-c` it was loaded with, **including windows larger than 32768 (e.g. `-c 65536`, `131072`)** — is planned for its real window and cutoffs are caught at exactly that boundary. **Discovery always wins over `context_window:` in `config.yml`**, so a stale/too-small config can never cap a bigger loaded window (config is used only when the server is unreachable). When discovery fails the engine default is 32768.
- **Context Safety Controller (CSC)** — `context_safety.py` turns the window into hard budgets so the local model never reaches an unusable context state. Safe input budget = discovered window − output reserve (default 12000) − safety margin (default 2000), configurable via `review.context_safety`. Before EVERY local call the prompt is measured and preflighted: **GREEN** <70% proceed, **YELLOW** 70–85% compress, **ORANGE** 85–95% aggressive compaction, **RED** >95% → the local model is **NOT called** and the task escalates to DeepSeek (`local_context_red_escalation`). Source context is compacted first, then terminal output is summarized (exit code / errors / stack traces / changed files / warnings — never raw 20k-token logs). The DeepSeek supervisor receives a `LOCAL MODEL CONTEXT` block (window, safe input, output reserve, step N/M, tools, zone) in every review, so the API model plans within the real usable context. Output states are classified distinctly: COMPLETED / OUTPUT_LIMIT_REACHED / CONTEXT_LIMIT_REACHED / TIMEOUT / MODEL_ERROR.
- **Per-model capability discovery** — `discover_model_capabilities()` reads model id, ACTUAL loaded window, max context, architecture, format, quantization, max output, vision, tool-use from `/api/v1/models`. Models are never assumed to behave alike. The output reserve is **dynamic**: `min(configured_reserve, model_max_output, task_required_output)` so a tiny task does not sacrifice 12k tokens of context (observed live: a 2-file task ran with reserve 512 → safe budget 30256).
- **Compaction invariants** — compaction NEVER removes the protected core: task requirements, BOUND, contract (files being modified, security requirements), supervisor fixes, engine protocol. Only bulk context (source tree, tool output) is compactable; if even the protected core exceeds the budget, the prompt is returned intact and the RED preflight escalates instead of truncating the task.
- **Retry budget** — continuation recovery after incomplete local output is hard-capped (`max_recovery_retries`, default 1) then escalates; a pathological model cannot burn unbounded time/tokens.
- **Context telemetry** — one structured line per local call: `MODEL | ARCH | WINDOW | SAFE_INPUT | INPUT | OUT_RESERVE | ZONE | COMPACTION | OUTPUT | STATUS | STEP | MAX_OUTPUT | VISION | TOOLS`. Token estimation is conservative (3.5 chars/token — code tokenizes denser than prose; `HYBRID_CHARS_PER_TOKEN` overrides).
- **Durable task state** — `task_state.py` makes the Task Contract the permanent source of truth (JSON at `<root>/.hybrid-agent/tasks/TASK-<hash>.json`). Strict state machine: PLANNING → IMPLEMENTING → GENERATED → REVIEWING → REVIEWED → VERIFYING → VERIFIED → APPROVED → APPLIED → DEPLOY_AUTHORIZED → PUSHED → DEPLOYED (FAILED/REJECTED terminal). Illegal transitions (FAILED→DEPLOYED, GENERATED→APPROVED, REVIEWED→APPROVED, TRUNCATED→APPROVED) raise `IllegalTransition` at code level. Evidence ledger (E-001…: command, exit code, output hash, files, source) makes approvals auditable. Batch transactions are explicit (SUCCESS / PARTIAL_FAILURE / FAILED — 3/4 steps OK is never success). Apply/push/deploy are idempotent via `OperationLog` (a completed op never re-executes). `--resume` continues from durable state (step N/M, evidence, completed ops). Deployment requires the `DEPLOY_AUTHORIZED` trust boundary — passing tests alone never authorizes it. `--turbo` adjudicates provider verdicts deterministically (strictest wins; REJECTED > FIX_REQUIRED > UNKNOWN > APPROVED), never "longest response".

## Known pitfall: chat degeneration ("garbage output")

LM Studio chat produced `!!!�"""...Side""Side` garbage on `qwen2.5-coder-14b-instruct-mlx`. Root cause (confirmed via `~/.lmstudio/conversations/*.json`): a stale thinking-mode custom field from the retired 12B model config (`ext.virtualModel.customField.*.enableThinking`) had leaked onto qwen chats, plus no repetition penalty — the model looped on repeated tokens. Fixed 2026-08-20: stripped the foreign field from qwen chats, added `llm.prediction.repeatPenalty: 1.15` + `llm.prediction.temperature: 0.2` to every conversation (backup: `~/.lmstudio/conversations-backup-2026-08-20/`).

**Bridge path:** `ask.py` talks to `/api/v1/chat` and does NOT see LM Studio's per-conversation prediction config, so the chat-side fix does NOT protect bridge generations (that is where the 2026-08-20 `longest_palindrome` garbage `== ((x + y) ** 2, ...` repetition came from). The bridge now sends `request_extra: {repeat_penalty: 1.15}` on the local providers in `config.yml` (`providers.local.*`) — `repeat_penalty` (snake_case) is the only repetition-penalty key the native endpoint accepts (`repetition_penalty`, `repeatPenalty`, `frequency_penalty` are all rejected with `unrecognized_keys`).

Prevention rules:
- Never set a thinking-mode `enableThinking` custom field on qwen (non-thinking) chats — it forces a mismatched thinking prompt template and degrades output.
- Keep `llm.prediction.repeatPenalty` ≥ 1.1 in every chat (guards against token-repetition loops).
- A coding model asked out-of-domain questions (e.g. "why is voice not available") will hallucinate — that is not a server bug. Use qwen for both general chat and code.
- Diagnose: `rg -i "error|exception" ~/.lmstudio/server-logs/YYYY-MM/*.log`; empty/garbled local output + a clean isolated `curl` to `/api/v1/chat` means the problem is session/context state, not the model.

## Known pitfall: Kilo chat crashes the MLX backend (MUST load with `--parallel 1`)

Kilo's chat talks to LM Studio via the **OpenAI-compatible** endpoint `POST http://localhost:1234/v1/chat/completions`. On this MLX model loaded with the default **`parallel: 4`**, every request crashes the backend scheduler:

```
ValueError: Slice indices must be 32-bit integers.
  .../mlx_engine/model_kit/batched_model_kit.py:410  token_logprob = r.logprobs[r.token].item()
```

The crash even **unloads the model** (next request then answers "No models loaded"). The native `POST /api/v1/chat` endpoint is unaffected, so `ask.py` (the hybrid bridge) keeps working — this only breaks Kilo's own chat.

**Fix:** load the model with `--parallel 1` (single-sequence; bypasses the batched scheduler). Verified working for both `stream: false` and `stream: true` on 2026-08-20, LM Studio 0.4.21+2 / MLX backend `app-mlx-generate-mac14-arm64@34`.

```bash
python3 "Agents /hybrid-agent/reload-local.sh"   # unloads + reloads qwen with --parallel 1 -c 32768 + verifies
# manual: lms unload qwen2.5-coder-14b-instruct-mlx && lms load qwen2.5-coder-14b-instruct-mlx --parallel 1 -c 32768 -y
```

Check current state with `lms ps` — the qwen row must show `PARALLEL 1` and `CONTEXT 32768`. Any load from the LM Studio UI may revert to `parallel: 4`, so re-run the script after reloading in the app. Kilo config for the local provider lives in `.kilo/kilo.jsonc` and `~/.config/kilo/kilo.jsonc` (`provider.lmstudio` → `http://localhost:1234/v1`).

> **Do NOT load qwen with a smaller context window (e.g. `-c 16384`).** The engine plans every coding step for a 32768-token window (`supervise.py` `LOCAL_CONTEXT_TOKENS`, `config.yml` `context_window: 32768`) and detects server-side truncation at that exact boundary. A smaller load makes LM Studio cut off multi-file output mid-generation — the "constantly truncated" symptom. The model supports the full 32768 (`config.json` `max_position_embeddings`).

## Known pitfall: indexing embedding baseUrl (MUST include `/v1`)

`~/.config/kilo/kilo.jsonc` → `indexing.openai-compatible.baseUrl` must end in **`/v1`**:

```jsonc
"indexing": {
  "enabled": true,
  "provider": "openai-compatible",
  "model": "text-embedding-nomic-embed-text-v1.5",
  "dimension": 768,
  "openai-compatible": {
    "baseUrl": "http://127.0.0.1:1234/v1"   // /v1 is REQUIRED
  }
}
```

Kilo's embedding provider appends **only `/embeddings`** to the baseUrl. So `baseUrl` ending in `/v1` → `/v1/embeddings` (works); a host-root baseUrl (no `/v1`) → `/embeddings`, which LM Studio answers with HTTP 200 + `{"error":"Unexpected endpoint or method."}` — indexing fails silently ("stuck indexing / initializing", "never initiates"). Diagnose via `~/.lmstudio/server-logs/2026-08/*.log`: a bare `POST /embeddings` error entry means the baseUrl is missing `/v1`.

Verify with:

```bash
curl -s http://127.0.0.1:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5","input":"x"}' \
  # expect data[0].embedding (768-dim); an error body means the path is wrong
```

## Project: `vvvxvvxvvx` (full-stack todo app)

- Backend: Express + TypeScript + Prisma + PostgreSQL, port 5000.
- Frontend: React + Vite + TS + Tailwind, port 3000 (dev proxy `/api` → 5000).
- Run: `docker compose up --build` from `vvvxvvxvvx/`. Local dev: `backend` (`.env` → `npm run dev`) and `frontend` (`npm run dev`).
- Default demo accounts (seed): `alice@example.com` / `password123`, `bob@example.com` / `password123`.
