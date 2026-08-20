# AGENTS.md — Workspace Knowledge

Operational conventions and known pitfalls for this workspace (`/Users/user/Desktop/VS`). This is a Kilo/hybrid-agent setup that manages the project at `vvvxvvxvvx/`.

## Local AI / LM Studio

- Local model: `qwen2.5-coder-14b-instruct-mlx` via the MLX-style chat API at `http://localhost:1234/api/v1` (POST {base}/chat with `system_prompt`/`input`; embeddings stay on the legacy `http://localhost:1234/v1/embeddings`). Run the bridge with `python3 hybrid-agent/ask.py` — it self-heals into `hybrid-agent/.venv` for PyYAML/openai, so `config.yml` (180s local timeout) is always loaded.
- The hybrid agent runs the 80/20 law: local model generates first, DeepSeek reviews (needs `DEEPSEEK_API_KEY`; when unset the Kilo brain is the review gate).
- Local inference is slow (~10-15 tok/s). The bridge sends NO `max_tokens` — the `/api/v1/chat` endpoint rejects it (`Unrecognized key(s) in object: 'max_tokens'` → empty response). `exclude_keys` handles this.

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
python3 "Agents /hybrid-agent/reload-local.sh"   # unloads + reloads qwen with --parallel 1 + verifies
# manual: lms unload qwen2.5-coder-14b-instruct-mlx && lms load qwen2.5-coder-14b-instruct-mlx --parallel 1 -c 16384 -y
```

Check current state with `lms ps` — the qwen row must show `PARALLEL 1`. Any load from the LM Studio UI may revert to `parallel: 4`, so re-run the script after reloading in the app. Kilo config for the local provider lives in `.kilo/kilo.jsonc` and `~/.config/kilo/kilo.jsonc` (`provider.lmstudio` → `http://localhost:1234/v1`).

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
