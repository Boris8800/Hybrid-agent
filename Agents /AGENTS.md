# AGENTS.md — Workspace Knowledge

Operational conventions and known pitfalls for this workspace (`/Users/user/Desktop/VS`). This is a Kilo/hybrid-agent setup that manages the project at `vvvxvvxvvx/`.

## Local AI / LM Studio

- Local model: `qwen2.5-coder-14b-instruct-mlx` via the MLX-style chat API at `http://localhost:1234/api/v1` (POST {base}/chat with `system_prompt`/`input`; embeddings stay on the legacy `http://localhost:1234/v1/embeddings`). Run the bridge with `python3 hybrid-agent/ask.py` — it self-heals into `hybrid-agent/.venv` for PyYAML/openai, so `config.yml` (180s local timeout) is always loaded.
- The hybrid agent runs the 80/20 law: local model generates first, DeepSeek reviews (needs `DEEPSEEK_API_KEY`; when unset the Kilo brain is the review gate).

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
