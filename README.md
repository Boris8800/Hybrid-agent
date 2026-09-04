# Hermes — Self-Contained Hybrid Agent (macOS / Apple Silicon)

Hermes is a local-first **hybrid coding agent** for macOS. A cheap/free **local model**
does the real work on your machine, while an **online AI (DeepSeek)** is used only to
*plan* and *supervise* — so you get strong results without burning tokens on every
edit, file read, or command.

It is delivered as a **single installer** that, each time you run it, creates a fresh,
fully self-contained runtime folder on your Desktop (`~/Desktop/Hermes-<date>`).
Everything — Python venv, config, CLI agents, web dashboard, and an **MCP tool server**
with ~46 local tools — lives inside that one folder. Nothing is installed into
`~/Agents`, `~/.zprofile`, or elsewhere on your machine.

> **Repository layout (installer only).** This repository is the *installation kit*.
> `install.sh` sits at the top level; all agent source files live inside the single
> `Hermes/` subfolder (`mcp_server.py`, `start_mcp.sh`, `ask_mcp.sh`,
> `tools_manifest.json`). `install.sh` is fully self-contained — it embeds every
> payload — so it works even if only `install.sh` is copied alone. Your actual
> running agent lives in the `Hermes-<date>` folder that `install.sh` creates;
> keep this repo clean as your installer source.

---

## What you get

| Piece | Purpose |
|---|---|
| `ask.py` | Quick one-shot: local Qwen implements, optional DeepSeek review. |
| `run_agent.py` | Full pipeline: online **planner** → local **implementer** → online **supervisor** → **recorder**. |
| `web_dashboard.py` | Flask + Socket.IO dashboard (`start_dashboard.sh`, port 8660). |
| `mcp_server.py` | **MCP tool server** exposing ~46 local tools (stdlib JSON-RPC). |
| `config.yml` | Full configuration (backends, router, review, cache, circuit breaker, memory, safety, guardrails, secrets scan). |
| `.env` | API keys + overrides. |
| `memory/` | Per-install long-term memory (see below). |

### Architecture

```
┌─────────────┐   plan + supervise   ┌──────────────────────┐
│  Online AI  │◄────────────────────│                      │
│  (DeepSeek) │                     │      run_agent.py     │
└──────┬──────┘                     │   ┌────────────────┐   │
       │ prompts/supervises          │   │  local Qwen    │   │
       ▼                             │   │  (LM Studio /  │   │
┌─────────────┐   implements steps   │   │   Ollama)      │   │
│ Local model │─────────────────────►│   └────────┬───────┘   │
└──────┬──────┘                     │            │ calls    │
       │                            │            ▼           │
       └────────────────────────────│    mcp_server.py       │
                                    │  (files, terminal,     │
                                    │   web, browser, docs,  │
                                    │   memory, cron, ...)   │
                                    └──────────────────────┘
```

The point: the **online AI never touches your files or runs commands**. It only decides
*what* to do and checks the work. Everything heavy is executed locally — which is what
keeps online-token spend low.

---

## Quick start

Requirements (already on most Macs):

- macOS on **Apple Silicon** (Intel works with small caveats)
- A working `python3` (used only to build the in-folder venv)
- **Either** [LM Studio](https://lmstudio.ai) **or** [Ollama](https://ollama.com)
  running a local model (e.g. Qwen 2.5 Coder 14B) on `localhost:1234`
- A [DeepSeek API key](https://platform.deepseek.com/) for the online planner/supervisor

### 1. Clone the installer kit

```bash
git clone https://github.com/Boris8800/Hybrid-agent.git
cd Hybrid-agent
```

### 2. Run the installer

```bash
bash install.sh
```

It creates a new dated runtime folder each run:

```
~/Desktop/Hermes-2026-09-04
```

Install to a specific folder or base path if you prefer:

```bash
HERMES_DIR=/path/to/folder bash install.sh     # exact folder
HERMES_BASE=~/Somewhere   bash install.sh      # where the dated folder goes
```

### 3. Configure your key

```bash
cd ~/Desktop/Hermes-<date>
nano .env          # set DEEPSEEK_API_KEY
```

### 4. Try it

```bash
./ask.py --task "write a hello world in python"

./run_agent.py --task "build a fibonacci module with tests"
# planner = online DeepSeek · implementer = local Qwen · supervisor = DeepSeek
```

### 5. Start the dashboard or the MCP tool server

```bash
./start_dashboard.sh          # http://localhost:8660
./start_mcp.sh                # MCP server over stdio (attach to an MCP host)
./start_mcp.sh --list         # list the built-in tools
./ask_mcp.sh read_file '{"path":"config.yml"}'   # test a single tool directly
```

---

## The tools (why you save tokens)

`mcp_server.py` is a dependency-light MCP server (newline-delimited JSON-RPC over
stdio, standard library only). It implements the **zero-credential** tools that do the
grunt work locally, so the online AI only plans and supervises.

Implemented now (no keys needed):

- **Filesystem** — read/write/patch/search/list/info/move/copy/delete (safety-gated)
- **Terminal & processes** — run commands, background, list/kill, env
- **Web** — DuckDuckGo search (no key), fetch, URL validate, link extract, metadata, RSS
- **Browser** — headless Chromium via Playwright (`browser_navigate`)
- **Documents** — read PDF / DOCX / XLSX (needs `pypdf`, `python-docx`, `openpyxl`)
- **Data** — CSV / JSON read + conversion
- **Code** — run Python, static analysis
- **Memory** — store / recall / stats (uses this install's `memory/`)
- **Security** — secret scan, password gen, hashing, Fernet encrypt/decrypt
- **Database** — local SQLite
- **Todo / Cron / Notifications** — file-backed todo, persisted cron jobs, webhook + macOS notify

Everything else (Slack/Discord/Telegram/email, AWS/GCP/Azure CLIs, Kubernetes,
Terraform, Docker, messaging, media models, and cloud APIs) is **covered two ways**:
account-based services activate when their key/CLI is present, and anything runnable on
a shell (git, docker, kubectl, pytest, ffmpeg, ssh, …) is already reachable through the
local `exec` tool. See `tools_manifest.json` for a full item-by-item coverage audit.

### Safety model

Destructive tools refuse to run unless you opt in for the session:

```bash
HERMES_SAFE=1 ./start_mcp.sh            # allow delete/kill/destructive tools
HERMES_ALLOW=delete_file,kill_process ./start_mcp.sh   # allowlist specific tools
HERMES_PROJECT=/abs/project ./start_mcp.sh             # confine file tools to a root
```

File tools never escape the project root; nothing destructive runs silently.

---

## Memory

**Memory is per-install and lives inside the runtime folder** (`<runtime>/memory/`):

- Every ask/run is recorded under `memory/logs/`; a distilled long-term index is kept
  in `memory/index.md`.
- `run_agent.py` reads recent memory back and feeds it to the planner as context.
- **Delete the runtime folder → deletes everything, including memory.** Nothing is
  stored elsewhere.

To keep memory for a new install, copy the `memory/` folder into the new runtime
folder:

```bash
cp -R "/Users/<you>/Desktop/Hermes-OLD/memory" "/Users/<you>/Desktop/Hermes-NEW/memory"
```

`memory/` is self-contained, so you can also rename or back it up freely. Set
`HERMES_MEMORY=/abs/path` in `.env` to point memory anywhere you like.

---

## Configuration highlights (`config.yml`)

| Section | Purpose |
|---|---|
| `backends.local` / `backends.deepseek` | Model endpoints, timeouts, retries |
| `router` | When to prefer local vs. online (thresholds, weights, target local rate) |
| `review` | Verification + regression settings, token budgets |
| `cache` / `circuit_breaker` | Result caching and failure handling |
| `memory` | Memory root + embedding settings |
| `bound` / `guardrails` | Safety: danger zones, never-do commands, iron laws, approval-required |
| `secrets_scan` | Redact/block detected secrets |
| `providers` | DeepSeek, Groq (disabled by default), and local (Qwen) |

---

## Removing Hermes

```bash
# 1. stop the dashboard/MCP processes you started
# 2. delete the runtime folder (this removes everything, memory included)
rm -rf ~/Desktop/Hermes-<date>
```

The installer kit repo is separate — delete it only if you no longer want it.

---

## Troubleshooting

- **`DEEPSEEK_API_KEY not set`** → edit `.env` in the runtime folder and set the key.
- **Local model fails / connection refused** → LM Studio: load
  `qwen2.5-coder-14b-instruct-mlx` and start the local server on `:1234`. Ollama:
  `ollama pull qwen2.5-coder:14b && ollama serve`.
- **MCP tools blocked** → that is the safety gate. Run with `HERMES_SAFE=1` or allowlist.
- **PDF/DOCX/XLSX read errors** → `pip install pypdf python-docx openpyxl` in the venv.

---

## License

MIT
