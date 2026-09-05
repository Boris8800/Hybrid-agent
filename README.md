# Hermes Agent Installer

A single, self-contained, **idempotent installer** for the real
**[Hermes Agent](https://github.com/NousResearch/hermes-agent)** (NousResearch) —
the AI agent that ships with its own full **web dashboard**
(`hermes dashboard` → http://127.0.0.1:9119), browser chat, sessions, config editor,
API-key manager, logs, analytics and cron.

You run one file once. It clones Hermes Agent into `~/.hermes/hermes-agent`, creates a
venv, installs the web + chat extras, and gives you a menu. Re-running it later is fast
(no reinstall) — it just opens the menu so you can launch the dashboard, chat, or
model config.

## Install

```bash
git clone https://github.com/Boris8800/Hybrid-agent.git
cd Hybrid-agent
bash install.sh
```

Shortcuts (install if needed, then run straight into a mode):

```bash
bash install.sh dashboard   # open the web dashboard at http://127.0.0.1:9119
bash install.sh chat        # start the chat TUI
bash install.sh setup       # pick your default model/provider
```

## Requirements

- macOS (any architecture) with Homebrew available (used to install `uv` if missing)
- `git`
- Internet access (clone + PyPI). Hermes Agent picks its own Python (3.11–3.13) via `uv`.
- Node.js + npm (to build the web UI on first `dashboard` launch)

## What it installs and where

| Item | Location |
|---|---|
| Agent repo | `~/.hermes/hermes-agent` |
| Virtual env + `hermes` | `~/.hermes/hermes-agent/.venv/bin/hermes` |
| Extras | `web` (FastAPI/Uvicorn dashboard), `pty` (browser chat) |
| Config / keys / sessions | `~/.hermes/` (managed by the agent) |

## Using it (daily)

Hermes stays installed. To use it again, run `install.sh` (menu) or use the binaries
directly:

```bash
export PATH="$HOME/.hermes/hermes-agent/.venv/bin:$PATH"
hermes model       # set default model (pick your LOCAL model for local-first/hybrid use)
hermes chat        # text agent
hermes dashboard   # web control panel -> http://127.0.0.1:9119
```

## Hybrid mode — online plans · local works · online is the last gate

`hybrid.sh` uses the REAL Hermes Agent for **both** roles, so:

1. **Online supervisor = whichever model you choose in Hermes** (its default —
   set via `hermes model`, and it can be a free online model).
2. **Your local model is the worker** and does ALL the real work — edits, commands,
   tool calls — via `hermes -m <local>`.
3. **Online supervisor inspects** each round and asks for fixes if needed.
4. **Online supervisor is the LAST GATE**: final `APPROVED` / `REJECTED`.

No separate API keys are needed — Hermes already holds the model credentials, so the
supervisor can be **any model you can pick in Hermes** (free or paid).

One-time setup — pick your models:

```bash
# in Hermes, set your ONLINE (supervisor) model as the default:
hermes model

# tell hybrid which LOCAL model is the worker (add to the folder's .env):
echo 'HYBRID_LOCAL_MODEL=your-local-model-name' >> ~/Desktop/Hermes-<date>/.env
```

Run a task:

```bash
bash hybrid.sh "build a fibonacci module with tests"
echo "some task" | bash hybrid.sh
bash hybrid.sh --file task.txt
```

Overrides (defaults shown):

```
HYBRID_LOCAL_MODEL=<your-local-model>   # worker model for `hermes -m`
HYBRID_MAX_ROUNDS=3                     # plan -> run -> inspect fix rounds
HYBRID_WORK=~/.hermes/hybrid_work       # artifacts (plan, transcript, verdicts)
```

The online (supervisor) model is exactly what Hermes' default is set to. Transcripts
and verdicts are saved under `~/.hermes/hybrid_work/` for review.

## Two ways to chat (pick per chat)

Use the Desktop launcher or `install.sh` menu:
- **Start chat** → a normal Hermes chat. Set its model to your **local** model for a
  free local conversation.
- **Run a task in HYBRID mode** → online supervisor plans/inspects/gates while the
  local model does the work.

## Uninstall

```bash
rm -rf ~/Desktop/Hermes-<date>     # deletes the whole one-folder install
```
