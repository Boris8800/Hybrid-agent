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

## Hybrid / local-first tip

To avoid spending tokens on routine work, set your **default model to a local one**
(LM Studio on `:1234`, or Ollama) via `hermes model`. Keep DeepSeek / an online model
available and switch to it only when you want the stronger model or need supervision.
Hermes Agent loads providers by probing their `/v1/models`, so a running local server
is auto-detected.

## Uninstall

```bash
bash ~/.hermes/hermes-agent/uninstall   # or remove the folder + ~/.hermes config you no longer need
rm -rf ~/.hermes/hermes-agent
```
