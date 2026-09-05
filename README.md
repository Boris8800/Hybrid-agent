# Hermes Agent Installer

A single, self-contained installer for the real **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** (NousResearch) — the AI agent with its own **web dashboard** (`hermes dashboard` → http://127.0.0.1:9119), browser chat, sessions, config editor, API-key manager, logs and analytics.

`install.sh` is all you need. Run it once: it creates a fresh **`~/Desktop/Hermes-<date>`** folder on the Desktop and installs *everything* inside that one folder (software, venv, config, sessions, memory, logs). It also bundles two hybrid helpers and puts a **double-click launcher** on your Desktop.

## Install

```bash
git clone https://github.com/Boris8800/Hybrid-agent.git
cd Hybrid-agent
bash install.sh
```

Shortcuts (install if needed, then go straight to a mode):

```bash
bash install.sh dashboard   # web control panel -> http://127.0.0.1:9119
bash install.sh chat        # start chat
bash install.sh setup       # choose default model/provider
```

## Requirements

- macOS, Homebrew (to install `uv` if missing), `git`, internet.
- Node.js + npm (to build the web UI on first dashboard launch).

## Layout — ONE dated folder

| Item | Location |
|---|---|
| Everything (software + data) | `~/Desktop/Hermes-<date>` |
| Agent software + venv | `~/Desktop/Hermes-<date>/hermes-agent` |
| Config / sessions / memory | `~/Desktop/Hermes-<date>` (Hermes writes here) |
| Desktop launcher | `~/Desktop/Hermes Agent.command` |

Delete the `Hermes-<date>` folder = delete the whole install (memory included).

## Using it (daily)

Double-click **`Hermes Agent.command`**, or run `bash install.sh`. Menu:

```
1) Open web dashboard
2) Start chat              (normal Hermes chat)
3) Set default model
4) Run a HYBRID task       (online plan -> local work -> online gate)
5) SUPERVISED chat         (continuous chat, every message supervised)
0) Quit
```

## Hybrid & supervised chat

Both helpers are bundled into the install and use Hermes for **both** roles:

- **Online supervisor** = whichever model you set as Hermes' default (`hermes model`) — any free or paid model.
- **Local worker** = your local model, run via `hermes -m <local>`.

One-time setup:

```bash
hermes model                                          # set ONLINE (supervisor) model as default
echo 'HYBRID_LOCAL_MODEL=<your-local-model>' >> ~/Desktop/Hermes-<date>/.env
```

Then, per task or per chat, choose from the launcher menu (options 4 and 5). Transcripts/verdicts land in `~/.hermes/hybrid_work/`.

## Uninstall

```bash
rm -rf ~/Desktop/Hermes-<date>
```
