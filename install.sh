#!/bin/bash
# ============================================================
# HERMES AGENT INSTALLER — NousResearch Hermes Agent
# ============================================================
# A single, idempotent installer for the real Hermes Agent
# (https://github.com/NousResearch/hermes-agent), which includes its
# own web dashboard (hermes dashboard -> http://127.0.0.1:9119).
#
# What it does:
#   1. checks macOS / prerequisites
#   2. ensures `uv` is installed (Homebrew)
#   3. creates ONE dated folder ~/Desktop/Hermes-<date> and clones Hermes Agent
#      into it (software + all data stay inside that single folder)
#   4. creates a venv and installs the web + pty extras
#   5. offers a menu to configure the model, chat, or open the dashboard
#
# Idempotent: re-running is fast — it skips what already exists and opens
# the menu, so you NEVER reinstall to use it again.
#
# Usage:  bash install.sh            # install + menu
#         bash install.sh dashboard  # install if needed, then open the dashboard
#         bash install.sh chat       # install if needed, then start chat
#         bash install.sh setup      # install if needed, then run model config
# ============================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log(){ echo -e "${BLUE}$1${NC}"; }
ok(){ echo -e "${GREEN}  ✓ $1${NC}"; }
warn(){ echo -e "${YELLOW}  ! $1${NC}"; }
fail(){ echo -e "${RED}  ✗ $1${NC}"; }

# --- Location: ONE dated, self-contained folder on the Desktop ---
# Each run makes a NEW folder ~/Desktop/Hermes-<YYYYMMDD> (-2, -3 … if reused).
# ALL Hermes data (software, venv, config, sessions, memory, logs) lives inside it.
# Delete the folder = delete the whole install (memory included).
BASE="${HERMES_BASE:-$HOME/Desktop}"
if [ -n "${HERMES_DIR:-}" ]; then
    AGENT_DIR="${HERMES_DIR}"
else
    AGENT_DIR="$BASE/Hermes-$(date +%Y%m%d)"
    n=1
    while [ -e "$AGENT_DIR" ]; do
        n=$((n+1))
        AGENT_DIR="$BASE/Hermes-$(date +%Y%m%d)-$n"
    done
fi
export HERMES_HOME="$AGENT_DIR"          # Hermes stores ALL its data under here
HERMES_ROOT="$AGENT_DIR"
REPO="$HERMES_ROOT/hermes-agent"
BIN="$REPO/.venv/bin/hermes"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERMES_ROOT"

log "============================================================"
log "   HERMES AGENT (NousResearch) — INSTALLER"
log "============================================================"
log "  ONE folder     : $AGENT_DIR  (created this run)"
log "  Agent software : $REPO"
log "  venv           : $REPO/.venv"
log "  Hermes data    : $AGENT_DIR  (config/sessions/memory/logs stay here)"
log "============================================================"

# --- 1. system ---
echo -e "${BLUE}--- 1/5 System requirements ---${NC}"
ok "macOS: $(sw_vers -productVersion) ($(uname -m))"

# --- 2. uv ---
echo -e "${BLUE}--- 2/5 uv (Python env manager) ---${NC}"
if command -v uv >/dev/null 2>&1; then
    ok "uv: $(uv --version 2>&1 | head -1)"
else
    if command -v brew >/dev/null 2>&1; then
        echo "  Installing uv via Homebrew..."
        brew install uv
        ok "uv installed"
    else
        echo "  Installing uv via the standalone installer..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        ok "uv installed"
    fi
fi

# --- 3. clone ---
echo -e "${BLUE}--- 3/5 Clone Hermes Agent ---${NC}"
if [ ! -d "$REPO" ]; then
    mkdir -p "$HERMES_ROOT"
    echo "  Cloning (shallow) into $REPO ..."
    git clone --depth 1 https://github.com/NousResearch/hermes-agent.git "$REPO"
    ok "cloned"
else
    ok "already cloned: $REPO"
fi
cd "$REPO"

# --- 4. venv + deps ---
echo -e "${BLUE}--- 4/5 Virtual env + dependencies (web, pty) ---${NC}"
if [ ! -x "$BIN" ]; then
    if [ ! -d ".venv" ]; then
        echo "  Creating venv..."
        uv venv
    fi
    echo "  Installing '.[web,pty]' — this pulls dependencies and may take a while..."
    uv pip install -e ".[web,pty]"
    ok "Hermes Agent installed"
else
    ok "already installed ($("$BIN" --version 2>&1 | head -1))"
fi

# --- 4c. Bundle hybrid (part of the install, not a separate thing) ---
# hybrid.sh uses the REAL Hermes Agent for both roles:
#   ONLINE supervisor = the model you choose in Hermes (its default),
#   LOCAL worker      = your local model (hermes -m <local>).
if [ -f "$SCRIPT_DIR/hybrid.sh" ]; then
    cp "$SCRIPT_DIR/hybrid.sh" "$HERMES_HOME/"
    chmod +x "$HERMES_HOME/hybrid.sh"
    ok "Hybrid mode bundled in this folder (hybrid.sh)"
else
    warn "hybrid.sh not beside install.sh — hybrid not bundled (re-run from the full repo)"
fi

# --- 4b. Desktop launcher icon ---
DESKTOP_LAUNCHER="$HOME/Desktop/Hermes Agent.command"
cat > "$DESKTOP_LAUNCHER" <<'LAUNCHER'
#!/bin/bash
# Hermes Agent — desktop launcher (double-click me). ONE dated folder.
export HERMES_HOME="__AGENT_DIR__"
export PATH="__AGENT_DIR__/hermes-agent/.venv/bin:$PATH"
cd "$HOME" || exit 1
while :; do
  clear 2>/dev/null || true
  echo "=============================================="
  echo "   HERMES AGENT   (folder: $HERMES_HOME)"
  echo "=============================================="
  echo "  1) Open web dashboard   (http://127.0.0.1:9119)"
  echo "  2) Start chat"
  echo "  3) Set default model    (hybrid: local-first)"
  echo "  4) Run a task in HYBRID mode (online plans -> local works -> online gate)"
  echo "  0) Quit"
  printf 'Choose [0-4]: '
  read -r c
  case "$c" in
    1) hermes dashboard ;;
    2) hermes chat ;;
    3) hermes model ;;
    4) printf 'Task: '; read -r t; [ -n "$t" ] && bash "$HERMES_HOME/hybrid.sh" "$t"; printf '[enter] back to menu'; read -r _ ;;
    0|q|Q) echo "bye"; break ;;
    *) echo "  invalid: $c"; sleep 1 ;;
  esac
done
LAUNCHER
chmod +x "$DESKTOP_LAUNCHER"
# bake in the actual dated folder path
sed -i '' "s|__AGENT_DIR__|$AGENT_DIR|g" "$DESKTOP_LAUNCHER"
ok "Desktop launcher created: $DESKTOP_LAUNCHER"

# --- 5. launch / menu ---
echo -e "${BLUE}--- 5/5 Ready ---${NC}"
if [ -x "$BIN" ]; then
    export PATH="$REPO/.venv/bin:$PATH"
fi

ACT="${1:-}"
case "$ACT" in
    dashboard) echo "Opening the web dashboard..."; hermes dashboard; exit 0 ;;
    chat)      echo "Starting chat...";           hermes chat;    exit 0 ;;
    setup|model) echo "Model/provider setup...";  hermes model;   exit 0 ;;
esac

echo ""
log "  Hermes Agent is ready."
echo "    ONE folder : $AGENT_DIR   (everything lives here; delete it to remove Hermes)"
echo "    Activate   :  export PATH=\"$REPO/.venv/bin:\$PATH\"   (or: source $REPO/.venv/bin/activate)"
echo "    Configure  :  hermes model      (pick default model — choose your LOCAL one for hybrid/local-first)"
echo "    API keys   :  add DEEPSEEK_API_KEY=sk-... to  $HERMES_HOME/.env   (for hybrid online plan/gate)"
echo ""
echo "  What do you want to do?"
echo "    1) Configure model/provider   (hermes model)"
echo "    2) Start chat                 (hermes chat)"
echo "    3) Open web dashboard         (hermes dashboard -> http://127.0.0.1:9119)"
echo "    4) Status / doctor            (hermes status)"
echo "    0) Exit (Hermes stays installed — run install.sh again to launch)"
printf 'Choose: '
read -r c
case "$c" in
    1) hermes model ;;
    2) hermes chat ;;
    3) hermes dashboard ;;
    4) hermes status ;;
    *) echo "bye" ;;
esac
echo ""
echo -e "${GREEN}Done. To use Hermes again later, just run:  bash install.sh${NC}"
