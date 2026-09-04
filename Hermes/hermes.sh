#!/bin/bash
# ============================================================
# HERMES — interactive menu launcher
# Pick a number instead of typing commands.
# ============================================================
cd "$(dirname "$0")" || exit 1
if [ -f ".env" ]; then set -a; source ".env"; set +a; fi
VENV=".venv/bin/python"; [ -x "$VENV" ] || VENV="python3"

ask_task() { printf '\nTask: '; read -r t; printf 'Using task: %s\n' "$t"; }

while :; do
  clear 2>/dev/null || true
  echo "=========================================================="
  echo "   HERMES — main menu"
  echo "   runtime folder: $(pwd)"
  echo "=========================================================="
  echo "  1) Edit .env (add API keys)"
  echo "  2) Quick ask            (ask.py --task ...)"
  echo "  3) Full pipeline        (plan -> local implement -> supervise)"
  echo "  4) Web dashboard        (http://localhost:8660)"
  echo "  5) MCP tools            list available local tools"
  echo "  6) MCP tools            test a single tool"
  echo "  7) MCP server           start (attach to an MCP host)"
  echo "  8) Self-test            verify this install"
  echo "  9) Memory               where memory lives + how to keep it"
  echo "  0) Exit"
  echo "----------------------------------------------------------"
  printf 'Choose [0-9]: '
  read -r choice
  case "$choice" in
    1) "${EDITOR:-nano}" .env ;;
    2) printf '\nWhat do you want built? '; read -r t; "$VENV" ask.py --task "$t"; read -r -p "[enter] back to menu" _ ;;
    3) printf '\nWhat is the overall task? '; read -r t; "$VENV" run_agent.py --task "$t"; read -r -p "[enter] back to menu" _ ;;
    4) echo "Starting dashboard at http://localhost:8660 — Ctrl-C to stop."; "$VENV" web_dashboard.py ;;
    5) "$VENV" mcp_server.py --list; echo; read -r -p "[enter] back to menu" _ ;;
    6) printf 'Tool name: '; read -r tool
       printf 'JSON params (default {}): '; read -r args
       ./ask_mcp.sh "$tool" "${args:-{}}"; echo; read -r -p "[enter] back to menu" _ ;;
    7) echo "Starting MCP server over stdio — Ctrl-C to stop."
       echo "Dangerous tools need HERMES_SAFE=1 or HERMES_ALLOW=... ."
       "$VENV" mcp_server.py ;;
    8) "$VENV" test_hybrid.py; read -r -p "[enter] back to menu" _ ;;
    9) echo "Memory lives here : $(pwd)/memory"
       echo "Delete this folder removes memory too. To keep it for a new install:"
       echo "  cp -R \"$(pwd)/memory\" \"/path/to/Hermes-<newdate>/memory\""
       read -r -p "[enter] back to menu" _ ;;
    0|q|Q) echo "bye"; break ;;
    *) echo "  Invalid choice: $choice"; sleep 1 ;;
  esac
done
