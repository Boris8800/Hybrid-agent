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

# --- 4c. Bundle hybrid + supervised chat (embedded, self-contained) ---
# hybrid.sh:          one-shot task -> online plan / local work / online gate
# supervised_chat.sh: a continuous chat where EVERY message is supervised
# Both use the real Hermes Agent: ONLINE supervisor = Hermes default model;
# LOCAL worker = your local model (hermes -m). No separate API keys.
cat > "$HERMES_HOME/hybrid.sh" <<'HYBRID_SRC'
#!/bin/bash
# ============================================================
# hybrid.sh — ONLINE SUPERVISOR (Hermes default) · LOCAL WORKER
# ============================================================
# Uses the REAL Hermes Agent for BOTH roles, so:
#   - The ONLINE SUPERVISOR = whichever model you choose in Hermes (its default).
#     Pick it with:  hermes model     (set an ONLINE model as default)
#   - The LOCAL WORKER    = your local model, run via `hermes -m <local>`.
#     Set it once:  HYBRID_LOCAL_MODEL=<your-local-model-name>  in the folder's .env
#
# Flow per task:
#   1) ONLINE (Hermes default) writes a PLAN.
#   2) LOCAL worker (hermes -m <local>) EXECUTES the plan — real edits/commands/tools.
#   3) ONLINE inspects the result; asks for fixes if needed (loops).
#   4) ONLINE is the LAST GATE: final APPROVED / REJECTED.
#
# No separate API keys: Hermes already holds the model credentials. Any model you
# can pick in Hermes — free or paid — can be the supervisor.
#
# Usage:
#   bash hybrid.sh "build a fibonacci module with tests"
#   echo "task" | bash hybrid.sh
#   bash hybrid.sh --file task.txt
# ============================================================
set -u
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'

# newest dated one-folder install
D="$(ls -d "$HOME"/Desktop/Hermes-* 2>/dev/null | sort -r | head -1)"
HERMES_BIN="${HERMES_BIN:-}"
if [ -z "$HERMES_BIN" ]; then
    if [ -n "$D" ] && [ -x "$D/hermes-agent/.venv/bin/hermes" ]; then
        HERMES_BIN="$D/hermes-agent/.venv/bin/hermes"
    else
        HERMES_BIN="$HOME/.hermes/hermes-agent/.venv/bin/hermes"
    fi
fi
ENVF="${HYBRID_ENV:-}"
[ -z "$ENVF" ] && [ -n "$D" ] && [ -f "$D/.env" ] && ENVF="$D/.env"
[ -z "$ENVF" ] && [ -f "$HOME/.hermes/.env" ] && ENVF="$HOME/.hermes/.env"

# local worker model (required unless Hermes default already IS your local model)
LOCAL="${HYBRID_LOCAL_MODEL:-}"
if [ -z "$LOCAL" ] && [ -n "$ENVF" ]; then
    LOCAL="$(grep -E '^HYBRID_LOCAL_MODEL=' "$ENVF" | head -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi
# online SUPERVISOR model — blank = Hermes' default (what you pick in `hermes model`)
SUP="${HYBRID_SUPERVISOR_MODEL:-}"
if [ -z "$SUP" ] && [ -n "$ENVF" ]; then
    SUP="$(grep -E '^HYBRID_SUPERVISOR_MODEL=' "$ENVF" | head -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi

WORK="${HYBRID_WORK:-$HOME/.hermes/hybrid_work}"
mkdir -p "$WORK"
MAX_ROUNDS="${HYBRID_MAX_ROUNDS:-3}"

if [ ! -x "$HERMES_BIN" ]; then
    echo -e "${RED}Hermes Agent not found at $HERMES_BIN. Run install.sh first.$NC" >&2; exit 2
fi
if [ -z "$LOCAL" ]; then
    echo -e "${YELLOW}No HYBRID_LOCAL_MODEL set. The worker will run on Hermes' default model.${NC}" >&2
    echo "  If your default is the ONLINE supervisor, the worker would not be local." >&2
    echo "  Set the local model in $ENVF :  HYBRID_LOCAL_MODEL=<local-model-name>" >&2
fi

# sup() runs the ONLINE supervisor (explicit model if set, else Hermes default)
sup(){ if [ -n "$SUP" ]; then "$HERMES_BIN" -m "$SUP" -z "$1"; else "$HERMES_BIN" -z "$1"; fi; }

# task
TASK=""
if [ "${1:-}" = "--file" ]; then TASK="$(cat "${2:?need file}")"
elif [ -n "${1:-}" ]; then TASK="$1"
else TASK="$(cat)"
fi
[ -n "$TASK" ] || { echo "no task given" >&2; exit 2; }

# ---------- 1) ONLINE plans ----------
echo -e "${BLUE}== ONLINE supervisor (Hermes default) is planning... ==$NC"
PLAN_PROMPT="You are the planning engine in a hybrid system. A LOCAL worker model (which
has real file/terminal/tool access) will execute your plan; you never do the work.
Task: $TASK

Return ONLY the plan: numbered, concrete, actionable steps, plus any risks. No
preamble, no markdown fences."
if ! sup "$PLAN_PROMPT" > "$WORK/plan.txt" 2>&1; then
    echo -e "${RED}online plan failed$NC"; exit 1
fi
echo -e "${GREEN}Plan:${NC}"; sed 's/^/    /' "$WORK/plan.txt"

# ---------- 2+3) LOCAL works, ONLINE inspects ----------
VERDICT="FIX"; FEED=""
for round in $(seq 1 "$MAX_ROUNDS"); do
    echo -e "${BLUE}== round $round: LOCAL worker is doing the work... ==$NC"
    EXEC_PROMPT="TASK:
$TASK

PLAN (follow it):
$(cat "$WORK/plan.txt")
$([ -n "$FEED" ] && printf '\nSUPERVISOR FEEDBACK FROM LAST ROUND — address these now:\n%s' "$FEED")

Do the work now using your tools. When done, give a concise summary of what you did."
    printf '%s' "$EXEC_PROMPT" > "$WORK/exec_prompt.txt"
    PROMPT_ARG="$(cat "$WORK/exec_prompt.txt")"
    if [ -n "$LOCAL" ]; then
        "$HERMES_BIN" -m "$LOCAL" -z "$PROMPT_ARG" > "$WORK/run_out.txt" 2>&1 || true
    else
        "$HERMES_BIN" -z "$PROMPT_ARG" > "$WORK/run_out.txt" 2>&1 || true
    fi
    { echo "### LOCAL TRANSCRIPT ###"; tail -n 300 "$WORK/run_out.txt"; echo; echo "### FILES ###"; ls -la "$(pwd)"; } > "$WORK/evidence.txt"

    echo -e "${BLUE}== ONLINE supervisor is inspecting round $round... ==$NC"
    INSPECT_PROMPT="You are the inspector in a hybrid system. The LOCAL worker executed the
plan. Review the evidence below against the task and plan. Decide if the work is done
correctly and completely.

TASK:
$TASK

PLAN:
$(cat "$WORK/plan.txt")

EVIDENCE:
$(cat "$WORK/evidence.txt")

Return EXACTLY one line starting with VERDICT: APPROVED or VERDICT: FIX, then concise
FEEDBACK lines for what to fix next."
    sup "$INSPECT_PROMPT" > "$WORK/inspect.txt" 2>&1
    echo "    $(head -1 "$WORK/inspect.txt")"
    if head -1 "$WORK/inspect.txt" | grep -q "APPROVED"; then
        echo -e "${GREEN}  round $round APPROVED$NC"; break
    fi
    FEED="$(tail -n +2 "$WORK/inspect.txt" | head -n 40)"
    echo -e "${YELLOW}  round $round needs fixes; looping.$NC"
done

# ---------- 4) ONLINE is the LAST GATE ----------
echo -e "${BLUE}== ONLINE supervisor is the LAST GATE... ==$NC"
GATE_PROMPT="You are the LAST GATE in a hybrid system. The local worker finished and the
inspector approved it. Make the FINAL call using task, plan, and final transcript.

TASK:
$TASK

PLAN:
$(cat "$WORK/plan.txt")

FINAL TRANSCRIPT:
$(tail -n 200 "$WORK/run_out.txt")

Return EXACTLY one line: FINAL: APPROVED or FINAL: REJECTED, then a one-line rationale."
sup "$GATE_PROMPT" > "$WORK/gate.txt" 2>&1
FINAL="$(head -1 "$WORK/gate.txt")"
echo ""
if echo "$FINAL" | grep -q "APPROVED"; then
    echo -e "${GREEN}== FINAL: APPROVED — task complete. ==$NC"
else
    echo -e "${RED}== $FINAL ==$NC"
    echo -e "${RED}== FINAL: REJECTED — inspect artifacts below. ==$NC"
fi
tail -n +2 "$WORK/gate.txt" | sed 's/^/  /'
echo "  artifacts in: $WORK/"
exit 0
HYBRID_SRC
chmod +x "$HERMES_HOME/hybrid.sh"
ok "Bundled: hybrid.sh (embedded)"
cat > "$HERMES_HOME/supervised_chat.sh" <<'SUPERVISED_SRC'
#!/bin/bash
# ============================================================
# supervised_chat.sh — in-chat supervision
# ============================================================
# A continuous chat where EVERY message you send is supervised:
#   1) ONLINE supervisor (Hermes default model) approves/plans the turn
#   2) LOCAL worker (your local model, `hermes -m <local>`) does the real work
#   3) ONLINE supervisor GATES it before you see the answer (fix loops if needed)
#
# Same setup as hybrid.sh:
#   - Hermes default model  = ONLINE supervisor (choose in `hermes model`)
#   - HYBRID_LOCAL_MODEL     = your local worker model (set in the folder .env)
#
# Type normally.  Type:  exit / quit  to leave.  Save: CTRL-C
# A running conversation transcript is kept in ~/.hermes/hybrid_work/.
# ============================================================
set -u
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'

D="$(ls -d "$HOME"/Desktop/Hermes-* 2>/dev/null | sort -r | head -1)"
HERMES_BIN="${HERMES_BIN:-}"
if [ -z "$HERMES_BIN" ]; then
    if [ -n "$D" ] && [ -x "$D/hermes-agent/.venv/bin/hermes" ]; then
        HERMES_BIN="$D/hermes-agent/.venv/bin/hermes"
    else
        HERMES_BIN="$HOME/.hermes/hermes-agent/.venv/bin/hermes"
    fi
fi
ENVF="${HYBRID_ENV:-}"
[ -z "$ENVF" ] && [ -n "$D" ] && [ -f "$D/.env" ] && ENVF="$D/.env"
[ -z "$ENVF" ] && [ -f "$HOME/.hermes/.env" ] && ENVF="$HOME/.hermes/.env"

LOCAL="${HYBRID_LOCAL_MODEL:-}"
if [ -z "$LOCAL" ] && [ -n "$ENVF" ]; then
    LOCAL="$(grep -E '^HYBRID_LOCAL_MODEL=' "$ENVF" | head -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi
MAX_ROUNDS="${HYBRID_MAX_ROUNDS:-3}"
WORK="${HYBRID_WORK:-$HOME/.hermes/hybrid_work}"
mkdir -p "$WORK"
TRANS="$WORK/chat_transcript.txt"
: > "$TRANS"

if [ ! -x "$HERMES_BIN" ]; then
    echo -e "${RED}Hermes Agent not found at $HERMES_BIN.$NC" >&2; exit 2
fi
if [ -z "$LOCAL" ]; then
    echo -e "${YELLOW}WARNING: HYBRID_LOCAL_MODEL not set — the worker will use Hermes' default (not necessarily local).$NC" >&2
fi

ctx(){ # last ~60 lines of transcript
    tail -n 60 "$TRANS"
}

echo -e "${GREEN}SUPERVISED CHAT${NC} (supervisor = Hermes default model)"
echo -e "  worker        = ${LOCAL:-Hermes default}"
echo -e "  type a message, or 'exit' to quit"
echo "======================================================"

while :; do
    printf '\nyou> '
    IFS= read -r msg || break
    [ -z "$msg" ] && continue
    case "$msg" in
        exit|quit|q) echo -e "${GREEN}bye${NC}"; break ;;
    esac
    echo "user: $msg" >> "$TRANS"

    # --- supervisor plans/approves this turn ---
    echo -e "${BLUE}  supervisor: planning this turn...${NC}"
    PLAN_PROMPT="You are the online SUPERVISOR in a supervised chat. A LOCAL worker model
(which has real tools) does the work; you only decide/approve.

Conversation so far:
$(ctx)

The user just said:
$msg

Give the LOCAL worker ONE concise, actionable instruction that satisfies the user's
latest message (continue prior work if this is a follow-up). Return only that
instruction, no preamble."
    "$HERMES_BIN" -z "$PLAN_PROMPT" > "$WORK/turn_plan.txt" 2>&1
    INST="$(cat "$WORK/turn_plan.txt")"

    # --- local worker does the work, supervisor gates ---
    FEED=""
    ANSWER=""
    for round in $(seq 1 "$MAX_ROUNDS"); do
        echo -e "${BLUE}  worker: doing the work (round $round)...${NC}"
        WP="Conversation so far:
$(ctx)

Approved instruction for THIS turn:
$INST
$([ -n "$FEED" ] && printf '\nSupervisor feedback to address:\n%s' "$FEED")

Act on the instruction now using your tools. Reply concisely as the assistant."
        printf '%s' "$WP" > "$WORK/worker_prompt.txt"
        if [ -n "$LOCAL" ]; then
            "$HERMES_BIN" -m "$LOCAL" -z "$(cat "$WORK/worker_prompt.txt")" > "$WORK/turn_work.txt" 2>&1 || true
        else
            "$HERMES_BIN" -z "$(cat "$WORK/worker_prompt.txt")" > "$WORK/turn_work.txt" 2>&1 || true
        fi

        echo -e "${BLUE}  supervisor: gating this turn...${NC}"
        GP="You are the GATE in a supervised chat. Did the local worker properly satisfy
the approved instruction, in the context of the conversation?

Instruction:
$INST

Worker result:
$(tail -n 200 "$WORK/turn_work.txt")

If OK: reply with EXACTLY the assistant's final answer to the user (concise, natural).
If NOT OK: reply starting with GATE_FIX: then specific feedback for the worker."
        "$HERMES_BIN" -z "$GP" > "$WORK/turn_gate.txt" 2>&1
        GATE="$(head -1 "$WORK/turn_gate.txt")"
        if echo "$GATE" | grep -q "^GATE_FIX:"; then
            FEED="$(tail -n +2 "$WORK/turn_gate.txt" | head -n 30)"
            echo -e "${YELLOW}  gate wants a fix (round $round)$NC"
        else
            ANSWER="$(cat "$WORK/turn_gate.txt")"
            break
        fi
    done
    [ -z "$ANSWER" ] && ANSWER="$(tail -n 100 "$WORK/turn_work.txt")"

    echo "assistant: $ANSWER" >> "$TRANS"
    echo -e "\n${GREEN}Hermes>${NC} $ANSWER"
done
exit 0
SUPERVISED_SRC
chmod +x "$HERMES_HOME/supervised_chat.sh"
ok "Bundled: supervised_chat.sh (embedded)"
cat > "$HERMES_HOME/hybrid_config.sh" <<'HYBRID_CONFIG'
#!/bin/bash
# ============================================================
# hybrid_config.sh — configure HYBRID mode (like Hermes' MoA configure)
# ============================================================
# Choose:
#   1) the ONLINE SUPERVISOR model (plans / inspects / last gate)
#   2) the LOCAL WORKER model      (does the real work)
# Saves them to the Hermes folder's .env, used by hybrid.sh / supervised_chat.sh.
# Leave a prompt empty to keep the current value; enter "-" to clear to default.
# ============================================================
set -u
BLUE=$'\033[0;34m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'

D="$(ls -d "$HOME"/Desktop/Hermes-* 2>/dev/null | sort -r | head -1)"
if [ -n "${HERMES_HOME:-}" ]; then ENVF="${HERMES_HOME:-}/.env"
elif [ -n "$D" ]; then ENVF="$D/.env"
else ENVF="$HOME/.hermes/.env"; fi
touch "$ENVF" 2>/dev/null || true

readv(){ # $1=key
    grep -E "^$1=" "$ENVF" 2>/dev/null | head -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}
setkey(){ # $1=key $2=value  ("-" => remove)
    if [ "$2" = "-" ] || [ -z "$2" ]; then
        grep -v "^$1=" "$ENVF" > "$ENVF.tmp" 2>/dev/null || true
        [ -f "$ENVF.tmp" ] && mv "$ENVF.tmp" "$ENVF"
    elif grep -q "^$1=" "$ENVF" 2>/dev/null; then
        sed -i '' "s|^$1=.*|$1=$2|" "$ENVF"
    else
        printf '%s=%s\n' "$1" "$2" >> "$ENVF"
    fi
}

echo -e "${BLUE}Configure HYBRID mode${NC}   (folder: ${ENVF})"
echo "======================================================"
echo "  ONLINE supervisor = plans / inspects / is the LAST gate."
echo "  LOCAL worker      = does the real work (your local model)."
echo "  Leave blank = keep current.  Enter '-' = clear to Hermes default."
echo ""
cur_sup="$(readv HYBRID_SUPERVISOR_MODEL)"; cur_loc="$(readv HYBRID_LOCAL_MODEL)"
echo "  current SUPERVISOR = ${cur_sup:-<Hermes default>}"
echo "  current WORKER     = ${cur_loc:-<Hermes default>}"
echo ""
printf 'Online SUPERVISOR model [%s]: ' "${cur_sup:-<default>}"; read -r sup
printf 'Local WORKER model     [%s]: ' "${cur_loc:-<default>}"; read -r loc
[ -n "$sup" ] && setkey HYBRID_SUPERVISOR_MODEL "$sup"
[ -n "$loc" ] && setkey HYBRID_LOCAL_MODEL "$loc"

echo ""
echo -e "${GREEN}Saved to ${ENVF}:${NC}"
grep -E '^(HYBRID_SUPERVISOR_MODEL|HYBRID_LOCAL_MODEL)=' "$ENVF" || echo "  (both on Hermes default)"
echo ""
echo "Run:"
echo "  bash $D/hybrid.sh \"<your task>\"        # one-shot hybrid"
echo "  bash $D/supervised_chat.sh              # supervised chat"
exit 0
HYBRID_CONFIG
chmod +x "$HERMES_HOME/hybrid_config.sh"
ok "Bundled: hybrid_config.sh (embedded)"

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
  echo "  5) SUPERVISED chat      (every message is online-supervised)"
  echo "  6) Configure hybrid      (choose supervisor + local worker)"
  echo "  0) Quit"
  printf 'Choose [0-6]: '
  read -r c
  case "$c" in
    1) hermes dashboard ;;
    2) hermes chat ;;
    3) hermes model ;;
    4) printf 'Task: '; read -r t; [ -n "$t" ] && bash "$HERMES_HOME/hybrid.sh" "$t"; printf '[enter] back to menu'; read -r _ ;;
    5) bash "$HERMES_HOME/supervised_chat.sh" ;;
    6) bash "$HERMES_HOME/hybrid_config.sh" ;;
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
