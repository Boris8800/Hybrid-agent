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
