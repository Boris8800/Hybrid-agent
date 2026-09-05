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
if ! "$HERMES_BIN" -z "$PLAN_PROMPT" > "$WORK/plan.txt" 2>&1; then
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
    "$HERMES_BIN" -z "$INSPECT_PROMPT" > "$WORK/inspect.txt" 2>&1
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
"$HERMES_BIN" -z "$GATE_PROMPT" > "$WORK/gate.txt" 2>&1
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
