#!/bin/bash
# ============================================================
# hybrid.sh — ONLINE PLANS · LOCAL DOES THE WORK · ONLINE GATES
# ============================================================
# Wraps the real Hermes Agent so that:
#   1) an ONLINE AI (DeepSeek by default) writes a plan for the task,
#   2) the LOCAL model (Hermes Agent's configured default) does ALL the real
#      work — edits, commands, tool calls — right on your machine,
#   3) the ONLINE AI inspects each run and can ask for fixes,
#   4) the ONLINE AI is the LAST GATE: final APPROVED/REJECTED verdict.
#
# Works with ANY local model: just point Hermes at it first via `hermes model`
# (LM Studio, Ollama, whatever). Online is any OpenAI-compatible API.
#
# Env overrides:
#   HYBRID_ONLINE_BASE  default https://api.deepseek.com/v1
#   HYBRID_ONLINE_MODEL default deepseek-chat
#   HYBRID_ONLINE_KEY   default reads DEEPSEEK_API_KEY from ~/.hermes/.env
#   HYBRID_MAX_ROUNDS   default 3   (plan->run->inspect fix rounds)
#
# Usage:
#   bash hybrid.sh "build a fibonacci module with tests"
#   echo "build ..." | bash hybrid.sh
#   bash hybrid.sh --file task.txt
# ============================================================

set -u
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'

HERMES_BIN="${HERMES_BIN:-$HOME/.hermes/hermes-agent/.venv/bin/hermes}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/hybrid_ai.py"
WORK="${HYBRID_WORK:-$HOME/.hermes/hybrid_work}"
mkdir -p "$WORK"

# ---------- resolve online (DeepSeek) ----------
ONLINE_BASE="${HYBRID_ONLINE_BASE:-https://api.deepseek.com/v1}"
ONLINE_MODEL="${HYBRID_ONLINE_MODEL:-deepseek-chat}"
ONLINE_KEY="${HYBRID_ONLINE_KEY:-}"
if [ -z "$ONLINE_KEY" ] && [ -f "$HOME/.hermes/.env" ]; then
    ONLINE_KEY="$(grep -E '^DEEPSEEK_API_KEY=' "$HOME/.hermes/.env" | head -1 | cut -d= -f2-)"
    # trim surrounding whitespace and quotes, and a possible CR
    ONLINE_KEY="${ONLINE_KEY%$'\r'}"
    ONLINE_KEY="${ONLINE_KEY#\"}"; ONLINE_KEY="${ONLINE_KEY%\"}"
    ONLINE_KEY="${ONLINE_KEY#\'}"; ONLINE_KEY="${ONLINE_KEY%\'}"
    ONLINE_KEY="$(printf '%s' "$ONLINE_KEY" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi
if [ -z "$ONLINE_KEY" ]; then
    echo -e "${RED}No online key. Set DEEPSEEK_API_KEY in ~/.hermes/.env or HYBRID_ONLINE_KEY.$NC" >&2
    exit 2
fi
if [ ! -x "$HERMES_BIN" ]; then
    echo -e "${RED}Hermes Agent not found at $HERMES_BIN. Run install.sh first.$NC" >&2
    exit 2
fi

# ---------- task ----------
TASK=""
if [ "${1:-}" = "--file" ]; then TASK="$(cat "${2:?need file}")"
elif [ -n "${1:-}" ]; then TASK="$1"
else TASK="$(cat)"
fi
[ -n "$TASK" ] || { echo "no task given" >&2; exit 2; }

MAX_ROUNDS="${HYBRID_MAX_ROUNDS:-3}"

# ---------- prompts ----------
cat > "$WORK/system_plan.txt" <<'PLAN'
You are the planning engine of a hybrid system. You NEVER do the work yourself.
You produce a plan that the LOCAL model (which has real file/terminal/tool access)
will execute. Return ONLY the plan: numbered, concrete, actionable steps. Note any
risks. No preamble, no markdown fences.
PLAN

cat > "$WORK/system_inspect.txt" <<'INSP'
You are the inspector in a hybrid system. The LOCAL model executed the plan. Review
the evidence (tool transcript) against the task and plan. Decide if the work is done
correctly and completely. Return EXACTLY one line that begins with VERDICT: APPROVED
or VERDICT: FIX, then on the following lines concise FEEDBACK for what to fix next
(the local model will act on it). Be specific and actionable.
INSP

cat > "$WORK/system_gate.txt" <<'GATE'
You are the LAST GATE in a hybrid system. The local model finished and an inspector
already approved the work. Now make the FINAL call on the overall result using the
task, the plan, and the final output. Return EXACTLY one line: FINAL: APPROVED or
FINAL: REJECTED, then a one-line rationale.
GATE

cat > "$WORK/user_plan.txt" <<EOF
Task for the local model to execute:
$TASK
EOF

# ---------- 1) ONLINE plans ----------
echo -e "${BLUE}== [1/4] ONLINE AI is planning... ==$NC"
if ! "$PY" "$ONLINE_BASE" "$ONLINE_KEY" "$ONLINE_MODEL" "$WORK/system_plan.txt" "$WORK/user_plan.txt" > "$WORK/plan.txt"; then
    echo -e "${RED}planning failed$NC"; exit 1
fi
echo -e "${GREEN}Plan:${NC}"; sed 's/^/    /' "$WORK/plan.txt"

# ---------- 2+3) LOCAL does work, ONLINE inspects ----------
VERDICT="FIX"; FEED=""
for round in $(seq 1 "$MAX_ROUNDS"); do
    echo -e "${BLUE}== [$((round+1))/4] round $round: LOCAL model is doing the work... ==$NC"
    {
        echo "TASK:"
        echo "$TASK"
        echo
        echo "PLAN (follow it):"
        cat "$WORK/plan.txt"
        if [ -n "$FEED" ]; then
            echo
            echo "INSPECTOR FEEDBACK FROM THE LAST ROUND — address these now:"
            echo "$FEED"
        fi
        echo
        echo "Do the work now using your tools. When done, give a concise summary."
    } > "$WORK/user_run.txt"

    # run Hermes with the LOCAL default model; capture a transcript of what it did
    if ! "$HERMES_BIN" -z "$(cat "$WORK/user_run.txt")" > "$WORK/run_out.txt" 2>&1; then
        echo -e "${YELLOW}  local run exited non-zero (continuing to inspection anyway)$NC"
    fi
    # prefer a fuller evidence file: transcript + files we can list
    {
        echo "### LOCAL TRANSCRIPT (what the local model did / said) ###"
        tail -n 300 "$WORK/run_out.txt"
        echo
        echo "### WORKING DIRECTORY FILES ###"
        ls -la "$(pwd)" 2>/dev/null
    } > "$WORK/evidence.txt"

    echo -e "${BLUE}== inspecting round $round with ONLINE AI... ==$NC"
    {
        echo "TASK:"
        echo "$TASK"
        echo
        echo "PLAN:"
        cat "$WORK/plan.txt"
        echo
        echo "EVIDENCE:"
        cat "$WORK/evidence.txt"
    } > "$WORK/user_inspect.txt"

    if ! "$PY" "$ONLINE_BASE" "$ONLINE_KEY" "$ONLINE_MODEL" "$WORK/system_inspect.txt" "$WORK/user_inspect.txt" > "$WORK/inspect.txt"; then
        echo -e "${RED}inspection call failed$NC"; break
    fi
    echo "    $(head -1 "$WORK/inspect.txt")"
    VERDICT="$(head -1 "$WORK/inspect.txt")"
    if printf '%s' "$VERDICT" | grep -q "APPROVED"; then
        echo -e "${GREEN}  round $round APPROVED$NC"
        break
    fi
    FEED="$(tail -n +2 "$WORK/inspect.txt" | head -n 40 | tr '\n' '\n')"
    echo -e "${YELLOW}  round $round needs fixes; looping.$NC"
done

# ---------- 4) ONLINE is the LAST GATE ----------
echo -e "${BLUE}== [4/4] ONLINE AI is the LAST GATE... ==$NC"
{
    echo "TASK:"; echo "$TASK"; echo
    echo "PLAN:"; cat "$WORK/plan.txt"; echo
    echo "FINAL OUTPUT / TRANSCRIPT:"; tail -n 200 "$WORK/run_out.txt"
} > "$WORK/user_gate.txt"
if ! "$PY" "$ONLINE_BASE" "$ONLINE_KEY" "$ONLINE_MODEL" "$WORK/system_gate.txt" "$WORK/user_gate.txt" > "$WORK/gate.txt"; then
    echo -e "${RED}final gate call failed$NC"; exit 1
fi
FINAL="$(head -1 "$WORK/gate.txt")"
echo
if printf '%s' "$FINAL" | grep -q "APPROVED"; then
    echo -e "${GREEN}== FINAL: APPROVED — task complete. ==$NC"
else
    echo -e "${RED}== $FINAL ==$NC"
    echo -e "${RED}== FINAL: REJECTED — inspect the artifacts below. ==$NC"
fi
echo "  artifacts in: $WORK/"
tail -n +2 "$WORK/gate.txt" | sed 's/^/  /'
exit 0
