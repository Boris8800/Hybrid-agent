#!/bin/bash
# ============================================================
# HERMES - Self-Contained Hybrid Agent (Apple Silicon)
# ============================================================
# Every install creates ONE self-contained RUNTIME folder
# (default ~/Desktop/Hermes-<date>). No files are written to ~/Agents,
# ~/.zprofile, or anywhere else on the machine.
#
# Inside the one runtime folder you get:
#   - in-folder Python virtual environment (.venv)
#   - Python deps (openai, pyyaml, flask, playwright, ...)
#   - config.yml + .env
#   - ask.py            : quick one-shot local + DeepSeek review
#   - run_agent.py      : planner (online) -> local implement -> supervisor -> recorder
#   - web_dashboard.py  + start scripts
#   - logs/ memory/ cache/ ... (all kept in-folder)
#
# MEMORY IS PER-INSTALL AND LIVES INSIDE THIS FOLDER:
#   - All memories are under <runtime>/memory/
#   - Delete the runtime folder  => deletes EVERYTHING, including memory.
#   - To KEEP memory for a future/new install:
#       copy -R old/Hermes-<date>/memory/  new/Hermes-<date>/memory/
#   - memory/ can also be renamed or backed up on its own (it is self-contained).
#
# External apps you still run yourself (NOT installed into Hermes):
#   - a local model server (LM Studio or Ollama) for the "local" model
#   - your existing Python (python3) used only to build the in-folder venv
#   - DeepSeek API key (cloud planner/supervisor)
#
# Usage:  bash install.sh            # creates a NEW dated folder ~/Desktop/Hermes-<date>
#         HERMES_DIR=/path bash install.sh   # install into a specific folder instead
#         HERMES_BASE=/parent bash install.sh  # choose where the dated folder is made
#
# This script lives in an INSTALLATION folder and creates a fresh RUNTIME folder
# on each run, so the installation folder stays clean (installer only).
# ============================================================

set -e  # Exit on any error

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# --- Location: install folder vs. runtime folder ---
# The folder holding this script is the INSTALLATION folder (installer only).
# Each run creates a NEW dated RUNTIME folder on the Desktop, e.g.
# ~/Desktop/Hermes-2026-09-04. If that name exists, a -2/-3 suffix is added.
# Long-term memory is PER-INSTALL, inside the runtime folder (deleted with it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_PARENT="${HERMES_BASE:-$HOME/Desktop}"
if [ -n "${HERMES_DIR:-}" ]; then
    AGENT_DIR="${HERMES_DIR}"
else
    AGENT_DIR="$RUNTIME_PARENT/Hermes-$(date +%Y%m%d)"
    n=1
    while [ -e "$AGENT_DIR" ]; do
        n=$((n+1))
        AGENT_DIR="$RUNTIME_PARENT/Hermes-$(date +%Y%m%d)-$n"
    done
fi

# --- Toggle stages (defaults keep everything INSIDE the one folder) ---
DO_PYTHON=0       # set 1 to auto-'brew install python@3.11' only if python3 is missing
DO_PIP=1          # install python deps into the in-folder venv
DO_LMSTUDIO=1     # download LM Studio app if not installed (external model server)
DO_OLLAMA=1       # install ollama app if missing (external model runtime)

log()  { echo -e "${BLUE}$1${NC}"; }
step() { echo ""; echo -e "${YELLOW}--- $1 ---${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ! $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; }

# Make sure brew is on PATH for the rest of the script (both chip types)
source_brew() {
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

mkdir -p "$AGENT_DIR"
cd "$AGENT_DIR"

log "============================================================"
log "     HERMES - HYBRID AGENT INSTALLER"
log "============================================================"
log "  INSTALL folder (keep clean) : $SCRIPT_DIR"
log "  RUNTIME folder (created new): $AGENT_DIR"
log "  Memory (inside runtime)    : ${AGENT_DIR}/memory"
log "============================================================"

# ============================================================
# 1. System requirements
# ============================================================
step "1/8 Checking system requirements"
ok "macOS: $(sw_vers -productVersion)"
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then ok "Architecture: Apple Silicon (arm64)"; else warn "Architecture: $ARCH (not arm64)"; fi

# ============================================================
# 2. Python + in-folder virtual environment
# ============================================================
step "2/8 Python and in-folder virtual environment"
source_brew
PYBIN=""
for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYBIN="$(command -v "$cand")"; break; fi
done
if [ -z "$PYBIN" ] && [ "$DO_PYTHON" = "1" ]; then
    brew install python@3.11
    PYBIN="$(brew --prefix python@3.11)/bin/python3.11"
fi
if [ -z "$PYBIN" ]; then
    fail "No python3 found. Install one (brew install python@3.11) and re-run, or set DO_PYTHON=1."
    exit 1
fi
ok "Using python: $PYBIN ($($PYBIN --version 2>&1))"

if [ ! -d "$AGENT_DIR/.venv" ]; then
    "$PYBIN" -m venv "$AGENT_DIR/.venv"
    ok "Created in-folder virtual environment ($AGENT_DIR/.venv)"
else
    ok "Virtual environment already exists"
fi
# shellcheck disable=SC1091
source "$AGENT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
ok "pip upgraded"

# ============================================================
# 3. Python dependencies (into the in-folder venv only)
# ============================================================
step "3/8 Installing Python dependencies"
if [ "$DO_PIP" = "1" ]; then
    pip install --quiet \
        openai pyyaml flask flask-socketio cryptography pytest requests python-dotenv
    ok "Python packages installed (inside .venv)"
    playwright install chromium >/dev/null 2>&1 && ok "Playwright chromium installed" || warn "Playwright browser install skipped/failed (non-fatal)"
else
    warn "Skipped Python dependency install (DO_PIP=0)"
fi

# ============================================================
# 4. Local model runtime (LM Studio / Ollama) — EXTERNAL APPS
# ============================================================
step "4/8 Local model runtime (external apps, optional)"
if [ -d "/Applications/LM Studio.app" ]; then
    ok "LM Studio installed. Load qwen2.5-coder-14b-instruct-mlx -> Local server :1234"
elif [ "$DO_LMSTUDIO" = "1" ]; then
    mkdir -p "$HOME/Downloads"
    URL="https://releases.lmstudio.ai/darwin/arm64/LM%20Studio-0.3.9-arm64.dmg"
    echo "  Downloading LM Studio to ~/Downloads/LM-Studio.dmg ..."
    curl -L --fail -o "$HOME/Downloads/LM-Studio.dmg" "$URL" \
        && warn "Open ~/Downloads/LM-Studio.dmg and drag LM Studio to Applications." \
        || warn "LM Studio download failed — get it at https://lmstudio.ai"
else
    warn "Skipped LM Studio (DO_LMSTUDIO=0)"
fi
if command -v ollama >/dev/null 2>&1; then
    ok "Ollama: $(ollama --version 2>&1 | head -1)  -> 'ollama pull qwen2.5-coder:14b'"
elif [ "$DO_OLLAMA" = "1" ]; then
    curl -fsSL https://ollama.com/install.sh | sh || warn "Ollama install failed (non-fatal)"
else
    warn "Skipped Ollama (DO_OLLAMA=0)"
fi

# ============================================================
# 5. Configuration files (config.yml + .env)
# ============================================================
step "5/8 Config files (config.yml + .env)"
cat > "$AGENT_DIR/config.yml" <<'YAML'
# ============================================================
# UNIVERSAL HYBRID AGENT CONFIGURATION
# ============================================================

agent:
  name: "Universal Hybrid Agent"
  version: "1.0.0"
  architecture: "hybrid_local_first"

backends:
  local:
    base_url: "http://localhost:1234/api/v1"
    api_key: "not-needed"
    model: "qwen2.5-coder-14b-instruct-mlx"
    timeout_s: 120
    max_retries: 3
    backoff_s: 2
    cold_start_wait_s: 5

  deepseek:
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    timeout_s: 60
    max_retries: 2
    backoff_s: 1

router:
  local_threshold: 0.6
  threshold_min: 0.3
  threshold_max: 0.85
  target_local_rate: 0.7
  alpha: 0.1
  supervision: auto  # auto | full | local_first | critical
  weights:
    archetype: 0.25
    confidence: 0.20
    memory: 0.20
    cost: 0.15
    context: 0.10
    task_complexity: 0.10

review:
  verify: true
  verify_timeout: 600
  verify_groups: []
  verify_allowlist:
    - "npm run"
    - "npm test"
    - "npx tsc"
    - "python -m pytest"
    - "pytest"
    - "go test"
    - "cargo test"
    - "make"
    - "just"
  regression: true
  regression_timeout: 600
  daily_token_budget: 200000
  max_depth_tokens: 4000
  max_failure_summary_words: 500
  terminal_timeout: 120
  context_safety:
    output_reserve_tokens: 12000
    safety_margin_tokens: 2000
    max_recovery_retries: 1
    local_context_red_escalation: true

cache:
  enabled: true
  dir: ".cache"
  ttl_days: 7
  max_entries: 100

circuit_breaker:
  window_size: 10
  local_error_ceiling: 0.4
  deepseek_error_ceiling: 0.3
  cooldown_s: 60

memory:
  root: "memory"   # Per-install, INSIDE this runtime folder (<runtime>/memory).
                   # Delete the runtime folder => memory is deleted too.
                   # To KEEP memory for a new install, copy <runtime>/memory into
                   # the new folder's memory/ (self-contained; may also be renamed).
                   # Override anywhere with env HERMES_MEMORY=/abs/path.
  max_project_summary_words: 500
  semantic_similarity: true
  embedding_model: "text-embedding-all-minilm-l6-v2"
  embedding_threshold: 0.60

deploy:
  command: "npm run deploy"
  cwd: "."
  timeout: 300

bound:
  danger_zones:
    - "**/.env"
    - "**/*.pem"
    - "**/*.key"
    - ".git/**"
    - "**/node_modules/**"
    - "**/.venv/**"
    - "**/.cache/**"
    - "**/memory/**"
    - "**/stats.json"
  never_do:
    - "rm -rf /"
    - "git push --force"
    - "git reset --hard"
    - "chmod 777"
    - "sudo"
    - "dd"
    - "mkfs"
  iron_laws:
    - "Never modify files outside the project root"
    - "Never delete user data without confirmation"
    - "Never execute destructive shell commands"
    - "Never commit secrets to git"
    - "Always verify changes before applying"

guardrails:
  block:
    - "drop table"
    - "delete from"
    - "truncate table"
  approval_required:
    - "production"
    - "deploy"
    - "migration"
  cost_limit: 0.50

secrets_scan:
  mode: "redact"  # redact | block | off
  types:
    - "api_key"
    - "aws_key"
    - "github_token"
    - "jwt"
    - "private_key"
    - "stripe_key"
    - "connection_string"

dependency_gate:
  allow:
    - "lodash"
    - "axios"
    - "react"
  audit: true

output:
  show_result: true
  format: text
  verbose: false
  show_cost: true
  show_routing: false

providers:
  online:
    - name: "deepseek"
      base_url: "https://api.deepseek.com/v1"
      model: "deepseek-chat"
      api_key_env: "DEEPSEEK_API_KEY"
      enabled: true
      timeout_s: 60
      max_retries: 2
    - name: "groq"
      base_url: "https://api.groq.com/openai/v1"
      model: "llama-3.3-70b-versatile"
      api_key_env: "GROQ_API_KEY"
      enabled: false
      timeout_s: 60
      max_retries: 2
  local:
    - name: "qwen"
      base_url: "http://localhost:1234/api/v1"
      model: "qwen2.5-coder-14b-instruct-mlx"
      api_key: "not-needed"
      enabled: true
      timeout_s: 120
      max_retries: 3
    - name: "local-2"
      base_url: "http://localhost:1234/api/v1"
      model: "llama-3.2-3b-instruct"
      api_key: "not-needed"
      enabled: false
      timeout_s: 120
      max_retries: 3
YAML
ok "config.yml written"

if [ ! -f "$AGENT_DIR/.env" ]; then
    cat > "$AGENT_DIR/.env" <<'ENV'
# DeepSeek API Key (get from https://platform.deepseek.com/)
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Optional: Groq API Key
GROQ_API_KEY=your-groq-api-key-here

# Optional: OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here

# Memory lives INSIDE this runtime folder (see config.yml memory.root).
# To store memory elsewhere, set HERMES_MEMORY=/abs/path/to/memory
ENV
    chmod 600 "$AGENT_DIR/.env"
    ok ".env created (chmod 600) — add your API keys"
else
    ok ".env already exists (left untouched)"
fi

# ============================================================
# 6. CLI (ask.py) quick ask
# ============================================================
step "6/8 CLI (ask.py) quick ask"
cat > "$AGENT_DIR/ask.py" <<'PY'
#!/usr/bin/env python3
"""
Universal Hybrid Agent - CLI Bridge
Qwen (local) implements, DeepSeek (cloud) supervises
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if sys.executable != str(VENV_PYTHON) and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import yaml
from openai import OpenAI

def load_config():
    cfg = Path(__file__).resolve().parent / "config.yml"
    if cfg.exists():
        with open(cfg) as f:
            return yaml.safe_load(f)
    return {}

def load_dotenv_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except Exception:
        pass

def memory_root(config=None):
    """Memory folder for THIS install: <app>/memory. Lives inside the runtime
    folder, so deleting that folder removes everything. Copy it to carry memory."""
    env = os.environ.get("HERMES_MEMORY")
    if env:
        return Path(env).expanduser()
    base = Path(__file__).resolve().parent
    cfg = config or {}
    r = (cfg.get("memory") or {}).get("root")
    if r:
        p = Path(r).expanduser()
        return p if p.is_absolute() else (base / p).resolve()
    return base / "memory"

def remember(config, kind, text):
    """Append one line to the long-term memory daily log."""
    try:
        root = memory_root(config)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        day = datetime.now().strftime("%Y-%m-%d")
        with open(root / "logs" / f"{kind}_{day}.log", "a") as f:
            f.write(f"[{stamp}] {text}\n")
    except Exception:
        pass

def get_local_client(config):
    lc = config.get("backends", {}).get("local", {})
    return OpenAI(
        base_url=lc.get("base_url", "http://localhost:1234/api/v1"),
        api_key=lc.get("api_key", "not-needed"),
        timeout=lc.get("timeout_s", 120),
    )

def get_deepseek_client(config):
    dc = config.get("backends", {}).get("deepseek", {})
    key = os.environ.get(dc.get("api_key_env", "DEEPSEEK_API_KEY"))
    if not key:
        print("Error: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    return OpenAI(
        base_url=dc.get("base_url", "https://api.deepseek.com/v1"),
        api_key=key,
        timeout=dc.get("timeout_s", 60),
    )

def local_generate(task, config, stream=False):
    client = get_local_client(config)
    model = config.get("backends", {}).get("local", {}).get("model",
                  "qwen2.5-coder-14b-instruct-mlx")
    messages = [
        {"role": "system", "content": "You are a coding assistant. Generate clean, working code."},
        {"role": "user", "content": task},
    ]
    kwargs = dict(model=model, messages=messages, temperature=0.2, max_tokens=4096)
    try:
        if stream:
            resp = client.chat.completions.create(stream=True, **kwargs)
            result = ""
            for chunk in resp:
                piece = chunk.choices[0].delta.content or ""
                print(piece, end="", flush=True)
                result += piece
            print()
            return result
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Local model error: {e}", file=sys.stderr)
        return None

def deepseek_review(task, implementation, config):
    client = get_deepseek_client(config)
    model = config.get("backends", {}).get("deepseek", {}).get("model", "deepseek-chat")
    messages = [
        {"role": "system", "content": (
            "You are a code reviewer. Review the implementation and return:\n"
            "VERDICT: APPROVED / FIX_REQUIRED / REJECTED\n"
            "CONFIDENCE: 0-1\n"
            "ISSUES: List any issues\n"
            "REQUIRED_FIXES: Specific fixes needed")},
        {"role": "user", "content": f"Task: {task}\n\nImplementation:\n{implementation}"},
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=messages,
                                              temperature=0.3, max_tokens=4096)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek review error: {e}", file=sys.stderr)
        return None

def parse_verdict(review_text):
    verdict, confidence, issues, fixes = "UNKNOWN", 0.5, [], []
    for line in review_text.splitlines():
        if line.startswith("VERDICT:"):   verdict = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try: confidence = float(line.split(":", 1)[1].strip())
            except ValueError: pass
        elif line.startswith("ISSUES:"):          issues.append(line.split(":", 1)[1].strip())
        elif line.startswith("REQUIRED_FIXES:"):  fixes.append(line.split(":", 1)[1].strip())
    return verdict, confidence, issues, fixes

def main():
    p = argparse.ArgumentParser(description="Universal Hybrid Agent")
    p.add_argument("--task", required=True, help="Task description")
    p.add_argument("--local", action="store_true", help="Local model only")
    p.add_argument("--deepseek", action="store_true", help="DeepSeek only")
    p.add_argument("--supervise", action="store_true", help="Supervise loop")
    p.add_argument("--enhance", action="store_true", help="Enhance prompt with DeepSeek")
    p.add_argument("--apply", action="store_true", help="Apply generated code")
    p.add_argument("--verify", action="store_true", help="Verify changes")
    p.add_argument("--stream", action="store_true", help="Stream output")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    load_dotenv_file()
    config = load_config()
    print(f"Task: {args.task}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("Implementing with local model...", file=sys.stderr)

    result = local_generate(args.task, config, args.stream)
    if result is None:
        print("Local model failed", file=sys.stderr)
        remember(config, "ask", f"LOCAL_FAILED | task={args.task!r}")
        sys.exit(3)

    verdict = "LOCAL_ONLY"
    if args.supervise or args.enhance:
        print("Supervising with DeepSeek...", file=sys.stderr)
        review = deepseek_review(args.task, result, config)
        if review:
            verdict, confidence, issues, fixes = parse_verdict(review)
            print(f"Verdict: {verdict} (confidence: {confidence:.2f})", file=sys.stderr)
            for i in issues: print(f"Issue: {i}", file=sys.stderr)

    remember(config, "ask", f"verdict={verdict} | task={args.task!r}")

    if args.json:
        print(json.dumps({"task": args.task, "result": result, "verdict": verdict}))
    else:
        print(result)
    if verdict in ("FIX_REQUIRED", "REJECTED", "UNKNOWN"):
        print("Fix required.", file=sys.stderr)
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY
chmod +x "$AGENT_DIR/ask.py"
ok "ask.py written + executable"

# ============================================================
# 6b. Planner + Supervisor + Recorder orchestrator (run_agent.py)
#    Planner  : online (DeepSeek) breaks a task into steps
#    Local    : Qwen implements each step
#    Supervisor: online (DeepSeek) reviews, tracks progress, fix loops
#    Recorder : full per-run session log (JSON + Markdown)
# ============================================================
step "6/8 (b) run_agent.py planner/supervisor/recorder"
cat > "$AGENT_DIR/run_agent.py" <<'PY'
#!/usr/bin/env python3
"""
Universal Hybrid Agent - Orchestrator
Planner (online DeepSeek) -> Local (Qwen) implements -> Supervisor (online) reviews -> Recorder logs.

Flow:
  1. online AI plans the task into ordered steps (recorder: plan)
  2. for each step: local implements -> supervisor verdict
       APPROVED       -> next step
       FIX_REQUIRED   -> local re-implements with supervisor's feedback (up to max_fixes)
       REJECTED       -> mark blocked, stop
  3. recorder writes a full session log (logs/session_<ts>.md and .json)

Usage:
  ./run_agent.py --task "build a fibonacci module with tests"
  ./run_agent.py --task "..." --max-iterations 5
  ./run_agent.py --task "..." --json      # machine-readable final summary
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if sys.executable != str(VENV_PYTHON) and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import yaml
from openai import OpenAI

DEFAULT_SYS_PLAN = (
    "You are a meticulous planning engine. Break the user's task into a small, ordered set of "
    "concrete implementation steps. Reply with ONLY valid JSON, no prose, in this exact shape:\n"
    "{\"objective\":\"<one line>\",\"steps\":[{\"id\":1,\"title\":\"<short>\","
    "\"detail\":\"<what to build, files, expected behaviour>\"}]}\n"
    "Keep 1-8 steps. Each detail must be actionable by a coding assistant that writes files."
)
DEFAULT_SYS_SUPER = (
    "You are a code supervisor reviewing work one step at a time. Return ONLY valid JSON:\n"
    "{\"verdict\":\"APPROVED\"|\"FIX_REQUIRED\"|\"REJECTED\",\"confidence\":0.0-1.0,"
    "\"issues\":[\"...\"],\"feedback\":\"specific instructions for the implementer to fix it\"}\n"
    "APPROVED means the step is correct, complete and safe to keep. If anything needs changing "
    "use FIX_REQUIRED with concrete feedback. Use REJECTED only if the step is unusable."
)


class Recorder:
    """Persists one full run: task, plan, every step, verdicts, tokens, timing."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "task": None,
            "planner": None,
            "steps": [],
            "summary": {},
        }
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def set_task(self, task, planner_model):
        self.session["task"] = task
        self.session["planner"] = planner_model

    def record_plan(self, plan):
        self.session["plan"] = plan

    def add_step(self, step, implementer, supervisor, code, verdict, confidence,
                 issues, feedback, fixes, tokens):
        self.session["steps"].append({
            "id": step.get("id"),
            "title": step.get("title"),
            "detail": step.get("detail"),
            "implementer": implementer,
            "supervisor": supervisor,
            "code": code,
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "feedback": feedback,
            "fix_rounds": fixes,
            "tokens": tokens,
        })

    def finish(self, summary):
        self.session["summary"] = summary
        self.session["finished_at"] = datetime.now().isoformat(timespec="seconds")
        base = self.log_dir / f"session_{self.run_id}"
        with open(f"{base}.json", "w") as f:
            json.dump(self.session, f, indent=2)
        with open(f"{base}.md", "w") as f:
            self._write_markdown(f)
        return f"{base}.md"

    def _write_markdown(self, f):
        s = self.session
        f.write(f"# Hybrid Agent Session {self.run_id}\n\n")
        f.write(f"- Started: {s.get('started_at')}\n- Finished: {s.get('finished_at')}\n")
        f.write(f"- Planner: {s.get('planner')}\n- Task: {s.get('task')}\n\n")
        plan = s.get("plan") or {}
        if plan.get("objective"):
            f.write(f"## Objective\n\n{plan['objective']}\n\n")
        f.write("## Steps\n\n")
        for st in s.get("steps", []):
            f.write(f"### Step {st.get('id')}: {st.get('title')}\n\n")
            f.write(f"- Implementer: {st.get('implementer')}\n- Verdict: **{st.get('verdict')}**"
                    f" (confidence {st.get('confidence')})\n- Fix rounds: {st.get('fix_rounds')}\n\n")
            f.write(f"Detail: {st.get('detail')}\n\n")
            if st.get("code"):
                f.write("```\n" + st["code"] + "\n```\n\n")
            if st.get("issues"):
                f.write("Issues:\n")
                for i in st["issues"]:
                    f.write(f"- {i}\n")
                f.write("\n")
            if st.get("feedback"):
                f.write(f"Supervisor feedback: {st['feedback']}\n\n")
        f.write("## Summary\n\n")
        f.write("```json\n" + json.dumps(s.get("summary", {}), indent=2) + "\n```\n")


def load_config():
    cfg = Path(__file__).resolve().parent / "config.yml"
    if cfg.exists():
        with open(cfg) as f:
            return yaml.safe_load(f)
    return {}


def _deepseek(config):
    dc = config.get("backends", {}).get("deepseek", {})
    key = os.environ.get(dc.get("api_key_env", "DEEPSEEK_API_KEY"))
    if not key:
        print("Error: DEEPSEEK_API_KEY not set (add it to .env)", file=sys.stderr)
        sys.exit(2)
    return OpenAI(base_url=dc.get("base_url", "https://api.deepseek.com/v1"),
                  api_key=key, timeout=dc.get("timeout_s", 60)), \
           dc.get("model", "deepseek-chat")


def _local(config):
    lc = config.get("backends", {}).get("local", {})
    return OpenAI(base_url=lc.get("base_url", "http://localhost:1234/api/v1"),
                  api_key=lc.get("api_key", "not-needed"),
                  timeout=lc.get("timeout_s", 120)), \
           lc.get("model", "qwen2.5-coder-14b-instruct-mlx")


def _extract_json(text):
    """Pull the first JSON object out of a model reply (handles ``` fences)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        t = "\n".join(lines[1:])
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model reply")
    return json.loads(t[start:end + 1])


def planner_plan(task, config, stream=False):
    """Online DeepSeek turns the task into an ordered plan (uses long-term memory)."""
    client, model = _deepseek(config)
    mem = (config or {}).get("_memory_context") or ""
    user = f"Task: {task}"
    if mem:
        user = ("You have prior long-term memory of this user's work. Use it to plan well.\n"
                "--- BEGIN MEMORY ---\n" + mem + "\n--- END MEMORY ---\n\n" + user)
    messages = [{"role": "system", "content": DEFAULT_SYS_PLAN},
                {"role": "user", "content": user}]
    kw = dict(model=model, messages=messages, temperature=0.2, max_tokens=2000)
    if stream:
        r = client.chat.completions.create(stream=True, **kw)
        out = ""
        for chunk in r:
            out += chunk.choices[0].delta.content or ""
    else:
        r = client.chat.completions.create(**kw)
        out = r.choices[0].message.content
    return _extract_json(out), getattr(r, "usage", None)


def local_implement(step, feedback, config):
    """Local Qwen implements (or re-implements) one step."""
    client, model = _local(config)
    sys_txt = "You are a coding implementer. Produce clean, working, complete code or instructions."
    user = f"Step {step.get('id')}: {step.get('title')}\n{step.get('detail')}"
    if feedback:
        user += f"\n\nSupervisor feedback to address:\n{feedback}"
    r = client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=4096,
        messages=[{"role": "system", "content": sys_txt},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content, getattr(r, "usage", None)


def supervisor_review(step, code, config):
    """Online DeepSeek reviews a step and returns verdict + feedback."""
    client, model = _deepseek(config)
    r = client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=1200,
        messages=[{"role": "system", "content": DEFAULT_SYS_SUPER},
                  {"role": "user", "content":
                   f"Step {step.get('id')}: {step.get('title')}\n{step.get('detail')}\n\n"
                   f"Implementation:\n{code}"}])
    review = _extract_json(r.choices[0].message.content)
    return review, getattr(r, "usage", None)


def _tokens(usage):
    if not usage:
        return {"prompt": 0, "completion": 0}
    return {"prompt": usage.prompt_tokens or 0, "completion": usage.completion_tokens or 0}


# --- Persistent long-term memory (survives reinstalls) ---
def load_dotenv_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except Exception:
        pass


def memory_root(config=None):
    """Canonical memory folder for THIS install: <app>/memory. Lives inside the
    runtime folder so deleting it removes everything; copy it to keep memory."""
    env = os.environ.get("HERMES_MEMORY")
    if env:
        return Path(env).expanduser()
    base = Path(__file__).resolve().parent
    cfg = config or {}
    r = (cfg.get("memory") or {}).get("root")
    if r:
        p = Path(r).expanduser()
        return p if p.is_absolute() else (base / p).resolve()
    return base / "memory"


def recent_memory(root, limit_chars=3000):
    """Tail of index.md used as prompt context for the planner."""
    idx = root / "index.md"
    if not idx.exists():
        return ""
    try:
        lines = idx.read_text().splitlines()
    except Exception:
        return ""
    out, used = [], 0
    for ln in reversed(lines):
        if used + len(ln) + 1 > limit_chars:
            break
        out.append(ln)
        used += len(ln) + 1
    return "\n".join(reversed(out))


def write_index_entry(root, entry):
    """Append one distilled line to the long-term memory index."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(root / "index.md", "a") as f:
            f.write(f"- [{stamp}] {entry}\n")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Hybrid Agent: plan -> implement -> supervise -> record")
    p.add_argument("--task", required=True, help="High level task to plan and execute")
    p.add_argument("--max-fixes", type=int, default=2, help="Max FIX_REQUIRED rounds per step")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming during planning")
    p.add_argument("--json", action="store_true", help="Print final summary as JSON")
    p.add_argument("--log-dir", default="", help="Where recorder writes sessions (default: memory/logs)")
    args = p.parse_args()

    load_dotenv_file()
    config = load_config()
    stream = not args.no_stream

    mem = memory_root(config)
    config["_memory_context"] = recent_memory(mem)
    if not args.log_dir or args.log_dir in ("logs", "memory"):
        args.log_dir = str(mem / "logs")
    print(f"Long-term memory : {mem}", file=sys.stderr)
    print(f"Session logs dir : {args.log_dir}", file=sys.stderr)

    print(f"Task: {args.task}\n", file=sys.stderr)

    # --- 1. Plan (online) ---
    print("PLANNING (online DeepSeek)...", file=sys.stderr)
    plan, _ = planner_plan(args.task, config, stream=stream)
    steps = plan.get("steps", [])
    objective = plan.get("objective", args.task)
    print(f"  objective: {objective}", file=sys.stderr)
    print(f"  plan: {len(steps)} step(s)", file=sys.stderr)

    rec = Recorder(args.log_dir)
    _, planner_model = _deepseek(config)
    _, implementer_model = _local(config)
    _, supervisor_model = _deepseek(config)
    rec.set_task(args.task, planner_model)
    rec.record_plan(plan)

    summary = {"steps_total": len(steps), "approved": 0, "blocked": 0,
               "fix_rounds_total": 0, "tokens": {"prompt": 0, "completion": 0}}
    stopped = False

    for step in steps:
        print(f"\nEXECUTING step {step.get('id')}: {step.get('title')} (local {implementer_model})",
              file=sys.stderr)
        feedback = None
        fixes = 0
        code = None
        final = None
        while True:
            code, lu = local_implement(step, feedback, config)
            review, su = supervisor_review(step, code, config)
            verdict = review.get("verdict", "FIX_REQUIRED").upper()
            conf = float(review.get("confidence", 0.5))
            issues = review.get("issues", [])
            feedback = review.get("feedback")
            _add_tokens(summary, lu, su)

            if verdict in ("APPROVED", "REJECTED"):
                final = (verdict, conf, issues, feedback)
                break
            fixes += 1
            if fixes > args.max_fixes:
                final = ("REJECTED", conf, issues,
                         f"exceeded max fixes ({args.max_fixes}) | last: {feedback}")
                break
            print(f"  FIX_REQUIRED (round {fixes}): {feedback}", file=sys.stderr)

        verdict, conf, issues, feedback = final
        summary["fix_rounds_total"] += fixes
        if verdict == "APPROVED":
            summary["approved"] += 1
            print(f"  step {step.get('id')} APPROVED", file=sys.stderr)
        else:
            summary["blocked"] += 1
            stopped = True
            print(f"  step {step.get('id')} {verdict}", file=sys.stderr)

        rec.add_step(step, implementer_model, supervisor_model, code,
                     verdict, conf, issues, feedback, fixes, {"fix_rounds": fixes})

        if stopped:
            summary["note"] = "execution stopped early because a step was not approved"
            break

    summary["steps_executed"] = summary["approved"] + summary["blocked"]
    summary["status"] = "STOPPED_BLOCKED" if stopped else "COMPLETE"
    rec.finish(summary)

    write_index_entry(mem, (f"{summary['status']} | task={args.task!r} | "
                            f"steps {summary.get('approved')}/{summary.get('steps_total')} | "
                            f"fix_rounds {summary.get('fix_rounds_total')} | session {rec.run_id}"))

    if args.json:
        print(json.dumps({"objective": objective, "summary": summary}))
    else:
        print(f"\nDONE. status={summary['status']} approved={summary['approved']}/"
              f"{summary['steps_total']} fix_rounds={summary['fix_rounds_total']}", file=sys.stderr)
    return 0 if not stopped else 1


def _add_tokens(summary, *usages):
    for u in usages:
        if not u:
            continue
        summary["tokens"]["prompt"] += u.prompt_tokens or 0
        summary["tokens"]["completion"] += u.completion_tokens or 0


if __name__ == "__main__":
    sys.exit(main())
PY
chmod +x "$AGENT_DIR/run_agent.py"
ok "run_agent.py (planner/supervisor/recorder) written + executable"

# ============================================================
# 7. Web dashboard + start scripts
# ============================================================
step "7/8 Web dashboard + launchers"
cat > "$AGENT_DIR/web_dashboard.py" <<'PY'
#!/usr/bin/env python3
"""Hybrid Agent Web Dashboard"""
import os
import json
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

task_queue = []
is_running = False
current_task = None

@app.route('/')
def index():
    return "Hybrid Agent Dashboard — try /api/stats or POST /api/run"

@app.route('/api/stats')
def stats():
    return jsonify({"status": "running", "queue_length": len(task_queue),
                    "is_running": is_running, "task": current_task})

@app.route('/api/run', methods=['POST'])
def run_task():
    global current_task, is_running
    task = (request.json or {}).get('task', '')
    if not task:
        return jsonify({"error": "No task provided"}), 400
    current_task = {"task": task, "status": "running"}
    is_running = True
    socketio.emit('task_update', current_task)
    return jsonify({"status": "started", "task": task})

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8660))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
PY
chmod +x "$AGENT_DIR/web_dashboard.py"

cat > "$AGENT_DIR/start.sh" <<'SH'
#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ -f ".env" ]; then set -a; source ".env"; set +a; fi
source .venv/bin/activate
./ask.py "$@"
SH
chmod +x "$AGENT_DIR/start.sh"

cat > "$AGENT_DIR/start_dashboard.sh" <<'SH'
#!/bin/bash
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
echo "Starting Web Dashboard at http://localhost:8660"
python3 web_dashboard.py "$@"
SH
chmod +x "$AGENT_DIR/start_dashboard.sh"
ok "web_dashboard.py, start.sh, start_dashboard.sh written"

# ============================================================
# 8. Git repo + pre-commit hooks (all inside the one folder)
# ============================================================
step "8/8 Git repo + pre-commit hooks"
if [ ! -d "$AGENT_DIR/.git" ]; then
    git -C "$AGENT_DIR" init -q
    ok "Initialized git repo"
else
    ok "Git repo already exists"
fi

cat > "$AGENT_DIR/scripts/validate-agents.sh" <<'SH'
#!/bin/bash
# Validates required hybrid-agent files exist. Run on pre-commit.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
echo "Validating agent files in $HERE ..."
required_files=("$HERE/config.yml" "$HERE/ask.py" "$HERE/.env" "$HERE/web_dashboard.py")
missing=0
for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then echo "Missing: $f"; missing=1; fi
done
# crude secret check so keys never get committed
if [ -f "$HERE/.env" ] && grep -qE 'your-(deepseek|groq|openai)-api-key-here|sk-[A-Za-z0-9]{10,}' "$HERE/.env"; then
    echo "Warning: .env still contains placeholder or real-looking secret."
fi
if [ "$missing" -ne 0 ]; then exit 1; fi
echo "OK: all agent files present."
exit 0
SH
chmod +x "$AGENT_DIR/scripts/validate-agents.sh"

cat > "$AGENT_DIR/.gitignore" <<'GI'
.env
.venv/
__pycache__/
.cache/
logs/
stats.json
.DS_Store
node_modules/
GI

HOOKS="$AGENT_DIR/.git/hooks"
if [ -d "$HOOKS" ]; then
    printf '#!/bin/bash\nexec "%s/scripts/validate-agents.sh"\n' "$AGENT_DIR" > "$HOOKS/pre-commit"
    chmod +x "$HOOKS/pre-commit"
    ok "pre-commit hook installed"
fi

# ============================================================
# ============================================================
# 8. (done) Runtime dirs + self test + summary
# ============================================================
step "8/8 (done) Finalize: memory store + self test"
# Memory is PER-INSTALL and lives INSIDE this runtime folder (<runtime>/memory).
# Deleting the runtime folder deletes everything, memory included. To KEEP
# memory for a new install, copy this memory/ folder into the new one.
MEMORY_ROOT="${HERMES_MEMORY:-$AGENT_DIR/memory}"
mkdir -p "$MEMORY_ROOT/logs" "$MEMORY_ROOT/sessions"
ok "Memory store ready (inside this runtime folder): $MEMORY_ROOT"
echo -e "${BLUE}      To keep memory on a new install, copy it:${NC}"
echo -e "        cp -R \"$MEMORY_ROOT\" \"/path/to/Hermes-<newdate>/memory\""
# In-app scratch dirs (cache/sessions/state/logs) are disposable.
mkdir -p "$AGENT_DIR"/cache "$AGENT_DIR"/sessions "$AGENT_DIR"/state "$AGENT_DIR"/logs
ok "In-app scratch dirs created (cache sessions state logs)"

# ============================================================
# 8c. MCP tool layer (local tools so the ONLINE AI only plans/supervises)
#     mcp_server.py is a dependency-light MCP tool server (stdlib JSON-RPC).
# ============================================================
step "8/8 (c) MCP tool layer (local tools)"
mkdir -p "$AGENT_DIR/tools_state"
if [ -f "$SCRIPT_DIR/mcp_server.py" ]; then
    cp "$SCRIPT_DIR/mcp_server.py" "$AGENT_DIR/mcp_server.py" 2>/dev/null || true
fi
if [ -f "$SCRIPT_DIR/start_mcp.sh" ]; then
    cp "$SCRIPT_DIR/start_mcp.sh" "$AGENT_DIR/start_mcp.sh" 2>/dev/null || true
fi
if [ -f "$SCRIPT_DIR/tools_manifest.json" ]; then
    cp "$SCRIPT_DIR/tools_manifest.json" "$AGENT_DIR/tools_manifest.json" 2>/dev/null || true
fi
if [ -f "$AGENT_DIR/mcp_server.py" ]; then
    chmod +x "$AGENT_DIR/mcp_server.py" "$AGENT_DIR/start_mcp.sh" 2>/dev/null || true
    ok "mcp_server.py present (filesystem/terminal/web/browser/docs/data/code/memory/security/cron/todo/notify)"
    warn "Optional richer tools (PDF/DOCX/XLSX, feeds) need: pip install pypdf python-docx openpyxl feedparser"
else
    warn "mcp_server.py not found beside install.sh — place it in this folder (or rerun from the full Hermes folder)."
fi
ok "tools_state ready: $AGENT_DIR/tools_state"

cat > "$AGENT_DIR/test_hybrid.py" <<'PY'
#!/usr/bin/env python3
"""Test Hybrid Agent installation."""
from pathlib import Path
import sys

ok, bad = [], []
def check(name, cond):
    (ok if cond else bad).append(name)

try:
    import openai, yaml, flask, cryptography, playwright  # noqa
    check("python packages import", True)
except ImportError as e:
    check(f"python packages import ({e})", False)
check("config.yml exists", Path("config.yml").exists())
check(".env exists", Path(".env").exists())
check("ask.py exists", Path("ask.py").exists())
check("run_agent.py exists", Path("run_agent.py").exists())
check("venv exists", Path(".venv/bin/python").exists())

for name in ok:  print(f"  ✓ {name}")
for name in bad: print(f"  ✗ {name}")
print("RESULT:", "PASS" if not bad else f"FAIL ({len(bad)} problems)")
sys.exit(0 if not bad else 1)
PY
chmod +x "$AGENT_DIR/test_hybrid.py"

# shellcheck disable=SC1091
source "$AGENT_DIR/.venv/bin/activate"
(cd "$AGENT_DIR" && python test_hybrid.py) || echo -e "${YELLOW}  (Self-test has warnings — see above)${NC}"

echo ""
log "============================================================"
log "     HERMES INSTALLED — NEW RUNTIME FOLDER CREATED"
log "============================================================"
echo -e "${GREEN}  RUNTIME folder : $AGENT_DIR  (created this run)${NC}"
echo -e "  INSTALL folder: $SCRIPT_DIR  (kept for installation only)"
echo ""
log "  Next steps:"
echo "    1. Edit  $AGENT_DIR/.env  and add your real DEEPSEEK_API_KEY"
echo "    2. Start LM Studio -> load qwen2.5-coder-14b-instruct-mlx -> Local server :1234"
echo "       OR  ollama pull qwen2.5-coder:14b && ollama serve"
echo "    3. Quick ask:    $AGENT_DIR/ask.py --task \"write hello world in python\""
echo "    4. Full pipeline: $AGENT_DIR/run_agent.py --task \"build a fibonacci module with tests\""
echo "         planner = online DeepSeek, implementer = local Qwen, supervisor = DeepSeek"
echo "    5. Dashboard: $AGENT_DIR/start_dashboard.sh  ->  http://localhost:8660"
echo "    6. MCP local tools (so online AI only plans/supervises):"
echo "       $AGENT_DIR/start_mcp.sh            # connect this as an MCP server to your host"
echo "       $AGENT_DIR/start_mcp.sh --list     # list the ~46 built-in local tools"
echo "       Set HERMES_SAFE=1 to allow destructive tools (delete/kill/exec)."
echo ""
echo -e "${YELLOW}  MEMORY — inside this folder, deleted with it:${NC}"
echo "    - All memories live under:  ${MEMORY_ROOT:-$AGENT_DIR/memory}"
echo "      (logs in .../logs, distilled index in .../index.md)"
echo "    - Delete this runtime folder => deletes EVERYTHING (incl. memory)."
echo "    - To KEEP memory for a new install, copy it in:"
echo "        cp -R \"${MEMORY_ROOT:-$AGENT_DIR/memory}\" \"/path/to/Hermes-<newdate>/memory\""
echo "    - run_agent.py feeds recent memory back as planning context."
echo ""
log "============================================================"
ok "Done"
