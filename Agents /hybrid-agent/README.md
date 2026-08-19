# hybrid-agent

Bridge CLI that lets a coding agent delegate work out-of-band to two external models — a local LM Studio model and the DeepSeek cloud API. It does **not** use the Kilo provider system.

## Models

- **Local (Gemma)** — `google/gemma-4-12b-qat` via LM Studio at `http://localhost:1234/v1`. Fast, cheap, no API key. Best for mechanical, well-specified edits.
- **DeepSeek** — `deepseek-chat`, requires `DEEPSEEK_API_KEY` (or a key under `deepseek.key` in Kilo's `auth.json`). Best for architecture, design, and ambiguous debugging.

## Running

Use the venv python (the system `python3` may not have `openai`/`yaml`):

```bash
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --help
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --models
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --local --task "<task>" --stream
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --deepseek --task "<task>"
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --supervise --task "<task>" --max-iterations 4
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --supervise --enhance --task "<task>" --apply
```

Key flags: `--local` / `--deepseek` / `--auto`, `--review` (DeepSeek supervisor review), `--supervise` (Gemma-primary / DeepSeek-supervisor loop), `--enhance` (DeepSeek enhances the prompt and plans around Gemma's context/output limits BEFORE implementing — the improved prompt + reasoning + plan are shown, then sent to the local model; if the task is unclear DeepSeek asks clarifying questions and the user can adjust the prompt before it proceeds), `--models`, `--stream`, `--json`, `--route-only`.

## Project Context Feature

The hybrid agent can automatically build a project index (files, dependencies, structure, entry points, git info) and use it for smarter routing and suggestions.

```bash
# Initial scan (auto-runs on agent load if context.json is missing)
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/scan.py" --project-root .

# See context
"Agents /hybrid-agent/manage-context.sh" status

# Get task suggestions
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/scan.py" --project-root . --suggest-tasks

# Refresh after changes
"Agents /hybrid-agent/manage-context.sh" update

# Ask a context-aware question (context is auto-appended to every request)
"Agents /hybrid-agent/.venv/bin/python" "Agents /hybrid-agent/ask.py" --route-only --task "Add tests for the main module"
```

Context is stored in `hybrid-agent/context.json` (git-ignored) and auto-appended to every model request by `ask.py`. The scanner skips `.venv`, `node_modules`, `.git`, and caches.

## Quick Setup for New Developers

1. **Clone and enter the repo**
   ```bash
   git clone <url>
   cd <repo>
   ```

2. **Create the bridge venv** (system `python3` cannot pip-install globally on macOS, so use a venv)
   ```bash
   python3 -m venv hybrid-agent/.venv
   hybrid-agent/.venv/bin/pip install openai pyyaml
   ```

3. **Install the pre-commit hook** (git hooks are not versioned, so run the installer after cloning)
   ```bash
   ./scripts/install-hooks.sh
   ```

4. **Verify everything works**
   ```bash
   ./scripts/validate-agents.sh   # -> "All agent files valid."
   kilo config check              # -> "No config warnings."
   ```

5. **Start using the hybrid agent**
   - VS Code frontmatter warnings? **Ignore them** (known false positive — see `.kilo/known-false-positives`).
   - Run the agent with `hybrid-agent/.venv/bin/python hybrid-agent/ask.py --help`.

## Agent File Validation

Agent definitions live in `.kilo/agent/*.md`.

- **Source of truth:** `kilo config check` — should show "No config warnings." (There is **no** `--file` flag; it validates the whole config.)
- **VS Code warnings:** ignore. A "Failed to parse frontmatter" squiggle is a known false positive from the extension's whole-file YAML parser — the files are valid and `kilo config check` passes.
- **Validation script:** `./scripts/validate-agents.sh` — run before commits or in CI.
- **Known false positives:** see `.kilo/known-false-positives` for the current list.
- **Checks use the venv:** `hybrid-agent/.venv/bin/python` (system `python3` has no `yaml`).
