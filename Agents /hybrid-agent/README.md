# hybrid-agent

Bridge CLI that lets a coding agent delegate work out-of-band to two external
models — a local LM Studio model (Qwen) and the DeepSeek cloud API. Qwen
implements, DeepSeek supervises, with routing, prompt enhancement, parallel
step execution, caching, token budgeting, persistent memory, a hardened
file-application engine, a **Context Safety Controller** (per-model capability
discovery, hard budgets, preflight zones, compaction, retry budgets,
telemetry), and a **durable task-state machine** (Task Contract as permanent
state: strict legal transitions, evidence ledger, batch transactions,
idempotent operations, restart/resume, and a deploy trust boundary).

**Full documentation lives in the repository root: [`README.md`](../../README.md).**

Quick start:

```bash
cd "Agents /hybrid-agent"
./.venv/bin/python ask.py --help
./.venv/bin/python ask.py --local --task "Add input validation to validate_email"
./.venv/bin/python ask.py --supervise --enhance --task "<task>" --apply
```

Run the tests with:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
