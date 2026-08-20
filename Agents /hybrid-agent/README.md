# hybrid-agent

Bridge CLI that lets a coding agent delegate work out-of-band to two external
models — a local LM Studio model (Qwen) and the DeepSeek cloud API. Qwen
implements, DeepSeek supervises, with routing, prompt enhancement, parallel
step execution, caching, token budgeting, persistent memory, a hardened
file-application engine, and a **Context Safety Controller** that prevents the
local model from ever reaching an unusable context state (per-model capability
discovery, hard input/output budgets, preflight GREEN/YELLOW/ORANGE/RED zones,
invariant-preserving compaction, tool-output summarization, retry budgets,
output-state classification, and per-call context telemetry).

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
