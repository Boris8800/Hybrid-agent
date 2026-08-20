# hybrid-agent

Bridge CLI that lets a coding agent delegate work out-of-band to two external
models — a local LM Studio model (Gemma) and the DeepSeek cloud API. Gemma
implements, DeepSeek supervises, with routing, prompt enhancement, parallel
step execution, caching, token budgeting, persistent memory, and a hardened
file-application engine.

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
