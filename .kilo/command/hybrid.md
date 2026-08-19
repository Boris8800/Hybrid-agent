---
description: Run the hybrid Gemma-primary / DeepSeek-supervisor loop on a task via hybrid-agent/ask.py --supervise --apply
mode: all
---
Run the hybrid bridge directly on the task: Gemma (local, LM Studio) implements, DeepSeek reviews, loops until APPROVED or max iterations, then writes the approved files to the repo root via --apply.

```bash
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRIDGE="$ROOT/Agents /hybrid-agent"
if [ ! -x "$BRIDGE/.venv/bin/python" ]; then
  echo "error: hybrid-agent bridge not found (expected $BRIDGE)" >&2
  exit 2
fi
if [ "$#" -eq 0 ]; then
  echo "usage: /hybrid <task>" >&2
  exit 2
fi
cd "$BRIDGE" || exit 2
exec .venv/bin/python ask.py --supervise --mode hybrid --apply --root "$ROOT" --task "$*"
```
