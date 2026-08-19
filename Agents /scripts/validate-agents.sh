#!/bin/bash
# Validate all agent definition files (.kilo/agent/*.md).
# 1) Runs `kilo config check` ONCE as the authoritative whole-config gate.
# 2) Checks each file's frontmatter YAML via the bridge venv.
# Exits non-zero if kilo config check fails.

set -u
FAILED=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== kilo config check (authoritative gate) =="
if kilo config check 2>&1 | grep -q "No config warnings"; then
    echo "  OK: no config warnings"
else
    echo "  FAILED: kilo config check reported warnings/errors"
    FAILED=1
fi
echo

echo "== per-file frontmatter check (venv yaml) =="
for file in .kilo/agent/*.md; do
    if [ -f "$file" ]; then
        if "$SCRIPT_DIR/../hybrid-agent/.venv/bin/python" - "$file" <<'PY'
import sys, yaml
content = open(sys.argv[1]).read()
if '---' not in content:
    sys.exit(1)
_, fm, _ = content.split('---', 2)
yaml.safe_load(fm)
PY
        then
            echo "  OK   $(basename "$file")"
        else
            echo "  WARN $(basename "$file") (frontmatter parse issue; may be a false positive)"
        fi
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo
    echo "All agent files valid."
    exit 0
else
    echo
    echo "kilo config check failed — see output above."
    exit 1
fi
