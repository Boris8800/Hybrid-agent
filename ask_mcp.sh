#!/bin/bash
# ask_mcp.sh <tool_name> [json-arguments]
# Send a SINGLE tool call to the Hermes MCP server and print the result.
# Example:
#   ./ask_mcp.sh read_file '{"path":"config.yml"}'
#   ./ask_mcp.sh memory_stats '{}'
cd "$(dirname "$0")" || exit 1
[ -f ".env" ] && { set -a; source ".env"; set +a; }
TOOL="$1"; shift || true
ARGS="${1:-{}}"

REQ="$(python3 - "$TOOL" "$ARGS" <<'PY'
import sys, json
try:
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else {}
except Exception:
    args = {}
print(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": sys.argv[1], "arguments": args}}))
PY
)"
printf '%s\n' "$REQ" | python3 mcp_server.py
