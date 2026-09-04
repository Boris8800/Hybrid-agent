#!/bin/bash
# Start the Hermes MCP tool server over stdio.
# Connect from an MCP host (Claude Code / VS Code / Cursor / your supervisor):
#   command: bash /path/to/Hermes/start_mcp.sh
# OR run directly:  python3 mcp_server.py
#
# Safety:
#   HERMES_SAFE=1            -> allow destructive tools (delete, kill, exec, db write)
#   HERMES_ALLOW=delete_file,kill_process  -> allowlist specific tools
#   HERMES_PROJECT=/path     -> confine filesystem tools to this root (default: app dir)
#   HERMES_MEMORY=/path      -> memory folder (default: <app>/memory)
#
# Examples:
#   ./start_mcp.sh                       # stdio (for an MCP client)
#   ./start_mcp.sh --list                # just list the available tools
#   ./start_mcp.sh --manifest            # print the full tool manifest as JSON

cd "$(dirname "$0")" || exit 1
if [ -f ".env" ]; then set -a; source ".env"; set +a; fi
# Run with the venv python so auto-installed libs (pypdf, docx, openpyxl, feedparser)
# are visible to the MCP server. Fall back to python3 if the venv is missing.
VENV_PY="$(dirname "$0")/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"
exec "$VENV_PY" mcp_server.py "$@"
