#!/bin/bash
# manage-context.sh - simple wrapper around the hybrid agent project scanner.
# Usage: hybrid-agent/manage-context.sh [project-root] {scan|update|status|clear}
#   scan    - initial scan (default)
#   update  - refresh existing context
#   status  - show context info / freshness
#   clear   - remove the context file

set -e

PROJECT_ROOT="${1:-.}"
COMMAND="${2:-scan}"

cd "$PROJECT_ROOT"

PY="hybrid-agent/.venv/bin/python"
SCAN="hybrid-agent/scan.py"

case "$COMMAND" in
    scan|"")
        echo "Scanning project..."
        "$PY" "$SCAN" --project-root . --suggest-tasks
        ;;
    update)
        echo "Updating project context..."
        "$PY" "$SCAN" --project-root . --update --suggest-tasks
        ;;
    status)
        if [ -f hybrid-agent/context.json ]; then
            echo "Context exists:"
            grep -E '"project_root"|"scan_time"' hybrid-agent/context.json | sed 's/^/  /'
            if [ -n "$(find hybrid-agent/context.json -mmin +60 2>/dev/null)" ]; then
                echo "  (warning: over 1 hour old - run: hybrid-agent/manage-context.sh update)"
            else
                echo "  (context is fresh)"
            fi
        else
            echo "No context found. Run: hybrid-agent/manage-context.sh scan"
        fi
        ;;
    clear)
        rm -f hybrid-agent/context.json
        echo "Context cleared."
        ;;
    *)
        echo "Usage: $0 [project-root] {scan|update|status|clear}"
        echo "  scan    - initial scan (default)"
        echo "  update  - refresh existing context"
        echo "  status  - show context info"
        echo "  clear   - remove context file"
        exit 1
        ;;
esac
