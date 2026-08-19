#!/bin/bash
# Install the repo's git hooks (scripts/hooks/*) into .git/hooks/.
# Git hooks are not versioned, so new clones must run this to get them.
set -e

HOOK_DIR=".git/hooks"
if [ ! -d "$HOOK_DIR" ]; then
    echo "error: not a git repository (no $HOOK_DIR)" >&2
    exit 1
fi

for hook in scripts/hooks/*; do
    [ -e "$hook" ] || continue
    name="$(basename "$hook")"
    target="$HOOK_DIR/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        cp "$target" "$target.bak"
        echo "backed up existing $target -> $target.bak"
    fi
    ln -sf "$(pwd)/$hook" "$target"
    echo "installed $target -> $hook"
done
