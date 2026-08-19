"""Build compact structured diffs for the reviewer (ARCHITECTURE.md §5)."""

import re
from dataclasses import dataclass, field

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.M)


@dataclass
class Hunk:
    old_start: int
    new_start: int
    lines: list[str]          # ' ' context, '-' removed, '+' added


@dataclass
class FileDiff:
    path: str
    hunks: list[Hunk] = field(default_factory=list)

    def render(self, context_lines: int = 3) -> str:
        """Render compact hunks with limited sibling context."""
        parts = [f"--- a/{self.path}", f"+++ b/{self.path}"]
        for hunk in self.hunks:
            parts.append(f"@@ -{hunk.old_start} +{hunk.new_start} @@")
            head, tail = hunk.lines, []
            if context_lines > 0 and head:
                head = head[:context_lines] + ["..."]
            parts.extend(head)
        return "\n".join(parts)


def parse_unified_diff(raw: str) -> list[FileDiff]:
    """Parse `git diff`-style unified output into FileDiff objects."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: Hunk | None = None

    for line in raw.splitlines():
        if line.startswith("diff --git"):
            if current is not None:
                files.append(current)
            match = re.search(r" b/(.+)$", line)
            current = FileDiff(path=match.group(1) if match else "unknown")
            current_hunk = None
        elif line.startswith("@@") and current is not None:
            if current_hunk is not None:
                current.hunks.append(current_hunk)
            m = _HUNK_HEADER.search(line)
            current_hunk = Hunk(
                old_start=int(m.group(1)) if m else 0,
                new_start=int(m.group(2)) if m else 0,
                lines=[],
            )
        elif current is not None and current_hunk is not None:
            if line.startswith(("+", "-", " ")):
                current_hunk.lines.append(line)

    if current_hunk is not None and current is not None:
        current.hunks.append(current_hunk)
    if current is not None:
        files.append(current)
    return files


def token_estimate(diff_set: list[FileDiff]) -> int:
    """Rough token estimate (4 chars/token) for budget accounting."""
    total_chars = sum(len(f.render()) for f in diff_set)
    return max(1, total_chars // 4)