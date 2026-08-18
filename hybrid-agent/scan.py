#!/usr/bin/env python3
"""
Project scanner for the hybrid agent.

Builds context about the project structure, dependencies, and entry points and
stores it in a JSON file (default hybrid-agent/context.json) that the agent can
use for smarter routing and suggestions.

Usage:
    python3 hybrid-agent/scan.py --project-root . [--output context.json]
    python3 hybrid-agent/scan.py --project-root . --suggest-tasks
    python3 hybrid-agent/scan.py --project-root . --update
    python3 hybrid-agent/scan.py --project-root . --json
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Directories that must never be scanned: they are machine-specific, generated,
# or third-party and would produce huge, useless results.
IGNORE_DIRS = {
    ".venv", "venv", "env",
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    ".next", ".nuxt", ".cache", "build", "dist", ".tox", ".mypy_cache",
}


class ProjectScanner:
    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self.context = {
            "project_root": str(self.root),
            "scan_time": datetime.now().isoformat(),
            "files": {},
            "dependencies": {},
            "structure": {},
            "entry_points": [],
            "suggestions": [],
        }

    def _is_ignored(self, path: Path) -> bool:
        """True if any part of the path's relative components is an ignored dir."""
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return True
        return any(part in IGNORE_DIRS for part in rel.parts[:-1])

    def _matching_files(self, patterns: List[str]) -> List[str]:
        """All files matching any pattern, excluding ignored dirs, sorted."""
        found: List[Path] = []
        for pattern in patterns:
            for p in self.root.rglob(pattern):
                if p.is_file() and not self._is_ignored(p):
                    found.append(p)
        # de-dup while preserving order
        seen = set()
        unique = []
        for p in found:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        unique.sort()
        return [str(p.relative_to(self.root)) for p in unique]

    def scan_files(self) -> Dict:
        """Scan project files by type."""
        file_types = {
            "python": ["*.py", "*.pyi"],
            "javascript": ["*.js", "*.mjs", "*.cjs"],
            "typescript": ["*.ts", "*.tsx"],
            "react": ["*.jsx", "*.tsx"],
            "go": ["*.go"],
            "rust": ["*.rs"],
            "c": ["*.c", "*.h"],
            "cpp": ["*.cpp", "*.hpp", "*.cc"],
            "java": ["*.java"],
            "ruby": ["*.rb"],
            "php": ["*.php"],
            "html": ["*.html", "*.htm"],
            "css": ["*.css", "*.scss", "*.sass"],
            "markdown": ["*.md"],
            "config": ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"],
        }

        result = {}
        for lang, patterns in file_types.items():
            files = self._matching_files(patterns)
            if files:
                result[lang] = files[:10]  # limit for readability

        self.context["files"] = result
        return result

    def detect_dependencies(self) -> Dict:
        """Detect project dependencies from common files."""
        deps: Dict = {}

        # Python
        req_file = self.root / "requirements.txt"
        if req_file.exists():
            with open(req_file) as f:
                deps["python"] = [line.strip() for line in f
                                  if line.strip() and not line.startswith("#")]

        if (self.root / "pyproject.toml").exists():
            deps["python_poetry"] = "pyproject.toml found"

        # Node/JavaScript
        package_file = self.root / "package.json"
        if package_file.exists():
            try:
                with open(package_file) as f:
                    data = json.load(f)
                deps["node"] = {
                    "dependencies": list(data.get("dependencies", {}).keys()),
                    "devDependencies": list(data.get("devDependencies", {}).keys()),
                    "scripts": list(data.get("scripts", {}).keys()),
                }
            except (json.JSONDecodeError, OSError):
                deps["node"] = "package.json present but unreadable"

        # Go
        go_mod = self.root / "go.mod"
        if go_mod.exists():
            with open(go_mod) as f:
                deps["go"] = [line.strip() for line in f if line.startswith("require")]

        # Rust
        if (self.root / "Cargo.toml").exists():
            deps["rust"] = "Cargo.toml found"

        self.context["dependencies"] = deps
        return deps

    def detect_structure(self) -> Dict:
        """Detect project structure patterns."""
        structure = {}

        patterns = {
            "src": self.root / "src",
            "tests": self.root / "tests",
            "docs": self.root / "docs",
            "examples": self.root / "examples",
            "scripts": self.root / "scripts",
            "assets": self.root / "assets",
            "static": self.root / "static",
            "templates": self.root / "templates",
        }
        for name, path in patterns.items():
            if path.exists():
                structure[name] = str(path.relative_to(self.root))

        # Detect framework
        if (self.root / "manage.py").exists():
            structure["framework"] = "Django"
        elif (self.root / "app.py").exists() and (self.root / "templates").exists():
            structure["framework"] = "Flask"
        elif (self.root / "package.json").exists():
            try:
                data = json.loads((self.root / "package.json").read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
            blob = json.dumps(data).lower()
            if "react" in blob:
                structure["framework"] = "React"
            elif "next" in blob:
                structure["framework"] = "Next.js"
            elif "vue" in blob:
                structure["framework"] = "Vue"

        self.context["structure"] = structure
        return structure

    def detect_entry_points(self) -> List[str]:
        """Find likely entry point files."""
        entry_points = []
        candidates = [
            "main.py", "app.py", "index.py",
            "index.js", "app.js", "server.js", "main.js",
            "index.ts", "main.ts",
            "main.go", "main.rs",
            "index.html", "index.php",
        ]

        for candidate in candidates:
            if (self.root / candidate).exists():
                entry_points.append(candidate)

        src_dir = self.root / "src"
        if src_dir.exists():
            for candidate in candidates:
                if (src_dir / candidate).exists():
                    entry_points.append(f"src/{candidate}")

        self.context["entry_points"] = entry_points
        return entry_points

    def git_info(self) -> Dict:
        """Get git repository information."""
        try:
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=self.root, capture_output=True, text=True,
            )
            recent_commits = log.stdout.strip().split("\n") if log.stdout else []
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.root, capture_output=True, text=True,
            ).stdout.strip()
            return {"branch": branch, "recent_commits": recent_commits, "has_git": True}
        except Exception:
            return {"has_git": False}

    def suggest_tasks(self) -> List[str]:
        """Suggest tasks based on project analysis."""
        suggestions = []
        deps = self.context.get("dependencies", {})

        if "python" in deps:
            suggestions.append("Update dependencies in requirements.txt")
            if "pytest" in str(deps):
                suggestions.append("Write tests for the main module")

        if "node" in deps:
            suggestions.append("Update npm dependencies")
            node_deps = deps.get("node", {})
            if isinstance(node_deps, dict) and "react" in node_deps.get("dependencies", []):
                suggestions.append("Add a new React component")

        structure = self.context.get("structure", {})
        if "tests" not in structure:
            suggestions.append("Create a tests directory")
        if "docs" not in structure:
            suggestions.append("Create a docs directory")

        git = self.context.get("git", {})
        if git.get("has_git") and git.get("recent_commits"):
            suggestions.append("Review recent changes and update documentation")

        self.context["suggestions"] = suggestions
        return suggestions

    def scan(self, suggest: bool = False) -> Dict:
        """Run full project scan."""
        self.scan_files()
        self.detect_dependencies()
        self.detect_structure()
        self.detect_entry_points()
        self.context["git"] = self.git_info()
        if suggest:
            self.suggest_tasks()
        return self.context

    def save(self, output_file: str = "context.json") -> None:
        """Save context to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.context, f, indent=2)
        print(f"Context saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project scanner for hybrid agent")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "context.json"),
        help="Output JSON file (default: hybrid-agent/context.json next to scan.py)",
    )
    parser.add_argument("--suggest-tasks", action="store_true",
                        help="Suggest tasks based on project analysis")
    parser.add_argument("--update", action="store_true",
                        help="Re-scan and refresh the context file")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout")
    args = parser.parse_args()

    scanner = ProjectScanner(args.project_root)
    context = scanner.scan(suggest=args.suggest_tasks)

    if args.json:
        print(json.dumps(context, indent=2))
    else:
        scanner.save(args.output)
        print("Project Summary:")
        print(f"  Files: {sum(len(files) for files in context['files'].values())}")
        print(f"  Languages: {list(context['files'].keys())}")
        print(f"  Entry points: {context.get('entry_points', [])}")
        print(f"  Framework: {context.get('structure', {}).get('framework', 'Unknown')}")
        if args.suggest_tasks:
            print("Suggested tasks:")
            for i, suggestion in enumerate(context.get("suggestions", []), 1):
                print(f"  {i}. {suggestion}")


if __name__ == "__main__":
    main()
