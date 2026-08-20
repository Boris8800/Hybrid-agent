"""Project Context Awareness module for hybrid-agent.

Scans a project and renders a compact prompt so DeepSeek/Qwen understand the
codebase (structure, dependencies, architecture, coding standards, code
examples) before implementing. Self-contained: stdlib only.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


_IGNORE_DIRS = {
    '.venv', 'venv', 'env', 'node_modules', '.git', '__pycache__',
    '.pytest_cache', '.next', '.nuxt', '.cache', 'build', 'dist',
    '.tox', '.mypy_cache', '.idea', '.vscode',
}

_SOURCE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb', '.php',
    '.java', '.html', '.css', '.scss', '.json', '.md', '.yml', '.yaml',
}

_MAX_FILE_SIZE = 200_000  # bytes


def _is_ignored(rel_path: Path) -> bool:
    """True if any part of the relative path is an ignored/hidden dir."""
    for part in rel_path.parts:
        if part in _IGNORE_DIRS or part.startswith('.'):
            return True
    return False


def _safe_read(path: Path, max_size: int = _MAX_FILE_SIZE) -> Optional[str]:
    """Read a file safely; None on error or if too large."""
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > max_size:
            return None
        return path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None


class ProjectContext:
    """Scans a project and builds a context dictionary for prompt injection."""

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.context: Dict = {}
        self._relevant_files: Optional[List[str]] = None

    # -- Public API ---------------------------------------------------------

    def scan(self) -> dict:
        """Perform a full scan and populate self.context."""
        # Assign progressively: each step reads the fields the previous step
        # produced (e.g. architecture detection needs dependencies). A single
        # dict-literal would evaluate all values BEFORE self.context is set.
        self.context = {'root': str(self.root)}
        self.context['structure'] = self._scan_structure()
        self.context['dependencies'] = self._read_dependencies()
        self.context['architecture'] = self._detect_architecture()
        self.context['coding_standards'] = self._detect_coding_standards()
        self.context['examples'] = self._collect_examples()
        self.context['estimate_tokens'] = estimate_tokens(self.to_prompt())
        return self.context

    def find_relevant_files(self, task: str, limit: int = 5) -> List[str]:
        """Score files by keyword overlap with the task; return top matches."""
        task_words = set(re.findall(r'\b[a-z0-9_]+\b', task.lower()))
        if not task_words:
            return []

        scored: List[Tuple[int, str]] = []
        files = self.context.get('structure', {}).get('files', [])

        for rel_path_str in files:
            full_path = self.root / rel_path_str
            try:
                if full_path.stat().st_size > _MAX_FILE_SIZE:
                    continue
            except Exception:
                continue

            # Score from path (weighted higher).
            path_words = set(re.findall(r'\b[a-z0-9_]+\b', rel_path_str.lower()))
            score = len(task_words & path_words) * 2

            # Score from content (best-effort).
            content = _safe_read(full_path, max_size=50_000)
            if content:
                content_words = set(re.findall(r'\b[a-z0-9_]+\b', content.lower()))
                score += len(task_words & content_words)

            if score > 0:
                scored.append((score, rel_path_str))

        scored.sort(key=lambda x: (-x[0], x[1]))
        self._relevant_files = [f for _, f in scored[:limit]]
        return self._relevant_files

    def recommend_libraries(self, task: str) -> List[str]:
        """Dependency-aware library suggestions based on task keywords."""
        suggestions: List[str] = []
        deps = self.context.get('dependencies', {})
        all_deps: Set[str] = set()
        for lang_deps in deps.values():
            all_deps.update(lang_deps.keys())
        task_lower = task.lower()

        if 'auth' in task_lower or 'login' in task_lower or 'jwt' in task_lower:
            if not any('jwt' in d or 'auth' in d or 'passport' in d for d in all_deps):
                suggestions.append('jwt (JSON Web Token) for authentication')

        if ('database' in task_lower or 'orm' in task_lower or 'model' in task_lower
                or 'query' in task_lower or 'sql' in task_lower):
            if not any(d in all_deps for d in ['sqlalchemy', 'prisma', 'sequelize', 'mongoose']):
                suggestions.append('SQLAlchemy (Python) or Prisma (Node) for ORM')

        if 'form' in task_lower and any(d in all_deps for d in ['react', 'next']):
            if 'formik' not in all_deps and 'react-hook-form' not in all_deps:
                suggestions.append('formik or react-hook-form for form handling')

        if 'test' in task_lower or 'testing' in task_lower:
            if 'pytest' not in all_deps and 'jest' not in all_deps:
                suggestions.append('pytest (Python) or jest (Node) for testing')

        if 'http' in task_lower or 'api' in task_lower or 'request' in task_lower:
            if not any(d in all_deps for d in ['requests', 'axios', 'fetch']):
                suggestions.append('requests (Python) or axios (Node) for HTTP calls')

        return suggestions

    def to_prompt(self) -> str:
        """Render a compact markdown prompt from the context."""
        if not self.context:
            self.scan()

        lines = ['## Project Context', '']
        lines.append(f'**Root:** `{self.context["root"]}`')
        lines.append('')

        lines.append('### Structure')
        structure = self.context.get('structure', {})
        dirs = structure.get('directories', [])
        files = structure.get('files', [])
        if dirs:
            lines.append('**Directories:** ' + ', '.join(dirs[:20]))
        if files:
            lines.append('**Files:** ' + ', '.join(files[:30]))
        lines.append('')

        lines.append('### Dependencies')
        deps = self.context.get('dependencies', {})
        if deps:
            for lang, packages in deps.items():
                if packages:
                    lines.append('**%s:** %s' % (lang, ', '.join(
                        ('%s==%s' % (k, v)) if v != 'latest' else k
                        for k, v in list(packages.items())[:15]
                    )))
        else:
            lines.append('_No dependencies detected_')
        lines.append('')

        lines.append('### Architecture')
        arch = self.context.get('architecture', {})
        if arch:
            for key, value in arch.items():
                if value:
                    lines.append('- **%s:** %s' % (key.replace('_', ' ').title(), value))
        else:
            lines.append('_Unknown_')
        lines.append('')

        lines.append('### Coding Standards')
        standards = self.context.get('coding_standards', {})
        if standards:
            for key, value in standards.items():
                lines.append('- **%s:** %s' % (key.replace('_', ' ').title(), value))
        lines.append('')

        lines.append('### Code Examples')
        examples = self.context.get('examples', {})
        for label in ['model', 'route', 'test']:
            code = examples.get(label)
            if code:
                lines.append('**%s:**' % label.title())
                lines.append('```')
                lines.append(code)
                lines.append('```')
        lines.append('')

        if self._relevant_files:
            lines.append('### Relevant Files')
            for f in self._relevant_files:
                lines.append('- `%s`' % f)
            lines.append('')

        prompt = '\n'.join(lines)
        if estimate_tokens(prompt) > 1500:
            prompt = prompt[:6000] + '\n... (truncated)'
        return prompt

    # -- Internal scan methods ----------------------------------------------

    def _scan_structure(self) -> dict:
        """Return top-level dirs and source files (ignoring build/vendored dirs)."""
        directories: Set[str] = set()
        files: List[str] = []

        try:
            for root, dirs, filenames in os_walk_skip_hidden(self.root):
                rel_root = Path(root).relative_to(self.root)
                if _is_ignored(rel_root):
                    dirs[:] = []
                    continue
                if len(rel_root.parts) == 1:
                    directories.add(rel_root.parts[0])
                for filename in filenames:
                    rel_path = rel_root / filename
                    if _is_ignored(rel_path):
                        continue
                    if rel_path.suffix.lower() not in _SOURCE_EXTENSIONS:
                        continue
                    full_path = self.root / rel_path
                    try:
                        if full_path.stat().st_size > _MAX_FILE_SIZE:
                            continue
                    except Exception:
                        continue
                    files.append(str(rel_path))
        except Exception:
            pass

        return {
            'directories': sorted(directories)[:50],
            'files': sorted(files)[:150],
        }

    def _read_dependencies(self) -> dict:
        """Parse common dependency files; return {lang: {name: version}}.

        Reads the root manifest plus any found in immediate subdirectories
        (depth <= 2, ignoring vendored dirs), so monorepo layouts where the
        app lives under a subfolder are still detected.
        """
        manifests: List[Path] = []
        for manifest in ('requirements.txt', 'pyproject.toml', 'package.json',
                         'go.mod', 'Cargo.toml'):
            candidate = self.root / manifest
            if candidate.is_file():
                manifests.append(candidate)
        # Depth-limited sub-directory manifests.
        try:
            for sub in self.root.iterdir():
                if not sub.is_dir() or _is_ignored(Path(sub.name)):
                    continue
                for manifest in ('package.json', 'requirements.txt',
                                 'pyproject.toml', 'go.mod', 'Cargo.toml'):
                    candidate = sub / manifest
                    if candidate.is_file():
                        manifests.append(candidate)
        except Exception:
            pass

        deps: Dict[str, Dict[str, str]] = {}
        seen_files: Set[str] = set()
        for path in manifests:
            key = str(path)
            if key in seen_files:
                continue
            seen_files.add(key)
            self._parse_manifest(path, deps)
        return deps

    def _parse_manifest(self, path: Path, deps: Dict[str, Dict[str, str]]) -> None:
        """Parse a single manifest file into the deps dict."""
        name = path.name
        if name == 'requirements.txt':
            content = _safe_read(path)
            if not content:
                return
            py_deps = deps.setdefault('python', {})
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                if '==' in line:
                    pkg, version = line.split('==', 1)
                    py_deps.setdefault(pkg.strip(), version.strip())
                elif '>=' in line:
                    pkg, version = line.split('>=', 1)
                    py_deps.setdefault(pkg.strip(), version.strip())
                else:
                    py_deps.setdefault(line, 'latest')
        elif name == 'pyproject.toml':
            content = _safe_read(path)
            if not content:
                return
            try:
                in_deps = False
                py_deps = deps.setdefault('python', {})
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('[project]') or line.startswith('[tool.poetry.dependencies]'):
                        in_deps = True
                        continue
                    if line.startswith('[') and in_deps:
                        in_deps = False
                    if in_deps and '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        pkg = parts[0].strip().strip('"').strip("'")
                        version = parts[1].strip().strip('"').strip("'")
                        if pkg and pkg not in ('python', 'name', 'version'):
                            py_deps.setdefault(pkg, version)
            except Exception:
                pass
        elif name == 'package.json':
            content = _safe_read(path)
            if not content:
                return
            try:
                data = json.loads(content)
                js_deps = deps.setdefault('node', {})
                for section in ('dependencies', 'devDependencies'):
                    for pkg, version in data.get(section, {}).items():
                        js_deps.setdefault(pkg, version)
            except Exception:
                pass
        elif name == 'go.mod':
            content = _safe_read(path)
            if not content:
                return
            go_deps = deps.setdefault('go', {})
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('require') or line.startswith('\t'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] != 'require':
                        go_deps.setdefault(parts[0], parts[1] if len(parts) > 1 else 'latest')
        elif name == 'Cargo.toml':
            content = _safe_read(path)
            if not content:
                return
            try:
                in_deps = False
                rust_deps = deps.setdefault('rust', {})
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('[dependencies]'):
                        in_deps = True
                        continue
                    if line.startswith('[') and in_deps:
                        in_deps = False
                    if in_deps and '=' in line:
                        parts = line.split('=', 1)
                        pkg = parts[0].strip().strip('"').strip("'")
                        version = parts[1].strip().strip('"').strip("'")
                        if pkg:
                            rust_deps.setdefault(pkg, version)
            except Exception:
                pass

    def _detect_architecture(self) -> dict:
        """Detect architecture from dependencies and file heuristics."""
        arch: Dict[str, str] = {}
        deps = self.context.get('dependencies', {})
        all_deps: Set[str] = set()
        for lang_deps in deps.values():
            all_deps.update(lang_deps.keys())

        if any(d in all_deps for d in ['express', 'koa', 'nest']):
            arch['backend_framework'] = 'NestJS' if 'nest' in all_deps \
                else ('Koa' if 'koa' in all_deps else 'Express')
        elif 'fastapi' in all_deps:
            arch['backend_framework'] = 'FastAPI'
        elif 'flask' in all_deps:
            arch['backend_framework'] = 'Flask'
        elif (self.root / 'manage.py').exists():
            arch['backend_framework'] = 'Django'

        if 'next' in all_deps:
            arch['frontend_framework'] = 'Next.js'
        elif 'react' in all_deps:
            arch['frontend_framework'] = 'React'
        elif 'vue' in all_deps:
            arch['frontend_framework'] = 'Vue'
        elif 'angular' in all_deps:
            arch['frontend_framework'] = 'Angular'

        if 'prisma' in all_deps:
            arch['orm'] = 'Prisma'
        elif 'sqlalchemy' in all_deps:
            arch['orm'] = 'SQLAlchemy'
        elif 'sequelize' in all_deps:
            arch['orm'] = 'Sequelize'

        if any(d in all_deps for d in ['postgres', 'pg']):
            arch['database'] = 'PostgreSQL'
        elif 'mysql' in all_deps:
            arch['database'] = 'MySQL'
        elif 'sqlite' in all_deps:
            arch['database'] = 'SQLite'
        elif any(d in all_deps for d in ['mongodb', 'mongoose']):
            arch['database'] = 'MongoDB'

        if 'backend_framework' in arch and 'frontend_framework' in arch:
            arch['type'] = 'fullstack'
        elif 'backend_framework' in arch:
            arch['type'] = 'backend'
        elif 'frontend_framework' in arch:
            arch['type'] = 'frontend'
        else:
            arch['type'] = 'unknown'

        dirs = self.context.get('structure', {}).get('directories', [])
        layers = []
        if any('controller' in d.lower() for d in dirs):
            layers.append('controllers')
        if any('service' in d.lower() for d in dirs):
            layers.append('services')
        if any('model' in d.lower() for d in dirs):
            layers.append('models')
        if layers:
            arch['layers'] = ' -> '.join(layers)

        return arch

    def _detect_coding_standards(self) -> dict:
        """Sample files to detect coding standards."""
        files = self.context.get('structure', {}).get('files', [])
        python_files = [f for f in files if f.endswith('.py')]
        js_files = [f for f in files if f.endswith(('.js', '.ts', '.tsx', '.jsx'))]
        sample_files = python_files[:6] if python_files else js_files[:6]
        empty = {
            'type_hints': False, 'docstrings': False, 'import_style': 'unknown',
            'quote_style': 'unknown', 'semicolons': False,
        }
        if not sample_files:
            return empty

        type_hints = docstrings = semicolons = total = 0
        import_styles: List[str] = []
        quote_styles: List[str] = []

        for rel_path in sample_files:
            content = _safe_read(self.root / rel_path)
            if not content:
                continue
            total += 1

            if rel_path.endswith('.py'):
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if node.returns or any(a.annotation for a in node.args.args):
                                type_hints += 1
                            break
                except Exception:
                    pass
                if '"""' in content or "'''" in content:
                    docstrings += 1
                try:
                    tree = ast.parse(content)
                    imports = [n for n in ast.walk(tree)
                               if isinstance(n, (ast.Import, ast.ImportFrom))]
                    if len(imports) > 1:
                        lines = content.splitlines()
                        import_lines = [i for i, line in enumerate(lines)
                                        if line.startswith('import ') or line.startswith('from ')]
                        if import_lines:
                            gaps = sum(1 for i in range(1, len(import_lines))
                                       if import_lines[i] - import_lines[i - 1] > 1)
                            import_styles.append('grouped' if gaps > 0 else 'single')
                        else:
                            import_styles.append('unknown')
                    else:
                        import_styles.append('single')
                except Exception:
                    import_styles.append('unknown')

            single_quotes = content.count("'")
            double_quotes = content.count('"')
            quote_styles.append('single' if single_quotes > double_quotes
                                else ('double' if double_quotes > single_quotes else 'unknown'))

            if rel_path.endswith(('.js', '.ts', '.tsx', '.jsx')):
                lines = content.splitlines()
                semi = sum(1 for line in lines if line.rstrip().endswith(';'))
                if semi > len(lines) * 0.3:
                    semicolons += 1

        if total == 0:
            return empty

        return {
            'type_hints': type_hints > total // 2,
            'docstrings': docstrings > total // 2,
            'import_style': max(set(import_styles), key=import_styles.count)
            if import_styles else 'unknown',
            'quote_style': max(set(quote_styles), key=quote_styles.count)
            if quote_styles else 'unknown',
            'semicolons': semicolons > total // 2,
        }

    def _collect_examples(self) -> dict:
        """Extract code examples from model, route, and test files."""
        files = self.context.get('structure', {}).get('files', [])

        def first_match(patterns):
            for f in files:
                if any(re.search(p, f) for p in patterns):
                    return f
            return None

        model = self._extract_example(
            first_match([r'models?/.*\.py$', r'models?\.py$']), 'class')
        route = self._extract_example(
            first_match([r'routes?/.*\.py$', r'routes?\.py$', r'controllers?/.*\.py$']),
            'function')
        test = self._extract_example(
            first_match([r'test_.*\.py$', r'.*_test\.py$']), 'test')

        return {
            'model': model,
            'route': route,
            'test': test,
        }

    def _extract_example(self, rel_path: Optional[str], kind: str) -> str:
        """Extract a code snippet from a file by kind ('class'|'function'|'test')."""
        if not rel_path:
            return ''
        content = _safe_read(self.root / rel_path)
        if not content:
            return ''
        try:
            tree = ast.parse(content)
            if kind == 'class':
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        return ast.unparse(node)[:600]
            elif kind == 'test':
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and node.name.startswith('test_'):
                        return ast.unparse(node)[:600]
            else:  # function (route handler)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return ast.unparse(node)[:600]
        except Exception:
            pass
        return ''


def os_walk_skip_hidden(root: Path):
    """os.walk but pruning hidden directories to avoid vendored/cache dirs."""
    import os as _os
    for root_dir, dirs, filenames in _os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in _IGNORE_DIRS]
        yield root_dir, dirs, filenames
