"""parallel.py — Parallel step execution for the hybrid agent.

Pure stdlib implementation (typing, concurrent.futures, re, threading).
No third-party dependencies.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from backends.base import ModelRequest


# ---------------------------------------------------------------------------
# 1) Plan parsing
# ---------------------------------------------------------------------------

def parse_plan_steps(plan_text: str) -> List[Dict]:
    """Parse a DeepSeek plan block into a list of step dicts.

    Tolerant: never crashes, falls back to a single whole-task step.
    """
    if not plan_text or not isinstance(plan_text, str):
        return []

    lines = plan_text.splitlines()
    steps: List[Dict] = []
    current_step: Optional[Dict] = None

    step_header_re = re.compile(r'^(?:Step|STEP)\s*(\d+)\s*:?\s*(.*)$')
    numbered_re = re.compile(r'^(\d+)\.\s+(.*)$')
    marker_re = re.compile(r'^===\s*STEP\s*(\d+)\s*:?\s*(.*?)\s*===?$', re.IGNORECASE)
    deps_re = re.compile(r'^(?:Dependencies|Depends on)\s*:?\s*(.*)$', re.IGNORECASE)
    files_re = re.compile(r'^Files?\s*:?\s*(.*)$', re.IGNORECASE)
    desc_re = re.compile(r'^Description\s*:?\s*(.*)$', re.IGNORECASE)

    def flush_current():
        nonlocal current_step
        if current_step is not None:
            current_step.setdefault('id', len(steps) + 1)
            current_step.setdefault('name', f"Step {current_step['id']}")
            current_step.setdefault('description', '')
            current_step.setdefault('files', [])
            current_step.setdefault('dependencies', [])
            steps.append(current_step)
            current_step = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = marker_re.match(line)
        if m:
            flush_current()
            step_id = int(m.group(1))
            current_step = {
                'id': step_id,
                'name': m.group(2).strip() or f"Step {step_id}",
                'description': '', 'files': [], 'dependencies': [],
            }
            continue

        m = step_header_re.match(line)
        if m:
            flush_current()
            step_id = int(m.group(1))
            current_step = {
                'id': step_id,
                'name': m.group(2).strip() or f"Step {step_id}",
                'description': '', 'files': [], 'dependencies': [],
            }
            continue

        if current_step is None:
            m = numbered_re.match(line)
            if m:
                step_id = int(m.group(1))
                current_step = {
                    'id': step_id,
                    'name': m.group(2).strip() or f"Step {step_id}",
                    'description': '', 'files': [], 'dependencies': [],
                }
                continue
            continue  # header/non-step line before any step

        # Inside a step: parse metadata / accumulate description.
        m = deps_re.match(line)
        if m:
            deps_str = m.group(1).strip()
            current_step['dependencies'] = [int(x) for x in re.findall(r'\d+', deps_str)]
            continue
        m = files_re.match(line)
        if m:
            files_str = m.group(1).strip()
            current_step['files'] = [f.strip() for f in re.split(r'[,\s]+', files_str)
                                     if f.strip()]
            continue
        m = desc_re.match(line)
        if m:
            current_step['description'] = m.group(1).strip()
            continue
        if current_step['description']:
            current_step['description'] += '\n' + line
        else:
            current_step['description'] = line

    flush_current()

    if not steps:
        return [{
            'id': 1, 'name': 'whole task', 'description': plan_text,
            'files': [], 'dependencies': [],
        }]
    return steps


# ---------------------------------------------------------------------------
# 2) DependencyAnalyzer
# ---------------------------------------------------------------------------

class DependencyAnalyzer:
    """Topological level-order grouping of steps based on dependencies."""

    def __init__(self, steps: List[Dict]):
        self.steps = steps
        self.steps_by_id: Dict[int, Dict] = {}
        self.dependencies: Dict[int, List[int]] = {}
        self.dependents: Dict[int, List[int]] = {}
        for step in steps:
            sid = step.get('id')
            if sid is None:
                continue
            self.steps_by_id[sid] = step
            deps = [d for d in (step.get('dependencies', []) or []) if d in self.steps_by_id]
            self.dependencies[sid] = deps
            for dep in deps:
                self.dependents.setdefault(dep, []).append(sid)

    def get_parallel_groups(self) -> List[List[Dict]]:
        """Topological level-order groups; steps within a group can run in parallel."""
        if not self.steps:
            return []

        all_ids = set(self.steps_by_id.keys())
        scheduled: set = set()
        groups: List[List[Dict]] = []
        remaining_deps = {sid: set(self.dependencies.get(sid, [])) for sid in all_ids}

        for _ in range(len(all_ids) + 1):  # cap to avoid infinite loop
            ready = [sid for sid in all_ids
                     if sid not in scheduled and not remaining_deps.get(sid, set())]
            if not ready:
                break
            group = []
            for sid in ready:
                scheduled.add(sid)
                group.append(self.steps_by_id[sid])
                for dependent in self.dependents.get(sid, []):
                    if dependent in remaining_deps:
                        remaining_deps[dependent].discard(sid)
            groups.append(group)

        remaining = [sid for sid in all_ids if sid not in scheduled]
        if remaining:  # cycle / unresolvable: run sequentially
            groups.append([self.steps_by_id[sid] for sid in remaining])
        return groups

    def has_dependencies(self, group: List[Dict]) -> bool:
        """True if the group has >1 step (parallelizable)."""
        return len(group) > 1


# ---------------------------------------------------------------------------
# 3) ParallelExecutor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Execute steps in parallel using a thread pool."""

    def __init__(self, qwen_generate: Callable, max_workers: int = 4,
                 max_tokens: int = 4096, temperature: float = 0.2,
                 timeout_s: float = 180.0):
        self.qwen_generate = qwen_generate
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._lock = threading.Lock()

    def build_step_request(self, step: Dict, task: str,
                           project_context: str = '') -> ModelRequest:
        """Build a ModelRequest for a single step."""
        system = (
            "You are a precise engineer. Implement ONLY this step. "
            "Output complete files in fenced code blocks labeled with paths. "
            "Do not implement other steps."
        )
        step_name = step.get('name', f"Step {step.get('id', '?')}")
        step_desc = step.get('description', '')
        files = step.get('files', [])
        files_str = '\n'.join(files) if files else '(none specified)'
        user = (f"{task}\n\nPROJECT CONTEXT:\n{project_context}\n\n"
                f"STEP:\n{step_name}\n{step_desc}\n\nFILES:\n{files_str}")
        return ModelRequest(
            system=system, user=user, max_tokens=self.max_tokens,
            temperature=self.temperature, timeout_s=self.timeout_s,
        )

    def execute_parallel(self, steps: List[Dict], task: str,
                         project_context: str = '') -> List[Dict]:
        """Run each step in parallel; return results in input order."""
        if not steps:
            return []
        workers = max(1, min(len(steps), self.max_workers))
        results: List[Optional[Dict]] = [None] * len(steps)

        def run_step(index: int, step: Dict) -> Dict:
            try:
                request = self.build_step_request(step, task, project_context)
                resp = self.qwen_generate(request)
                text = getattr(resp, 'text', '') or ''
                return {'step': step.get('name', 'step'), 'id': step.get('id'),
                        'status': 'success', 'text': text}
            except Exception as exc:  # noqa: BLE001
                return {'step': step.get('name', 'step'), 'id': step.get('id'),
                        'status': 'failed', 'error': str(exc)}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(run_step, i, step): i
                for i, step in enumerate(steps)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[idx] = {'step': 'unknown', 'id': None,
                                    'status': 'failed', 'error': str(exc)}

        return [r if r is not None else {'step': 'unknown', 'id': None,
                                         'status': 'failed', 'error': 'no result'}
                for r in results]


# ---------------------------------------------------------------------------
# 4) Summarize
# ---------------------------------------------------------------------------

def summarize(results: List[Dict]) -> str:
    """Return a short human summary of execution results."""
    if not results:
        return "0 steps OK, 0 failed · 0 tokens"
    ok = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
    failed = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'failed')
    total_chars = sum(len(r.get('text', '') or '') for r in results if isinstance(r, dict))
    return f"{ok} steps OK, {failed} failed · {total_chars // 4 // 1000}k tokens"
