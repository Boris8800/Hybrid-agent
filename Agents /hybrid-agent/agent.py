"""
Hybrid coding agent orchestrator (ARCHITECTURE.md §9-§10).

P0 skeleton: routing (archetypes + confidence + breaker), backend clients,
and local→DeepSeek escalation. P2+ adds the diff builder, always-on DeepSeek
review, testing, and memory.

Run:  python agent.py --task "add validation to validate_email"
      python agent.py --task "make checkout more robust" --route-only
"""

import argparse
import importlib
import os
import time

from backends.base import Backend, ModelRequest
from backends.circuit_breaker import CircuitBreaker
from backends.deepseek import DeepSeekBackend
from backends.local_gemma import GemmaBackend
from memory import TaskMemory, TaskRecord, memory_root_from_cfg
from embed import memory_embed_callable
from providers import backend_for, get_local, get_online, load_providers
from router import archetypes
from router.confidence import MemoryView, score
from router.threshold import ThresholdController

# Inline defaults so P0 runs with zero config files. `--config config.yml`
# overrides these when a file exists (see ARCHITECTURE.md §10).
DEFAULT_CONFIG = {
    "router": {
        "local_threshold": 0.85,
        "threshold_min": 0.60,
        "threshold_max": 0.85,
        "target_local_rate": 0.93,
        "alpha": 0.01,
        "weights": {"clarity": 0.30, "specificity": 0.25, "size_penalty": 0.15,
                    "history_bonus": 0.20, "novelty": 0.10},
    },
    "backends": {
        "local": {
            "base_url": "http://localhost:1234/api/v1",
            "api_key": "lm-studio",
            "model": "qwen2.5-coder-14b-instruct-mlx",
            "timeout_s": 180,
            "max_retries": 3,
            "backoff_s": [0.5, 1.0, 2.0],
            "cold_start_wait_s": 30,
        },
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "timeout_s": 30,
            "max_retries": 4,
            "backoff_s": [1.0, 2.0, 4.0],
        },
    },
    "review": {"max_local_retries": 2, "max_depth_tokens": 600,
                "max_failure_summary_words": 150, "daily_token_budget": 200000,
                "verify": [], "verify_timeout": 600,
                "regression": [], "regression_timeout": 600},
    "circuit_breaker": {"window_size": 20, "local_error_ceiling": 0.40,
                         "deepseek_error_ceiling": 0.30, "cooldown_s": 60},
    "memory": {"root": "./memory", "max_project_summary_words": 200},
}


class HybridAgent:
    def __init__(self, cfg: dict, online_provider: str | None = None,
                 local_provider: str | None = None):
        self.cfg = cfg
        self.online_provider = online_provider
        self.local_provider = local_provider
        r_cfg = cfg["router"]
        self.threshold = ThresholdController(
            threshold=r_cfg["local_threshold"],
            threshold_min=r_cfg["threshold_min"],
            threshold_max=r_cfg["threshold_max"],
            target_local_rate=r_cfg["target_local_rate"],
            alpha=r_cfg["alpha"],
        )
        cb = cfg["circuit_breaker"]
        self.local_breaker = CircuitBreaker("local", cb["local_error_ceiling"], cb["window_size"])
        self.cloud_breaker = CircuitBreaker("cloud", cb["deepseek_error_ceiling"], cb["window_size"])
        self._local: Backend | None = None
        self._cloud: Backend | None = None

    def _backends(self) -> tuple[Backend, Backend]:
        if self._local is None or self._cloud is None:
            load_providers(self.cfg)  # validates the provider config shape
            lp = get_local(self.cfg, name=self.local_provider)
            op = get_online(self.cfg, name=self.online_provider)
            if lp is None and self.local_provider:
                lp = get_local(self.cfg)
            if op is None and self.online_provider:
                op = get_online(self.cfg)
            if lp is None or op is None:
                # No enabled providers: fall back to the legacy backends section.
                lc = self.cfg["backends"]["local"]
                dc = self.cfg["backends"]["deepseek"]
                self._local = GemmaBackend(
                    lc["base_url"], lc["model"],
                    api_key=lc.get("api_key") or "lm-studio",
                    timeout_s=lc["timeout_s"], max_retries=lc["max_retries"],
                )
                self._cloud = DeepSeekBackend(
                    dc["api_key_env"], dc["model"],
                    base_url=dc.get("base_url") or "https://api.deepseek.com",
                    timeout_s=dc["timeout_s"], max_retries=dc["max_retries"],
                )
            else:
                self._local = backend_for(lp)
                self._cloud = backend_for(op)
        return self._local, self._cloud

    # --- routing --------------------------------------------------------
    def route(self, task: str, context_chars: int, memory: MemoryView) -> tuple[str, str]:
        """Returns (route, reason) — hard pins beat confidence, breakers override."""
        cls = archetypes.classify(task)
        if cls.is_deepseek_pinned:
            return "deepseek", f"archetype:{cls.primary}"
        if cls.is_local_pinned:
            return "local", f"archetype:{cls.primary}"

        # Ambiguous (A11): breaker checks, then confidence score.
        if self.local_breaker.open:
            return "deepseek", "breaker:local"
        if self.cloud_breaker.open:
            return "local", "breaker:cloud"

        conf = score(task, context_chars, memory)
        return self.threshold.decide(conf), f"confidence:{conf:.2f}"

    # --- execution ------------------------------------------------------
    def run(self, task: str, context_chars: int = 2000,
            memory: MemoryView | None = None) -> dict:
        memory = memory or TaskMemory(
            memory_root_from_cfg(self.cfg),
            embed=memory_embed_callable(self.cfg)).memory_view(task)
        route, reason = self.route(task, context_chars, memory)
        local, cloud = self._backends()

        result = {"task": task, "route": route, "reason": reason, "text": ""}
        local_ok = True
        if route == "local":
            try:
                resp = local.generate(self._local_request(task))
                result["text"] = resp.text
                result["tokens"] = resp.token_usage
                result["latency_ms"] = resp.latency_ms
                self.local_breaker.record(False)
            except Exception:  # noqa: BLE001 - escalation path
                local_ok = False
                self.local_breaker.record(True)
                result["route"] = "deepseek"
                result["reason"] = "local_failure_escalation"
                resp = cloud.generate(self._cloud_plan_request(task))
                result["text"] = resp.text
                result["tokens"] = resp.token_usage
        else:
            resp = cloud.generate(self._cloud_plan_request(task))
            result["text"] = resp.text
            result["tokens"] = resp.token_usage
            result["latency_ms"] = resp.latency_ms
            self.cloud_breaker.record(False)

        # Adaptive threshold: feed REAL outcomes (local success = no escalation;
        # cloud success = the call itself returned). update() runs periodically
        # once the observation window has enough samples.
        self.threshold.record(
            routed_local=(result["route"] == "local"),
            success=(result["route"] == "local" and local_ok)
            or (result["route"] == "deepseek"),
        )
        self.threshold.maybe_update()

        # Persistent memory event so the router's confidence/novelty signals
        # reflect real history (memory.py keeps only recent records).
        try:
            TaskMemory(memory_root_from_cfg(self.cfg),
                       embed=memory_embed_callable(self.cfg)).record(TaskRecord(
                task=task, ts=time.time(), route=result["route"],
                verdict="APPROVED" if local_ok else result["reason"],
            ))
        except Exception:  # noqa: BLE001 - memory must never break the run
            pass
        return result

    # --- prompts (ARCHITECTURE.md §4) -----------------------------------
    def _local_request(self, task: str) -> ModelRequest:
        return ModelRequest(
            system=(
                "You are a precise senior software engineer in a low-latency environment.\n"
                "1. Change ONLY what the task requires. Never restructure unrelated code.\n"
                "2. Output the complete updated files in fenced code blocks labeled with paths.\n"
                "3. End with <CONFIDENCE>0.0-1.0</CONFIDENCE>."
            ),
            user=(
                f"TASK: {task}\n"
                "CONTEXT: project context is injected by the ask.py bridge "
                "(--context-scan / --context), not here.\n"
            ),
            max_tokens=4096,
            temperature=0.2,
            timeout_s=self.cfg["backends"]["local"]["timeout_s"],
        )

    def _cloud_plan_request(self, task: str) -> ModelRequest:
        return ModelRequest(
            system=(
                "You are the senior architect for a hybrid coding agent. Produce a "
                "step-by-step plan with file-level breakdown (max 400 words unless "
                "the task demands more)."
            ),
            user=f"OBJECTIVE: {task}\nPROJECT SUMMARY: injected from TaskMemory "
                 f"(hybrid-agent/memory.py) when history exists.\n",
            max_tokens=1200,
            temperature=0.1,
            timeout_s=self.cfg["backends"]["deepseek"]["timeout_s"],
        )


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _load_config(path: str | None) -> dict:
    if not path:
        return DEFAULT_CONFIG
    try:
        # Dynamic import so Pylance/type-checkers never flag a missing top-level
        # dependency, and so startup works without PyYAML installed.
        yaml = importlib.import_module("yaml")
        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}
        _deep_merge(DEFAULT_CONFIG, yaml_config)
        return DEFAULT_CONFIG
    except ImportError:
        print(f"warning: PyYAML not installed; ignoring {path} and using embedded defaults")
        return DEFAULT_CONFIG
    except Exception as exc:  # noqa: BLE001 - config must never break startup
        print(f"warning: could not load {path} ({exc}); using embedded defaults")
        return DEFAULT_CONFIG


def apply_env_overrides(cfg: dict) -> dict:
    """Apply provider-agnostic model/endpoint overrides from the environment.

    Roles stay architecture (local implementer, API supervisor); only the
    concrete model and endpoint are reconfigurable:
      LOCAL_MODEL     local backend model id
      LOCAL_BASE_URL  local OpenAI-compatible endpoint
      LOCAL_API_KEY   local endpoint key (defaults to "lm-studio")
      API_MODEL       API backend model id
      API_BASE_URL    API OpenAI-compatible endpoint (e.g. any vendor)
      API_KEY_ENV     env var holding the API key
    """
    lc = cfg["backends"]["local"]
    dc = cfg["backends"]["deepseek"]
    if os.environ.get("LOCAL_MODEL"):
        lc["model"] = os.environ["LOCAL_MODEL"]
    if os.environ.get("LOCAL_BASE_URL"):
        lc["base_url"] = os.environ["LOCAL_BASE_URL"].rstrip("/")
    if os.environ.get("LOCAL_API_KEY"):
        lc["api_key"] = os.environ["LOCAL_API_KEY"]
    if os.environ.get("API_MODEL"):
        dc["model"] = os.environ["API_MODEL"]
    if os.environ.get("API_BASE_URL"):
        dc["base_url"] = os.environ["API_BASE_URL"].rstrip("/")
    if os.environ.get("API_KEY_ENV"):
        dc["api_key_env"] = os.environ["API_KEY_ENV"]
    cfg.setdefault("roles", {"implementer": "local", "supervisor": "deepseek"})
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid coding agent (Gemma 4 12B + DeepSeek)")
    parser.add_argument("--task", required=True, help="coding task description")
    parser.add_argument("--file", help="(optional) primary file affected")
    parser.add_argument("--config", help="optional YAML config override")
    parser.add_argument("--route-only", action="store_true",
                        help="print routing decision without calling any model")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = apply_env_overrides(cfg)
    task = args.task if not args.file else f"{args.task} (in {args.file})"

    agent = HybridAgent(cfg)
    if args.route_only:
        route, reason = agent.route(task, context_chars=2000, memory=MemoryView())
        print(f"route: {route} ({reason})")
        return

    result = agent.run(task)
    print(f"route      : {result['route']} ({result['reason']})")
    print(f"tokens     : {result.get('tokens', {})}")
    print(f"latency_ms : {result.get('latency_ms', 'n/a')}")
    print("--- output (truncated to 2000 chars) ---")
    print(result["text"][:2000])


if __name__ == "__main__":
    main()
