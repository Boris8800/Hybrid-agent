"""providers.py — multi-model provider registry for the hybrid engine.

Supports ANY OpenAI-compatible endpoint in either role, with 2 online + 2 local
providers configured out of the box:

  online: deepseek (default), groq
  local:  gemma (default), local-2

Providers are configured under the `providers:` section of config.yml; when
that section is absent the legacy `backends:` section still defines the primary
pair and the remaining defaults are filled in, so nothing breaks.

API keys resolve in order: environment variable -> Kilo auth.json (online,
default key name only) -> the dashboard's encrypted secrets store.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from backends.deepseek import DeepSeekBackend
from backends.local_gemma import GemmaBackend

DEFAULT_ONLINE = [
    {"name": "deepseek", "base_url": "https://api.deepseek.com",
     "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
     "model": "llama-3.3-70b-versatile", "api_key_env": "GROQ_API_KEY"},
]
DEFAULT_LOCAL = [
    {"name": "gemma", "base_url": "http://localhost:1234/v1",
     "model": "google/gemma-4-12b-qat", "api_key": "lm-studio"},
    {"name": "local-2", "base_url": "http://localhost:1234/v1",
     "model": "qwen2.5-coder-7b-instruct", "api_key": "lm-studio"},
]


@dataclass
class Provider:
    name: str
    kind: str            # "online" | "local"
    base_url: str
    model: str
    api_key_env: str = ""
    api_key: str = ""    # literal key for local endpoints
    enabled: bool = True
    timeout_s: float = 30.0
    max_retries: int = 4
    backoff_s: list = field(default_factory=lambda: [1.0, 2.0, 4.0])

    def describe(self) -> str:
        return f"{self.name} ({self.kind}: {self.model} @ {self.base_url})"


def _provider(name, kind, base_url, model, api_key_env="", api_key="",
              enabled=True, timeout_s=30.0, max_retries=4, backoff_s=None):
    return Provider(
        name=name, kind=kind, base_url=base_url, model=model,
        api_key_env=api_key_env, api_key=api_key, enabled=enabled,
        timeout_s=timeout_s, max_retries=max_retries,
        backoff_s=backoff_s or ([1.0, 2.0, 4.0] if kind == "online" else [0.5, 1.0, 2.0]))


def _from_dict(d: dict, kind: str) -> Provider:
    return _provider(
        name=str(d.get("name", "unnamed")),
        kind=kind,
        base_url=str(d.get("base_url", "")),
        model=str(d.get("model", "")),
        api_key_env=str(d.get("api_key_env", "")),
        api_key=str(d.get("api_key", "")),
        enabled=bool(d.get("enabled", True)),
        timeout_s=float(d.get("timeout_s", 180.0 if kind == "local" else 30.0)),
        max_retries=int(d.get("max_retries", 3 if kind == "local" else 4)),
    )


def load_providers(cfg: dict) -> dict[str, list[Provider]]:
    """Return {"online": [...], "local": [...]} providers for the config.

    `providers:` section wins; otherwise the legacy `backends:` section defines
    the primary pair and defaults fill the other slots (backward compatible).
    Non-secret settings edited via the web dashboard (base_url/model/enabled/
    timeouts in $HYBRID_AGENT_HOME/providers.json) are merged on top.
    """
    if isinstance(cfg.get("providers"), dict):
        p = cfg["providers"]
        online = [_from_dict(d, "online") for d in p.get("online", []) if isinstance(d, dict)]
        local = [_from_dict(d, "local") for d in p.get("local", []) if isinstance(d, dict)]
        if online or local:
            online = online or [_from_dict(d, "online") for d in DEFAULT_ONLINE]
            local = local or [_from_dict(d, "local") for d in DEFAULT_LOCAL]
            return {
                "online": [_apply_overrides(prov) for prov in online],
                "local": [_apply_overrides(prov) for prov in local],
            }

    # Legacy fallback: backends.local -> local[0], backends.deepseek -> online[0].
    b = cfg.get("backends", {}) or {}
    lc = b.get("local", {}) or {}
    dc = b.get("deepseek", {}) or {}
    online = [
        _provider(
            name="deepseek", kind="online",
            base_url=str(dc.get("base_url", "https://api.deepseek.com")),
            model=str(dc.get("model", "deepseek-chat")),
            api_key_env=str(dc.get("api_key_env", "DEEPSEEK_API_KEY")),
            timeout_s=float(dc.get("timeout_s", 30.0)),
            max_retries=int(dc.get("max_retries", 4)),
        ),
        _from_dict(DEFAULT_ONLINE[1], "online"),
    ]
    local = [
        _provider(
            name="gemma", kind="local",
            base_url=str(lc.get("base_url", "http://localhost:1234/v1")),
            model=str(lc.get("model", "google/gemma-4-12b-qat")),
            api_key=str(lc.get("api_key", "lm-studio")),
            timeout_s=float(lc.get("timeout_s", 180.0)),
            max_retries=int(lc.get("max_retries", 3)),
        ),
        _from_dict(DEFAULT_LOCAL[1], "local"),
    ]
    return {"online": [_apply_overrides(p) for p in online],
            "local": [_apply_overrides(p) for p in local]}


def _overrides() -> dict:
    try:
        path = Path(os.environ.get(
            "HYBRID_AGENT_HOME", str(Path.home() / ".hybrid-agent"))) / "providers.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _apply_overrides(p: Provider, overrides: dict | None = None) -> Provider:
    """Merge dashboard-edited non-secret settings onto a provider."""
    d = (overrides if overrides is not None else _overrides()).get(p.name) or {}
    if not isinstance(d, dict):
        return p
    if "base_url" in d:
        p.base_url = str(d["base_url"])
    if "model" in d:
        p.model = str(d["model"])
    if "enabled" in d:
        p.enabled = bool(d["enabled"])
    if "timeout_s" in d:
        p.timeout_s = float(d["timeout_s"])
    if "max_retries" in d:
        p.max_retries = int(d["max_retries"])
    return p


def enabled_online(cfg: dict) -> list[Provider]:
    return [p for p in load_providers(cfg)["online"] if p.enabled]


def enabled_local(cfg: dict) -> list[Provider]:
    return [p for p in load_providers(cfg)["local"] if p.enabled]


def get_online(cfg: dict, name: str | None = None, index: int = 0) -> Provider | None:
    online = enabled_online(cfg)
    if not online:
        return None
    if name:
        for p in online:
            if p.name == name:
                return p
        return None
    return online[index] if 0 <= index < len(online) else online[0]


def get_local(cfg: dict, name: str | None = None, index: int = 0) -> Provider | None:
    local = enabled_local(cfg)
    if not local:
        return None
    if name:
        for p in local:
            if p.name == name:
                return p
        return None
    return local[index] if 0 <= index < len(local) else local[0]


def resolve_api_key(p: Provider, env: dict | None = None) -> str:
    """Resolve a provider's API key: env var -> Kilo auth.json -> dashboard
    secrets store. Returns '' when unavailable."""
    env = env if env is not None else os.environ
    if p.api_key_env and env.get(p.api_key_env):
        return env[p.api_key_env]
    if p.kind == "online":
        if p.api_key_env == "DEEPSEEK_API_KEY":
            try:
                from backends.deepseek import _resolve_api_key
                val = _resolve_api_key("DEEPSEEK_API_KEY")
                if val:
                    return val
            except Exception:  # noqa: BLE001 - best-effort resolution
                pass
        try:
            from dashboard import secrets
            return secrets.get_secret(p.name)
        except Exception:  # noqa: BLE001 - dashboard not installed is fine
            return ""
    return p.api_key


def has_api_key(p: Provider) -> bool:
    return bool(resolve_api_key(p))


def backend_for(p: Provider):
    """Build a working backend for a provider (DeepSeek for online, Gemma for
    local). The caller supplies the key at request time via env."""
    if p.kind == "online":
        return DeepSeekBackend(
            api_key_env=p.api_key_env or "DEEPSEEK_API_KEY", model=p.model,
            base_url=p.base_url, timeout_s=p.timeout_s, max_retries=p.max_retries)
    return GemmaBackend(
        base_url=p.base_url, model=p.model, api_key=p.api_key or "lm-studio",
        timeout_s=p.timeout_s, max_retries=p.max_retries)
