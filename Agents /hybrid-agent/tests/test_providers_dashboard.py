"""Providers + dashboard tests: 2-online/2-local registry, failover + turbo
cloud wrappers, encrypted secrets store, and the web dashboard API.

Run from hybrid-agent/:
    ./.venv/bin/python -m unittest tests.test_providers_dashboard -v
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ask  # noqa: E402
from backends.base import BackendError, ModelRequest, ModelResponse  # noqa: E402
from providers import (enabled_online, get_local, get_online, load_providers,
                       resolve_api_key)  # noqa: E402

HAS_FLASK = importlib.util.find_spec("flask") is not None
HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


PROVIDER_CFG = {
    "providers": {
        "online": [
            {"name": "deepseek", "base_url": "https://api.deepseek.com",
             "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY", "enabled": True},
            {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "api_key_env": "GROQ_API_KEY", "enabled": True},
        ],
        "local": [
            {"name": "qwen", "base_url": "http://localhost:1234/api/v1",
             "model": "qwen2.5-coder-14b-instruct-mlx", "api_key": "lm-studio", "enabled": True},
            {"name": "local-2", "base_url": "http://localhost:1234/api/v1",
             "model": "qwen2.5-coder-14b-instruct-mlx", "api_key": "lm-studio", "enabled": True},
        ],
    },
}


class TestProviderRegistry(unittest.TestCase):
    def test_defaults_are_two_online_and_two_local(self):
        prov = load_providers(PROVIDER_CFG)
        self.assertEqual([p.name for p in prov["online"]], ["deepseek", "groq"])
        self.assertEqual([p.name for p in prov["local"]], ["qwen", "local-2"])
        self.assertEqual(len(prov["online"]), 2)
        self.assertEqual(len(prov["local"]), 2)

    def test_legacy_backends_fallback(self):
        cfg = {"backends": {"local": {"base_url": "http://x", "model": "m",
                                      "timeout_s": 60, "max_retries": 2},
                            "deepseek": {"api_key_env": "K", "model": "d",
                                         "timeout_s": 15, "max_retries": 3}}}
        prov = load_providers(cfg)
        self.assertEqual(prov["online"][0].model, "d")
        self.assertEqual(prov["local"][0].model, "m")
        self.assertEqual(prov["local"][0].timeout_s, 60.0)

    def test_selection_and_enabled_filtering(self):
        cfg = {"providers": {"online": [
            {"name": "a", "base_url": "u", "model": "m", "enabled": False},
            {"name": "b", "base_url": "u", "model": "m", "enabled": True},
        ], "local": []}}
        self.assertEqual(get_online(cfg).name, "b")
        self.assertIsNone(get_online(cfg, name="a"))  # disabled -> not selectable
        self.assertEqual(get_online(cfg, name="b").name, "b")
        self.assertIsNone(get_online(cfg, name="missing"))
        self.assertEqual(get_local(cfg).name, "qwen")  # empty local slot -> default
        self.assertEqual([p.name for p in enabled_online(cfg)], ["b"])

    def test_resolve_api_key_env_wins(self):
        p = load_providers(PROVIDER_CFG)["online"][0]
        key = resolve_api_key(p, env={"DEEPSEEK_API_KEY": "env-key"})
        self.assertEqual(key, "env-key")

    def test_overrides_file_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = {"groq": {"model": "llama-4-sonnet", "enabled": False}}
            (Path(tmp) / "providers.json").write_text(json.dumps(overrides))
            old = os.environ.get("HYBRID_AGENT_HOME")
            os.environ["HYBRID_AGENT_HOME"] = tmp
            try:
                prov = load_providers(PROVIDER_CFG)
                groq = [p for p in prov["online"] if p.name == "groq"][0]
                self.assertEqual(groq.model, "llama-4-sonnet")
                self.assertFalse(groq.enabled)
                self.assertNotIn("groq", [p.name for p in enabled_online(prov)])
            finally:
                if old is None:
                    os.environ.pop("HYBRID_AGENT_HOME", None)
                else:
                    os.environ["HYBRID_AGENT_HOME"] = old


class TestFailoverAndTurbo(unittest.TestCase):
    def _cfg(self):
        return dict(PROVIDER_CFG)

    def test_failover_switches_provider_on_backend_error(self):
        class _Broken:
            def generate(self, req):
                raise BackendError("deepseek down")

        class _Working:
            def generate(self, req):
                return ModelResponse(text="from groq", backend="groq")

        import ask as ask_mod
        orig_backend_for, orig_resolve = ask_mod.backend_for, ask_mod.resolve_api_key
        ask_mod.backend_for = lambda p: _Working()
        ask_mod.resolve_api_key = lambda p: "key"
        try:
            wrapped = ask_mod._failover_cloud(_Broken(), self._cfg(),
                                              Namespace(router="auto"))
            resp = wrapped.generate(ModelRequest(system="s", user="u"))
            self.assertEqual(resp.text, "from groq")
        finally:
            ask_mod.backend_for, ask_mod.resolve_api_key = orig_backend_for, orig_resolve

    def test_single_provider_is_passthrough(self):
        cfg = {"providers": {"online": [
            {"name": "only", "base_url": "u", "model": "m", "enabled": True}], "local": []}}
        cloud = object()
        self.assertIs(ask._failover_cloud(cloud, cfg, Namespace(router="auto")), cloud)
        self.assertIs(ask._parallel_cloud(cloud, cfg, Namespace(router="auto")), cloud)

    def test_turbo_uses_longest_non_truncated(self):
        class _Short:
            def generate(self, req):
                return ModelResponse(text="ok", backend="primary")

        class _Long:
            def generate(self, req):
                return ModelResponse(text="much longer useful answer", backend="groq")

        orig_backend_for, orig_resolve = ask.backend_for, ask.resolve_api_key
        ask.backend_for = lambda p: _Long()
        ask.resolve_api_key = lambda p: "key"
        try:
            wrapped = ask._parallel_cloud(_Short(), self._cfg(), Namespace(router="auto"))
            resp = wrapped.generate(ModelRequest(system="s", user="u"))
            self.assertEqual(resp.text, "much longer useful answer")
        finally:
            ask.backend_for, ask.resolve_api_key = orig_backend_for, orig_resolve


@unittest.skipUnless(HAS_CRYPTO, "cryptography not installed")
class TestSecretsStore(unittest.TestCase):
    def _env(self, tmp):
        old_home = os.environ.get("HYBRID_AGENT_HOME")
        old_backend = os.environ.get("HYBRID_SECRETS_BACKEND")
        os.environ["HYBRID_AGENT_HOME"] = tmp
        os.environ["HYBRID_SECRETS_BACKEND"] = "file"
        return old_home, old_backend

    def _restore(self, old_home, old_backend):
        if old_home is None:
            os.environ.pop("HYBRID_AGENT_HOME", None)
        else:
            os.environ["HYBRID_AGENT_HOME"] = old_home
        if old_backend is None:
            os.environ.pop("HYBRID_SECRETS_BACKEND", None)
        else:
            os.environ["HYBRID_SECRETS_BACKEND"] = old_backend

    def test_roundtrip_encrypted(self):
        import dashboard.secrets as s
        with tempfile.TemporaryDirectory() as tmp:
            old = self._env(tmp)
            try:
                s.set_secret("groq", "sk-live-1234")
                self.assertTrue(s.has_secret("groq"))
                self.assertEqual(s.get_secret("groq"), "sk-live-1234")
                # ciphertext on disk, never the plaintext
                raw = (Path(tmp) / "secrets.json").read_text()
                self.assertNotIn("sk-live-1234", raw)
                s.delete_secret("groq")
                self.assertFalse(s.has_secret("groq"))
                self.assertEqual(s.get_secret("groq"), "")
            finally:
                self._restore(*old)


@unittest.skipUnless(HAS_FLASK, "flask not installed")
class TestDashboardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_dashboard
        cls.mod = web_dashboard

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        app, socketio = self.mod.create_app(root=self.tmp.name, start_worker=False)
        self.client = app.test_client()
        self.mod.TOKEN = ""

    def test_providers_list_has_no_keys(self):
        resp = self.client.get("/api/providers")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data), 4)
        self.assertTrue(all("key" not in p or not p.get("key")
                            for p in data))  # never the plaintext key
        self.assertTrue(all(p.get("key_masked") is not None for p in data))

    def test_stats_shape(self):
        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        s = resp.get_json()
        for k in ("tasks_completed", "success_rate", "cache", "phase_timings",
                  "verify", "api_tokens"):
            self.assertIn(k, s)

    def test_task_requires_text(self):
        resp = self.client.post("/api/task", json={"task": "  "})
        self.assertEqual(resp.status_code, 400)

    def test_config_rejects_invalid_yaml(self):
        resp = self.client.post("/api/config", json={"text": "backends: [unclosed"})
        self.assertEqual(resp.status_code, 400)

    def test_queue_ok(self):
        self.assertEqual(self.client.get("/api/queue").status_code, 200)

    def test_new_endpoints(self):
        mem = self.client.get("/api/memory")
        self.assertEqual(mem.status_code, 200)
        for k in ("count", "insights", "records"):
            self.assertIn(k, mem.get_json())
        sys = self.client.get("/api/system")
        self.assertEqual(sys.status_code, 200)
        self.assertIn("engine_dir", sys.get_json())
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        h = health.get_json()
        for k in ("local", "online"):
            self.assertIn(k, h)

    def test_auth_required_when_token_set(self):
        self.mod.TOKEN = "sekrit"
        try:
            resp = self.client.get("/api/stats")
            self.assertEqual(resp.status_code, 401)
            resp = self.client.get("/api/stats",
                                   headers={"Authorization": "Bearer sekrit"})
            self.assertEqual(resp.status_code, 200)
        finally:
            self.mod.TOKEN = ""


class TestEnhanceProceed(unittest.TestCase):
    """--proceed continues with the enhanced prompt when non-interactive."""

    RAW = ("=== ENHANCED PROMPT ===\nDo the thing.\n\n=== PLAN ===\nStep 1\n\n"
           "=== CLARIFYING QUESTIONS ===\n1. Which database?")

    def _run(self, proceed: bool):
        import contextlib
        import io
        fake_resp = ModelResponse(text=self.RAW, backend="deepseek")
        args = Namespace(cot=False, json=False, proceed=proceed, router="auto")
        cfg = {"backends": {"local": {}, "deepseek": {}}, "review": {}}
        with contextlib.redirect_stdout(io.StringIO()), \
                unittest.mock.patch.object(sys, "stdin",
                                           types.SimpleNamespace(isatty=lambda: False)), \
                unittest.mock.patch.object(ask, "_generate_with_retry",
                                           return_value=fake_resp):
            task_for_qwen, enhancement, clar_needed = ask._enhance_task(
                types.SimpleNamespace(), cfg, args, "check the web", "", cache=None)
        return task_for_qwen, clar_needed

    def test_aborts_when_not_proceeding(self):
        _, clar_needed = self._run(proceed=False)
        self.assertTrue(clar_needed)

    def test_proceeds_with_enhanced_prompt(self):
        task_for_qwen, clar_needed = self._run(proceed=True)
        self.assertFalse(clar_needed)
        self.assertIn("Do the thing.", task_for_qwen)


    def test_local_providers_exclude_max_tokens_by_default(self):
        prov = load_providers(PROVIDER_CFG)
        for p in prov["local"]:
            self.assertIn("max_tokens", p.request_exclude)  # MLX server rejects it
        for p in prov["online"]:
            self.assertEqual(p.request_exclude, [])

    def test_request_exclude_configurable(self):
        cfg = {"providers": {"local": [
            {"name": "x", "base_url": "u", "model": "m",
             "request_exclude": ["temperature"], "request_extra": {"top_p": 0.9}},
        ], "online": []}}
        p = load_providers(cfg)["local"][0]
        self.assertIn("temperature", p.request_exclude)
        self.assertEqual(p.request_extra, {"top_p": 0.9})


if __name__ == "__main__":
    unittest.main()
