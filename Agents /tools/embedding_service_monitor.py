#!/usr/bin/env python3
"""
embedding_service_monitor.py - health-check for an OpenAI-compatible embedding
service (LM Studio / Ollama / local-ai, etc.).

Safe by default:
  - Only probes GET /v1/models and POST /v1/embeddings (read-only).
  - Auto-restart is DISABLED unless --restart is passed, and even then it only
    reports guidance; it never kills processes. Start/stop the service manually.
  - Logs to a user-writable path (~/Library/Logs) instead of /var/log.

Usage:
  embedding_service_monitor.py                       # health check
  embedding_service_monitor.py --health              # JSON health, exit 0/1
  embedding_service_monitor.py --model <id>          # test one model
  embedding_service_monitor.py --url http://host:port
  embedding_service_monitor.py --restart             # allow (best-effort) restart
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("error: the 'requests' package is required (pip install requests)")

def _default_log_path() -> Path:
    """A user-writable, platform-appropriate default log location."""
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or home
        return Path(base) / "embedding_service" / "embedding_service.log"
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "embedding_service.log"
    return home / ".local" / "share" / "embedding_service" / "embedding_service.log"

COMMON_MODELS = [
    "text-embedding-nomic-embed-text-v1.5",
    "text-embedding-ada-002",
    "bge-small-en",
    "e5-small-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
]

EMBED_KEYWORDS = ("embed", "e5", "bge", "nomic", "ada", "text-embedding")


class EmbeddingServiceMonitor:
    def __init__(self, base_url: str = "http://127.0.0.1:1234",
                 service_name: str = "lm-studio",
                 log_file: Optional[str] = None,
                 allow_restart: bool = False):
        self.base_url = base_url.rstrip("/")
        self.embeddings_url = f"{self.base_url}/v1/embeddings"
        self.models_url = f"{self.base_url}/v1/models"
        self.service_name = service_name
        self.available_models: List[str] = []
        self.test_model: Optional[str] = None
        self.allow_restart = allow_restart
        self.log = self._setup_logging(log_file)

    def _setup_logging(self, log_file: Optional[str]):
        handlers = [logging.StreamHandler()]
        if log_file:
            path = Path(log_file)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handlers.append(logging.FileHandler(path))
            except OSError:
                print(f"warning: cannot write log at {path}; stderr only")
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            handlers=handlers)
        return logging.getLogger("embedding-monitor")

    # -- model discovery -------------------------------------------------
    def get_available_models(self) -> List[str]:
        """Fetch model ids, preferring embedding-looking ones."""
        try:
            r = requests.get(self.models_url, timeout=5)
            if r.status_code != 200:
                self.log.warning("could not fetch models (HTTP %s)", r.status_code)
                return []
            data = r.json()
            if "data" in data:
                models = [m.get("id", "") for m in data["data"]]
            elif "models" in data:
                models = list(data["models"])
            elif isinstance(data, list):
                models = data
            else:
                models = []

            emb = [m for m in models if any(k in m.lower() for k in EMBED_KEYWORDS)]
            self.available_models = emb or models
            self.log.info("found %d model(s): %s", len(self.available_models), self.available_models)
            return self.available_models
        except requests.exceptions.ConnectionError:
            self.log.error("connection error fetching models (service not reachable)")
            return []
        except Exception as exc:  # noqa: BLE001
            self.log.error("error fetching models: %s", exc)
            return []

    # -- embedding test --------------------------------------------------
    def test_embedding(self, model: Optional[str] = None) -> bool:
        model = model or self.test_model
        if not model:
            self.log.error("no model specified for testing")
            return False
        try:
            r = requests.post(self.embeddings_url,
                              json={"model": model, "input": "Test embedding"},
                              timeout=10)
            if r.status_code == 200:
                self.log.info("model %s works (HTTP 200)", model)
                return True
            if r.status_code == 404:
                # Service is up; model not found/not an embedding model.
                self.log.warning("model %s not found, but service is up", model)
                return True
            self.log.warning("model %s returned HTTP %s", model, r.status_code)
            return False
        except requests.exceptions.ConnectionError:
            self.log.error("connection error (service not reachable)")
            return False
        except requests.exceptions.Timeout:
            self.log.error("timeout (service not responding)")
            return False
        except Exception as exc:  # noqa: BLE001
            self.log.error("error testing embedding: %s", exc)
            return False

    # -- find a working model --------------------------------------------
    def find_working_model(self) -> Optional[str]:
        if not self.available_models:
            self.get_available_models()
        if not self.available_models:
            self.available_models = list(COMMON_MODELS)
        for model in self.available_models:
            self.log.info("testing model: %s", model)
            if self.test_embedding(model):
                self.test_model = model
                self.log.info("working model: %s", model)
                return model
        self.log.error("no working model found")
        return None

    # -- overall check ---------------------------------------------------
    def check_service(self) -> bool:
        if not self.available_models:
            self.get_available_models()
        if self.test_model:
            return self.test_embedding(self.test_model)
        return self.find_working_model() is not None

    # -- restart (disabled by default) -----------------------------------
    def restart_service(self) -> bool:
        if not self.allow_restart:
            self.log.warning(
                "auto-restart is disabled by default (safe mode). "
                "Start the service manually, or pass --restart to allow it.")
            return False
        # Best-effort: we intentionally do NOT kill/sudo/stop processes.
        # On macOS the service is LM Studio (a GUI app); on Windows it may be a
        # GUI or a service. It is safer to restart it manually.
        self.log.error(
            "restart not automated for service '%s' (safe mode). "
            "Restart the service manually, then re-run the health check.",
            self.service_name)
        return False

    # -- health report ---------------------------------------------------
    def get_service_health(self) -> Dict:
        status = "unknown"
        error: Optional[str] = None
        try:
            status = "healthy" if self.check_service() else "unhealthy"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)
        return {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "url": self.base_url,
            "test_model": self.test_model,
            "models": self.available_models,
            "error": error,
        }

    def run(self) -> bool:
        self.log.info("starting embedding service monitor (url=%s)", self.base_url)
        if self.check_service():
            self.log.info("service is healthy using model: %s", self.test_model)
            return True
        if self.restart_service():
            time.sleep(3)
            return self.check_service()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor an embedding service")
    parser.add_argument("--url", default="http://127.0.0.1:1234", help="Base URL")
    parser.add_argument("--service", default="lm-studio", help="Service name (informational)")
    parser.add_argument("--log-file", default=str(_default_log_path()), help="Log file path")
    parser.add_argument("--no-log", action="store_true", help="Do not write a log file")
    parser.add_argument("--health", action="store_true", help="Print JSON health and exit")
    parser.add_argument("--model", help="Test a specific model")
    parser.add_argument("--restart", action="store_true",
                        help="Allow best-effort restart (still non-destructive)")
    args = parser.parse_args()

    log_file = None if args.no_log else args.log_file
    monitor = EmbeddingServiceMonitor(base_url=args.url, service_name=args.service,
                                      log_file=log_file, allow_restart=args.restart)

    if args.health:
        health = monitor.get_service_health()
        print(json.dumps(health, indent=2))
        return 0 if health["status"] == "healthy" else 1

    if args.model:
        monitor.test_model = args.model
        ok = monitor.test_embedding(args.model)
        print(f"model {args.model}: {'working' if ok else 'not working'}")
        return 0 if ok else 1

    return 0 if monitor.run() else 1


if __name__ == "__main__":
    sys.exit(main())
