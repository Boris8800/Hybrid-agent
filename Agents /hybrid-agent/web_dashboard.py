#!/usr/bin/env python3
"""Web dashboard for the hybrid-agent engine.

Single-user local control panel: manage the 2+2 provider fleet (keys via macOS
Keychain / encrypted fallback), submit and queue tasks, watch live progress and
logs, and inspect stats, config, and the latest fix diff.

Run (from the hybrid-agent dir):
    .venv/bin/python web_dashboard.py --port 8660
    # open http://127.0.0.1:8660

Optional auth: export DASHBOARD_TOKEN=... (sent as Authorization: Bearer).
Optional API key storage home for tests: export HYBRID_AGENT_HOME=/tmp/...
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- Self-heal: flask/flask-socketio/cryptography live in the project venv
# (hybrid-agent/.venv). When invoked with a system python3 that lacks them,
# re-exec with the venv interpreter instead. Guarded so the re-exec cannot loop
# and never fires when the module is imported.
if __name__ == "__main__" and os.environ.get("HYBRID_REHEALED") != "1":
    try:
        import flask  # noqa: F401
        import flask_socketio  # noqa: F401
    except ImportError:
        venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        if venv_python.is_file():
            os.environ["HYBRID_REHEALED"] = "1"
            os.execv(str(venv_python), [str(venv_python),
                                        str(Path(__file__).resolve()), *sys.argv[1:]])
        print("error: flask/flask-socketio missing and no project venv found.\n"
              "Install them: cd 'Agents /hybrid-agent' && "
              "./.venv/bin/pip install flask flask-socketio cryptography",
              file=sys.stderr)
        raise SystemExit(2)

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

ENGINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ENGINE_DIR / "config.yml"
STATS_PATH = ENGINE_DIR / "stats.json"
ASK_PATH = ENGINE_DIR / "ask.py"
DASH_DIR = ENGINE_DIR / "dashboard"
TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

from dashboard import secrets  # noqa: E402
from providers import enabled_local, enabled_online, load_providers, resolve_api_key  # noqa: E402

# Machine-readable progress line emitted by ask.py (ProgressTracker).
_PROGRESS_RE = re.compile(r"\[hybrid\] progress phase=(\w+) (\d+)% ([\d.]+)s")


def _load_stats() -> dict:
    if STATS_PATH.is_file():
        try:
            data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    return {}


def _build_stats(data: dict) -> dict:
    verify = data.get("verify", {}) or {}
    cache = data.get("cache", {}) or {}
    api_tokens = data.get("api_tokens", {}) or {}
    phases = data.get("phase_timings", {}) or {}
    days = []
    for day in sorted(api_tokens.keys())[-7:]:
        days.append({"day": day, "tokens": api_tokens[day].get("tokens", 0)})
    reviews = data.get("deepseek_reviews", 0)
    approvals = data.get("approvals", 0)
    cache_total = cache.get("hits", 0) + cache.get("misses", 0)
    quality_count = data.get("selfeval_quality_count", 0) or 0
    return {
        "tasks_completed": data.get("tasks_completed", 0),
        "files_generated": data.get("files_generated", 0),
        "total_iterations": data.get("total_iterations", 0),
        "truncation_events": data.get("truncation_events", 0),
        "deepseek_reviews": reviews,
        "approvals": approvals,
        "rejections": data.get("rejections", 0),
        "fix_required": data.get("fix_required", 0),
        "deepseek_fallbacks": data.get("deepseek_fallbacks", 0),
        "avg_quality": round(data.get("selfeval_quality_sum", 0.0) / quality_count, 2)
        if quality_count else 0.0,
        "verify": {
            "runs": verify.get("runs", 0), "passed": verify.get("passed", 0),
            "failed": verify.get("failed", 0), "api_calls": verify.get("api_calls", 0),
            "tokens_used": verify.get("tokens_used", 0),
            "estimated_cost_usd": round(verify.get("estimated_cost_usd", 0.0), 5),
        },
        "cache": {
            "hits": cache.get("hits", 0), "misses": cache.get("misses", 0),
            "rate": round(cache.get("hits", 0) / cache_total, 4) if cache_total else 0.0,
        },
        "api_tokens": days,
        "phase_timings": {
            k: {"runs": v.get("runs", 0), "total_s": round(v.get("total_s", 0.0), 2)}
            for k, v in phases.items()
        },
        "success_rate": round(approvals / reviews, 4) if reviews else 0.0,
    }


def _mask(key: str) -> str:
    return ("*" * 8) + key[-4:] if key else ""


def _providers_payload() -> list[dict]:
    cfg = _read_config_dict()
    out = []
    for p in load_providers(cfg)["online"] + load_providers(cfg)["local"]:
        key = resolve_api_key(p)
        out.append({
            "name": p.name, "kind": p.kind, "base_url": p.base_url, "model": p.model,
            "api_key_env": p.api_key_env, "enabled": p.enabled,
            "timeout_s": p.timeout_s, "max_retries": p.max_retries,
            "has_key": bool(key), "key_masked": _mask(key),
        })
    return out


def _read_config_dict() -> dict:
    try:
        import yaml
        if CONFIG_PATH.is_file():
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

class TaskRunner:
    def __init__(self, socketio, root: str):
        self.socketio = socketio
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.current: dict | None = None
        self.history: list[dict] = []
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # --- public API -------------------------------------------------------
    def submit(self, payload: dict) -> str:
        task_id = f"t{int(time.time() * 1000)}"
        item = {"id": task_id, **payload}
        self.q.put(item)
        self._emit_queue()
        self._emit_log("info", f"[dashboard] queued {task_id}: {payload.get('task', '')[:80]}")
        return task_id

    def cancel(self) -> bool:
        with self._lock:
            proc = self.current.get("proc") if self.current else None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)  # graceful -> exit 130
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.terminate()
                except OSError:
                    pass
            self._emit_log("error", "[dashboard] cancel requested (SIGINT sent)")
            return True
        return False

    def status(self) -> dict:
        return {
            "current": self.current and {
                k: self.current[k] for k in ("id", "task", "mode")
                if k in self.current
            },
            "queued": [{"id": it["id"], "task": it.get("task", "")[:80]}
                       for it in list(self.q.queue)],
            "history": self.history[-10:],
        }

    # --- internals --------------------------------------------------------
    def _loop(self):
        while True:
            item = self.q.get()
            with self._lock:
                self.current = item
            self._emit_queue()
            try:
                self._run(item)
            finally:
                with self._lock:
                    self.current = None
                item["finished"] = time.strftime("%H:%M:%S")
                self.history.append(item)
                self.history = self.history[-50:]
                self.q.task_done()
                self._emit_queue()
                self.socketio.emit("done", {"id": item["id"],
                                            "status": item.get("status", "done")})

    def _run(self, item: dict):
        cmd = [sys.executable, str(ASK_PATH), "--task", item.get("task", ""),
               "--mode", item.get("mode", "hybrid"), "--root", self.root]
        for flag in ("enhance", "verify", "regression", "apply", "parallel",
                     "cot", "context_scan", "turbo", "proceed", "pull", "push",
                     "deploy"):
            if item.get(flag):
                cmd.append("--" + flag.replace("_", "-"))
        if item.get("max_iterations"):
            cmd += ["--max-iterations", str(item["max_iterations"])]
        if item.get("online_provider"):
            cmd += ["--online-provider", item["online_provider"]]
        if item.get("local_provider"):
            cmd += ["--local-provider", item["local_provider"]]

        cfg = _read_config_dict()
        env = os.environ.copy()
        for prov in enabled_online(cfg):
            key = resolve_api_key(prov)
            if key and prov.api_key_env:
                env[prov.api_key_env] = key

        self._emit_log("info", f"[dashboard] starting {item['id']}: "
                               f"{' '.join(cmd[len(cmd) - 12:])}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, cwd=str(ENGINE_DIR),
                start_new_session=True)
        except OSError as exc:
            item["status"] = "failed"
            self._emit_log("error", f"[dashboard] failed to launch ask.py: {exc}")
            return
        item["proc"] = proc

        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n")
            if not line:
                continue
            self._emit_log(self._classify(line), line)
            m = _PROGRESS_RE.search(line)
            if m:
                self.socketio.emit("progress", {
                    "phase": m.group(1), "percent": int(m.group(2)),
                    "elapsed": float(m.group(3)),
                })
        proc.wait()
        item["status"] = "passed" if proc.returncode == 0 else f"failed({proc.returncode})"
        self._emit_log("success" if proc.returncode == 0 else "error",
                       f"[dashboard] {item['id']} finished: {item['status']}")

    @staticmethod
    def _classify(line: str) -> str:
        low = line.lower()
        if any(m in line for m in ("⛔", "✗", "failed", "error", "⚠")) or \
                "budget exhausted" in low:
            return "error"
        if any(m in line for m in ("✅", "✓", "approved", "passed", "all agent files")):
            return "success"
        return "info"

    def _emit_log(self, typ: str, line: str):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        self.logs = self.logs[-2000:]
        self.socketio.emit("log", {"type": typ, "message": line})

    def _emit_queue(self):
        self.socketio.emit("queue", self.status())


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(root: str = str(ENGINE_DIR.parent), config_path: str | None = None,
               start_worker: bool = True):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("DASHBOARD_SECRET", "hybrid-agent-dashboard")
    app.config["AGENT_ROOT"] = root
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=[])
    runner = TaskRunner(socketio, root) if start_worker else None

    @app.before_request
    def _auth():
        if TOKEN and request.path.startswith("/api/"):
            ok = request.headers.get("Authorization") == f"Bearer {TOKEN}" \
                 or request.args.get("token") == TOKEN
            if not ok:
                return jsonify({"error": "unauthorized"}), 401

    @socketio.on("connect")
    def _connect(auth=None):
        if TOKEN and not (auth or {}).get("token") == TOKEN:
            return False
        return True

    # --- pages ------------------------------------------------------------
    @app.route("/")
    def index():
        return send_from_directory(DASH_DIR, "index.html")

    # --- providers --------------------------------------------------------
    @app.route("/api/providers")
    def providers():
        return jsonify(_providers_payload())

    @app.route("/api/providers", methods=["POST"])
    def save_provider():
        body = request.json or {}
        name = body.get("name", "")
        if not name:
            return jsonify({"error": "provider name required"}), 400
        ov = secrets_provider_overrides()
        entry = dict(ov.get(name, {}))
        for key in ("base_url", "model", "enabled", "timeout_s", "max_retries"):
            if key in body and body[key] is not None:
                entry[key] = body[key]
        if body.get("api_key"):
            secrets.set_secret(name, body["api_key"])
        ov[name] = entry
        secrets_save_provider_overrides(ov)
        return jsonify({"status": "ok", "provider": next(
            (p for p in _providers_payload() if p["name"] == name), {})})

    @app.route("/api/providers/test", methods=["POST"])
    def test_provider():
        body = request.json or {}
        name = body.get("name", "")
        prov = next((p for p in _provider_objects() if p.name == name), None)
        if prov is None:
            return jsonify({"ok": False, "error": "unknown provider"}), 404
        key = body.get("api_key") or resolve_api_key(prov)
        import urllib.request
        import urllib.error
        start = time.time()
        try:
            if prov.kind == "local":
                req = urllib.request.Request(prov.base_url.rstrip("/") + "/models",
                                             headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    payload = json.loads(resp.read().decode())
                models = [m.get("id") for m in payload.get("data", [])]
                return jsonify({"ok": True, "latency_ms": round((time.time() - start) * 1000),
                                "message": f"{prov.name} reachable", "models": models})
            body_payload = json.dumps({
                "model": prov.model,
                "messages": [{"role": "user", "content": "Reply with the single word ok"}],
                "max_tokens": 8, "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                prov.base_url.rstrip("/") + "/chat/completions", data=body_payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode())
            return jsonify({"ok": True, "latency_ms": round((time.time() - start) * 1000),
                            "message": f"{prov.name} responded",
                            "models": [prov.model]})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200] if exc.fp else str(exc)
            return jsonify({"ok": False, "error": f"HTTP {exc.code}: {detail}"})
        except Exception as exc:  # noqa: BLE001 - surface the failure to the UI
            return jsonify({"ok": False, "error": str(exc)})

    @app.route("/api/providers/delete-key", methods=["POST"])
    def delete_provider_key():
        name = (request.json or {}).get("name", "")
        secrets.delete_secret(name)
        return jsonify({"status": "ok"})

    # --- stats / config / diff --------------------------------------------
    @app.route("/api/stats")
    def stats():
        return jsonify(_build_stats(_load_stats()))

    @app.route("/api/memory")
    def memory():
        """Recent task-memory records + consolidated insights for the UI."""
        try:
            from memory import TaskMemory, memory_root_from_cfg
            mem = TaskMemory(memory_root_from_cfg(_read_config_dict(),
                                                  cwd=str(ENGINE_DIR)))
            records = mem._load()[-20:][::-1]  # most recent first
            return jsonify({
                "count": mem.count(),
                "insights": mem.insights_text(),
                "records": [{
                    "task": r.get("task", ""), "verdict": r.get("verdict", ""),
                    "route": r.get("route", ""), "ts": r.get("ts", 0.0),
                    "quality": r.get("quality", 0.0),
                } for r in records],
            })
        except Exception as exc:  # noqa: BLE001 - memory is best-effort
            return jsonify({"count": 0, "insights": "", "records": [],
                            "error": str(exc)})

    @app.route("/api/system")
    def system():
        return jsonify({
            "engine_dir": str(ENGINE_DIR),
            "root": app.config["AGENT_ROOT"],
            "python": sys.version.split()[0],
            "interpreter": sys.executable,
            "stats_path": str(STATS_PATH),
            "config_path": str(CONFIG_PATH),
            "providers_configured": len(_providers_payload()),
        })

    @app.route("/api/health")
    def health():
        """Local endpoints reachability + online key presence (no online calls)."""
        cfg = _read_config_dict()
        local_status = []
        for p in enabled_local(cfg):
            ok, models = _ping_local(p)
            local_status.append({"name": p.name, "ok": ok, "models": models[:6]})
        online_status = [{"name": p.name, "key": bool(resolve_api_key(p)),
                          "model": p.model} for p in enabled_online(cfg)]
        return jsonify({"local": local_status, "online": online_status})

    @app.route("/api/config")
    def config():
        text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.is_file() else ""
        return jsonify({"text": text})

    @app.route("/api/config", methods=["POST"])
    def save_config():
        text = (request.json or {}).get("text", "")
        import yaml
        try:
            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict):
                return jsonify({"error": "config must be a YAML mapping"}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"invalid YAML: {exc}"}), 400
        try:
            backup = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
            if CONFIG_PATH.is_file():
                backup.write_text(CONFIG_PATH.read_text(encoding="utf-8"),
                                  encoding="utf-8")
            _write_atomic(CONFIG_PATH, text)
        except OSError as exc:
            return jsonify({"error": f"write failed: {exc}"}), 500
        return jsonify({"status": "ok", "backup": str(backup)})

    @app.route("/api/diff")
    def diff():
        path = Path(app.config["AGENT_ROOT"]) / "hybrid-verify" / "fixes.diff"
        if not path.is_file():
            return jsonify({"text": "", "exists": False})
        return jsonify({"text": path.read_text(encoding="utf-8", errors="replace"),
                        "exists": True})

    # --- tasks ------------------------------------------------------------
    @app.route("/api/task", methods=["POST"])
    def submit_task():
        body = request.json or {}
        task = (body.get("task") or "").strip()
        if not task:
            return jsonify({"error": "task text required"}), 400
        if runner is None:
            return jsonify({"error": "worker not started"}), 500
        task_id = runner.submit({
            "task": task, "mode": body.get("mode", "hybrid"),
            "enhance": bool(body.get("enhance")), "verify": bool(body.get("verify")),
            "regression": bool(body.get("regression")), "apply": bool(body.get("apply")),
            "parallel": bool(body.get("parallel")),             "cot": bool(body.get("cot")),
            "context_scan": bool(body.get("context_scan")),             "turbo": bool(body.get("turbo")),
            "proceed": bool(body.get("proceed")),
            "pull": bool(body.get("pull")), "push": bool(body.get("push")),
            "deploy": bool(body.get("deploy")),
            "max_iterations": int(body["max_iterations"]) if body.get("max_iterations") else 0,
            "online_provider": body.get("online_provider") or "",
            "local_provider": body.get("local_provider") or "",
        })
        return jsonify({"status": "ok", "id": task_id})

    @app.route("/api/queue")
    def queue_status():
        return jsonify(runner.status() if runner else {"current": None, "queued": [], "history": []})

    @app.route("/api/cancel", methods=["POST"])
    def cancel_task():
        ok = runner.cancel() if runner else False
        return jsonify({"status": "ok" if ok else "nothing_to_cancel"})

    @app.route("/api/logs")
    def logs():
        return jsonify({"lines": runner.logs[-500:] if runner else []})

    @app.route("/api/logs/download")
    def logs_download():
        text = "\n".join(runner.logs) if runner else ""
        from flask import Response
        return Response(text, mimetype="text/plain",
                        headers={"Content-Disposition": "attachment; filename=hybrid-agent.log"})

    app.socketio = socketio
    app.runner = runner
    return app, socketio


def _ping_local(p) -> tuple[bool, list]:
    """Quick reachability probe of a local endpoint's /models. Never raises."""
    import urllib.request
    try:
        req = urllib.request.Request(
            p.base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {p.api_key or 'lm-studio'}"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        return True, [m.get("key") or m.get("id") or ""
                      for m in data.get("models", data.get("data", []))
                      if m and (m.get("key") or m.get("id"))]
    except Exception:  # noqa: BLE001
        return False, []


def _provider_objects():
    return load_providers(_read_config_dict())["online"] + \
        load_providers(_read_config_dict())["local"]


def secrets_provider_overrides() -> dict:
    path = Path(os.environ.get("HYBRID_AGENT_HOME", str(Path.home() / ".hybrid-agent"))) / "providers.json"
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def secrets_save_provider_overrides(overrides: dict) -> None:
    path = Path(os.environ.get("HYBRID_AGENT_HOME", str(Path.home() / ".hybrid-agent"))) / "providers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="hybrid-agent web dashboard")
    parser.add_argument("--port", type=int, default=8660)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--root", default=str(ENGINE_DIR.parent),
                        help="project root for task application (default: repo root)")
    args = parser.parse_args()

    app, socketio = create_app(root=args.root)
    print(f"🤖 hybrid-agent dashboard: http://{args.host}:{args.port}")
    print(f"   providers: 2 online (deepseek, groq) + 2 local (qwen, local-2)")
    if TOKEN:
        print("   auth: DASHBOARD_TOKEN is set — include it as Authorization: Bearer")
    socketio.run(app, host=args.host, port=args.port, debug=False,
                 allow_unsafe_werkzeug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
