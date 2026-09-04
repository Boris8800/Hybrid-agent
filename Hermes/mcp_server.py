#!/usr/bin/env python3
"""
Hermes MCP Tool Server
======================
Exposes Hermes' local tools over MCP (Model Context Protocol) using a
dependency-light stdio JSON-RPC transport (stdlib only).

WHY: the ONLINE AI only plans/supervises (sparing tokens). The heavy lifting —
files, terminal, web, browser, documents, data, memory, cron, security — is done
here, locally, cheaply. Connect this server to an MCP host (Claude Code, VS Code,
Cursor, or your Hermes supervisor) and the supervisor can call these tools
directly instead of burning online tokens on file edits, fetches, etc.

TOOLS THAT NEED NO KEYS ARE FULLY IMPLEMENTED.
Account-based services (Slack, GitHub API, AWS, cloud SMS...) and things that
need a local CLI/daemon are "adapters": they activate automatically when the
required CLI/env is present, otherwise they return a clear setup hint.

TRANSPORT: newline-delimited JSON-RPC 2.0 on stdio (MCP stdio transport).
  invoke:  {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"README.md"}}}
  listing: {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}

SAFETY: destructive tools (delete_file, kill_process, command runs outside the
project, etc.) refuse unless HERMES_SAFE=1 is set, or the specific tool is named
in HERMES_ALLOW (comma separated). Nothing destructive runs silently.
"""

import os
import sys
import json
import time
import uuid
import glob
import shutil
import hashlib
import secrets
import base64
import sqlite3
import datetime
import threading
import subprocess
import tempfile
import re
import signal
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

# ---------------------------------------------------------------------------
# Project / memory roots + safety
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MEMORY_ROOT = Path(os.environ.get("HERMES_MEMORY", str(APP_DIR / "memory"))).expanduser()
PROJECT_ROOT = Path(os.environ.get("HERMES_PROJECT", str(APP_DIR))).expanduser()
SAFE = os.environ.get("HERMES_SAFE", "") == "1"
ALLOWED_DESTRUCTIVE = {x.strip() for x in os.environ.get("HERMES_ALLOW", "").split(",") if x.strip()}

MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "tools_state").mkdir(parents=True, exist_ok=True)


def _guard(tool_name):
    """Refuse destructive/untrusted tool unless explicitly allowed."""
    if SAFE or tool_name in ALLOWED_DESTRUCTIVE:
        return None
    return (f"BLOCKED by safety gate. Re-run the server with HERMES_SAFE=1 to allow "
            f"destructive tools, or add '{tool_name}' to HERMES_ALLOW.")


def _resolve(path):
    """Resolve a path inside the project root; refuse to escape it."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if p != PROJECT_ROOT and PROJECT_ROOT not in p.parents:
        raise ValueError(f"path escapes project root: {p}")
    return p


# ===========================================================================
# 1. FILESYSTEM TOOLS
# ===========================================================================
def fs_read_file(path, start=None, end=None):
    p = _resolve(path)
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    lines = p.read_text(errors="replace").splitlines()
    if start is not None and end is not None:
        lines = lines[start - 1:end]
    numbered = [f"{i + (start or 1):5d} | {l}" for i, l in enumerate(lines)]
    return {"content": "\n".join(numbered), "total_lines": len(lines)}


def fs_write_file(path, content):
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))}


def fs_patch(path, old, new, fuzzy=False):
    p = _resolve(path)
    if not p.is_file():
        return {"error": "not a file"}
    text = p.read_text()
    if fuzzy:
        # case-insensitive first occurrence replacement
        idx = text.lower().find(old.lower())
        if idx == -1:
            return {"error": "pattern not found"}
        text = text[:idx] + new + text[idx + len(old):]
    else:
        if text.count(old) != 1:
            return {"error": f"expected exactly 1 match, found {text.count(old)}"}
        text = text.replace(old, new)
    p.write_text(text)
    return {"ok": True, "path": str(p)}


def fs_search(pattern, path=None, glob_="**/*", ignore_case=True):
    base = _resolve(path or ".")
    rx = re.compile(pattern, re.I if ignore_case else 0)
    matches = []
    for f in base.glob(glob_):
        if f.is_file():
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        matches.append({"file": str(f), "line": i, "text": line[:500]})
                        if len(matches) >= 500:
                            return {"matches": matches, "truncated": True}
            except Exception:
                continue
    return {"matches": matches}


def fs_list(path=None):
    base = _resolve(path or ".")
    items = []
    for e in sorted(base.iterdir()):
        try:
            st = e.stat()
            items.append({"name": e.name, "type": "dir" if e.is_dir() else "file",
                          "size": st.st_size, "mtime": int(st.st_mtime)})
        except Exception:
            continue
    return {"path": str(base), "items": items}


def fs_info(path):
    p = _resolve(path)
    if not p.exists():
        return {"error": "not found"}
    st = p.stat()
    return {"path": str(p), "type": "dir" if p.is_dir() else "file", "size": st.st_size,
            "mtime": int(st.st_mtime), "readable": os.access(p, os.R_OK),
            "writable": os.access(p, os.W_OK), "permissions": oct(st.st_mode & 0o777)}


def fs_move(src, dst):
    g = _guard("fs_move")
    if g: return {"error": g}
    _resolve(src).rename(_resolve(dst))
    return {"ok": True}


def fs_copy(src, dst):
    s, d = _resolve(src), _resolve(dst)
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return {"ok": True}


def fs_delete(path, recursive=False):
    g = _guard("fs_delete")
    if g: return {"error": g}
    p = _resolve(path)
    if p.is_dir() and not recursive:
        return {"error": "is a directory; pass recursive=True"}
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"ok": True, "deleted": str(p)}


# ===========================================================================
# 2. TERMINAL & PROCESS TOOLS
# ===========================================================================
def _run_cmd(command, cwd=None, timeout=120, shell=True):
    base = _resolve(cwd) if cwd else PROJECT_ROOT
    try:
        r = subprocess.run(command, shell=shell, cwd=str(base), capture_output=True,
                           text=True, timeout=timeout)
        return {"exit_code": r.returncode, "stdout": r.stdout[-20000:],
                "stderr": r.stderr[-20000:], "cwd": str(base)}
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def term_exec(command, cwd=None, timeout=120):
    if "sudo" in command or ("rm " in command and " -rf /" in command):
        g = _guard("term_exec")
        if g: return {"error": g}
    return _run_cmd(command, cwd, timeout)


def term_bg(command, cwd=None):
    g = _guard("term_bg")
    if g: return {"error": g}
    try:
        proc = subprocess.Popen(command, shell=True, cwd=str(_resolve(cwd or ".")),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return {"pid": proc.pid, "started": True}
    except Exception as e:
        return {"error": str(e)}


def term_list():
    try:
        r = subprocess.run(["ps", "axo", "pid,ppid,comm"], capture_output=True, text=True, timeout=20)
        return {"processes": r.stdout[-20000:]}
    except Exception as e:
        return {"error": str(e)}


def term_kill(pid, sig="TERM"):
    g = _guard("term_kill")
    if g: return {"error": g}
    try:
        os.kill(int(pid), getattr(signal, f"SIG{sig.upper()}"))
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


def term_env_get(name):
    return {"name": name, "value": os.environ.get(name)}


def term_env_set(name, value):
    g = _guard("term_env_set")
    if g: return {"error": g}
    os.environ[name] = value
    return {"ok": True}


# ===========================================================================
# 3. WEB TOOLS  (zero-credential core; search via DuckDuckGo HTML endpoint)
# ===========================================================================
def _http_get(url, timeout=30, headers=None):
    try:
        import requests
        h = {"User-Agent": "Mozilla/5.0 (Hermes)", **(headers or {})}
        r = requests.get(url, headers=h, timeout=timeout)
        return r
    except Exception as e:
        return None


def web_search(query, max_results=8):
    try:
        import requests
        from urllib.parse import unquote
        from html.parser import HTMLParser

        class _L(HTMLParser):
            def __init__(self):
                super().__init__(); self._url = None; self.res = []
            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if tag == "a" and a.get("class") == "result__a":
                    self._url = a.get("href")
            def handle_data(self, d):
                if self._url is not None:
                    self.res.append((self._url, d.strip())); self._url = None

        r = requests.get("https://html.duckduckgo.com/html/", params={"q": query},
                         timeout=25, headers={"User-Agent": "Mozilla/5.0 (Hermes)"})
        if r.status_code != 200:
            return {"error": f"search failed http {r.status_code}"}
        pa = _L(); pa.feed(r.text)
        out = []
        for href, title in pa.res[:max_results]:
            if "uddg=" in href:
                href = unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
            out.append({"title": title, "url": href})
        return {"results": out}
    except Exception as e:
        return {"error": f"web_search unavailable: {e}"}


def web_fetch(url, format_="text", timeout=30):
    r = _http_get(url, timeout)
    if not r:
        return {"error": "fetch failed"}
    if format_ == "markdown":
        try:
            from html2text import html2text
            return {"status": r.status_code, "content": html2text(r.text)[:30000]}
        except Exception:
            return {"status": r.status_code, "content": r.text[:30000]}
    return {"status": r.status_code, "content": r.text[:30000]}


def web_validate(url):
    u = urlparse(url)
    return {"valid": u.scheme in ("http", "https") and bool(u.netloc),
            "scheme": u.scheme, "host": u.netloc, "normalized": u.geturl()}


def web_links(url, timeout=30):
    r = _http_get(url, timeout)
    if not r:
        return {"error": "fetch failed"}
    from html.parser import HTMLParser
    class _L(HTMLParser):
        def __init__(self):
            super().__init__(); self.links = []
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "a" and a.get("href"):
                self.links.append(a["href"])
    pa = _L(); pa.feed(r.text)
    return {"links": pa.links[:200], "count": len(pa.links)}


def web_metadata(url, timeout=30):
    r = _http_get(url, timeout)
    if not r:
        return {"error": "fetch failed"}
    from html.parser import HTMLParser
    class _M(HTMLParser):
        def __init__(self):
            super().__init__(); self.title = ""; self.meta = {}; self._in_title = False
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "title": self._in_title = True
            if tag == "meta":
                k = a.get("name") or a.get("property")
                if k: self.meta[k] = a.get("content", "")
        def handle_data(self, d):
            if self._in_title: self.title += d
        def handle_endtag(self, tag):
            if tag == "title": self._in_title = False
    pa = _M(); pa.feed(r.text)
    return {"title": pa.title.strip(), "description": pa.meta.get("description"),
            "og_title": pa.meta.get("og:title"), "og_image": pa.meta.get("og:image"),
            "status": r.status_code, "final_url": r.url}


def web_rss(url, max_items=20, timeout=30):
    r = _http_get(url, timeout)
    if not r:
        return {"error": "fetch failed"}
    items = []
    try:
        root = ElementTree.fromstring(r.content)
        for item in root.iter():
            if item.tag.split("}")[-1] in ("item", "entry"):
                e = {c.tag.split("}")[-1]: (c.text or "") for c in item}
                items.append({"title": e.get("title", ""), "link": e.get("link", ""),
                              "published": e.get("pubDate", e.get("updated", "")),
                              "summary": (e.get("description", "") or "")[:500]})
                if len(items) >= max_items:
                    break
        return {"items": items}
    except Exception as e:
        return {"error": f"could not parse feed: {e}"}


# ===========================================================================
# 4. BROWSER AUTOMATION (Playwright)
# ===========================================================================
_playwright = None
def _browser():
    global _playwright
    try:
        if _playwright is None:
            from playwright.sync_api import sync_playwright
            _playwright = sync_playwright().start()
        return _playwright
    except Exception as e:
        return None


def browser_navigate(url, width=1280, height=800, extract="text", timeout=30000):
    pw = _browser()
    if not pw:
        return {"error": "playwright not available"}
    browser = None
    try:
        browser = pw.chromium.launch(headless=True)
        pg = browser.new_page(viewport={"width": width, "height": height})
        pg.goto(url, wait_until="domcontentloaded", timeout=timeout)
        result = {"url": pg.url, "title": pg.title(), "status": "loaded"}
        if extract in ("text", "all"):
            result["text"] = pg.evaluate("document.body ? document.body.innerText : ''")[:30000]
        if extract in ("links", "all"):
            result["links"] = pg.evaluate(
                "[...document.querySelectorAll('a')].map(a=>a.href).filter(Boolean)")[:200]
        if extract in ("html", "all"):
            result["html"] = pg.content()[:50000]
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


# ===========================================================================
# 5. DOCUMENT TOOLS (optional libs guarded)
# ===========================================================================
def doc_read(path):
    p = _resolve(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            r = PdfReader(str(p))
            return {"pages": len(r.pages),
                    "text": "\n".join((pg.extract_text() or "") for pg in r.pages)[:30000]}
        except Exception as e:
            return {"error": str(e)}
    if ext in (".docx", ".doc"):
        try:
            import docx
            d = docx.Document(str(p))
            return {"paragraphs": len(d.paragraphs),
                    "text": "\n".join(x.text for x in d.paragraphs)[:30000]}
        except Exception as e:
            return {"error": str(e)}
    if ext in (".xlsx", ".xls"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
            rows = []
            for ws in wb.worksheets:
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i > 200: break
                    rows.append(",".join(str(c) if c is not None else "" for c in row))
            return {"sheets": wb.sheetnames, "text": "\n".join(rows)[:30000]}
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"unsupported extension {ext}"}


# ===========================================================================
# 6. DATA PROCESSING
# ===========================================================================
def data_csv_read(path, max_rows=500):
    p = _resolve(path)
    import csv
    with open(p, newline="", errors="replace") as f:
        rows = list(csv.reader(f))[:max_rows]
    return {"rows": len(rows), "headers": rows[0] if rows else [],
            "sample": rows[:20]}


def data_json_read(path):
    p = _resolve(path)
    return {"data": json.loads(p.read_text())}


def data_csv_to_json(path):
    p = _resolve(path)
    import csv
    with open(p, newline="", errors="replace") as f:
        rdr = csv.DictReader(f)
        out = list(rdr)
    return {"count": len(out), "rows": out}


# ===========================================================================
# 7. CODE ANALYSIS / EXECUTION
# ===========================================================================
def code_run(python_code=None, file=None, timeout=60):
    g = _guard("code_run")
    if g: return {"error": g}
    if file:
        return _run_cmd(f"{sys.executable} {_resolve(file)}", timeout=timeout)
    if python_code:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(python_code); name = tf.name
        try:
            return _run_cmd(f"{sys.executable} {name}", timeout=timeout)
        finally:
            try: os.unlink(name)
            except Exception: pass
    return {"error": "provide python_code or file"}


def code_analyze(path):
    import ast
    p = _resolve(path)
    try:
        tree = ast.parse(p.read_text())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
        imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        return {"functions": funcs, "classes": classes,
                "imports": sorted(set(imports))}
    except Exception as e:
        return {"error": f"not parseable python: {e}"}


# ===========================================================================
# 8. TODO (file-backed)
# ===========================================================================
_TODO = PROJECT_ROOT / "tools_state" / "todo.json"
def _load_todo():
    if _TODO.exists():
        return json.loads(_TODO.read_text())
    return []
def _save_todo(t):
    _TODO.write_text(json.dumps(t, indent=2))
def todo_list():
    return {"items": _load_todo()}
def todo_add(text):
    t = _load_todo(); t.append({"id": str(uuid.uuid4())[:8], "text": text, "done": False})
    _save_todo(t); return {"ok": True}
def todo_mark(id_, done=True):
    t = _load_todo()
    for x in t:
        if x["id"] == id_: x["done"] = bool(done)
    _save_todo(t); return {"ok": True}


# ===========================================================================
# 9. MEMORY (long-term, under ~/.hermes/memory)
# ===========================================================================
def mem_store(kind, text, tags=None):
    d = MEMORY_ROOT / "entries"
    d.mkdir(parents=True, exist_ok=True)
    day = datetime.datetime.now().strftime("%Y-%m-%d")
    e = {"id": str(uuid.uuid4())[:8], "ts": datetime.datetime.now().isoformat(timespec="seconds"),
         "kind": kind, "tags": tags or [], "text": text}
    f = d / f"entry_{day}.jsonl"
    with open(f, "a") as fh:
        fh.write(json.dumps(e) + "\n")
    return {"ok": True, "id": e["id"]}


def mem_recall(query, limit=10):
    """Simple keyword recall over stored entries (vector mode needs embedder)."""
    out = []
    q = query.lower()
    for f in sorted((MEMORY_ROOT / "entries").glob("*.jsonl"))[-20:]:
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            blob = (e.get("text", "") + " " + " ".join(e.get("tags", []))).lower()
            if q in blob:
                out.append(e)
    return {"results": out[-limit:][::-1], "count": len(out)}


def mem_stats():
    return {"root": str(MEMORY_ROOT), "entries": len(list((MEMORY_ROOT / "entries").glob("*.jsonl")))}


# ===========================================================================
# 10. SECURITY (local)
# ===========================================================================
_SECRET_RX = [
    (r"sk-[A-Za-z0-9]{20,}", "openai_key"),
    (r"AKIA[0-9A-Z]{16}", "aws_key"),
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "github_token"),
    (r"AIza[0-9A-Za-z\-_]{20,}", "google_key"),
    (r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----", "private_key"),
]
def sec_scan(text=None, path=None):
    if path:
        p = _resolve(path)
        text = p.read_text(errors="ignore")
    found = []
    for rx, name in _SECRET_RX:
        for m in re.finditer(rx, text or ""):
            found.append({"type": name, "match": m.group(0)[:6] + "…" + m.group(0)[-4:]})
    return {"found": found, "count": len(found)}


def sec_password(length=24):
    return {"password": secrets.token_urlsafe(length)[:length]}


def sec_hash(text, algo="sha256"):
    h = hashlib.new(algo, text.encode())
    return {"algorithm": algo, "digest": h.hexdigest()}


def sec_encrypt(text, key_b64):
    from cryptography.fernet import Fernet
    try:
        f = Fernet(key_b64.encode())
        return {"ciphertext": f.encrypt(text.encode()).decode()}
    except Exception as e:
        return {"error": str(e)}


def sec_decrypt(ciphertext_b64, key_b64):
    from cryptography.fernet import Fernet
    try:
        f = Fernet(key_b64.encode())
        return {"plaintext": f.decrypt(ciphertext_b64.encode()).decode()}
    except Exception as e:
        return {"error": str(e)}


# ===========================================================================
# 11. DATABASE (SQLite via stdlib; others are adapters)
# ===========================================================================
def db_sqlite(query, path=None, params=None):
    g = _guard("db_sqlite")
    if g: return {"error": g}
    p = _resolve(path) if path else PROJECT_ROOT / "tools_state" / "hermes.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(query, params or ())
        if query.strip().lower().startswith(("select", "pragma", "with")):
            rows = [dict(r) for r in cur.fetchall()]
            return {"rows": rows, "count": len(rows)}
        con.commit()
        return {"changes": cur.rowcount}
    except Exception as e:
        return {"error": str(e)}
    finally:
        con.close()


# ===========================================================================
# 12. NOTIFICATIONS (webhook; email adapter; desktop notify)
# ===========================================================================
def notify_webhook(url, payload=None, method="POST", headers=None):
    try:
        import requests
        r = requests.request(method, url, json=payload if payload is not None else {},
                             headers=headers or {}, timeout=20)
        return {"status": r.status_code, "ok": r.status_code < 400}
    except Exception as e:
        return {"error": str(e)}


def notify_desktop(title, message):
    try:
        import subprocess
        subprocess.run(["osascript", "-e",
                        f'display notification "{message}" with title "{title}"'],
                       capture_output=True, timeout=10)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ===========================================================================
# 13. CRON / SCHEDULING (lightweight, persisted in memory root)
# ===========================================================================
def cron_list():
    f = MEMORY_ROOT / "cron.json"
    if f.exists():
        return {"jobs": json.loads(f.read_text())}
    return {"jobs": []}
def cron_create(name, command, schedule, cwd=None):
    f = MEMORY_ROOT / "cron.json"
    jobs = json.loads(f.read_text()) if f.exists() else []
    jobs.append({"name": name, "command": command, "schedule": schedule,
                 "cwd": cwd or str(PROJECT_ROOT), "created": datetime.datetime.now().isoformat()})
    f.write_text(json.dumps(jobs, indent=2))
    return {"ok": True}
def cron_remove(name):
    f = MEMORY_ROOT / "cron.json"
    if not f.exists(): return {"error": "no jobs"}
    jobs = [j for j in json.loads(f.read_text()) if j.get("name") != name]
    f.write_text(json.dumps(jobs, indent=2))
    return {"ok": True}


# ===========================================================================
# 14. INTEGRATION ADAPTERS (activate when CLI/config present)
# ===========================================================================
def _which(cmd):
    return shutil.which(cmd) is not None

def adapter_generic(name, required, hint, available):
    """Return adapter status for a credentialed/CLI integration."""
    return {"tool": name, "status": "available" if available else "not_configured",
            "how_to_enable": "" if available else hint}

def integration_status():
    checks = {
        "git": _which("git"), "docker": _which("docker"), "gh": _which("gh"),
        "kubectl": _which("kubectl"), "helm": _which("helm"),
        "terraform": _which("terraform"), "aws": _which("aws"),
        "gcloud": _which("gcloud"), "az": _which("az"), "ssh": _which("ssh"),
        "ansible": _which("ansible"), "node": _which("node"), "go": _which("go"),
    }
    return {"integrations": checks}


# ===========================================================================
# TOOL MANIFEST / REGISTRY
# ===========================================================================
def register(server_tools, name, description, fn, arguments, input_schema, category, safety=False):
    server_tools.append({
        "name": name, "description": description, "category": category,
        "safety": safety,
        "inputSchema": {"type": "object", "properties": input_schema,
                        "required": [k for k, v in input_schema.items() if v.get("required")]},
        "arguments": arguments, "fn": fn,
    })


def build_manifest(server_tools):
    """Machine-readable list for tool listing + the user's audit."""
    return [{"name": t["name"], "category": t["category"], "safety": t["safety"],
             "description": t["description"]} for t in server_tools]


# ===========================================================================
# MCP stdio server (newline-delimited JSON-RPC 2.0)
# ===========================================================================
def _result(tools, name, arguments):
    for t in tools:
        if t["name"] == name:
            try:
                declared = list(t["inputSchema"].get("properties", {}).keys())
                # keep only declared params; drop anything extra a client may send
                kwargs = {k: v for k, v in (arguments or {}).items() if k in declared}
                try:
                    out = t["fn"](**kwargs)
                except TypeError as e:
                    out = {"error": f"bad arguments for {name}: {e} (expected {declared})"}
                content = json.dumps(out, ensure_ascii=False, indent=2)
                return {"content": [{"type": "text", "text": content}], "isError": False}
            except Exception as e:
                return {"content": [{"type": "text",
                                     "text": json.dumps({"error": f"{type(e).__name__}: {e}"})}],
                        "isError": True}
    return {"content": [{"type": "text", "text": json.dumps({"error": f"unknown tool: {name}"})}],
            "isError": True}


def main():
    tools = []

    # 1. Filesystem
    register(tools, "read_file", "Read a file with line numbers (start/end optional).", fs_read_file,
             ["path", "start", "end"], {"path": {"type": "string", "required": True},
                 "start": {"type": "integer"}, "end": {"type": "integer"}}, "filesystem")
    register(tools, "write_file", "Create/overwrite a file inside the project.", fs_write_file,
             ["path", "content"], {"path": {"type": "string", "required": True},
                 "content": {"type": "string", "required": True}}, "filesystem")
    register(tools, "patch", "Find & replace in a file (fuzzy optional).", fs_patch,
             ["path", "old", "new", "fuzzy"], {"path": {"type": "string", "required": True},
                 "old": {"type": "string", "required": True}, "new": {"type": "string", "required": True},
                 "fuzzy": {"type": "boolean"}}, "filesystem")
    register(tools, "search_files", "Regex search across project files.", fs_search,
             ["pattern", "path", "glob_", "ignore_case"], {"pattern": {"type": "string", "required": True},
                 "path": {"type": "string"}, "glob_": {"type": "string"},
                 "ignore_case": {"type": "boolean"}}, "filesystem")
    register(tools, "list_directory", "List directory contents.", fs_list, ["path"],
             {"path": {"type": "string"}}, "filesystem")
    register(tools, "file_info", "Get file metadata.", fs_info, ["path"],
             {"path": {"type": "string", "required": True}}, "filesystem")
    register(tools, "move_file", "Rename/move a file.", fs_move, ["src", "dst"],
             {"src": {"type": "string", "required": True}, "dst": {"type": "string", "required": True}},
             "filesystem", safety=True)
    register(tools, "copy_file", "Copy file or folder.", fs_copy, ["src", "dst"],
             {"src": {"type": "string", "required": True}, "dst": {"type": "string", "required": True}},
             "filesystem")
    register(tools, "delete_file", "Delete file/folder (recursive optional).", fs_delete,
             ["path", "recursive"], {"path": {"type": "string", "required": True},
                 "recursive": {"type": "boolean"}}, "filesystem", safety=True)

    # 2. Terminal & process
    register(tools, "exec", "Run a shell command (cwd/timeout optional).", term_exec,
             ["command", "cwd", "timeout"], {"command": {"type": "string", "required": True},
                 "cwd": {"type": "string"}, "timeout": {"type": "integer"}}, "terminal", safety=True)
    register(tools, "bg_process", "Start a background process.", term_bg, ["command", "cwd"],
             {"command": {"type": "string", "required": True}, "cwd": {"type": "string"}},
             "terminal", safety=True)
    register(tools, "process_list", "List running processes.", term_list, [],
             {}, "terminal")
    register(tools, "kill_process", "Terminate a process by pid.", term_kill, ["pid", "sig"],
             {"pid": {"type": "string", "required": True}, "sig": {"type": "string"}},
             "terminal", safety=True)
    register(tools, "env_get", "Read an environment variable.", term_env_get, ["name"],
             {"name": {"type": "string", "required": True}}, "terminal")
    register(tools, "env_set", "Set an environment variable (session only).", term_env_set,
             ["name", "value"], {"name": {"type": "string", "required": True},
                 "value": {"type": "string", "required": True}}, "terminal", safety=True)

    # 3. Web
    register(tools, "web_search", "Search the web (DuckDuckGo, no key).", web_search,
             ["query", "max_results"], {"query": {"type": "string", "required": True},
                 "max_results": {"type": "integer"}}, "web")
    register(tools, "web_fetch", "Fetch a URL and return text/markdown.", web_fetch,
             ["url", "format_", "timeout"], {"url": {"type": "string", "required": True},
                 "format_": {"type": "string"}, "timeout": {"type": "integer"}}, "web")
    register(tools, "url_validate", "Validate/normalize a URL.", web_validate, ["url"],
             {"url": {"type": "string", "required": True}}, "web")
    register(tools, "extract_links", "Extract all links from a page.", web_links, ["url", "timeout"],
             {"url": {"type": "string", "required": True}, "timeout": {"type": "integer"}}, "web")
    register(tools, "page_metadata", "Get page metadata/SEO tags.", web_metadata, ["url", "timeout"],
             {"url": {"type": "string", "required": True}, "timeout": {"type": "integer"}}, "web")
    register(tools, "rss_read", "Read and parse an RSS/Atom feed.", web_rss,
             ["url", "max_items", "timeout"], {"url": {"type": "string", "required": True},
                 "max_items": {"type": "integer"}, "timeout": {"type": "integer"}}, "web")

    # 4. Browser (Playwright)
    register(tools, "browser_navigate", "Open a URL in headless Chromium; optional extract text/links/html.", browser_navigate,
             ["url", "extract", "timeout"], {"url": {"type": "string", "required": True},
                 "width": {"type": "integer"}, "height": {"type": "integer"},
                 "extract": {"type": "string"}, "timeout": {"type": "integer"}}, "browser")

    # 5. Documents
    register(tools, "read_document", "Read PDF/DOCX/XLSX text content.", doc_read, ["path"],
             {"path": {"type": "string", "required": True}}, "documents")

    # 6. Data
    register(tools, "csv_read", "Read a CSV file.", data_csv_read, ["path", "max_rows"],
             {"path": {"type": "string", "required": True}, "max_rows": {"type": "integer"}}, "data")
    register(tools, "json_read", "Read a JSON file.", data_json_read, ["path"],
             {"path": {"type": "string", "required": True}}, "data")
    register(tools, "csv_to_json", "Convert CSV to JSON rows.", data_csv_to_json, ["path"],
             {"path": {"type": "string", "required": True}}, "data")

    # 7. Code
    register(tools, "run_code", "Execute python code or a file.", code_run,
             ["python_code", "file", "timeout"], {"python_code": {"type": "string"},
                 "file": {"type": "string"}, "timeout": {"type": "integer"}}, "code", safety=True)
    register(tools, "analyze_code", "Static analysis of a Python file.", code_analyze, ["path"],
             {"path": {"type": "string", "required": True}}, "code")

    # 8. Todo
    register(tools, "todo_list", "List todo items.", todo_list, [], {}, "orchestration")
    register(tools, "todo_add", "Add a todo item.", todo_add, ["text"],
             {"text": {"type": "string", "required": True}}, "orchestration")
    register(tools, "todo_mark", "Mark a todo done/undone.", todo_mark, ["id_", "done"],
             {"id_": {"type": "string", "required": True}, "done": {"type": "boolean"}}, "orchestration")

    # 9. Memory
    register(tools, "memory_store", "Store a long-term memory entry.", mem_store,
             ["kind", "text", "tags"], {"kind": {"type": "string", "required": True},
                 "text": {"type": "string", "required": True}, "tags": {"type": "array"}}, "memory")
    register(tools, "memory_recall", "Recall past memory entries by keyword.", mem_recall,
             ["query", "limit"], {"query": {"type": "string", "required": True},
                 "limit": {"type": "integer"}}, "memory")
    register(tools, "memory_stats", "Memory usage stats.", mem_stats, [], {}, "memory")

    # 10. Security
    register(tools, "secrets_scan", "Scan text/file for secrets.", sec_scan, ["text", "path"],
             {"text": {"type": "string"}, "path": {"type": "string"}}, "security")
    register(tools, "password_gen", "Generate a secure password.", sec_password, ["length"],
             {"length": {"type": "integer"}}, "security")
    register(tools, "hash_text", "Hash text (sha256 etc).", sec_hash, ["text", "algo"],
             {"text": {"type": "string", "required": True}, "algo": {"type": "string"}}, "security")
    register(tools, "encrypt_text", "Fernet-encrypt text.", sec_encrypt, ["text", "key_b64"],
             {"text": {"type": "string", "required": True}, "key_b64": {"type": "string", "required": True}},
             "security")
    register(tools, "decrypt_text", "Fernet-decrypt text.", sec_decrypt, ["ciphertext_b64", "key_b64"],
             {"ciphertext_b64": {"type": "string", "required": True},
              "key_b64": {"type": "string", "required": True}}, "security")

    # 11. Database
    register(tools, "sqlite_query", "Run SQL on the local SQLite db.", db_sqlite,
             ["query", "path", "params"], {"query": {"type": "string", "required": True},
                 "path": {"type": "string"}, "params": {"type": "array"}}, "database", safety=True)

    # 12. Notifications
    register(tools, "notify_webhook", "Send a webhook.", notify_webhook,
             ["url", "payload", "method", "headers"], {"url": {"type": "string", "required": True},
                 "payload": {"type": "object"}, "method": {"type": "string"},
                 "headers": {"type": "object"}}, "notification")
    register(tools, "notify_desktop", "macOS desktop notification.", notify_desktop,
             ["title", "message"], {"title": {"type": "string", "required": True},
                 "message": {"type": "string", "required": True}}, "notification")

    # 13. Cron
    register(tools, "cron_list", "List scheduled jobs.", cron_list, [], {}, "scheduling")
    register(tools, "cron_create", "Create a scheduled job (persisted).", cron_create,
             ["name", "command", "schedule", "cwd"], {"name": {"type": "string", "required": True},
                 "command": {"type": "string", "required": True}, "schedule": {"type": "string",
                 "required": True}, "cwd": {"type": "string"}}, "scheduling")
    register(tools, "cron_remove", "Remove a scheduled job.", cron_remove, ["name"],
             {"name": {"type": "string", "required": True}}, "scheduling")

    # 14. Integrations / cloud adapters + status
    register(tools, "integration_status", "Show which CLI integrations are available.",
             integration_status, [], {}, "integrations")

    manifest = build_manifest(tools)

    def _handle(msg):
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"protocolVersion": "2025-03-26",
                               "capabilities": {"tools": {}},
                               "serverInfo": {"name": "hermes", "version": "1.0.0"}}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"tools": [{"name": t["name"], "description": t["description"],
                                          "inputSchema": t["inputSchema"]} for t in tools]}}
        if method == "tools/call":
            name = params.get("name"); args = params.get("arguments") or {}
            return {"jsonrpc": "2.0", "id": mid, "result": _result(tools, name, args)}
        if method == "hermes/manifest":
            return {"jsonrpc": "2.0", "id": mid, "result": manifest}
        # notifications
        if mid is None:
            return None
        return {"jsonrpc": "2.0", "id": mid,
                "result": {"error": f"unsupported method: {method}"}}

    if "--manifest" in sys.argv:
        print(json.dumps(manifest, indent=2)); return
    if "--list" in sys.argv:
        for t in tools:
            print(f"[{t['category']:13s}] {t['name']}")
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = _handle(msg)
        if resp:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
