"""dashboard/secrets.py — secure storage of provider API keys.

Storage backends, in order of preference:

1. macOS Keychain — via the `security` CLI (no Python dependencies). Keys never
   touch the filesystem unencrypted.
2. Fernet-encrypted file — fallback for non-macOS. The key lives in the same
   ~/.hybrid-agent dir with 0600 permissions. This protects against casual
   inspection, not against someone with full access to your account; on macOS
   the Keychain backend is always used instead.

The data directory is ~/.hybrid-agent (override with $HYBRID_AGENT_HOME for
tests). Keys are stored per provider name. Keys are NEVER exposed by the API —
callers only ever see has-key booleans.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:  # cryptography is a venv dep (used only by the fallback backend)
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover - dashboard requires cryptography
    Fernet = None

KEYCHAIN_ACCOUNT = "hybrid-agent"


def _home() -> Path:
    return Path(os.environ.get("HYBRID_AGENT_HOME", str(Path.home() / ".hybrid-agent")))


def KEY_FILE():
    return _home() / "secret.key"


def DATA_FILE():
    return _home() / "secrets.json"


def _keychain_available() -> bool:
    if sys.platform != "darwin" or os.environ.get("HYBRID_SECRETS_BACKEND") == "file":
        return False
    try:
        subprocess.run(["security", "help"], capture_output=True, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _keychain(name: str, action: str, value: str = "") -> str:
    """action: 'set' | 'get' | 'delete'. Returns the secret for 'get'."""
    if action == "set":
        cmd = ["security", "add-generic-password", "-U", "-a", KEYCHAIN_ACCOUNT,
               "-s", name, "-w", value]
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
        return ""
    if action == "delete":
        subprocess.run(["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT,
                        "-s", name], capture_output=True, timeout=10, check=False)
        return ""
    proc = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", name, "-w"],
        capture_output=True, text=True, timeout=10)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _fernet() -> "Fernet":
    if Fernet is None:  # pragma: no cover - guarded elsewhere
        raise RuntimeError("cryptography is required for the file secrets backend")
    if not KEY_FILE().exists():
        KEY_FILE().parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE().write_bytes(Fernet.generate_key())
        os.chmod(KEY_FILE(), 0o600)
    return Fernet(KEY_FILE().read_bytes())


def _load_file() -> dict:
    if not DATA_FILE().is_file():
        return {}
    try:
        data = json.loads(DATA_FILE().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_file(data: dict) -> None:
    DATA_FILE().parent.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_FILE().parent, 0o700)
    tmp = DATA_FILE().with_suffix(DATA_FILE().suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, DATA_FILE())
    os.chmod(DATA_FILE(), 0o600)


def _file_encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def _file_decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except Exception:  # noqa: BLE001 - corrupt entry behaves as missing
        return ""


def set_secret(name: str, value: str) -> None:
    """Store a provider API key (Keychain on macOS, Fernet file elsewhere)."""
    if not name or value is None:
        return
    if _keychain_available():
        _keychain(name, "set", value)
        return
    data = _load_file()
    data[name] = _file_encrypt(value)
    _save_file(data)


def get_secret(name: str) -> str:
    """Return the stored key for a provider, or '' when absent."""
    if not name:
        return ""
    if _keychain_available():
        return _keychain(name, "get")
    enc = _load_file().get(name, "")
    return _file_decrypt(enc) if enc else ""


def delete_secret(name: str) -> None:
    if _keychain_available():
        _keychain(name, "delete")
        return
    data = _load_file()
    if name in data:
        data.pop(name, None)
        _save_file(data)


def has_secret(name: str) -> bool:
    if _keychain_available():
        return bool(_keychain(name, "get"))
    enc = _load_file().get(name, "")
    return bool(enc) and bool(_file_decrypt(enc))
