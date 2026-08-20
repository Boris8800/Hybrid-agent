"""embed.py — Local semantic embeddings for the hybrid agent's memory.

Talks to an OpenAI-compatible /embeddings endpoint (LM Studio serves one at
http://localhost:1234/v1/embeddings) using the stdlib only. Every call is
error-safe: embed() returns None instead of raising, and the client disables
itself for the process after the first failure so a down endpoint cannot add
repeated timeouts to a long run. When embeddings are unavailable, callers fall
back to their previous heuristic (word-trigram) behavior.
"""

from __future__ import annotations

import json
import math
import urllib.request

DEFAULT_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_TIMEOUT_S = 10.0
# Cosine threshold for "same task domain" with the default embedding model.
# Calibrated against the real nomic model: paraphrased pairs land ~0.68-0.70,
# unrelated tasks ~0.32-0.44.
DEFAULT_SIMILARITY_THRESHOLD = 0.60


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1]; 0.0 for empty/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return max(0.0, min(1.0, dot / (na * nb)))


class EmbeddingClient:
    """Minimal OpenAI-compatible embeddings client (stdlib only)."""

    def __init__(self, base_url: str, api_key: str = "lm-studio",
                 model: str = DEFAULT_MODEL, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.url = (base_url or "").rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.disabled = False

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of texts; returns a list of vectors (input order) or
        None on any failure. After the first failure the client disables
        itself for the rest of the process."""
        if self.disabled or not self.url or not texts:
            return None
        try:
            payload = json.dumps({"model": self.model, "input": texts}).encode()
            req = urllib.request.Request(
                self.url, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode())
            entries = sorted(raw.get("data", []), key=lambda d: d.get("index", 0))
            vectors = [d.get("embedding") for d in entries]
            if not vectors or not all(vectors):
                raise ValueError("empty embedding payload")
            return vectors
        except Exception:  # noqa: BLE001 - embeddings are best-effort
            self.disabled = True
            return None


def make_embedding_client(cfg: dict):
    """Build an EmbeddingClient from the engine config (None when disabled by
    config or missing backend settings)."""
    try:
        mem = cfg.get("memory") or {}
        if not mem.get("semantic_similarity", True):
            return None
        lc = cfg["backends"]["local"]
        return EmbeddingClient(
            lc.get("base_url", ""),
            api_key=lc.get("api_key") or "lm-studio",
            model=mem.get("embedding_model", DEFAULT_MODEL),
        )
    except (KeyError, TypeError, AttributeError):
        return None


def memory_embed_callable(cfg: dict):
    """Return the embed() bound method for the configured client, or None when
    semantic memory is disabled/unavailable (callers fall back to trigrams)."""
    client = make_embedding_client(cfg)
    return client.embed if client is not None else None
