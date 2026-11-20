"""Offline stand-ins so the whole pipeline runs without network access or credentials.

Only embeddings need a fake *model*; every LLM-backed component (router, NL->SQL, generator,
judge) has its own deterministic heuristic fallback that is used when no LLM is configured.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from km.text import content_tokens


class HashingEmbeddings(Embeddings):
    """Bag-of-words feature hashing. Deterministic, no downloads, good enough for a demo."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in content_tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
