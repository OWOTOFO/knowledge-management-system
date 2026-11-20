"""Hybrid document retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank Fusion.

A relevance gate drops passages that are neither semantically close (cosine similarity) nor
lexically on-topic (query-term coverage).  An empty result is a *signal*, not an error: it is
what lets the pipeline say "Failed to find an answer" instead of guessing.
"""

from __future__ import annotations

import json
from itertools import zip_longest

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from km.config import Settings
from km.ingest.index import open_store
from km.text import content_tokens

RRF_K = 60
# The best few of each ranking always make the cut. Fusion alone favours paragraphs found by both rankings, so
# it dropped a paragraph ranked #1 on meaning but #111 on keywords (measured on eval/sec_embedding_ab.yaml).
KEEP_TOP = 3


class DocRetriever:
    def __init__(self, s: Settings, embeddings: Embeddings):
        self.s = s
        path = s.index_path / "chunks.json"
        self.chunks: list[dict] = json.loads(path.read_text()) if path.exists() else []
        self.idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        tokens = [content_tokens(c["text"]) for c in self.chunks]
        self.tokens = [set(t) for t in tokens]
        self.bm25 = BM25Okapi(tokens) if tokens else None
        self.store = open_store(s, embeddings) if self.chunks else None
        catalog_path = s.index_path / "catalog.json"
        catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
        self.catalog: list[dict] = catalog.get("documents", [])
        self.embedding_provider = catalog.get("embedding_provider")
        built = catalog.get("embedding_model")
        if built and built != s.embedding_id:   # vectors from another model would give a crash or nonsense
            raise RuntimeError(f"The index was built with {built}, but the settings now use {s.embedding_id}. "
                               "Rebuild the index (Library tab, or `km ingest`).")

    def search(self, query: str, k: int | None = None, tickers: list[str] | None = None) -> list[dict]:
        k = k or self.s.top_k
        if not self.chunks:
            return []
        if tickers and len(tickers) > 1:
            # A comparison needs evidence on every company: search each on its own (k each) and interleave,
            # otherwise the company with more matching paragraphs crowds the other out.
            per_company = [self._search_one(query, k, [t]) for t in tickers]
            return [h for group in zip_longest(*per_company) for h in group if h is not None]
        return self._search_one(query, k, tickers)

    def _search_one(self, query: str, k: int, tickers: list[str] | None) -> list[dict]:
        q_tokens = content_tokens(query)   # empty for a question without English words or digits: meaning-only search
        allowed = range(len(self.chunks))
        where = None
        if tickers:   # hard company filter: an Apple question must never cite a Meta paragraph
            allowed = [i for i, c in enumerate(self.chunks) if c["metadata"].get("ticker") in tickers]
            where = {"ticker": {"$in": list(tickers)}}
            if not allowed:
                return []
        n_candidates = min(len(allowed), k * 3)

        dense_hits = self.store.similarity_search_with_relevance_scores(query, k=n_candidates, filter=where)
        dense_hits = [(d, sc) for d, sc in dense_hits if d.metadata.get("chunk_id") in self.idx]
        dense_sim = {d.metadata["chunk_id"]: float(sc) for d, sc in dense_hits}
        dense_rank = [d.metadata["chunk_id"] for d, _ in dense_hits]

        sparse_rank = []
        if q_tokens:
            scores = self.bm25.get_scores(q_tokens)
            order = sorted(allowed, key=lambda i: scores[i], reverse=True)[:n_candidates]
            sparse_rank = [self.chunks[i]["id"] for i in order if scores[i] > 0]

        rrf: dict[str, float] = {}
        for ranking in (dense_rank, sparse_rank):
            for rank, cid in enumerate(ranking):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = [c for pair in zip_longest(dense_rank[:KEEP_TOP], sparse_rank[:KEEP_TOP]) for c in pair if c]
        ranked += sorted(rrf, key=rrf.get, reverse=True)
        q_set = set(q_tokens)
        hits = []
        for cid in dict.fromkeys(ranked):   # unique, order kept
            i = self.idx[cid]
            coverage = len(q_set & self.tokens[i]) / len(q_set) if q_set else 0.0
            sim = dense_sim.get(cid, 0.0)
            if sim < self.s.doc_min_similarity and coverage < self.s.doc_min_coverage:
                continue  # relevance gate
            c = self.chunks[i]
            hits.append({"chunk_id": cid, "text": c["text"], "metadata": c["metadata"],
                         "dense_similarity": round(sim, 3), "term_coverage": round(coverage, 3),
                         "rrf": round(rrf[cid], 5)})
            if len(hits) >= k:
                break
        return hits
