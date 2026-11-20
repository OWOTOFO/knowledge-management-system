"""Build the indexes: Chroma vector store (dense) + chunk file for BM25 (sparse) + SEC facts tables."""

from __future__ import annotations

import json
from collections import Counter
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from km.config import Settings
from km.ingest.chunker import chunk_sections
from km.ingest.loaders import load_documents
from km.sec import load_companyfacts

COLLECTION = "km_docs"
BATCH = 1000   # Chroma rejects very large single inserts; 3 years of 2 companies' 10-Ks is ~6k chunks


def open_store(s: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(s.index_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(s: Settings, embeddings: Embeddings,
                on_progress: Callable[[int, int], None] | None = None) -> dict:
    if not s.docs_dir.exists():
        raise FileNotFoundError(f"{s.docs_dir} not found - add documents first")
    embeddings.embed_query("ping")   # a wrong model or key fails here, before anything is rewritten
    sections = load_documents(s.docs_dir)
    if not sections:
        raise ValueError(f"No supported documents found in {s.docs_dir}")

    sec_summary = None
    json_paths = sorted(s.docs_dir.glob("CIK*.json"))
    if json_paths:
        s.db_path.parent.mkdir(parents=True, exist_ok=True)
        docs = {sec.source: sec.meta for sec in sections if sec.meta.get("cik")}
        sec_summary = load_companyfacts(s.db_path, json_paths, docs)
        for sec in sections:
            if sec.source in sec_summary["urls"]:
                sec.meta["url"] = sec_summary["urls"][sec.source]

    chunks = chunk_sections(sections, s.chunk_size, s.chunk_overlap)
    s.index_path.mkdir(parents=True, exist_ok=True)
    # chunks.json first: if embedding fails half-way, retrieval still knows every chunk id.
    (s.index_path / "chunks.json").write_text(
        json.dumps([{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks], indent=1))

    open_store(s, embeddings).delete_collection()   # full rebuild keeps it idempotent
    store = open_store(s, embeddings)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        store.add_texts(texts=[c.text for c in batch], metadatas=[c.metadata for c in batch],
                        ids=[c.id for c in batch])
        if on_progress:
            on_progress(min(i + BATCH, len(chunks)), len(chunks))

    counts = Counter(c.metadata["source"] for c in chunks)
    first = {}
    for c in chunks:
        first.setdefault(c.metadata["source"], c.metadata)
    documents = [{"source": src, "title": first[src]["title"], "paragraphs": n,
                  **{k: first[src][k] for k in ("ticker", "company", "fiscal_year", "url") if k in first[src]}}
                 for src, n in sorted(counts.items())]
    catalog = {"embedding_provider": s.effective_embedding_provider, "embedding_model": s.embedding_id,
               "documents": documents}
    (s.index_path / "catalog.json").write_text(json.dumps(catalog, indent=1))
    return {"documents": len(documents), "chunks": len(chunks), "catalog": documents, "sec": sec_summary}
