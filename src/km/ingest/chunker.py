"""Turn paragraph Sections into retrieval chunks: one chunk per paragraph, stable readable ids."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from km.ingest.loaders import Section


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _slug(s: str) -> str:
    m = re.match(r"(?:part\s+([iv]+),\s*)?item\s+(\d+[a-c]?)", s, re.I)
    if m:
        return (f"part-{m.group(1).lower()}-" if m.group(1) else "") + f"item-{m.group(2).lower()}"
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def location(meta: dict) -> str:
    """Human-readable position, e.g. 'Item 7. Management's Discussion…, paragraph 12'."""
    parts = [meta["heading"]] if meta.get("heading") else []
    if meta.get("page"):
        parts.append(f"page {meta['page']}")
    if meta.get("para"):
        parts.append(f"paragraph {meta['para']}")
    return ", ".join(parts)


def chunk_sections(sections: list[Section], size: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for sec in sections:
        where = _slug(sec.heading) if sec.heading else f"page{sec.page}" if sec.page else "body"
        # shortcut: a paragraph/table longer than chunk_size is split and later parts lose a table's header row.
        pieces = [sec.text] if len(sec.text) <= size else splitter.split_text(sec.text)
        for i, piece in enumerate(pieces):
            cid = f"{sec.source}::{where}::p{sec.para}" + (f"-{i + 1}" if len(pieces) > 1 else "")
            while cid in seen:
                cid += "b"
            seen.add(cid)
            meta = {"chunk_id": cid, "source": sec.source, "title": sec.title, "heading": sec.heading,
                    "page": sec.page, "para": sec.para, **sec.meta}
            meta = {k: v for k, v in meta.items() if v is not None}   # Chroma rejects None values
            # Contextual header: the year spelled out so a query mentioning "2024" matches.
            year = f" (fiscal year {sec.meta['fiscal_year']})" if sec.meta.get("fiscal_year") else ""
            header = f"{sec.title}{year} | {location(meta)}"
            chunks.append(Chunk(cid, f"{header}\n{piece}", meta))
    return chunks
