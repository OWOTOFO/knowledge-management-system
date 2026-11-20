"""Score the frozen SEC queries against one already-built document index."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

import yaml

from km.config import get_settings
from km.text import normalize_quote


def matches(group: dict, passage: dict) -> bool:
    meta = passage["metadata"]
    if Path(meta["source"]).stem != group["filing"]:
        return False
    if not normalize_quote(meta.get("heading") or "").startswith(normalize_quote(group["section_prefix"])):
        return False
    body = normalize_quote(passage["text"].split("\n", 1)[-1])
    return all(normalize_quote(anchor) in body for anchor in group["anchors"])


def score_case(case: dict, docs, k: int) -> dict:
    query = case.get("doc_query", case["question"])
    result = {"id": case["id"], "query": query}
    start = time.perf_counter()
    try:
        hits = docs.search(query, k=k, tickers=case["tickers"])
    except Exception as exc:
        return result | {"error": type(exc).__name__}
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    ranks = {}
    for group in case["evidence"]:
        ranks[group["id"]] = next(
            (rank for rank, hit in enumerate(hits, 1) if matches(group, hit)), None
        )
    found_ranks = [rank for rank in ranks.values() if rank is not None]
    return result | {
        "group_first_ranks": ranks,
        "hit": bool(found_ranks),
        "complete": len(found_ranks) == len(ranks),
        "group_coverage": len(found_ranks) / len(ranks),
        "reciprocal_rank": 1 / min(found_ranks) if found_ranks else 0.0,
        "latency_ms": elapsed_ms,
        "top_hits": [
            {"rank": rank, "chunk_id": hit["chunk_id"], **{
                field: hit["metadata"].get(field) for field in ("source", "heading", "para", "fiscal_year", "url")
            }}
            for rank, hit in enumerate(hits, 1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Save ranks and metrics as JSON")
    parser.add_argument("--verify-only", action="store_true", help="Check gold anchors locally without calling the model")
    args = parser.parse_args()
    settings = get_settings()
    suite = yaml.safe_load(Path(__file__).with_name("sec_embedding_ab.yaml").read_text(encoding="utf-8"))
    raw_chunks = (settings.index_path / "chunks.json").read_bytes()
    chunks = json.loads(raw_chunks)
    cases = suite["retrieval_cases"]
    diagnostics = suite["diagnostics"]
    groups = [(c["id"], g) for c in cases + diagnostics for g in c["evidence"]]
    missing = [f"{case}/{group['id']}" for case, group in groups if not any(matches(group, p) for p in chunks)]
    if missing:
        parser.exit(1, "Missing gold evidence in this index: " + ", ".join(missing) + "\n")
    if args.verify_only:
        print(f"Verified {len(groups)} evidence groups in {len(chunks)} indexed passages without model calls.")
        return 0
    if args.report is None:
        parser.error("--report is required unless using --verify-only")

    from km.llm.providers import get_embeddings
    from km.retrieval.docs import DocRetriever

    docs = DocRetriever(settings, get_embeddings(settings))
    provider = settings.effective_embedding_provider
    if docs.embedding_provider != provider:
        parser.exit(1, "The index was built for a different embedding provider. Rebuild in its own directory.\n")
    k = suite["k"]
    results = [score_case(c, docs, k) for c in cases]
    errors = [r["id"] for r in results if "error" in r]
    summary = {"cases": len(results), "query_errors": errors}
    if not errors:
        summary |= {
            f"hit_at_{k}": sum(r["hit"] for r in results),
            f"complete_at_{k}": sum(r["complete"] for r in results),
            f"mean_group_coverage_at_{k}": round(mean(r["group_coverage"] for r in results), 4),
            f"mrr_at_{k}": round(mean(r["reciprocal_rank"] for r in results), 4),
        }
    report = {
        "suite": suite["name"],
        "embedding_provider": provider,
        "embedding_model": settings.openrouter_embedding_model if provider == "openrouter" else None,
        "chunks_sha256": hashlib.sha256(raw_chunks).hexdigest(),
        "settings": {"k": k, "doc_min_similarity": settings.doc_min_similarity,
                     "doc_min_coverage": settings.doc_min_coverage},
        "summary": summary,
        "results": results,
        "diagnostics": [score_case(c, docs, k) for c in diagnostics],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
