"""Document search rules that keep answers on the right company, checked on a tiny hand-made index."""

import json

from km.config import Settings
from km.ingest.index import open_store
from km.llm.mock import HashingEmbeddings
from km.retrieval.docs import DocRetriever


def _retriever(tmp_path):
    s = Settings(llm_provider="mock", index_dir=tmp_path / "index", db_path=tmp_path / "km.db", _env_file=None)
    texts = {"AAPL": ["Apple paid dividends and dividend equivalents of $15.4 billion."],
             "META": [f"Meta dividends cash flow dividends paid note {i}." for i in range(30)]}
    chunks = [{"id": f"{t}-{i}", "text": f"{t} 10-K | Item 7, paragraph {i}\n{body}",
               "metadata": {"chunk_id": f"{t}-{i}", "source": f"{t}.htm", "ticker": t}}
              for t, bodies in texts.items() for i, body in enumerate(bodies)]
    s.index_path.mkdir(parents=True)
    (s.index_path / "chunks.json").write_text(json.dumps(chunks))
    open_store(s, HashingEmbeddings()).add_texts([c["text"] for c in chunks], [c["metadata"] for c in chunks],
                                                 ids=[c["id"] for c in chunks])
    return DocRetriever(s, HashingEmbeddings())


def test_company_filter_and_one_search_per_company(tmp_path):
    d = _retriever(tmp_path)
    q = "What dividends were paid?"
    assert {h["metadata"]["ticker"] for h in d.search(q, k=4, tickers=["META"])} == {"META"}   # never the other company
    both = d.search(q, k=4, tickers=["AAPL", "META"])
    # Meta has 30 matching paragraphs to Apple's one; each company is searched on its own, so Apple still shows
    assert [h["metadata"]["ticker"] for h in both[:2]] == ["AAPL", "META"]
    assert len({h["chunk_id"] for h in both}) == len(both)


def test_question_without_english_words_does_not_crash(tmp_path):
    assert _retriever(tmp_path).search("苹果公司的股息是多少？", k=4) == []   # offline embeddings: nothing similar, no error


def test_an_index_built_with_another_embedding_model_says_rebuild(tmp_path):
    import pytest
    d = _retriever(tmp_path)
    (d.s.index_path / "catalog.json").write_text(json.dumps({"embedding_provider": "mock",
                                                            "embedding_model": "openrouter:qwen/qwen3-embedding-8b"}))
    with pytest.raises(RuntimeError, match="Rebuild the index"):
        DocRetriever(d.s, HashingEmbeddings())


def test_each_embedding_model_keeps_its_own_saved_index(tmp_path):
    offline = Settings(llm_provider="mock", index_dir=tmp_path, _env_file=None)
    qwen = Settings(llm_provider="openrouter", openrouter_embedding_model="qwen/qwen3-embedding-8b",
                    index_dir=tmp_path, _env_file=None)
    assert offline.index_path == tmp_path / "mock"
    assert qwen.index_path == tmp_path / "openrouter_qwen_qwen3-embedding-8b"   # switching back reuses this one


def test_an_index_from_before_per_model_folders_is_still_used(tmp_path):
    import json as _json
    (tmp_path / "catalog.json").write_text(_json.dumps({"embedding_model": "openrouter:qwen/qwen3-embedding-8b"}))
    qwen = Settings(llm_provider="openrouter", openrouter_embedding_model="qwen/qwen3-embedding-8b",
                    index_dir=tmp_path, _env_file=None)
    assert qwen.index_path == tmp_path                                   # the existing build, no rebuild needed
    assert Settings(llm_provider="mock", index_dir=tmp_path, _env_file=None).index_path == tmp_path / "mock"
