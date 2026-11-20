import pytest

from km.config import Settings
from km.ingest.index import build_index
from km.llm.providers import get_embeddings
from km.sample_data import generate


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("km")
    s = Settings(
        llm_provider="mock",
        embedding_provider="mock",
        docs_dir=root / "docs",
        db_path=root / "km.db",
        index_dir=root / "index",
        audit_log=root / "audit.jsonl",
    )
    generate(s.docs_dir, s.db_path)
    build_index(s, get_embeddings(s))
    return s


@pytest.fixture(scope="session")
def pipeline(settings):
    from km.graph import KMPipeline

    return KMPipeline(settings)
