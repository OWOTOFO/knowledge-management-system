"""Central configuration, read from environment variables / a local .env file."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

Provider = Literal["mock", "azure_openai", "openai", "openrouter", "bedrock"]


class Settings(BaseSettings):
    # Empty values mean "use the default"; the literal "none" means None (e.g. LLM_TEMPERATURE=none).
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", env_ignore_empty=True,
                                      env_parse_none_str="none")

    # --- providers ---
    llm_provider: Provider = "mock"
    embedding_provider: Provider | None = None  # defaults to llm_provider
    llm_temperature: float | None = 0.0
    llm_thinking: bool = False   # OpenRouter models with a thinking mode: off is 3-7x faster per question

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    openrouter_api_key: str | None = None
    openrouter_chat_model: str = "deepseek/deepseek-v4.1-flash-1017"
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"

    aws_region: str = "us-east-1"
    bedrock_chat_model_id: str | None = None
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # --- storage ---
    docs_dir: Path = ROOT / "data" / "docs"
    db_path: Path = ROOT / "data" / "km.db"
    index_dir: Path = ROOT / ".index"
    audit_log: Path = ROOT / "logs" / "audit.jsonl"

    # --- ingestion (one chunk = one paragraph; only longer paragraphs/tables are split) ---
    chunk_size: int = 1500       # characters
    chunk_overlap: int = 150

    # --- retrieval ---
    top_k: int = 12              # paragraphs per company handed to the answer writer
    doc_min_similarity: float = 0.35   # dense cosine similarity gate
    doc_min_coverage: float = 0.5      # fraction of query terms present in chunk (lexical gate)

    # --- routing / SQL / validation ---
    router_min_confidence: float = 0.6
    sql_max_rows: int = 50
    sql_timeout_s: float = 5.0
    sql_allowed_tables: list[str] | None = None  # None = every table in the DB
    groundedness_threshold: float = 0.8

    @property
    def effective_embedding_provider(self) -> Provider:
        return self.embedding_provider or self.llm_provider

    @property
    def embedding_id(self) -> str:
        """Provider and model of the embeddings: an index only works with the model that built it."""
        p = self.effective_embedding_provider
        model = {"azure_openai": self.azure_openai_embedding_deployment, "openai": self.openai_embedding_model,
                 "openrouter": self.openrouter_embedding_model, "bedrock": self.bedrock_embedding_model_id}.get(p)
        return f"{p}:{model}" if model else p

    @property
    def index_path(self) -> Path:
        """One saved index per embedding model, so switching models (e.g. offline and back) reuses an earlier
        build instead of forcing a rebuild. The SQL database does not depend on embeddings and stays shared."""
        path = self.index_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", self.embedding_id)
        # An index built before per-model folders sits directly in index_dir: keep using it for its own model.
        legacy = self.index_dir / "catalog.json"
        if not path.exists() and legacy.exists():
            try:
                if json.loads(legacy.read_text()).get("embedding_model") == self.embedding_id:
                    return self.index_dir
            except ValueError:
                pass
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


def save_env(updates: dict[str, str], path: Path = ENV_FILE) -> None:
    """Update or add KEY=value lines in the .env file, keeping comments and other keys."""
    for key, value in updates.items():
        if "\n" in value or "'" in value:
            raise ValueError(f"{key}: value must not contain newlines or single quotes")
    remaining = dict(updates)
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in remaining:
            out.append(f"{key}='{remaining.pop(key)}'")
        else:
            out.append(line)
    out += [f"{k}='{v}'" for k, v in remaining.items()]
    path.write_text("\n".join(out) + "\n")
    get_settings.cache_clear()
