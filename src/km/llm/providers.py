"""Provider factory: one place that knows about Azure OpenAI / OpenAI / OpenRouter / Bedrock / mock.

Everything else in the codebase talks to the small `LLM` wrapper below, so swapping the
bank's approved endpoint is a config change, not a code change.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TypeVar
from urllib.parse import urlsplit
from urllib.request import urlopen

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from km.config import Settings
from km.llm.mock import HashingEmbeddings

T = TypeVar("T", bound=BaseModel)
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Embedding models trained to read a task instruction in front of the QUESTION (never the documents).
QUERY_INSTRUCTIONS = {
    "qwen3-embedding": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
}


class QueryInstruction(Embeddings):
    """Put the model's task instruction in front of questions; documents are embedded as they are,
    so adding or changing it needs no index rebuild."""

    def __init__(self, inner: Embeddings, instruction: str):
        self.inner, self.instruction = inner, instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.instruction + text)


class LLM:
    """Thin wrapper exposing the two calls the pipeline needs: free text and structured output."""

    def __init__(self, chat: BaseChatModel, name: str):
        self.chat = chat
        self.name = name

    def text(self, system: str, user: str) -> str:
        msg = self.chat.invoke([SystemMessage(system), HumanMessage(user)])
        content = msg.content
        if isinstance(content, list):  # e.g. Bedrock returns content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content).strip()

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        runnable = self.chat.with_structured_output(schema)
        return runnable.invoke([SystemMessage(system), HumanMessage(user)])


def _temperature(s: Settings) -> dict:
    return {} if s.llm_temperature is None else {"temperature": s.llm_temperature}


def _azure_auth(s: Settings) -> dict:
    if s.azure_openai_api_key:
        return {"api_key": s.azure_openai_api_key}
    # Keyless auth (Entra ID / managed identity) - the usual pattern inside a bank tenant.
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return {"azure_ad_token_provider": provider}


@lru_cache
def _has_thinking_mode(model: str) -> bool:
    """Whether OpenRouter lists a 'reasoning' switch for the model. Sending the switch to a model without one
    fails ("No endpoints found"), so on any doubt (offline, unknown model) it is not sent."""
    try:
        with urlopen(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=10) as r:
            endpoints = json.load(r)["data"]["endpoints"]
        return any("reasoning" in e.get("supported_parameters", []) for e in endpoints)
    except Exception:
        return False


def _openai_compatible(s: Settings, p: str) -> tuple[str | None, str | None]:
    """(api_key, base_url) for the OpenAI-protocol providers."""
    return (s.openrouter_api_key, OPENROUTER_URL) if p == "openrouter" else (s.openai_api_key, s.openai_base_url)


def get_llm(s: Settings) -> LLM | None:
    """Return the configured chat model, or None in mock mode (components use heuristics)."""
    p = s.llm_provider
    if p == "mock":
        return None
    if p == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        chat = AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_openai_chat_deployment,
            **_azure_auth(s),
            **_temperature(s),
        )
        return LLM(chat, f"azure_openai:{s.azure_openai_chat_deployment}")
    if p in ("openai", "openrouter"):
        from langchain_openai import ChatOpenAI

        api_key, base_url = _openai_compatible(s, p)
        model = s.openrouter_chat_model if p == "openrouter" else s.openai_chat_model
        gateway_options = {}
        if urlsplit(base_url or "").hostname == "openrouter.ai":
            # Route structured requests only to upstreams that support the requested parameters.
            gateway_options = {"use_responses_api": False,
                               "extra_body": {"provider": {"require_parameters": True}}}
            if not s.llm_thinking and _has_thinking_mode(model):
                gateway_options["extra_body"]["reasoning"] = {"enabled": False}
        chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=60,      # one slow upstream must not hold a question for minutes;
            max_retries=2,   # a retry usually lands on another provider
            **_temperature(s),
            **gateway_options,
        )
        return LLM(chat, f"{p}:{model}")
    if p == "bedrock":
        from langchain_aws import ChatBedrockConverse

        chat = ChatBedrockConverse(
            model=s.bedrock_chat_model_id, region_name=s.aws_region, **_temperature(s)
        )
        return LLM(chat, f"bedrock:{s.bedrock_chat_model_id}")
    raise ValueError(f"Unknown LLM_PROVIDER: {p}")


def get_embeddings(s: Settings) -> Embeddings:
    p = s.effective_embedding_provider
    if p == "mock":
        return HashingEmbeddings()
    if p == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_openai_embedding_deployment,
            **_azure_auth(s),
        )
    if p in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings

        api_key, base_url = _openai_compatible(s, p)
        model = s.openrouter_embedding_model if p == "openrouter" else s.openai_embedding_model
        instruction = next((text for key, text in QUERY_INSTRUCTIONS.items() if key in model.lower()), "")
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            # Gateways (OpenRouter, internal proxies) expect plain text; only OpenAI itself takes token ids.
            check_embedding_ctx_length=base_url is None,
            # Small requests and patient retries: upstream providers cap tokens per request / per minute (HTTP 429).
            chunk_size=32,
            max_retries=10,
        )
        return QueryInstruction(embeddings, instruction) if instruction else embeddings
    if p == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=s.bedrock_embedding_model_id, region_name=s.aws_region)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {p}")
