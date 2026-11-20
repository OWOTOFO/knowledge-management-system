"""Provider wiring, checked offline: building the clients makes no network call."""

from km.config import Settings
from km.llm.providers import OPENROUTER_URL, get_embeddings, get_llm


def test_openrouter_uses_its_own_key_url_and_models(monkeypatch):
    monkeypatch.setattr("km.llm.providers._has_thinking_mode", lambda model: False)   # offline: no lookup
    s = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test", openai_api_key="sk-other",
                 _env_file=None)
    llm = get_llm(s)
    assert llm.name == "openrouter:deepseek/deepseek-v4.1-flash-1017"
    assert llm.chat.openai_api_base == OPENROUTER_URL
    assert llm.chat.openai_api_key.get_secret_value() == "sk-or-test"
    assert llm.chat.extra_body == {"provider": {"require_parameters": True}}   # structured output needs it
    emb = get_embeddings(s).inner   # the default Qwen model comes wrapped with its question instruction
    assert (emb.model, emb.openai_api_base) == ("qwen/qwen3-embedding-8b", OPENROUTER_URL)
    assert emb.check_embedding_ctx_length is False                          # send text, not token ids
    assert (emb.chunk_size, emb.max_retries) == (32, 10)                    # stay under provider rate limits


def test_plain_openai_is_unchanged():
    s = Settings(llm_provider="openai", openai_api_key="sk-test", _env_file=None)
    assert get_llm(s).chat.openai_api_base is None and get_llm(s).chat.extra_body is None
    assert get_embeddings(s).check_embedding_ctx_length is True


def test_thinking_switch_is_sent_only_to_models_that_have_one(monkeypatch):
    s = Settings(llm_provider="openrouter", openrouter_api_key="k", _env_file=None)
    monkeypatch.setattr("km.llm.providers._has_thinking_mode", lambda model: True)
    assert get_llm(s).chat.extra_body["reasoning"] == {"enabled": False}           # off by default: faster
    assert "reasoning" not in get_llm(s.model_copy(update={"llm_thinking": True})).chat.extra_body
    monkeypatch.setattr("km.llm.providers._has_thinking_mode", lambda model: False)
    assert "reasoning" not in get_llm(s).chat.extra_body                           # e.g. GPT-4.1 would fail


def test_qwen_questions_carry_the_task_instruction_but_documents_do_not():
    from km.llm.providers import QueryInstruction

    class Inner:
        def embed_documents(self, texts):
            return texts

        def embed_query(self, text):
            return text

    s = Settings(llm_provider="mock", embedding_provider="openrouter", openrouter_api_key="k",
                 openrouter_embedding_model="qwen/qwen3-embedding-8b", _env_file=None)
    emb = get_embeddings(s)
    assert isinstance(emb, QueryInstruction) and emb.inner.chunk_size == 32
    emb.inner = Inner()
    assert emb.embed_query("revenue?").startswith("Instruct: ") and emb.embed_query("revenue?").endswith("Query:revenue?")
    assert emb.embed_documents(["a paragraph"]) == ["a paragraph"]
