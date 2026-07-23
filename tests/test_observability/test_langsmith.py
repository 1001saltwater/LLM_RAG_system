from contextlib import contextmanager

from pydantic import SecretStr

from app.observability import langsmith as langsmith_module
from app.rag.embedding.vector_store import (
    sanitize_vector_search_inputs,
    summarize_vector_search_outputs,
)
from app.rag.generation.langchain_generator import (
    sanitize_trace_inputs,
    serialize_trace_output,
)
from app.schemas.schema_rag import RAGGenerationResult
from app.schemas.schema_search import SearchResult


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_langsmith_client_is_disabled_without_tracing(monkeypatch):
    langsmith_module.get_langsmith_client.cache_clear()
    monkeypatch.setattr(
        langsmith_module.settings,
        "LANGSMITH_TRACING",
        False,
    )

    assert langsmith_module.get_langsmith_client() is None


def test_langsmith_context_uses_explicit_client_and_project(monkeypatch):
    calls = []

    @contextmanager
    def fake_context(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(langsmith_module, "Client", FakeClient)
    monkeypatch.setattr(langsmith_module, "tracing_context", fake_context)
    monkeypatch.setattr(
        langsmith_module.settings,
        "LANGSMITH_TRACING",
        True,
    )
    monkeypatch.setattr(
        langsmith_module.settings,
        "LANGSMITH_API_KEY",
        SecretStr("test-key"),
    )
    langsmith_module.get_langsmith_client.cache_clear()

    try:
        with langsmith_module.langsmith_tracing_context(
            tags=["rag"],
            metadata={"environment": "test"},
        ):
            pass

        client = langsmith_module.get_langsmith_client()
        assert client.kwargs["api_key"] == "test-key"
        assert calls[0]["client"] is client
        assert calls[0]["project_name"] == (
            langsmith_module.settings.LANGSMITH_PROJECT
        )
        assert calls[0]["tags"] == ["rag"]
    finally:
        langsmith_module.get_langsmith_client.cache_clear()


def test_trace_processors_remove_sessions_vectors_and_document_content():
    sanitized = sanitize_trace_inputs(
        {
            "self": object(),
            "db": object(),
            "question": "question",
            "top_k": 5,
        }
    )
    assert sanitized == {"question": "question", "top_k": 5}

    output = serialize_trace_output(
        RAGGenerationResult(answer="answer", sources=[])
    )
    assert output == {"answer": "answer", "sources": []}

    vector_inputs = sanitize_vector_search_inputs(
        {
            "db": object(),
            "query_vector": [0.1, 0.2],
            "top_k": 3,
            "max_distance": 0.5,
        }
    )
    assert vector_inputs == {
        "top_k": 3,
        "max_distance": 0.5,
        "query_vector_dimension": 2,
    }

    vector_outputs = summarize_vector_search_outputs(
        [
            SearchResult(
                article_id=1,
                chunk_id=2,
                content="sensitive content",
                chunk_index=0,
                page_number=1,
                chunk_metadata={},
                distance=0.2,
            )
        ]
    )
    assert vector_outputs == {
        "result_count": 1,
        "results": [{"chunk_id": 2, "distance": 0.2}],
    }
