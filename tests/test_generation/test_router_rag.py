import asyncio
from contextlib import contextmanager

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.app import app
from app.rag.generation.langchain_generator import PreparedRAGContext
from app.routers import router_rag
from app.schemas.schema_rag import (
    RAGQueryRequest,
    RAGSource,
)


class FakeGenerator:
    calls: list[dict] = []

    def prepare(self, **kwargs) -> PreparedRAGContext:
        FakeGenerator.calls.append(kwargs)
        return PreparedRAGContext(
            documents=[
                Document(
                    page_content="参考资料",
                    metadata={"chunk_id": 2},
                )
            ],
            sources=[
                RAGSource(
                    article_id=1,
                    chunk_id=2,
                    page_number=3,
                    distance=0.2,
                    rerank_score=0.9,
                )
            ],
        )

    async def astream(self, question, prepared):
        assert question == "问题"
        assert prepared.sources[0].chunk_id == 2
        yield "流式"
        yield "回答"


@contextmanager
def fake_tracing_context(**_kwargs):
    yield


async def fake_run_in_threadpool(function):
    return function()


def test_rag_query_route_calls_langchain_generator(monkeypatch):
    FakeGenerator.calls = []
    monkeypatch.setattr(router_rag, "LangChainGenerator", FakeGenerator)
    monkeypatch.setattr(
        router_rag,
        "langsmith_tracing_context",
        fake_tracing_context,
    )
    monkeypatch.setattr(
        router_rag,
        "run_in_threadpool",
        fake_run_in_threadpool,
    )
    request = RAGQueryRequest(
        question="  问题  ",
        top_k=10,
        max_distance=0.4,
        rerank_top_k=2,
        rerank_threshold=0.6,
    )

    async def invoke_stream():
        response = await router_rag.query(request)
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            chunks.append(chunk)
        return response, "".join(chunks)

    response, body = asyncio.run(invoke_stream())

    call = FakeGenerator.calls[0]
    assert call["question"] == "问题"
    assert call["top_k"] == 10
    assert call["max_distance"] == 0.4
    assert call["rerank_top_k"] == 2
    assert call["rerank_threshold"] == 0.6
    assert response.media_type == "text/event-stream"
    assert 'event: token\ndata: {"content": "流式"}' in body
    assert 'event: token\ndata: {"content": "回答"}' in body
    assert '"chunk_id": 2' in body
    assert "event: done" in body


def test_rag_request_validation_and_route_registration():
    with pytest.raises(ValidationError):
        RAGQueryRequest(question="   ")

    assert "/rag/query" in app.openapi()["paths"]
