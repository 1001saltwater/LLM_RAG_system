import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.app import app
from app.routers import router_rag
from app.schemas.schema_rag import (
    RAGGenerationResult,
    RAGQueryRequest,
    RAGSource,
)


class FakeGenerator:
    calls: list[dict] = []

    def generate(self, **kwargs) -> RAGGenerationResult:
        FakeGenerator.calls.append(kwargs)
        return RAGGenerationResult(
            answer="回答。[资料1]",
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


def test_rag_query_route_calls_langchain_generator(monkeypatch):
    FakeGenerator.calls = []
    monkeypatch.setattr(router_rag, "LangChainGenerator", FakeGenerator)
    db = Session()
    request = RAGQueryRequest(
        question="  问题  ",
        top_k=10,
        max_distance=0.4,
        rerank_top_k=2,
        rerank_threshold=0.6,
    )

    response = router_rag.query(request, db)

    assert FakeGenerator.calls == [
        {
            "db": db,
            "question": "问题",
            "top_k": 10,
            "max_distance": 0.4,
            "rerank_top_k": 2,
            "rerank_threshold": 0.6,
        }
    ]
    assert response.answer == "回答。[资料1]"
    assert response.sources[0].chunk_id == 2
    db.close()


def test_rag_request_validation_and_route_registration():
    with pytest.raises(ValidationError):
        RAGQueryRequest(question="   ")

    assert "/rag/query" in app.openapi()["paths"]
