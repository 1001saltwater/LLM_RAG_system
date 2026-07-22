from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.app import app
from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.rag.embedding.vector_store import VectorStore
from app.rag.retrieval.retriever import Retriever
from app.schemas.schema_search import SearchRequest, SearchResult


class FakeEmbedder:
    def __init__(self, dimension: int = settings.EMBEDDING_DIM):
        self.dimension = dimension
        self.questions: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.questions.append(question)
        return [0.1] * self.dimension


class FakeVectorStore:
    def __init__(self):
        self.calls: list[dict] = []

    def similarity_search(self, **kwargs) -> list[SearchResult]:
        self.calls.append(kwargs)
        return [
            SearchResult(
                article_id=1,
                chunk_id=2,
                content="matched content",
                chunk_index=0,
                page_number=1,
                chunk_metadata={"page": 0},
                distance=0.2,
            )
        ]


def test_retriever_embeds_query_and_forwards_search_parameters():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(
        embedder=cast(Embedder, embedder),
        vector_store=cast(VectorStore, vector_store),
    )
    db = cast(Session, object())

    results = retriever.retrieve(
        db=db,
        question="  test question  ",
        top_k=3,
        max_distance=0.4,
    )

    assert embedder.questions == ["test question"]
    assert results[0].distance == 0.2
    assert vector_store.calls[0]["db"] is db
    assert vector_store.calls[0]["top_k"] == 3
    assert vector_store.calls[0]["max_distance"] == 0.4
    assert len(vector_store.calls[0]["query_vector"]) == settings.EMBEDDING_DIM


def test_retriever_rejects_wrong_query_vector_dimension():
    retriever = Retriever(
        embedder=cast(Embedder, FakeEmbedder(settings.EMBEDDING_DIM - 1)),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )

    with pytest.raises(RuntimeError):
        retriever.retrieve(
            db=cast(Session, object()),
            question="question",
        )


def test_vector_search_statement_uses_cosine_distance_order_and_limit():
    statement = VectorStore.build_similarity_statement(
        query_vector=[0.1] * settings.EMBEDDING_DIM,
        top_k=5,
        max_distance=0.5,
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "<=>" in sql
    assert "JOIN embedding" in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_search_request_validation_and_route_registration():
    request = SearchRequest(
        question="  question  ",
        top_k=5,
        max_distance=0.5,
    )
    assert request.question == "question"

    with pytest.raises(ValidationError):
        SearchRequest(question="   ")

    assert "/search" in app.openapi()["paths"]
