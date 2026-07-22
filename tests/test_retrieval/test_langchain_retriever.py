import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.rag.embedding.vector_store import VectorStore
from app.rag.retrieval.langchain_retriever import LangChainRetriever
from app.schemas.schema_search import SearchResult


class FakeEmbedder(Embedder):
    def __init__(self, dimension: int = settings.EMBEDDING_DIM):
        self.dimension_value = dimension
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * self.dimension_value


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.calls: list[dict] = []

    def similarity_search(self, **kwargs) -> list[SearchResult]:
        self.calls.append(kwargs)
        return [
            SearchResult(
                article_id=1,
                chunk_id=2,
                content="matched content",
                chunk_index=3,
                page_number=4,
                chunk_metadata={"page": 3},
                distance=0.2,
            )
        ]


def test_langchain_retriever_returns_documents_with_search_metadata():
    db = Session()
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = LangChainRetriever(
        db=db,
        top_k=5,
        max_distance=0.4,
        embedder=embedder,
        vector_store=vector_store,
    )

    documents = retriever.invoke("  question  ")

    assert embedder.queries == ["question"]
    assert documents[0].page_content == "matched content"
    assert documents[0].metadata == {
        "article_id": 1,
        "chunk_id": 2,
        "chunk_index": 3,
        "page_number": 4,
        "chunk_metadata": {"page": 3},
        "distance": 0.2,
    }
    assert vector_store.calls[0]["db"] is db
    assert vector_store.calls[0]["top_k"] == 5
    assert vector_store.calls[0]["max_distance"] == 0.4
    db.close()


def test_langchain_retriever_validates_parameters_and_vector_dimension():
    db = Session()
    with pytest.raises(ValidationError):
        LangChainRetriever(db=db, top_k=0)

    retriever = LangChainRetriever(
        db=db,
        embedder=FakeEmbedder(settings.EMBEDDING_DIM - 1),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(RuntimeError):
        retriever.invoke("question")
    db.close()
