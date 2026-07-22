from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.routers import router_search
from app.schemas.schema_search import SearchRequest


class FakeLangChainRetriever:
    init_kwargs: dict = {}
    invoked_query: str | None = None

    def __init__(self, **kwargs):
        FakeLangChainRetriever.init_kwargs = kwargs

    def invoke(self, query: str) -> list[Document]:
        FakeLangChainRetriever.invoked_query = query
        return [
            Document(
                page_content="matched content",
                metadata={
                    "article_id": 1,
                    "chunk_id": 2,
                    "chunk_index": 3,
                    "page_number": 4,
                    "chunk_metadata": {"page": 3},
                    "distance": 0.2,
                },
            )
        ]


class FakeReranker:
    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeReranker.init_kwargs = kwargs


class FakeCompressionRetriever:
    def __init__(self, base_retriever, base_compressor):
        self.base_retriever = base_retriever
        self.base_compressor = base_compressor

    def invoke(self, query: str) -> list[Document]:
        documents = self.base_retriever.invoke(query)
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rerank_score": 0.9,
                },
            )
            for document in documents
        ]


def test_search_route_uses_langchain_retriever(monkeypatch):
    monkeypatch.setattr(
        router_search,
        "LangChainRetriever",
        FakeLangChainRetriever,
    )
    db = Session()
    request = SearchRequest(
        question="question",
        top_k=5,
        max_distance=0.4,
        rerank=False,
    )

    response = router_search.search(request, db)

    assert FakeLangChainRetriever.init_kwargs == {
        "db": db,
        "top_k": 5,
        "max_distance": 0.4,
    }
    assert FakeLangChainRetriever.invoked_query == "question"
    assert response.results[0].content == "matched content"
    assert response.results[0].chunk_id == 2
    assert response.results[0].distance == 0.2
    db.close()


def test_search_route_uses_contextual_compression_when_rerank_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        router_search,
        "LangChainRetriever",
        FakeLangChainRetriever,
    )
    monkeypatch.setattr(router_search, "LangChainReranker", FakeReranker)
    monkeypatch.setattr(
        router_search,
        "ContextualCompressionRetriever",
        FakeCompressionRetriever,
    )
    db = Session()
    request = SearchRequest(
        question="question",
        top_k=15,
        max_distance=0.5,
        rerank=True,
        rerank_top_k=3,
        rerank_threshold=0.6,
    )

    response = router_search.search(request, db)

    assert FakeReranker.init_kwargs == {
        "top_n": 3,
        "score_threshold": 0.6,
    }
    assert response.results[0].rerank_score == 0.9
    assert response.results[0].distance == 0.2
    db.close()
