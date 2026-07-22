import pytest
from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)
from langchain_core.documents import Document

from app.rag.rerank.langchain_rerank import LangChainReranker


class FakeCrossEncoder(BaseCrossEncoder):
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = text_pairs
        return self.scores


def test_reranker_sorts_filters_and_preserves_document_metadata():
    model = FakeCrossEncoder([0.7, 0.9, 0.2])
    reranker = LangChainReranker(
        model=model,
        top_n=2,
        score_threshold=0.5,
    )
    documents = [
        Document(page_content="same content", metadata={"chunk_id": 1}),
        Document(page_content="same content", metadata={"chunk_id": 2}),
        Document(page_content="other content", metadata={"chunk_id": 3}),
    ]

    results = reranker.rerank(" question ", documents)

    assert model.pairs == [
        ("question", "same content"),
        ("question", "same content"),
        ("question", "other content"),
    ]
    assert [document.metadata["chunk_id"] for document in results] == [2, 1]
    assert [document.metadata["rerank_score"] for document in results] == [
        0.9,
        0.7,
    ]
    assert documents[0].metadata == {"chunk_id": 1}


def test_reranker_handles_empty_documents_and_invalid_model_output():
    reranker = LangChainReranker(
        model=FakeCrossEncoder([]),
        top_n=1,
        score_threshold=None,
    )
    assert reranker.rerank("question", []) == []

    with pytest.raises(ValueError):
        reranker.rerank("   ", [])

    invalid_reranker = LangChainReranker(
        model=FakeCrossEncoder([]),
        top_n=1,
        score_threshold=None,
    )
    with pytest.raises(RuntimeError):
        invalid_reranker.rerank(
            "question",
            [Document(page_content="content")],
        )
