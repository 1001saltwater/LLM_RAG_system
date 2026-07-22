from collections.abc import Sequence
from functools import lru_cache

from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field

from app.config.config import settings


@lru_cache(maxsize=1)
def get_cross_encoder() -> BaseCrossEncoder:
    return HuggingFaceCrossEncoder(
        model_name=settings.RERANK_MODEL,
        model_kwargs={"device": settings.DEVICE},
    )


class LangChainReranker(BaseDocumentCompressor):
    """Rerank LangChain Documents while preserving scores and metadata."""

    model: BaseCrossEncoder = Field(default_factory=get_cross_encoder)
    top_n: int = Field(default=settings.RERANK_TOP_K, ge=1)
    score_threshold: float | None = settings.RERANK_THRESHOLD

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if not documents:
            return []

        scores = self.model.score(
            [(query, document.page_content) for document in documents]
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                "Rerank model returned an unexpected score count"
            )

        ranked_documents = []
        for document, score in zip(documents, scores):
            rerank_score = float(score)
            if (
                self.score_threshold is not None
                and rerank_score < self.score_threshold
            ):
                continue
            ranked_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "rerank_score": rerank_score,
                    },
                )
            )

        ranked_documents.sort(
            key=lambda document: document.metadata["rerank_score"],
            reverse=True,
        )
        return ranked_documents[: self.top_n]

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[Document]:
        return list(self.compress_documents(documents, query))
