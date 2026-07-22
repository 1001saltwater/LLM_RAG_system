from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.rag.embedding.vector_store import VectorStore


class LangChainRetriever(BaseRetriever):
    """Adapter that exposes the existing pgvector search as a LangChain retriever."""

    db: Session = Field(exclude=True)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=100)
    max_distance: float = Field(
        default=settings.THRESHOLD,
        ge=0.0,
        le=2.0,
    )
    embedder: Embedder = Field(default_factory=Embedder)
    vector_store: VectorStore = Field(default_factory=VectorStore)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")

        query_vector = self.embedder.embed_query(query)
        if len(query_vector) != settings.EMBEDDING_DIM:
            raise RuntimeError(
                "Query embedding dimension does not match the database dimension"
            )

        results = self.vector_store.similarity_search(
            db=self.db,
            query_vector=query_vector,
            top_k=self.top_k,
            max_distance=self.max_distance,
        )
        return [
            Document(
                page_content=result.content,
                metadata=result.model_dump(
                    exclude={"content"},
                    exclude_none=True,
                ),
            )
            for result in results
        ]
