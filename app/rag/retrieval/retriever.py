from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.rag.embedding.vector_store import VectorStore
from app.schemas.schema_search import SearchResult


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        db: Session,
        question: str,
        top_k: int = settings.TOP_K,
        max_distance: float = settings.THRESHOLD,
    ) -> list[SearchResult]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")

        query_vector = self.embedder.embed_query(question)
        if len(query_vector) != settings.EMBEDDING_DIM:
            raise RuntimeError(
                "Query embedding dimension does not match the database dimension"
            )

        return self.vector_store.similarity_search(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
            max_distance=max_distance,
        )
