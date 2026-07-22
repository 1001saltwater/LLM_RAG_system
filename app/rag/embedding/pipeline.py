from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.schemas.schema_embedding import CreateEmbedding
from app.services.service_article import ServiceArticle
from app.services.service_chunk import ServiceChunk
from app.services.service_embedding import ServiceEmbedding


class EmbeddingPipeline:
    def __init__(self, embedder: Embedder | None = None):
        self.article_service = ServiceArticle()
        self.chunk_service = ServiceChunk()
        self.embedding_service = ServiceEmbedding()
        self.embedder = embedder or Embedder()

    def embed_article(self, db: Session, article_id: int) -> int:
        article = self.article_service.get_article_by_id(db, article_id)
        if article is None:
            raise HTTPException(status_code=404, detail="Article not found")

        chunks = self.chunk_service.get_chunk_by_article_id(db, article_id)
        if not chunks:
            raise HTTPException(
                status_code=409,
                detail="Article has no chunks; ingest it first",
            )

        embedding_count = self.embedding_service.count_by_article_id(db, article_id)
        if embedding_count == len(chunks):
            raise HTTPException(
                status_code=409,
                detail="Article has already been embedded",
            )
        if embedding_count > 0:
            raise HTTPException(
                status_code=409,
                detail="Article embedding is incomplete; rebuild it before retrying",
            )

        vector_dimension = self.embedder.dimension()
        if vector_dimension != settings.EMBEDDING_DIM:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Embedding dimension mismatch: model={vector_dimension}, "
                    f"database={settings.EMBEDDING_DIM}"
                ),
            )

        embedding_data: list[CreateEmbedding] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents(
                [chunk.content for chunk in chunk_batch]
            )
            if len(vectors) != len(chunk_batch):
                raise HTTPException(
                    status_code=500,
                    detail="Embedding model returned an unexpected vector count",
                )

            for chunk, vector in zip(chunk_batch, vectors):
                if len(vector) != settings.EMBEDDING_DIM:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Embedding dimension mismatch for chunk {chunk.id}: "
                            f"expected={settings.EMBEDDING_DIM}, actual={len(vector)}"
                        ),
                    )
                embedding_data.append(
                    CreateEmbedding(
                        chunk_id=chunk.id,
                        embedding=vector,
                    )
                )

        try:
            self.embedding_service.create_embedding_batch(db, embedding_data)
        except IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="Article has already been embedded",
            ) from exc

        return len(embedding_data)