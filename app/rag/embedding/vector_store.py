# app/rag/embedding/vector_store.py

from sqlalchemy.orm import Session

from app.schemas.schema_embedding import (
    CreateEmbedding,
    ResponseEmbedding,
)

from app.services.service_embedding import ServiceEmbedding


class VectorStore:

    def __init__(self):
        self.embedding_service = ServiceEmbedding()

    def add_embedding_batch(
        self,
        db: Session,
        embeddings: list[CreateEmbedding],
    ) -> list[ResponseEmbedding]:

        return self.embedding_service.create_embedding_batch(
            db,
            embeddings,
        )

    def similarity_search(
        self,
        db: Session,
        query_vector: list[float],
        top_k: int,
    ) -> list[ResponseEmbedding]:

        return self.embedding_service.similarity_search(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
        )

    def get_embedding_by_chunk_id(
        self,
        db: Session,
        chunk_id: int,
    ) -> ResponseEmbedding | None:

        return self.embedding_service.get_embedding_by_chunk_id(
            db,
            chunk_id,
        )

    def delete_embedding(
        self,
        db: Session,
        chunk_id: int,
    ) -> bool:

        return self.embedding_service.delete_embedding(
            db,
            chunk_id,
        )

    def exists_embedding(
        self,
        db: Session,
        chunk_id: int,
    ) -> bool:

        return self.embedding_service.exists_embedding(
            db,
            chunk_id,
        )