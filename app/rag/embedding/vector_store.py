# app/rag/embedding/vector_store.py

from sqlalchemy.orm import Session

from app.schemas.schema_embedding import CreateEmbedding
from app.services.service_embedding import ServiceEmbedding


class VectorStore:

    def __init__(self):
        self.embedding_service = ServiceEmbedding()

    def add_embedding(
        self,
        db: Session,
        embeddings: list[CreateEmbedding],
    ):
        self.embedding_service.create_embedding_batch(
            db,
            embeddings,
        )