# app/services/service_embedding.py

from sqlalchemy.orm import Session

from app.models.model_embedding import Embedding
from app.schemas.schema_embedding import (
    CreateEmbedding,
    ResponseEmbedding,
)
from app.config.config import settings


class ServiceEmbedding:

    def create_embedding(self,db: Session,embedding_data: CreateEmbedding) -> ResponseEmbedding:

        db_embedding = Embedding(
            chunk_id=embedding_data.chunk_id,
            embedding=embedding_data.embedding,
        )

        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)

        return ResponseEmbedding.model_validate(db_embedding)

    def create_embedding_batch(self,db: Session,embedding_data: list[CreateEmbedding]) -> list[ResponseEmbedding]:

        db_embeddings = []
        for embedding_data in embedding_data:
            db_embedding = Embedding(
                chunk_id=embedding_data.chunk_id,
                embedding=embedding_data.embedding,
            )
            db_embeddings.append(db_embedding)
        db.add_all(db_embeddings)
        db.commit()

        return [ResponseEmbedding.model_validate(db_embedding) for db_embedding in db_embeddings]

    def get_embedding_by_chunk_id(self,db: Session,chunk_id: int) -> ResponseEmbedding | None:

        db_embedding = (
            db.query(Embedding).filter(Embedding.chunk_id == chunk_id).first()
        )

        if db_embedding is None:
            return None

        return ResponseEmbedding.model_validate(db_embedding)

    def delete_embedding(self,db: Session,chunk_id: int) -> bool:

        db_embedding = (
            db.query(Embedding).filter(Embedding.chunk_id == chunk_id).first()
        )

        if db_embedding is None:
            return False

        db.delete(db_embedding)
        db.commit()

        return True

    def exists_embedding(self,db: Session,chunk_id: int) -> bool:

        return (
            db.query(Embedding).filter(Embedding.chunk_id == chunk_id).first()
            is not None
        )

    def similarity_search(self, db: Session, query_vector: list[float], top_k: int | None = None,) -> list[ResponseEmbedding]:

        if top_k is None:
            top_k = settings.TOP_K

        db_embeddings = (db.query(Embedding).order_by(Embedding.embedding.cosine_distance(query_vector)).limit(top_k).all())

        return [
            ResponseEmbedding.model_validate(item)
            for item in db_embeddings
        ]