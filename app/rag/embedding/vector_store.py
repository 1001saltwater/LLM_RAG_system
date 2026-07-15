from app.models.model_embedding import Embedding
from app.schemas.schema_embedding import (
    ResponseEmbedding,
)
from app.config.config import settings
from sqlalchemy.orm import Session

class VectorStore:

    def similarity_search(self, db: Session, query_vector: list[float], top_k: int | None = None,) -> list[ResponseEmbedding]:

        if top_k is None:
            top_k = settings.TOP_K
        threshold = settings.THRESHOLD

        db_embeddings = (db.query(Embedding).filter(Embedding.embedding.cosine_distance(query_vector) <= threshold).order_by(Embedding.embedding.cosine_distance(query_vector)).limit(top_k).all())

        return [
            ResponseEmbedding.model_validate(item)
            for item in db_embeddings
        ]