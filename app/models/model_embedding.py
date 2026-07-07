from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column, Mapped

from app.config.config import settings
from sqlalchemy import ForeignKey
from app.models.base import Base

class Embedding(Base):
    __tablename__ = "embedding"

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunk.id"), index=True)

    embedding : Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIM))
   