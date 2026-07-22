from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column, Mapped

from app.config.config import settings
from sqlalchemy import ForeignKey, UniqueConstraint
from app.models.base import Base

class Embedding(Base):
    __tablename__ = "embedding"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_embedding_chunk_id"),
    )

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"),
        index=True,
    )

    embedding : Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIM))
