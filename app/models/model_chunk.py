# app/models/model_article_chunks.py
from app.models.base import Base
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy import ForeignKey, Text, DateTime, JSON, UniqueConstraint
import datetime

class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("article_id", "chunk_index", name="uq_chunk_article_index"),
    )

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    article_id : Mapped[int] = mapped_column(ForeignKey("article.id"), index=True)
    content : Mapped[str] = mapped_column(Text, index=True)
    chunk_index : Mapped[int] = mapped_column(index=True)
    page_number : Mapped[int] = mapped_column(index=True)
    created_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now)
    updated_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now, onupdate=datetime.datetime.now)
    chunk_metadata : Mapped[dict] = mapped_column(JSON)
